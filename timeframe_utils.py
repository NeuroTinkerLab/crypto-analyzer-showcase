# --- START OF FILE timeframe_utils.py ---
# timeframe_utils.py
"""
Modulo per analizzare timeframe superiori e fornire contesto Multi-Timeframe (MTF).
"""
import logging
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any, Optional

# Importa funzioni e configurazioni necessarie
try:
    from data_collector import get_data_with_cache
except ImportError:
     logging.critical("ERRORE CRITICO: Impossibile importare data_collector in timeframe_utils.")
     def get_data_with_cache(*args, **kwargs) -> Optional[pd.DataFrame]:
         logging.error("Funzione get_data_with_cache fittizia chiamata!")
         return None

try:
    # Usa nuovo nome costante da config.py
    from config import DATA_LOOKBACK_PERIODS as BACKTEST_PERIODS, MTF_TARGET_TIMEFRAMES
    if 'BACKTEST_PERIODS' not in locals() or not isinstance(BACKTEST_PERIODS, dict): raise ImportError
    if 'MTF_TARGET_TIMEFRAMES' not in locals() or not isinstance(MTF_TARGET_TIMEFRAMES, list): MTF_TARGET_TIMEFRAMES = ['1h', '4h', '1d', '1w'] # Fallback
except ImportError:
    logging.warning("config.py (o costanti necessarie) non trovato in timeframe_utils. Uso fallback.")
    BACKTEST_PERIODS = {tf: 365 for tf in ['1h', '4h', '1d']}
    BACKTEST_PERIODS.update({'1w': 1825, '1M': 2555}) # Lookback più lunghi
    MTF_TARGET_TIMEFRAMES = ['1h', '4h', '1d', '1w'] # Fallback

try: import pandas_ta as ta
except ImportError: logging.critical("ERRORE CRITICO: pandas_ta mancante."); ta = None

logger = logging.getLogger(__name__)
# Import helper
try:
    from statistical_analyzer_helpers import _safe_float
except ImportError:
    def _safe_float(v, d=None):
        return float(v) if pd.notna(v) else d  # fallback

def analyze_higher_timeframes(
    symbol: str,
    current_timeframe: str,
    higher_timeframes_to_analyze: List[str] = MTF_TARGET_TIMEFRAMES # Usa da config o fallback
) -> Dict[str, Dict[str, Any]]:
    """
    Analizza trend, indicatori e livelli chiave per i timeframe superiori specificati.

    Args:
        symbol (str): Simbolo dell'asset (es. 'BTC/USDT').
        current_timeframe (str): Timeframe corrente dell'analisi principale (per logging e filtro).
        higher_timeframes_to_analyze (List[str]): Lista dei timeframe superiori da analizzare.

    Returns:
        Dict[str, Dict[str, Any]]: Dizionario {htf: {risultati_analisi}}.
                                   Include {'error': ...} se l'analisi per un TF fallisce.
    """
    htf_base_log_msg = f"Analisi HTF per {symbol} (Base TF: {current_timeframe})"

    # --- CORREZIONE: Filtra timeframe corrente dalla lista ---
    filtered_htfs = [htf for htf in higher_timeframes_to_analyze if htf != current_timeframe]
    if not filtered_htfs:
        logger.info(f"{htf_base_log_msg} - Nessun timeframe superiore specificato o tutti uguali al corrente. Salto analisi MTF.")
        return {} # Restituisce dizionario vuoto se non ci sono HTF da analizzare

    logger.debug(f"{htf_base_log_msg} - Avvio analisi per HTFs effettivi: {filtered_htfs}")
    htf_results: Dict[str, Dict[str, Any]] = {}
    if ta is None:
        logger.error(f"{htf_base_log_msg} - pandas_ta mancante."); return {htf: {'error': 'pandas_ta missing'} for htf in filtered_htfs}

    for htf in filtered_htfs: # Itera sulla lista filtrata
        htf_results[htf] = {'error': None}
        logger.debug(f"{htf_base_log_msg} - Analisi Timeframe: {htf}")

        # --- Calcolo Giorni Necessari ---
        candles_needed_rough = 250; days_needed = 365 # Default
        try:
            days_needed_config = BACKTEST_PERIODS.get(htf)
            if days_needed_config is not None and days_needed_config > 0:
                days_needed = days_needed_config
                logger.debug(f"({htf}) Usando lookback config: {days_needed} giorni.")
            else: # Stima se non in config
                unit = htf[-1].lower(); val_str = htf[:-1]; val = int(val_str) if val_str.isdigit() else 1
                # --- CORREZIONE H -> h --- (Già presente implicitamente se unit è 'h')
                if unit == 'm': days_per_candle = val / (24 * 60)
                elif unit == 'h': days_per_candle = val / 24
                elif unit == 'd': days_per_candle = val
                elif unit == 'w': days_per_candle = val * 7
                elif unit == 'M': days_per_candle = val * 30.5 # Stima
                else: raise ValueError("Unità timeframe non riconosciuta")
                days_needed = math.ceil(candles_needed_rough * days_per_candle)
                logger.debug(f"({htf}) Lookback stimato: {days_needed} giorni.")

            # --- CORREZIONE: Rimuovi buffer +30 ---
            # Assicura un minimo ragionevole, ma senza buffer aggiuntivo
            days_needed = max(days_needed, 90) # Minimo 90 giorni

        except Exception as e: logger.warning(f"({htf}) Errore calcolo days_needed: {e}. Uso default 365."); days_needed = 365
        logger.debug(f"({htf}) Giorni storico finali richiesti: {days_needed}")

        # --- Recupero Dati HTF ---
        df_htf: Optional[pd.DataFrame] = None
        try:
            # Passa il days_needed calcolato SENZA buffer aggiuntivo
            df_htf = get_data_with_cache(symbol, htf, days=days_needed, force_fetch=False)
            if df_htf is None or df_htf.empty: raise ValueError(f"Dati {htf} vuoti.")
            min_htf_rows = 50 # Minimo per indicatori come SMA 50/ADX etc.
            if len(df_htf) < min_htf_rows: raise ValueError(f"Dati {htf} insuff. ({len(df_htf)} < {min_htf_rows}).")
            # (Il resto della preparazione dati rimane invariato)
            df_htf.columns = [str(col).lower().replace(' ','_') for col in df_htf.columns]
            required_cols = ['open', 'high', 'low', 'close'];
            if not all(c in df_htf.columns for c in required_cols): raise ValueError(f"Colonne OHLC mancanti dati {htf}.")
            for col in required_cols + ['volume']:
                 if col in df_htf.columns: df_htf[col] = pd.to_numeric(df_htf[col], errors='coerce')
            df_htf.dropna(subset=required_cols, inplace=True)
            if df_htf.empty: raise ValueError(f"Dati {htf} vuoti dopo pulizia.")
        except Exception as e: logger.error(f"Errore dati HTF {htf}: {e}"); htf_results[htf]['error'] = f"Errore dati: {e}"; continue

        # --- Calcolo Indicatori HTF (Invariato) ---
        try:
            logger.debug(f"({htf}) Calcolo indicatori pandas_ta...")
            custom_strategy = ta.Strategy(
                name="HTF_Context_Analysis", description="SMA, EMA, RSI, MACD, ADX, ATR",
                ta=[ {"kind": "sma", "length": 50}, {"kind": "sma", "length": 200},
                     {"kind": "ema", "length": 50}, {"kind": "rsi", "length": 14},
                     {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
                     {"kind": "adx", "length": 14}, {"kind": "atr", "length": 14} ])
            df_htf.ta.strategy(custom_strategy, append=True)

            # --- DEBUG COLONNE ATR (Mantenuto) ---
            logger.debug(f"({htf}) Colonne DataFrame DOPO ta.strategy: {df_htf.columns.tolist()}")
            atr_col_name = None
            potential_atr_cols = ['ATR_14', 'ATRr_14', 'ATRe_14']
            for col in potential_atr_cols:
                if col in df_htf.columns: atr_col_name = col; logger.debug(f"({htf}) Colonna ATR trovata: '{atr_col_name}'"); break
            if atr_col_name is None: logger.warning(f"({htf}) Colonna ATR non trovata tra {potential_atr_cols}. ATR sarà None.")
            # --- FINE DEBUG COLONNE ATR ---

            expected_bases = ['SMA_50', 'SMA_200', 'EMA_50', 'RSI_14', 'ADX_14']
            expected_macd = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
            missing_ta_cols = [c for c in expected_bases if c not in df_htf.columns]
            if not any(c in df_htf.columns for c in expected_macd): missing_ta_cols.append("MACD Group")
            if atr_col_name is None: missing_ta_cols.append("ATR (nome non trovato)")
            if missing_ta_cols: logger.warning(f"({htf}) Colonne pandas_ta mancanti: {missing_ta_cols}.")

        except Exception as e: logger.error(f"Errore indicatori pandas_ta HTF {htf}: {e}", exc_info=True); htf_results[htf]['error'] = f"Errore indicatori: {e}"; continue

        # --- Estrazione Risultati (Invariata) ---
        try:
            latest = df_htf.iloc[-1]
            indicators_htf = {
                'sma_50': _safe_float(latest.get('SMA_50')), 'sma_200': _safe_float(latest.get('SMA_200')),
                'ema_50': _safe_float(latest.get('EMA_50')), 'rsi': _safe_float(latest.get('RSI_14')),
                'macd': _safe_float(latest.get('MACD_12_26_9')), 'macd_hist': _safe_float(latest.get('MACDh_12_26_9')),
                'macd_signal': _safe_float(latest.get('MACDs_12_26_9')), 'adx': _safe_float(latest.get('ADX_14')),
                'plus_di': _safe_float(latest.get('DMP_14')), 'minus_di': _safe_float(latest.get('DMN_14')),
                'atr': _safe_float(latest.get(atr_col_name)) if atr_col_name else None
            }
            htf_results[htf]['indicators'] = indicators_htf
            # Trend MA, ADX, Combinato (logica invariata)
            sma50, sma200, last_close_htf = indicators_htf.get('sma_50'), indicators_htf.get('sma_200'), _safe_float(latest.get('close'))
            htf_ma_trend = "Unknown"
            if all(v is not None for v in [last_close_htf, sma50, sma200]):
                 if last_close_htf > sma50 > sma200: htf_ma_trend = "Strong Bullish (Price > SMA50 > SMA200)"
                 elif last_close_htf < sma50 < sma200: htf_ma_trend = "Strong Bearish (Price < SMA50 < SMA200)"
                 # Rimosse condizioni intermedie ridondanti, semplificato
                 elif last_close_htf > sma50 and last_close_htf > sma200 : htf_ma_trend = "Bullish (Price > SMA50 & SMA200)"
                 elif last_close_htf < sma50 and last_close_htf < sma200 : htf_ma_trend = "Bearish (Price < SMA50 & SMA200)"
                 elif last_close_htf > sma50: htf_ma_trend = "Weak Bullish (Price > SMA50 only)"
                 elif last_close_htf < sma50: htf_ma_trend = "Weak Bearish (Price < SMA50 only)"
                 else: htf_ma_trend = "Neutral/Sideways"
            elif all(v is not None for v in [last_close_htf, sma50]): htf_ma_trend = "Bullish (vs SMA50)" if last_close_htf > sma50 else "Bearish (vs SMA50)"
            htf_results[htf]['trend_ma'] = htf_ma_trend

            adx_value, plus_di, minus_di = indicators_htf.get('adx'), indicators_htf.get('plus_di'), indicators_htf.get('minus_di')
            htf_adx_strength, htf_adx_direction = "Unknown", "Unknown"
            if adx_value is not None:
                if adx_value < 15: htf_adx_strength = "Very Weak/Absent"
                elif adx_value < 20: htf_adx_strength = "Weak/Developing"
                elif adx_value < 40: htf_adx_strength = "Moderate"
                elif adx_value < 50: htf_adx_strength = "Strong"
                else: htf_adx_strength = "Very Strong"
            if plus_di is not None and minus_di is not None: htf_adx_direction = "Bullish" if plus_di > minus_di else "Bearish" if minus_di > plus_di else "Neutral"
            htf_results[htf]['adx_strength'] = htf_adx_strength; htf_results[htf]['adx_direction'] = htf_adx_direction

            combined_trend = "Unknown"
            # Logica combinata migliorata: Priorità ad ADX forte, poi MA, poi ADX debole
            if htf_adx_strength in ("Strong", "Very Strong") and htf_adx_direction != 'Neutral':
                combined_trend = f"{htf_adx_direction} ({htf_adx_strength} ADX)"
            elif htf_ma_trend not in ('Unknown', 'Neutral/Sideways'):
                ma_base_trend = "Bullish" if "Bullish" in htf_ma_trend else "Bearish" if "Bearish" in htf_ma_trend else "Neutral"
                if htf_adx_strength == "Moderate" and htf_adx_direction == ma_base_trend:
                    combined_trend = f"{ma_base_trend} (MA Confirmed by Moderate ADX)"
                else:
                    combined_trend = f"{ma_base_trend} (Based on MA, ADX {htf_adx_strength})"
            elif htf_adx_strength == "Moderate" and htf_adx_direction != 'Neutral':
                 combined_trend = f"{htf_adx_direction} (Moderate ADX, MA Neutral/Unknown)"
            else: combined_trend = "Neutral/Sideways"
            htf_results[htf]['trend_combined'] = combined_trend

            # Livelli Chiave (logica invariata)
            recent_period_htf = 30; recent_high, recent_low = None, None
            if len(df_htf) >= recent_period_htf: recent_data_htf = df_htf.iloc[-recent_period_htf:]; recent_high = recent_data_htf['high'].max(); recent_low = recent_data_htf['low'].min()
            else: recent_high = df_htf['high'].max(); recent_low = df_htf['low'].min()
            htf_results[htf]['key_levels'] = {'sma_50': sma50, 'sma_200': sma200, 'ema_50': indicators_htf.get('ema_50'), 'recent_high_30p': _safe_float(recent_high), 'recent_low_30p': _safe_float(recent_low)}
            logger.info(f"Analisi HTF {htf} completata. Trend Combinato: {combined_trend}, ADX: {htf_adx_strength}/{htf_adx_direction}")
            htf_results[htf].pop('error', None)
        except Exception as e: logger.error(f"Errore estrazione risultati HTF {htf}: {e}", exc_info=True); htf_results[htf]['error'] = f"Errore estrazione: {e}"

    logger.debug(f"{htf_base_log_msg} - Analisi timeframe superiori completata.")
    return htf_results

# --- END OF FILE timeframe_utils.py ---