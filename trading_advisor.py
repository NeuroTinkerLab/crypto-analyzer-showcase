# --- START OF FILE trading_advisor.py ---
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta # Aggiunto timedelta
import logging
from typing import Optional, Dict, Any, List, Tuple
import ccxt # type: ignore
import sys
import time
import re
from collections import defaultdict
import os # Aggiunto per gestione path file storico

# Importa moduli locali necessari
from statistical_analyzer_advanced import StatisticalAnalyzerAdvanced
from statistical_analyzer_patterns import PatternAnalyzer
from fibonacci_calculator import (
    get_fibonacci_retracements,
    get_fibonacci_extensions,
    calculate_fibonacci_time_zones
)
from data_collector import get_last_n_candles_multiple_tf, get_exchange_instance
from timeframe_utils import analyze_higher_timeframes

# Import Deribit (con fallback)
try:
    from deribit_collector import (
        get_deribit_ticker_data,
        get_deribit_funding_rate_history,
        get_deribit_liquidations,
        get_deribit_trade_volumes,
        find_instrument_details,
        get_deribit_options_summary
    )
    DERIBIT_COLLECTOR_LOADED = True
except ImportError:
    logging.error("ERRORE CRITICO: Impossibile importare deribit_collector. I dati Deribit non saranno disponibili.")
    # Funzioni fallback
    def get_deribit_ticker_data(*args, **kwargs): return {"error": "deribit_collector not loaded"} # type: ignore
    def get_deribit_funding_rate_history(*args, **kwargs): return {"history": [], "error": "deribit_collector not loaded"} # type: ignore
    def get_deribit_liquidations(*args, **kwargs): return {"liquidations": [], "error": "deribit_collector not loaded"} # type: ignore
    def get_deribit_trade_volumes(*args, **kwargs): return {"futures_volume_usd": None, "options_volume_usd": None, "error": "deribit_collector not loaded"} # type: ignore
    def find_instrument_details(*args, **kwargs): return [{"error": "deribit_collector not loaded"}] # type: ignore
    def get_deribit_options_summary(*args, **kwargs): return [{"error": "deribit_collector not loaded"}] # type: ignore
    DERIBIT_COLLECTOR_LOADED = False

# Importa config e helpers (con fallback)
try:
    # Importa le costanti da config
    import config
    FIB_LOOKBACK_PERIOD = config.FIB_LOOKBACK_PERIOD
    MTF_TARGET_TIMEFRAMES = config.MTF_TARGET_TIMEFRAMES
    HISTORICAL_STATS_DIR = config.HISTORICAL_STATS_DIR
    DERIBIT_HISTORY_MAX_ENTRIES = config.DERIBIT_HISTORY_MAX_ENTRIES
    DERIBIT_HISTORY_AVG_PERIOD = config.DERIBIT_HISTORY_AVG_PERIOD
    TA_ATR_TARGET_MULTIPLIERS = config.TA_ATR_TARGET_MULTIPLIERS
    FIB_NEARNESS_THRESHOLD = config.FIB_NEARNESS_THRESHOLD
    VP_NEARNESS_THRESHOLD = config.VP_NEARNESS_THRESHOLD
    SR_NEARNESS_THRESHOLD = config.SR_NEARNESS_THRESHOLD
    MA_NEARNESS_THRESHOLD = config.MA_NEARNESS_THRESHOLD
    LEVEL_NEARNESS_THRESHOLD_ATR = config.LEVEL_NEARNESS_THRESHOLD_ATR
    CONFIG_LOADED = True
except (ImportError, AttributeError):
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.warning("Import/Accesso costanti da config fallito in trading_advisor. Uso fallback.")
    FIB_LOOKBACK_PERIOD = 90
    MTF_TARGET_TIMEFRAMES = ['1h', '4h', '1d', '1w']
    HISTORICAL_STATS_DIR = "historical_stats"
    DERIBIT_HISTORY_MAX_ENTRIES = 180
    DERIBIT_HISTORY_AVG_PERIOD = 30
    TA_ATR_TARGET_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
    FIB_NEARNESS_THRESHOLD = 0.015
    VP_NEARNESS_THRESHOLD = 0.010
    SR_NEARNESS_THRESHOLD = 0.010
    MA_NEARNESS_THRESHOLD = 0.005
    LEVEL_NEARNESS_THRESHOLD_ATR = 0.25
    CONFIG_LOADED = False

# Importa helper JSON/TXT e _safe_get + NUOVI HELPER STORICI
try:
    from statistical_analyzer_helpers import (
         _safe_get, safe_strftime, format_number, _safe_float,
         load_deribit_history, save_deribit_history,
         safe_get_last_value,
         load_historical_stats_from_json
    )
except ImportError:
    logger_critical = logging.getLogger(__name__)
    logger_critical.critical("ERRORE CRITICO IMPORT statistical_analyzer_helpers.")
    # Fallback helpers
    def _safe_get(data: Optional[Dict], keys: List[Any], default: Any = None) -> Any: return default # type: ignore
    def safe_strftime(date_obj, fmt="%Y-%m-%d %H:%M:%S", fallback="N/A"): # type: ignore
         if hasattr(date_obj, 'strftime'): return date_obj.strftime(fmt)
         return fallback
    def format_number(value, decimals=2, fallback="N/A"): return fallback # type: ignore
    def _safe_float(value, default=None): # type: ignore
        try: return float(value) if pd.notna(value) and not np.isinf(value) else default
        except: return default
    # Fallback funzioni storiche
    def load_deribit_history(filename: str) -> List[Dict[str, Any]]: return [] # type: ignore
    def save_deribit_history(history_data: List[Dict[str, Any]], new_entry: Dict[str, Any], filename: str, max_entries: int) -> bool: return False # type: ignore
    def safe_get_last_value(series, default=None): # type: ignore
            try:
                return series.iloc[-1] if series is not None and len(series) > 0 and pd.notna(series.iloc[-1]) else default
            except Exception:
                return default
    def load_historical_stats_from_json(filename: str) -> Optional[Dict]: return None # type: ignore

# Setup logging
logger = logging.getLogger(__name__)

# --- Classe TradingAdvisor ---
class TradingAdvisor:
    def __init__(self, data: Optional[pd.DataFrame] = None, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        self.data: Optional[pd.DataFrame] = None
        self.timeframe: Optional[str] = timeframe
        self.symbol: Optional[str] = symbol
        self.analyzer: Optional[StatisticalAnalyzerAdvanced] = None
        self.analysis_results: Dict[str, Any] = {}
        self.deribit_available = DERIBIT_COLLECTOR_LOADED
        if data is not None:
            self.set_data(data, symbol, timeframe)
        logger.debug(f"TradingAdvisor init: Symbol={self.symbol}, TF={self.timeframe}, Dati?={self.data is not None and not self.data.empty}, Deribit?={self.deribit_available}")

    def set_data(self, data: pd.DataFrame, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        if data is None or data.empty:
            logger.warning("set_data: dati vuoti/None forniti.")
            self.data = None; self.analyzer = None; self.analysis_results = {}; return
        if not isinstance(data.index, pd.DatetimeIndex):
            try: data.index = pd.to_datetime(data.index, utc=True)
            except Exception as e: logger.error(f"set_data: conversione indice fallita: {e}"); self.data = None; self.analyzer = None; return
        elif data.index.tz is None or str(data.index.tz).upper() != 'UTC':
            logger.warning(f"set_data: indice non UTC ({data.index.tz}). Forzo UTC.")
            try: data.index = data.index.tz_localize('UTC', ambiguous='infer') if data.index.tz is None else data.index.tz_convert('UTC')
            except Exception as tz_err: logger.error(f"Errore conversione UTC set_data: {tz_err}"); self.data = None; self.analyzer = None; return
        self.data = data.copy()
        if symbol: self.symbol = symbol
        if timeframe: self.timeframe = timeframe
        self.analyzer = None; self.analysis_results = {}
        try:
            self.analyzer = StatisticalAnalyzerAdvanced(self.data.copy())
            if self.analyzer and self.symbol and self.timeframe:
                 self.analyzer.set_symbol_timeframe(self.symbol, self.timeframe)
        except (ValueError, TypeError) as e:
             logger.error(f"Errore init/set AnalyzerAdvanced in set_data per {self.symbol}: {e}")
             self.data = None
             self.analyzer = None
        logger.debug(f"TradingAdvisor: Dati aggiornati per {self.symbol} ({self.timeframe}). Righe: {len(self.data if self.data is not None else [])}")

    def _run_mtf_analysis(self) -> Dict[str, Any]:
        if not self.symbol or not self.timeframe:
            logger.warning("Analisi MTF: Simbolo o Timeframe corrente mancante.")
            return {'error': 'Symbol or Current Timeframe missing'}
        try:
            mtf_results = analyze_higher_timeframes(
                symbol=self.symbol,
                current_timeframe=self.timeframe,
                higher_timeframes_to_analyze=MTF_TARGET_TIMEFRAMES
            )
            return mtf_results
        except Exception as e:
            logger.error(f"Errore chiamata analyze_higher_timeframes: {e}", exc_info=True)
            return {'error': f'MTF Analysis Failed: {e}'}

    def _get_dwm_ohlc_data(self) -> Dict[str, Any]:
        if not self.symbol:
            logger.warning("Estrazione D/W/M: Simbolo mancante.")
            return {'error': 'Symbol missing'}
        try:
            dwm_data = get_last_n_candles_multiple_tf(
                symbol=self.symbol, timeframes=['1d', '1w', '1M'], n_candles=3, force_fetch=False
            )
            return dwm_data
        except Exception as e:
            logger.error(f"Errore chiamata get_last_n_candles_multiple_tf: {e}", exc_info=True)
            return {'error': f'Failed to fetch D/W/M data: {e}'}

    def _run_deribit_data_collection(self) -> Dict[str, Any]:
        deribit_master_results = {"fetch_error": None}

        if not self.deribit_available:
            deribit_master_results["fetch_error"] = "deribit_collector module not loaded"
            return deribit_master_results

        supported_assets = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"}
        base_currency = None
        if self.symbol:
            base_currency_candidate = self.symbol.split('/')[0].upper()
            if base_currency_candidate in supported_assets:
                base_currency = supported_assets[base_currency_candidate]
                logger.info(f"Recupero dati Deribit per: {base_currency}")
            else:
                logger.info(f"Asset {base_currency_candidate} non supportato per Deribit. Salto.")
                deribit_master_results["fetch_error"] = f"Asset {base_currency_candidate} not in Deribit scope"
                return deribit_master_results
        else:
            logger.warning("Simbolo non definito. Salto dati Deribit.")
            deribit_master_results["fetch_error"] = "Symbol not defined"
            return deribit_master_results

        history_filename = os.path.join(HISTORICAL_STATS_DIR, f"deribit_history_{base_currency}.json")
        deribit_history = load_deribit_history(history_filename)

        deribit_master_results[base_currency] = {
            "perpetual_futures": {"error": None},
            "options_analysis": {"error": None, "historical_context": {"error": "History not processed yet"}},
            "diagnostic_ticker_test": {"error": "Test not run"}
        }
        perp_results = deribit_master_results[base_currency]["perpetual_futures"]
        opts_analysis = deribit_master_results[base_currency]["options_analysis"]
        hist_context = opts_analysis["historical_context"]
        diag_results = deribit_master_results[base_currency]["diagnostic_ticker_test"]

        perp_instrument_name = None; dated_future_instrument_name = None
        try:
            instruments = find_instrument_details(currency=base_currency, kind="future", expired=False)
            if not instruments or (isinstance(instruments, list) and len(instruments) > 0 and isinstance(instruments[0], dict) and "error" in instruments[0]):
                 log_msg = f"Nessuno strumento future trovato per {base_currency} via API."
                 if instruments and isinstance(instruments, list) and len(instruments) > 0 and isinstance(instruments[0], dict) and "error" in instruments[0]: log_msg += f" Errore API: {instruments[0]['error']}"
                 else: log_msg += " Risposta vuota o non valida."
                 logger.error(log_msg); perp_results["error"] = "No active future instruments found via API"
                 perp_instrument_name = f"{base_currency}-PERPETUAL"; dated_future_instrument_name = None
            else:
                found_perp = False; found_dated = False
                for inst in instruments:
                    if isinstance(inst, dict) and inst.get("is_active"):
                        inst_name = inst.get("instrument_name", "")
                        if inst_name.endswith("-PERPETUAL") and not found_perp: perp_instrument_name = inst_name; logger.info(f"Trovato perpetual attivo: {perp_instrument_name}"); found_perp = True
                        elif not inst_name.endswith("-PERPETUAL") and not found_dated: dated_future_instrument_name = inst_name; logger.info(f"Trovato future datato attivo per test: {dated_future_instrument_name}"); found_dated = True
                    if found_perp and found_dated: break
                if not perp_instrument_name: perp_instrument_name = f"{base_currency}-PERPETUAL"; logger.warning(f"Perpetual non trovato, uso fallback: {perp_instrument_name}")
                if not dated_future_instrument_name: logger.warning(f"Nessun future datato attivo trovato per test.")
        except Exception as e:
            logger.error(f"Errore ricerca strumenti per {base_currency}: {e}", exc_info=True); perp_results["error"] = f"Failed to find instruments: {e}"
            perp_instrument_name = f"{base_currency}-PERPETUAL"; dated_future_instrument_name = None

        if perp_instrument_name and perp_results["error"] is None:
             try:
                 logger.debug(f"Tentativo Ticker (Book Summary) per Perpetual: {perp_instrument_name}")
                 ticker_data_perp = get_deribit_ticker_data(perp_instrument_name)
                 if ticker_data_perp and not ticker_data_perp.get("error"): perp_results.update(ticker_data_perp)
                 else: err_ticker = ticker_data_perp.get("error", "Unknown") if ticker_data_perp else "None"; perp_results["error"] = err_ticker; logger.error(f"Errore ticker {perp_instrument_name}: {err_ticker}")
                 fetch_others = perp_results["error"] is None
                 if fetch_others: logger.debug(f"Tentativo Funding History per: {perp_instrument_name}"); funding_data = get_deribit_funding_rate_history(perp_instrument_name); perp_results["recent_funding_rates"] = funding_data.get("history", [{"error": funding_data.get("error", "Unknown")}]) if funding_data else [{"error":"Call failed"}]
                 if fetch_others: logger.debug(f"Tentativo Liquidations per: {perp_instrument_name}"); liq_data = get_deribit_liquidations(perp_instrument_name); perp_results["recent_liquidations"] = liq_data.get("liquidations", [{"error": liq_data.get("error", "Unknown")}]) if liq_data else [{"error":"Call failed"}]
                 if fetch_others and not perp_results.get("error") and \
                    not (isinstance(perp_results.get("recent_funding_rates"), list) and perp_results["recent_funding_rates"] and isinstance(perp_results["recent_funding_rates"][0], dict) and perp_results["recent_funding_rates"][0].get("error")) and \
                    not (isinstance(perp_results.get("recent_liquidations"), list) and perp_results["recent_liquidations"] and isinstance(perp_results["recent_liquidations"][0], dict) and perp_results["recent_liquidations"][0].get("error")):
                     perp_results.pop("error", None)

             except Exception as e: logger.error(f"Errore fetch dati perpetual {base_currency}: {e}", exc_info=True); perp_results["error"] = f"Unexpected error: {e}"
        elif not perp_instrument_name and perp_results["error"] is None: perp_results["error"] = "Perpetual instrument name could not be determined"

        try:
            logger.info(f"Recupero e processo dati Opzioni Deribit per {base_currency}...")
            options_summaries = get_deribit_options_summary(currency=base_currency)

            if not options_summaries or (isinstance(options_summaries, list) and len(options_summaries) > 0 and isinstance(options_summaries[0], dict) and options_summaries[0].get("error")):
                err_msg = options_summaries[0].get("error", "Failed to fetch options summaries") if options_summaries else "Empty response for options summaries"
                logger.error(f"Errore recupero options summary per {base_currency}: {err_msg}")
                opts_analysis["error"] = err_msg
            else:
                total_call_volume = 0.0; total_put_volume = 0.0; total_call_oi = 0.0; total_put_oi = 0.0
                volume_by_expiry = defaultdict(lambda: {'call': 0.0, 'put': 0.0}); oi_by_expiry = defaultdict(lambda: {'call': 0.0, 'put': 0.0})
                volume_by_strike = defaultdict(lambda: {'call': 0.0, 'put': 0.0}); oi_by_strike = defaultdict(lambda: {'call': 0.0, 'put': 0.0})
                iv_data = []; current_underlying_price = None

                for summary in options_summaries:
                    if not isinstance(summary, dict): continue
                    vol = _safe_float(summary.get("volume"), 0.0); oi = _safe_float(summary.get("open_interest"), 0.0)
                    strike = summary.get("strike_price"); expiry_ts = summary.get("expiration_timestamp"); option_type = summary.get("option_type")
                    mark_iv = _safe_float(summary.get("mark_iv")); expiry_date_str = 'N/A'
                    if current_underlying_price is None: current_underlying_price = _safe_float(summary.get("underlying_price"))
                    if option_type == 'C': total_call_volume += vol; total_call_oi += oi
                    elif option_type == 'P': total_put_volume += vol; total_put_oi += oi

                    if expiry_ts is not None: expiry_date_str = safe_strftime(expiry_ts, '%Y-%m-%d')

                    if expiry_date_str != 'N/A':
                        if option_type == 'C': volume_by_expiry[expiry_date_str]['call'] += vol; oi_by_expiry[expiry_date_str]['call'] += oi
                        elif option_type == 'P': volume_by_expiry[expiry_date_str]['put'] += vol; oi_by_expiry[expiry_date_str]['put'] += oi
                    if strike is not None:
                        strike_key = format_number(strike, 0)
                        if option_type == 'C': volume_by_strike[strike_key]['call'] += vol; oi_by_strike[strike_key]['call'] += oi
                        elif option_type == 'P': volume_by_strike[strike_key]['put'] += vol; oi_by_strike[strike_key]['put'] += oi
                    if strike is not None and expiry_date_str != 'N/A' and mark_iv is not None:
                         iv_data.append({
                             'strike': strike,
                             'expiry_str': expiry_date_str,
                             'expiry_ts': expiry_ts,
                             'type': option_type,
                             'mark_iv': mark_iv,
                             'volume': vol,
                             'open_interest': oi
                         })

                pcr_volume = _safe_float(total_put_volume / total_call_volume) if total_call_volume > 1e-9 else None
                pcr_oi = _safe_float(total_put_oi / total_call_oi) if total_call_oi > 1e-9 else None
                vol_pcr_sentiment = "Bearish (More Put Volume)" if pcr_volume is not None and pcr_volume > 1.0 else "Bullish (More Call Volume)" if pcr_volume is not None and pcr_volume < 0.7 else "Neutral"
                oi_pcr_sentiment = "Bearish (More Put OI)" if pcr_oi is not None and pcr_oi > 1.0 else "Bullish (More Call OI)" if pcr_oi is not None and pcr_oi < 0.7 else "Neutral"
                max_pain_strike = None; max_oi_strike_data = None; max_vol_strike_data = None; max_oi_expiry_data = None; max_vol_expiry_data = None
                if oi_by_strike:
                    all_strikes_oi = defaultdict(float); [all_strikes_oi.update({sk: d.get('call', 0.0) + d.get('put', 0.0)}) for sk, d in oi_by_strike.items()]
                    if all_strikes_oi: max_oi_strike_key = max(all_strikes_oi, key=all_strikes_oi.get); max_pain_strike = _safe_float(max_oi_strike_key); max_oi_strike_data = {"strike": max_pain_strike, "total_oi": all_strikes_oi[max_oi_strike_key], "call_oi": oi_by_strike[max_oi_strike_key].get('call', 0.0), "put_oi": oi_by_strike[max_oi_strike_key].get('put', 0.0)}
                if volume_by_strike:
                    all_strikes_vol = defaultdict(float); [all_strikes_vol.update({sk: d.get('call', 0.0) + d.get('put', 0.0)}) for sk, d in volume_by_strike.items()]
                    if all_strikes_vol: max_vol_strike_key = max(all_strikes_vol, key=all_strikes_vol.get); max_vol_strike_data = {"strike": _safe_float(max_vol_strike_key), "total_volume": all_strikes_vol[max_vol_strike_key], "call_volume": volume_by_strike[max_vol_strike_key].get('call', 0.0), "put_volume": volume_by_strike[max_vol_strike_key].get('put', 0.0)}
                if oi_by_expiry:
                    all_expiry_oi = defaultdict(float); [all_expiry_oi.update({ex: d.get('call', 0.0) + d.get('put', 0.0)}) for ex, d in oi_by_expiry.items()]
                    if all_expiry_oi: max_oi_expiry_key = max(all_expiry_oi, key=all_expiry_oi.get); max_oi_expiry_data = {"expiry": max_oi_expiry_key, "total_oi": all_expiry_oi[max_oi_expiry_key], "call_oi": oi_by_expiry[max_oi_expiry_key].get('call', 0.0), "put_oi": oi_by_expiry[max_oi_expiry_key].get('put', 0.0)}
                if volume_by_expiry:
                    all_expiry_vol = defaultdict(float); [all_expiry_vol.update({ex: d.get('call', 0.0) + d.get('put', 0.0)}) for ex, d in volume_by_expiry.items()]
                    if all_expiry_vol: max_vol_expiry_key = max(all_expiry_vol, key=all_expiry_vol.get); max_vol_expiry_data = {"expiry": max_vol_expiry_key, "total_volume": all_expiry_vol[max_vol_expiry_key], "call_volume": volume_by_expiry[max_vol_expiry_key].get('call', 0.0), "put_volume": volume_by_expiry[max_vol_expiry_key].get('put', 0.0)}

                atm_iv = None
                if current_underlying_price is not None and iv_data:
                    try:
                        iv_data.sort(key=lambda x: (x.get('expiry_ts', float('inf')), abs(x.get('strike', float('inf')) - current_underlying_price)))
                        if iv_data: atm_iv = iv_data[0].get('mark_iv'); logger.debug(f"Trovato ATM IV (strike: {iv_data[0].get('strike')}, expiry: {iv_data[0].get('expiry_str')}): {atm_iv}")
                        else: logger.warning("Lista iv_data vuota dopo l'ordinamento.")
                    except Exception as atm_err: logger.warning(f"Errore durante l'ordinamento o l'accesso per ATM IV: {atm_err}", exc_info=False)

                hist_context = {"error": None}
                lookback_days_short = 1; lookback_days_medium = DERIBIT_HISTORY_AVG_PERIOD
                try:
                    if deribit_history:
                        last_entry = deribit_history[-1] if deribit_history else None
                        last_entry_ts = last_entry.get('timestamp') if last_entry else None
                        if last_entry and last_entry_ts:
                             time_diff = datetime.now(timezone.utc).timestamp() * 1000 - last_entry_ts
                             if timedelta(hours=12).total_seconds()*1000 < time_diff < timedelta(hours=36).total_seconds()*1000:
                                  prev_pcr_vol = last_entry.get('volume_put_call_ratio'); prev_pcr_oi = last_entry.get('open_interest_put_call_ratio')
                                  prev_total_oi = last_entry.get('total_open_interest'); prev_atm_iv = last_entry.get('atm_implied_volatility_pct')
                                  current_total_oi = _safe_float(total_call_oi + total_put_oi)
                                  if pcr_volume is not None and prev_pcr_vol is not None: hist_context['pcr_volume_change_1d'] = _safe_float(pcr_volume - prev_pcr_vol)
                                  if pcr_oi is not None and prev_pcr_oi is not None: hist_context['pcr_oi_change_1d'] = _safe_float(pcr_oi - prev_pcr_oi)
                                  if current_total_oi is not None and prev_total_oi is not None:
                                       hist_context['total_oi_change_1d'] = _safe_float(current_total_oi - prev_total_oi)
                                       if prev_total_oi > 1e-9: hist_context['total_oi_change_1d_pct'] = _safe_float(((current_total_oi - prev_total_oi) / prev_total_oi) * 100)
                                  if atm_iv is not None and prev_atm_iv is not None: hist_context['atm_iv_change_1d'] = _safe_float(atm_iv - prev_atm_iv)
                                  hist_context['comparison_timestamp_1d'] = safe_strftime(last_entry_ts, fallback=str(last_entry_ts))
                             else: logger.debug(f"Ultima entry storica Deribit troppo vecchia/recente ({time_diff/1000/3600:.1f}h) per confronto 1d.")
                        history_df = pd.DataFrame(deribit_history)
                        if not history_df.empty and 'timestamp' in history_df.columns:
                             numeric_cols_hist = ['volume_put_call_ratio', 'open_interest_put_call_ratio', 'total_open_interest', 'atm_implied_volatility_pct']
                             for col in numeric_cols_hist:
                                 if col in history_df.columns: history_df[col] = pd.to_numeric(history_df[col], errors='coerce')
                             lookback_medium_entries = min(len(history_df), lookback_days_medium)
                             if lookback_medium_entries > 1:
                                  recent_history = history_df.iloc[-lookback_medium_entries:]
                                  hist_context['average_period_days'] = lookback_medium_entries
                                  for col in numeric_cols_hist:
                                      if col in recent_history.columns and not recent_history[col].isnull().all():
                                          avg_val = recent_history[col].mean(); current_val = None
                                          if col == 'volume_put_call_ratio': current_val = pcr_volume
                                          elif col == 'open_interest_put_call_ratio': current_val = pcr_oi
                                          elif col == 'total_open_interest': current_val = _safe_float(total_call_oi + total_put_oi)
                                          elif col == 'atm_implied_volatility_pct': current_val = atm_iv
                                          if pd.notna(avg_val):
                                               avg_key = f"{col}_avg_{lookback_medium_entries}d"; hist_context[avg_key] = _safe_float(avg_val)
                                               if current_val is not None and avg_val > 1e-9:
                                                   ratio_key = f"{col}_vs_avg_{lookback_medium_entries}d_ratio"; hist_context[ratio_key] = _safe_float(current_val / avg_val)
                    else: hist_context['error'] = "No historical data loaded"
                    hist_context.pop("error", None)
                except Exception as hist_err:
                     logger.error(f"Errore calcolo contesto storico Deribit: {hist_err}", exc_info=True); hist_context['error'] = f"Failed to process history: {hist_err}"

                opts_analysis.update({
                    "total_call_volume": _safe_float(total_call_volume), "total_put_volume": _safe_float(total_put_volume),
                    "volume_put_call_ratio": pcr_volume, "volume_pcr_sentiment": vol_pcr_sentiment,
                    "total_call_open_interest": _safe_float(total_call_oi), "total_put_open_interest": _safe_float(total_put_oi),
                    "open_interest_put_call_ratio": pcr_oi, "oi_pcr_sentiment": oi_pcr_sentiment,
                    "atm_implied_volatility_pct": atm_iv,
                    "max_pain_strike_approx": max_pain_strike,
                    "max_interest_strike": max_oi_strike_data, "max_volume_strike": max_vol_strike_data,
                    "max_interest_expiry": max_oi_expiry_data, "max_volume_expiry": max_vol_expiry_data,
                    "historical_context": hist_context,
                    "distribution_by_expiry": dict(sorted(volume_by_expiry.items())), "open_interest_by_expiry": dict(sorted(oi_by_expiry.items())),
                    "distribution_by_strike": dict(sorted(volume_by_strike.items(), key=lambda item: _safe_float(item[0], float('inf')))),
                    "open_interest_by_strike": dict(sorted(oi_by_strike.items(), key=lambda item: _safe_float(item[0], float('inf'))))
                })
                opts_analysis.pop('error', None)
                logger.info(f"Analisi opzioni Deribit per {base_currency} completata (con contesto storico).")

                current_timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                new_history_entry = {
                    "timestamp": current_timestamp_ms,
                    "volume_put_call_ratio": pcr_volume,
                    "open_interest_put_call_ratio": pcr_oi,
                    "total_open_interest": _safe_float(total_call_oi + total_put_oi),
                    "atm_implied_volatility_pct": atm_iv,
                }
                save_deribit_history(deribit_history, new_history_entry, history_filename, DERIBIT_HISTORY_MAX_ENTRIES)

        except Exception as e:
            logger.error(f"Errore imprevisto durante l'analisi delle opzioni Deribit {base_currency}: {e}", exc_info=True)
            opts_analysis["error"] = f"Options analysis processing failed: {e}"
            if not hist_context.get("error"): hist_context["error"] = "Processing failed before history calc"

        if dated_future_instrument_name:
             try:
                 logger.debug(f"--- TEST DIAGNOSTICO: Tentativo Ticker per Future Datato: {dated_future_instrument_name} ---")
                 ticker_data_dated = get_deribit_ticker_data(dated_future_instrument_name)
                 diag_results['instrument_tested'] = dated_future_instrument_name
                 if ticker_data_dated and not ticker_data_dated.get("error"):
                     logger.info(f"--- TEST DIAGNOSTICO SUCCESSO: {dated_future_instrument_name} ---"); diag_results['result'] = "Success"; diag_results['data_sample'] = {k: ticker_data_dated.get(k) for k in ['mark_price', 'open_interest_usd', 'last_price']}; diag_results.pop("error", None)
                 else: err_diag = ticker_data_dated.get("error", "Unknown") if ticker_data_dated else "None"; logger.error(f"--- TEST DIAGNOSTICO FALLITO: {dated_future_instrument_name}: {err_diag} ---"); diag_results['result'] = "Failed"; diag_results['error'] = err_diag
             except Exception as diag_e: logger.error(f"--- TEST DIAGNOSTICO ECCEZIONE: {dated_future_instrument_name}: {diag_e} ---", exc_info=True); diag_results['result'] = "Exception"; diag_results['error'] = f"Exception: {diag_e}"
        else: diag_results['result'] = "Skipped"; diag_results['error'] = "No dated future instrument found"

        if perp_results.get("error") is None and opts_analysis.get("error") is None:
             deribit_master_results.pop("fetch_error", None)

        return deribit_master_results

    def _calculate_potential_targets(self):
        """
        Calcola i potenziali target da diverse fonti (ATR, Fibo Ext, Cycle Extremes)
        e li salva nella sezione 'potential_targets' del risultato.
        Questo metodo viene chiamato DOPO che analyzer e fibonacci sono stati eseguiti.
        """
        logger.debug("Calculating potential targets...")
        target_results = {'error': None}
        # --- MODIFICA: Assegna direttamente a self.analysis_results ---
        self.analysis_results['potential_targets'] = target_results
        # --- FINE MODIFICA ---

        # 1. Target basati su ATR (presi da TechnicalAnalyzer)
        atr_targets = _safe_get(self.analysis_results, ['technical_analysis', 'price_level_targets', 'atr_based'], {'error': 'ATR targets not available'})
        target_results['atr_based'] = atr_targets

        # 2. Target basati su Estensioni Fibonacci (presi da analisi Fibonacci)
        fibo_ext = _safe_get(self.analysis_results, ['fibonacci', 'extensions'], {'error': 'Fibonacci extensions not available'})
        target_results['fibonacci_extensions'] = fibo_ext

        # 3. Target basati su Estremi Ciclo Precedente (richiede l'analyzer)
        prev_high, prev_low = None, None
        if self.analyzer:
            try:
                prev_high, prev_low = self.analyzer._find_previous_cycle_extremes()
            except Exception as e:
                logger.warning(f"Errore nel trovare estremi ciclo precedente: {e}")
        target_results['previous_cycle_extremes'] = {
            'previous_major_high': prev_high,
            'previous_major_low': prev_low
        }

        # Rimuovi l'errore generale se almeno una fonte di target è valida
        if (atr_targets and not atr_targets.get('error')) or \
           (fibo_ext and not fibo_ext.get('error')) or \
           prev_high is not None or prev_low is not None:
             target_results.pop('error', None)
        elif not target_results.get('error'): # Aggiungi errore solo se tutto fallisce e non c'è già
             target_results['error'] = "No target data available"

        logger.debug(f"Potential targets calculated: {target_results}")

    def _populate_level_proximity(self):
        """Popola la sezione level_proximity usando i dati calcolati."""
        logger.debug("Populating level_proximity section...")
        proximity_results = {}
        current_price = _safe_get(self.analysis_results, ['market_info', 'execution_time_current_price'])
        current_atr = _safe_get(self.analysis_results, ['technical_analysis', 'technical_indicators', 'atr'])

        if current_price is None or current_price <= 0:
            logger.warning("Prezzo corrente non valido per calcolo prossimità.")
            if 'level_proximity' not in self.analysis_results or not isinstance(self.analysis_results.get('level_proximity'), dict) or not self.analysis_results['level_proximity'].get('error'):
                self.analysis_results['level_proximity'] = {'error': 'Invalid current price'}
            return

        self.analysis_results.setdefault('level_proximity', {})

        valid_atr = current_atr if current_atr is not None and current_atr > 1e-9 else None

        def _add_level(level_name: str, level_value: Optional[float]):
            if level_value is None:
                proximity_results[level_name] = {"value": None, "distance_pct": None, "distance_atr": None}
                return
            distance = level_value - current_price
            distance_pct = _safe_float((distance / current_price) * 100)
            distance_atr = _safe_float(distance / valid_atr) if valid_atr is not None else None
            proximity_results[level_name] = {
                "value": _safe_float(level_value),
                "distance_pct": distance_pct,
                "distance_atr": distance_atr
            }

        # 1. Nearest S/R
        sr_data = _safe_get(self.analysis_results, ['technical_analysis', 'support_resistance'], {})
        _add_level("nearest_support", sr_data.get('nearest_support'))
        _add_level("nearest_resistance", sr_data.get('nearest_resistance'))

        # 2. Fibonacci Levels
        fibo_data = self.analysis_results.get('fibonacci', {})
        retracements = fibo_data.get('retracements', {})
        if isinstance(retracements, dict):
            _add_level("fib_retr_0_500", retracements.get('0.500'))
            _add_level("fib_retr_0_618", retracements.get('0.618'))
        extensions = fibo_data.get('extensions', {})
        if isinstance(extensions, dict):
            _add_level("fib_ext_1_618", extensions.get('1.618'))

        # 3. Moving Averages (SMA/EMA da TF Corrente)
        ma_data = _safe_get(self.analysis_results, ['technical_analysis', 'technical_indicators', 'moving_averages'], {})
        if isinstance(ma_data, dict):
            sma_periods_to_use = getattr(config, 'SMA_PERIODS', [20, 50, 200])
            ema_periods_to_use = getattr(config, 'EMA_PERIODS', [12, 26, 50])
            for p in sma_periods_to_use: _add_level(f"sma_{p}", ma_data.get(f'sma_{p}'))
            for p in ema_periods_to_use: _add_level(f"ema_{p}", ma_data.get(f'ema_{p}'))

        # 4. Volume Profile Levels (TF Corrente e Daily/Weekly)
        vp_base = _safe_get(self.analysis_results, ['technical_analysis', 'volume_profile'], {})
        if isinstance(vp_base, dict):
            _add_level("vp_poc", vp_base.get('poc_price'))
            _add_level("vp_vah", vp_base.get('vah_price'))
            _add_level("vp_val", vp_base.get('val_price'))
        for period in ['daily', 'weekly']:
            vp_periodic = _safe_get(self.analysis_results, ['technical_analysis', 'volume_profile_periodic', period], {})
            if isinstance(vp_periodic, dict):
                _add_level(f"vp_{period}_poc", vp_periodic.get('poc_price'))
                _add_level(f"vp_{period}_vah", vp_periodic.get('vah_price'))
                _add_level(f"vp_{period}_val", vp_periodic.get('val_price'))

        # 5. HTF Key Levels (SMA50/200, High/Low)
        mtf_data = self.analysis_results.get('multi_timeframe_analysis', {})
        if isinstance(mtf_data, dict):
            for htf, htf_levels in mtf_data.items():
                if isinstance(htf_levels, dict) and 'key_levels' in htf_levels:
                    levels = htf_levels['key_levels']
                    if isinstance(levels, dict):
                        _add_level(f"htf_{htf}_sma50", levels.get('sma_50'))
                        _add_level(f"htf_{htf}_sma200", levels.get('sma_200'))
                        _add_level(f"htf_{htf}_high", levels.get('recent_high_30p'))
                        _add_level(f"htf_{htf}_low", levels.get('recent_low_30p'))

        # Aggiungere i risultati consolidati, preservando eventuali errori precedenti
        existing_proximity = self.analysis_results.get('level_proximity', {})
        if isinstance(existing_proximity, dict):
            proximity_results['error'] = existing_proximity.get('error') # Mantieni errore esistente
            if any(p.get('value') is not None for p in proximity_results.values()):
                 proximity_results.pop('error', None)

        self.analysis_results['level_proximity'] = proximity_results
        logger.debug(f"Populated level_proximity with {len(proximity_results)} levels.")


    def analyze(self) -> Optional[Dict[str, Any]]:
        analysis_start_time = time.time()
        logger.info(f"Avvio analisi completa per {self.symbol} ({self.timeframe})...")
        self.analysis_results = {}

        # Controllo dati e inizializzazione analyzer
        if self.data is None or self.data.empty:
            logger.error("Analyze: Nessun dato valido per l'analisi.")
            return {'error': 'No valid data available for analysis'}
        if self.analyzer is None:
            logger.warning("Analyze: Istanza StatisticalAnalyzerAdvanced non inizializzata. Tentativo...")
            try:
                self.analyzer = StatisticalAnalyzerAdvanced(self.data.copy())
                if self.symbol and self.timeframe:
                    self.analyzer.set_symbol_timeframe(self.symbol, self.timeframe)
                else:
                    logger.error("Analyze: Symbol o Timeframe non definiti, impossibile impostarli nell'analyzer.")
                    return {'error': 'Symbol or Timeframe not set in TradingAdvisor'}
                logger.info("Analyzer inizializzato tardivamente in analyze().")
            except Exception as e:
                logger.error(f"Errore inizializzazione tardiva AnalyzerAdvanced: {e}")
                return {'error': f'Failed to initialize Analyzer: {e}'}
        elif not self.analyzer.symbol or not self.analyzer.timeframe:
             if self.symbol and self.timeframe:
                  logger.warning("Re-imposto symbol/timeframe nell'analyzer esistente...")
                  self.analyzer.set_symbol_timeframe(self.symbol, self.timeframe)
             else:
                  logger.error("Analyze: Analyzer esiste ma Symbol/Timeframe non sono definiti nel TradingAdvisor.")
                  return {'error': 'Symbol or Timeframe not set for existing analyzer'}

        # Recupero Prezzo Corrente
        execution_time_price = safe_get_last_value(self.data.get('close'))
        logger.debug(f"Prezzo iniziale fallback: {execution_time_price}")
        try:
            exchange = get_exchange_instance()
            if exchange and self.symbol:
                formatted_symbol_for_fetch = self.symbol
                if '/' not in formatted_symbol_for_fetch:
                    base = formatted_symbol_for_fetch.upper()
                    if hasattr(exchange, 'markets') and exchange.markets and exchange.markets:
                         quotes = ['USDT', 'BUSD', 'FDUSD', 'USDC', 'TUSD']; found_market = False
                         for quote in quotes:
                             market_symbol = f"{base}/{quote}"
                             if market_symbol in exchange.markets: formatted_symbol_for_fetch = market_symbol; found_market = True; break
                         if not found_market: formatted_symbol_for_fetch = f"{base}/USDT"
                    else: formatted_symbol_for_fetch = f"{base}/USDT"
                logger.debug(f"Tentativo fetch ticker per: {formatted_symbol_for_fetch}")
                try:
                    ticker_data = exchange.fetch_ticker(formatted_symbol_for_fetch)
                    if ticker_data:
                        price_raw = ticker_data.get('last') or ticker_data.get('close') or ticker_data.get('info',{}).get('lastPrice')
                        fetched_price = _safe_float(price_raw)
                        if fetched_price is not None:
                            execution_time_price = fetched_price
                            logger.info(f"[TA.analyze] Prezzo corrente recuperato: {execution_time_price}")
                        else: logger.warning(f"[TA.analyze Fetch] Prezzo 'last' non valido nel ticker per {formatted_symbol_for_fetch}. Uso fallback.")
                    else: logger.warning(f"[TA.analyze Fetch] fetch_ticker ha restituito None per {formatted_symbol_for_fetch}. Uso fallback.")
                except ccxt.NetworkError as ne: logger.warning(f"[TA.analyze Fetch NetworkError] {formatted_symbol_for_fetch}: {ne}. Uso fallback.")
                except ccxt.ExchangeError as ee: logger.warning(f"[TA.analyze Fetch ExchangeError] {formatted_symbol_for_fetch}: {ee}. Uso fallback.")
                except Exception as fetch_err: logger.warning(f"[TA.analyze Fetch] Errore generico fetch ticker {formatted_symbol_for_fetch}: {fetch_err}. Uso fallback.")
            else: logger.warning(f"[TA.analyze Fetch] Exchange non disponibile o simbolo mancante. Uso fallback.")
        except Exception as fetch_init_err:
            logger.error(f"[TA.analyze Fetch Init] Errore: {fetch_init_err}", exc_info=True)

        # Inizializza struttura report
        self.analysis_results = {
            "market_info": {
                'symbol': self.symbol, 'timeframe': self.timeframe,
                'analysis_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
                'data_start_time': safe_strftime(self.data.index.min()),
                'data_end_time': safe_strftime(self.data.index.max()),
                'data_points': len(self.data),
                'execution_time_current_price': execution_time_price
            },
            "statistical_analysis": {'error': 'Not run yet'},
            "technical_analysis": {'error': 'Not run yet'},
            "advanced_statistical_analysis": {'error': 'Not run yet'},
            "cycle_analysis": {'error': 'Not run yet'},
            "intracandle_analysis": {'error': 'Not run yet'},
            "timing_estimates": {'error': 'Not run yet'},
            "potential_targets": {'error': 'Not run yet'},
            "level_proximity": {'error': 'Not run yet'},
            "entry_exit_refinement": {'error': 'Not run yet'},
            "patterns": {'error': 'Not run yet'},
            "fibonacci": {
                'retracements': {'error': 'Not run yet'},
                'extensions': {'error': 'Not run yet'},
                'fibonacci_time_zones_from_low': {'error': 'Not run yet'},
                'fibonacci_time_zones_from_high': {'error': 'Not run yet'},
                'error': 'Not run yet'
            },
            "multi_timeframe_analysis": {'error': 'Not run yet'},
            "dwm_ohlc_data": {'error': 'Not run yet'},
            "hourly_volume_alert": {'error': "Analysis not run yet"},
            "deribit_data": {'error': 'Not run yet'}
        }
        logger.debug(f"Struttura report inizializzata. Prezzo corrente usato: {execution_time_price}")

        # --- ESECUZIONE ANALISI ---
        # 1. Esegui l'analisi completa di StatisticalAnalyzerAdvanced
        try:
            logger.debug("Esecuzione StatisticalAnalyzerAdvanced.run_analysis()...")
            if not self.analyzer: raise RuntimeError("Analyzer non è inizializzato")
            full_stat_results = self.analyzer.run_analysis()

            if not full_stat_results or not isinstance(full_stat_results, dict):
                logger.error("StatisticalAnalyzerAdvanced.run_analysis() non ha restituito risultati validi.")
                keys_to_error = ['statistical_analysis', 'technical_analysis',
                                 'advanced_statistical_analysis', 'cycle_analysis',
                                 'intracandle_analysis', 'timing_estimates', 'hourly_volume_alert']
                for key in keys_to_error:
                     self.analysis_results[key] = {'error': 'Analyzer run failed or returned invalid results.'}
            else:
                # Popola le sezioni
                self.analysis_results['statistical_analysis'] = {k: full_stat_results.get(k, {}) for k in ['descriptive', 'vwap_daily', 'market_condition']}; self.analysis_results['statistical_analysis']['error'] = full_stat_results.get('error')
                tech_raw = full_stat_results.get('technical_analysis', {})
                self.analysis_results['technical_analysis'] = {k: tech_raw.get(k, {}) for k in ['technical_indicators', 'trend_analysis', 'support_resistance', 'volume_analysis', 'volatility_analysis', 'price_level_targets', 'volume_profile', 'volume_profile_periodic', 'combined_features']}; self.analysis_results['technical_analysis']['error'] = tech_raw.get('error')
                adv_raw = full_stat_results.get('advanced_statistical_analysis', {})
                keys_for_advanced = ['rolling_stats', 'consecutive_stats', 'stationarity', 'autocorrelation', 'volatility_clustering', 'garch_model', 'normality', 'volatility', 'returns', 'risk', 'temporal_bias', 'hourly_volume_analysis'];
                self.analysis_results['advanced_statistical_analysis'] = {k: adv_raw.get(k, {}) for k in keys_for_advanced};
                adv_err = adv_raw.get('error') or adv_raw.get('final_metrics_error');
                self.analysis_results['advanced_statistical_analysis']['error'] = adv_err
                self.analysis_results['cycle_analysis'] = full_stat_results.get('cycle_analysis', {'error': 'Cycle analysis results missing'})
                self.analysis_results['intracandle_analysis'] = full_stat_results.get('intracandle_analysis', {'error': 'Intracandle analysis results missing'})
                self.analysis_results['timing_estimates'] = full_stat_results.get('timing_estimates', {'error': 'Timing estimates results missing'})
                hourly_vol_raw = adv_raw.get('hourly_volume_analysis', {}) if isinstance(adv_raw, dict) else {'error': 'Advanced analysis results missing or invalid'}
                self.analysis_results['hourly_volume_alert'] = {
                    'is_significantly_above_average': hourly_vol_raw.get('is_significantly_above_average'),
                    'last_volume': hourly_vol_raw.get('last_candle_volume'),
                    'average_volume_for_hour': hourly_vol_raw.get('last_candle_avg_volume_for_hour'),
                    'ratio_vs_average': hourly_vol_raw.get('last_vs_avg_ratio'),
                    'threshold_factor': hourly_vol_raw.get('volume_alert_threshold_factor'),
                    'error': hourly_vol_raw.get('error')
                }
                if full_stat_results.get('analysis_warnings'):
                     self.analysis_results['analysis_warnings'] = full_stat_results['analysis_warnings']
                self._populate_level_proximity()

        except Exception as stat_err:
            logger.error(f"Errore critico chiamata StatisticalAnalyzerAdvanced.run_analysis(): {stat_err}", exc_info=True);
            keys_to_error = ['statistical_analysis', 'technical_analysis',
                             'advanced_statistical_analysis', 'cycle_analysis',
                             'intracandle_analysis', 'timing_estimates', 'hourly_volume_alert']
            for key in keys_to_error:
                 self.analysis_results[key] = {'error': f'Analyzer run crashed: {stat_err}'}

        # 2. Analisi Pattern
        try:
            logger.debug("Esecuzione PatternAnalyzer...");
            pattern_analyzer = PatternAnalyzer(self.data.copy());
            self.analysis_results['patterns'] = pattern_analyzer.detect_all_patterns()
        except Exception as pattern_err: logger.error(f"Errore PatternAnalyzer: {pattern_err}", exc_info=True); self.analysis_results['patterns'] = {'error': f'Failed: {pattern_err}'}

        # 3. Analisi Fibonacci (Prezzo e Tempo)
        try:
            logger.debug("Esecuzione Fibonacci (Prezzo e Tempo)...")
            fibo_results = self.analysis_results.setdefault('fibonacci', {})
            fibo_results.pop('error', None)
            trend_analysis = _safe_get(self.analysis_results, ['technical_analysis', 'trend_analysis'], {})
            up_trend_fibo = None
            combined_trend_fibo = trend_analysis.get('trend_combined', trend_analysis.get('trend', 'Unknown'))
            if 'Uptrend' in combined_trend_fibo or 'Bullish' in combined_trend_fibo: up_trend_fibo = True
            elif 'Downtrend' in combined_trend_fibo or 'Bearish' in combined_trend_fibo: up_trend_fibo = False
            if up_trend_fibo is None:
                 try:
                     if len(self.data) >= FIB_LOOKBACK_PERIOD: up_trend_fibo = self.data['close'].iloc[-1] > self.data['close'].iloc[-FIB_LOOKBACK_PERIOD]
                     else: up_trend_fibo = True
                     logger.debug("Fibo: Usato fallback trend TF corrente.")
                 except Exception: up_trend_fibo = True

            fibo_results['retracements'] = get_fibonacci_retracements(self.data, period=FIB_LOOKBACK_PERIOD, up_trend=up_trend_fibo)
            fibo_results['extensions'] = get_fibonacci_extensions(self.data, period=FIB_LOOKBACK_PERIOD, up_trend=up_trend_fibo)
            fibo_results['trend_used'] = 'up' if up_trend_fibo else 'down'
            fibo_results['period_used'] = FIB_LOOKBACK_PERIOD
            fibo_results['fibonacci_time_zones_from_low'] = calculate_fibonacci_time_zones(self.data, period=FIB_LOOKBACK_PERIOD, start_from='low')
            fibo_results['fibonacci_time_zones_from_high'] = calculate_fibonacci_time_zones(self.data, period=FIB_LOOKBACK_PERIOD, start_from='high')

            if not fibo_results.get('retracements', {}).get('error') or \
               not fibo_results.get('extensions', {}).get('error') or \
               not fibo_results.get('fibonacci_time_zones_from_low', {}).get('error') or \
               not fibo_results.get('fibonacci_time_zones_from_high', {}).get('error'):
                fibo_results.pop('error', None)

            self._populate_level_proximity() # Richiama DOPO aver calcolato fibo

        except Exception as fibo_err:
            logger.error(f"Errore Fibonacci (Prezzo o Tempo): {fibo_err}", exc_info=True)
            self.analysis_results['fibonacci'] = {'error': f'Failed: {fibo_err}'}

        # 4. Analisi MTF
        try: logger.debug("Esecuzione Analisi MTF..."); self.analysis_results['multi_timeframe_analysis'] = self._run_mtf_analysis()
        except Exception as mtf_err: logger.error(f"Errore Analisi MTF: {mtf_err}", exc_info=True); self.analysis_results['multi_timeframe_analysis'] = {'error': f'Failed: {mtf_err}'}

        # 5. Recupero Dati OHLC D/W/M
        try: logger.debug("Recupero Dati DWM OHLC..."); self.analysis_results['dwm_ohlc_data'] = self._get_dwm_ohlc_data()
        except Exception as dwm_err: logger.error(f"Errore recupero Dati DWM: {dwm_err}", exc_info=True); self.analysis_results['dwm_ohlc_data'] = {'error': f'Failed: {dwm_err}'}

        # 6. Recupero Dati Deribit
        try: logger.debug("Recupero Dati Deribit..."); self.analysis_results['deribit_data'] = self._run_deribit_data_collection()
        except Exception as deribit_err: logger.error(f"Errore recupero Dati Deribit: {deribit_err}", exc_info=True); self.analysis_results['deribit_data'] = {'fetch_error': f'Failed: {deribit_err}'}

        # 7. --- MODIFICA: Chiamata Corretta ---
        # Calcola Potential Targets ALLA FINE, dopo che tutte le altre sezioni sono state popolate
        try:
            logger.debug("Esecuzione Calcolo Potential Targets...")
            self._calculate_potential_targets() # Questa funzione ora modifica self.analysis_results['potential_targets']
        except Exception as target_err:
            logger.error(f"Errore Calcolo Potential Targets: {target_err}", exc_info=True)
            # Assicura che la chiave esista e abbia l'errore
            self.analysis_results.setdefault('potential_targets', {})['error'] = f'Failed: {target_err}'
        # --- FINE MODIFICA ---

        # Rivedi Controllo Errori Finale
        final_errors = []
        critical_errors_found = False
        non_critical_warnings = []

        sections_to_check = [
            'statistical_analysis', 'technical_analysis', 'advanced_statistical_analysis',
            'cycle_analysis', 'intracandle_analysis', 'timing_estimates',
            'potential_targets', 'level_proximity',
            'patterns', 'fibonacci', 'multi_timeframe_analysis', 'dwm_ohlc_data',
            'hourly_volume_alert', 'deribit_data'
        ]

        non_critical_keys_long_tf = [
            ('advanced_statistical_analysis', 'temporal_bias'),
            ('advanced_statistical_analysis', 'hourly_volume_analysis'),
            ('hourly_volume_alert', None)
        ]
        is_current_tf_long = getattr(self.analyzer, '_is_long_timeframe', False) if self.analyzer else False

        for section in sections_to_check:
            data = self.analysis_results.get(section)
            error_msg = None
            is_potentially_non_critical = is_current_tf_long and (section, None) in non_critical_keys_long_tf

            if isinstance(data, dict) and data.get('error') and "Not run yet" not in str(data.get('error')):
                error_msg = f"Section '{section}': {data['error']}"
                if not is_potentially_non_critical: critical_errors_found = True
            elif section == 'deribit_data' and isinstance(data, dict) and data.get('fetch_error'):
                error_msg = f"Section 'deribit_data': {data['fetch_error']}"
                critical_errors_found = True
            elif isinstance(data, dict):
                 for sub_key, sub_data in data.items():
                      if sub_key == 'error': continue
                      if isinstance(sub_data, dict) and sub_data.get('error') and "Not run yet" not in str(sub_data.get('error')):
                           error_msg_sub = f"Subsection '{section}.{sub_key}': {sub_data['error']}"
                           is_sub_non_critical = is_current_tf_long and (section, sub_key) in non_critical_keys_long_tf
                           if not is_sub_non_critical: critical_errors_found = True
                           final_errors.append(error_msg_sub)

            if error_msg:
                final_errors.append(error_msg)

        analyzer_warnings = self.analysis_results.get('analysis_warnings')
        if isinstance(analyzer_warnings, list):
             non_critical_warnings.extend(analyzer_warnings)
             self.analysis_results.pop('analysis_warnings', None)

        if final_errors:
            unique_errors = sorted(list(set(final_errors)))
            logger.warning(f"Analysis completed with {len(unique_errors)} issue(s). Details: {'; '.join(unique_errors)}")

            if critical_errors_found:
                 error_summary = f"Analysis completed with critical error(s)."
                 self.analysis_results['error'] = error_summary
                 unique_warnings = sorted(list(set(non_critical_warnings)))
                 if unique_warnings and unique_warnings != unique_errors:
                      self.analysis_results['analysis_warnings'] = unique_warnings
                 elif 'analysis_warnings' in self.analysis_results:
                      self.analysis_results.pop('analysis_warnings', None)
            else:
                 self.analysis_results.pop('error', None)
                 all_warnings = sorted(list(set(final_errors + non_critical_warnings)))
                 if all_warnings:
                     self.analysis_results['analysis_warnings'] = all_warnings
        else:
             self.analysis_results.pop('error', None)
             self.analysis_results.pop('analysis_warnings', None)

        analysis_exec_time = time.time() - analysis_start_time
        logger.info(f"Analisi completa per {self.symbol} ({self.timeframe}) terminata in {analysis_exec_time:.2f} sec.")
        return self.analysis_results

# --- END OF FILE trading_advisor.py ---