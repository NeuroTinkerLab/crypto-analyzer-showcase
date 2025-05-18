# --- START OF FILE statistical_analyzer.py ---
# statistical_analyzer.py
"""
Classe per l'analisi statistica di base dei dati finanziari.
Si occupa di statistiche descrittive, calcolo VWAP,
identificazione preliminare della condizione di mercato e
orchestra la chiamata a TechnicalAnalyzer per ottenere gli indicatori base.
Le analisi più avanzate (rolling, bias, test statistici, risk)
sono gestite dalla classe derivata StatisticalAnalyzerAdvanced.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

# Importa helper SOLO da statistical_analyzer_helpers
try:
    from statistical_analyzer_helpers import safe_get_last_value, _safe_get, _safe_float
except ImportError:
    logger_fallback = logging.getLogger(__name__) # Ottieni logger se import fallisce
    logger_fallback.critical("ERRORE CRITICO: statistical_analyzer_helpers non trovato!")
    # Definizioni fallback minimali
    def _safe_get(d: Optional[Dict], keys: list, default=None): # type: ignore
        if d is None: return default
        temp = d
        for key in keys:
            if isinstance(temp, dict): temp = temp.get(key)
            elif isinstance(key, int) and isinstance(temp, (list, tuple)):
                if -len(temp) <= key < len(temp):
                    try: temp = temp[key]
                    except IndexError: return default
                else: return default
            else: return default
            if temp is None: return default
        if isinstance(temp, (float, int, np.number)) and (pd.isna(temp) or np.isinf(temp)): return default
        return temp
    def safe_get_last_value(series, default=None): # type: ignore
        try: return series.iloc[-1] if not pd.isna(series.iloc[-1]) else default
        except: return default
    def _safe_float(value, default=None): # type: ignore
        try: return float(value) if pd.notna(value) and not np.isinf(value) else default
        except: return default

# Setup logging
logger = logging.getLogger(__name__)

# --- Definizione Fallback Locale (per gestire ImportError di TechnicalAnalyzer nel metodo) ---
class _FallbackTechnicalAnalyzer:
    def __init__(self, data):
        logger.error("Errore critico: Impossibile importare TechnicalAnalyzer.")
        self.results = {'error': "TechnicalAnalyzer non importato"}
    def run_analysis(self):
        return self.results
# --- Fine Fallback ---

class StatisticalAnalyzer:
    """
    Classe per l'analisi statistica di base dei dati finanziari.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inizializza l'analizzatore statistico di base.

        Args:
            data (pd.DataFrame): DataFrame con dati OHLCV e DatetimeIndex UTC.
        """
        logger.debug(f"{self.__class__.__name__}.__init__() - INIZIO")
        if data is None or data.empty:
             raise ValueError("I dati forniti a StatisticalAnalyzer non possono essere None o vuoti.")
        # Validazione Indice (Già fatta a monte, ma doppia verifica non nuoce)
        if not isinstance(data.index, pd.DatetimeIndex):
             try: data.index = pd.to_datetime(data.index, utc=True)
             except Exception as e: raise TypeError(f"L'indice dati deve essere DatetimeIndex. Errore conversione: {e}")
        if data.index.tz is None or str(data.index.tz).upper() != 'UTC':
             logger.warning(f"Indice dati non è UTC in StatisticalAnalyzer. Forzo UTC.")
             try: data.index = data.index.tz_localize('UTC') if data.index.tz is None else data.index.tz_convert('UTC')
             except Exception as tz_err: raise TypeError(f"Impossibile forzare indice a UTC in StatisticalAnalyzer: {tz_err}")

        self.data = data.copy() # Lavora sempre su una copia
        self.results: Dict[str, Any] = {} # Risultati specifici di questa classe

        # Calcola rendimenti base se mancano (necessari per alcune analisi base/derivate)
        self._calculate_basic_returns()

        logger.debug(f"{self.__class__.__name__}.__init__() - FINE")

    def _calculate_basic_returns(self):
        """Calcola simple e log returns interni su self.data se mancano."""
        if 'close' in self.data.columns:
            recalculated = False
            if 'simple_return' not in self.data.columns:
                try:
                    close_valid = self.data['close'].replace(0, np.nan)
                    self.data['simple_return'] = close_valid.pct_change().replace([np.inf, -np.inf], np.nan)
                    recalculated = True
                except Exception as e: logger.error(f"Errore calcolo simple_return: {e}")
            if 'log_return' not in self.data.columns:
                 try:
                     close_valid = self.data['close'].replace(0, np.nan)
                     self.data['log_return'] = np.log(close_valid / close_valid.shift(1)).replace([np.inf, -np.inf], np.nan)
                     recalculated = True
                 except Exception as e: logger.error(f"Errore calcolo log_return: {e}")
            if recalculated: logger.debug("Rendimenti base (simple/log) calcolati.")
        else:
            logger.warning("Colonna 'close' non trovata, impossibile calcolare rendimenti base.")

    def calculate_descriptive_statistics(self) -> Dict[str, Any]:
        """Calcola statistiche descrittive per prezzo e volume."""
        logger.debug(f"{self.__class__.__name__}.calculate_descriptive_statistics() - INIZIO")
        stats_dict: Dict[str, Any] = {'price': {}, 'volume': {}}
        if self.data.empty: logger.warning("DataFrame vuoto per statistiche descrittive."); return stats_dict
        try:
            if 'close' in self.data.columns:
                close_prices = self.data['close'].dropna()
                if not close_prices.empty:
                    price_stats = {
                        'mean': _safe_float(close_prices.mean()), 'median': _safe_float(close_prices.median()),
                        'std': _safe_float(close_prices.std()), 'min': _safe_float(close_prices.min()),
                        'max': _safe_float(close_prices.max()), 'current': _safe_float(close_prices.iloc[-1]),
                        'count': int(len(close_prices)) }
                    try:
                        percentiles = { p: _safe_float(close_prices.quantile(float(p)/100)) for p in [10, 25, 50, 75, 90] }
                        # Filtra None dai percentili prima di assegnarli
                        price_stats['percentiles'] = {k: v for k, v in percentiles.items() if v is not None}
                    except Exception as perc_e: logger.warning(f"Errore calcolo percentili prezzo: {perc_e}"); price_stats['percentiles'] = {}
                    stats_dict['price'] = price_stats
                else: logger.warning("Nessun prezzo 'close' valido per stats.")
            else: logger.warning("Colonna 'close' mancante per stats prezzo.")

            if 'volume' in self.data.columns:
                volume = self.data['volume'].dropna()
                # Considera volumi zero come validi a meno che non siano *tutti* zero
                if not volume.empty and not (volume == 0).all():
                    volume_stats = {
                        'mean': _safe_float(volume.mean()), 'median': _safe_float(volume.median()),
                        'std': _safe_float(volume.std()), 'min': _safe_float(volume.min()),
                        'max': _safe_float(volume.max()), 'current': _safe_float(volume.iloc[-1]),
                        'count': int(len(volume)) }
                    stats_dict['volume'] = volume_stats
                else: logger.warning("Nessun volume valido (>0) o colonna volume vuota.")
            else: logger.warning("Colonna 'volume' mancante per stats volume.")

        except Exception as e:
            logger.error(f"Errore statistiche descrittive: {str(e)}", exc_info=True)
            return {'price': {}, 'volume': {}} # Ritorna struttura vuota in caso di errore
        logger.debug(f"{self.__class__.__name__}.calculate_descriptive_statistics() - FINE")
        return stats_dict

    def identify_market_condition(self) -> Dict[str, Any]:
        """
        Identifica condizione di mercato preliminare (trend e volatilità)
        basandosi sui risultati disponibili in self.results (principalmente da TechnicalAnalyzer).

        Returns:
            Dict[str, Any]: Dizionario con 'trend_strength', 'volatility', 'adx', 'relative_atr'.
                            I valori possono essere None o 'Unknown' se i dati base mancano.
        """
        logger.debug(f"{self.__class__.__name__}.identify_market_condition() - INIZIO")
        adx_value = None; trend_strength = 'Unknown'; volatility = 'Unknown'; relative_atr = None

        # Recupera risultati analisi tecnica da self.results (che sarà stato popolato da run_analysis)
        technical_results = self.results.get('technical_analysis', {})
        if not technical_results or 'error' in technical_results:
             logger.warning("Risultati analisi tecnica non trovati o contengono errore in self.results per identify_market_condition.")
        else:
            # Estrai ADX
            trend_analysis = technical_results.get('trend_analysis', {})
            adx_data = trend_analysis.get('adx', {})
            adx_value = adx_data.get('adx')
            if adx_value is not None and pd.notna(adx_value):
                if adx_value > 25: trend_strength = 'forte'
                elif adx_value > 20: trend_strength = 'moderato'
                else: trend_strength = 'debole' # Include < 20
            else: trend_strength = 'sconosciuta (ADX N/A)'

            # Estrai ATR e prezzo corrente per calcolare volatilità relativa
            # Usa _safe_get per accedere ad ATR nel dizionario nidificato
            atr = _safe_get(technical_results, ['technical_indicators', 'atr'])
            # Il prezzo corrente è calcolato dalle statistiche descrittive (o direttamente dai dati)
            current_close = _safe_get(self.results, ['descriptive', 'price', 'current'])
            if current_close is None: # Fallback se non ancora calcolato
                current_close = safe_get_last_value(self.data.get('close'))

            if atr is not None and pd.notna(atr) and atr > 0 and \
               current_close is not None and pd.notna(current_close) and current_close > 0:
                try:
                    relative_atr = atr / current_close
                    if relative_atr > 0.04: volatility = 'alta'
                    elif relative_atr > 0.015: volatility = 'media'
                    else: volatility = 'bassa'
                except TypeError as te: logger.error(f"Errore tipo calcolo ATR relativo: ATR={atr}, Close={current_close}. {te}"); volatility = 'sconosciuta (Errore Tipo)'; relative_atr = None
            else:
                volatility = 'sconosciuta (ATR/Prezzo N/A)'

        logger.debug(f"Condizione Mercato: Trend={trend_strength}, Volatilità={volatility}, ATR Rel={relative_atr}")
        logger.debug(f"{self.__class__.__name__}.identify_market_condition() - FINE")
        return {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'adx': _safe_float(adx_value), # Assicura sia float o None
            'relative_atr': _safe_float(relative_atr) # Assicura sia float o None
        }

    def calculate_vwap(self, period: str = 'D') -> Optional[pd.Series]:
        """
        Calcola il Volume Weighted Average Price (VWAP) giornaliero ('D')
        o per una finestra mobile numerica (es. '20').

        Args:
            period (str): Periodo per il calcolo ('D' per giornaliero, o un numero per finestra mobile).

        Returns:
            Optional[pd.Series]: Serie Pandas con i valori VWAP, o None se fallisce.
        """
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Dati OHLCV incompleti per VWAP (periodo: {period}).")
            return None

        # Lavora su una copia pulita
        df_vwap = self.data[required_cols].copy()
        df_vwap = df_vwap.dropna(subset=required_cols) # Rimuovi righe con NaN in OHLCV
        if df_vwap.empty: logger.warning(f"Nessun dato valido per VWAP dopo pulizia (periodo: {period})."); return None

        # Calcola Typical Price * Volume
        df_vwap['typical_price'] = (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3
        df_vwap['tp_vol'] = df_vwap['typical_price'] * df_vwap['volume']

        # Gestisci NaN/Inf introdotti dai calcoli
        df_vwap = df_vwap.replace([np.inf, -np.inf], np.nan).dropna(subset=['tp_vol', 'volume'])
        if df_vwap.empty: logger.warning(f"Nessun dato valido per VWAP dopo calcolo tp_vol (periodo: {period})."); return None

        vwap: Optional[pd.Series] = None
        try:
            if period.upper() == 'D':
                 if not isinstance(df_vwap.index, pd.DatetimeIndex):
                     logger.error("Indice non DatetimeIndex per VWAP giornaliero.")
                     return None
                 # Resetta il calcolo all'inizio di ogni giorno
                 daily_cumsum = df_vwap.groupby(df_vwap.index.date)[['tp_vol', 'volume']].cumsum()
                 # Calcola VWAP, gestendo divisione per zero
                 valid_volume = daily_cumsum['volume'].replace(0, np.nan)
                 vwap = daily_cumsum['tp_vol'] / valid_volume
            else:
                # Prova a interpretare il periodo come finestra mobile numerica
                try: window_size = int(period)
                except (ValueError, TypeError): logger.warning(f"Periodo VWAP non valido: {period}. Deve essere 'D' o un numero."); return None
                if window_size <= 0: logger.warning("Window size VWAP deve essere > 0."); return None
                if len(df_vwap) < window_size: logger.warning(f"Dati insuff. ({len(df_vwap)}) per VWAP window {window_size}."); return pd.Series(index=df_vwap.index, dtype=float) # Restituisci NaN Series?

                # Calcola somme rolling
                rolling_tp_vol = df_vwap['tp_vol'].rolling(window=window_size, min_periods=1).sum()
                rolling_volume = df_vwap['volume'].rolling(window=window_size, min_periods=1).sum()
                # Calcola VWAP, gestendo divisione per zero
                valid_volume = rolling_volume.replace(0, np.nan)
                vwap = rolling_tp_vol / valid_volume

            # Pulizia finale del risultato VWAP
            if vwap is not None:
                vwap = vwap.replace([np.inf, -np.inf], np.nan)
            return vwap

        except Exception as e:
            logger.error(f"Errore calcolo VWAP (periodo: {period}): {e}", exc_info=True)
            return None

    def run_analysis(self) -> Dict[str, Any]:
        """
        Esegue le analisi statistiche di base e chiama TechnicalAnalyzer.
        """
        logger.debug(f"{self.__class__.__name__}.run_analysis() - INIZIO")
        self.results = {} # Resetta risultati interni

        if self.data is None or self.data.empty:
            logger.error("Dati vuoti, impossibile eseguire l'analisi.")
            return {'error': 'Dati input mancanti o vuoti'}

        try:
            # 1. Esegui Analisi Tecnica *PRIMA* (Import locale)
            tech_results_dict = {}
            try:
                from technical_analyzer import TechnicalAnalyzer
                logger.debug("Istanza TechnicalAnalyzer...")
                technical_analyzer = TechnicalAnalyzer(self.data.copy())
                tech_results_dict = technical_analyzer.run_analysis()
                self.results['technical_analysis'] = tech_results_dict
                logger.debug("Analisi tecnica completata.")
            except ImportError:
                 logger.error("Errore critico importando TechnicalAnalyzer in run_analysis.")
                 self.results['technical_analysis'] = {'error': "TechnicalAnalyzer non importato"}
            except Exception as ta_err:
                 logger.error(f"Errore durante analisi tecnica: {ta_err}", exc_info=True)
                 self.results['technical_analysis'] = {'error': str(ta_err)}

            # 2. Calcola Statistiche Descrittive
            self.results['descriptive'] = self.calculate_descriptive_statistics()
            logger.debug("Statistiche descrittive calcolate.")

            # 3. Calcolo VWAP (solo giornaliero per semplicità nel report base)
            vwap_series = self.calculate_vwap(period='D')
            last_vwap = safe_get_last_value(vwap_series, default=None)
            last_close = _safe_get(self.results, ['descriptive', 'price', 'current'])
            if last_vwap is not None and last_close is not None:
                # Aggiungi controllo per divisione per zero
                distance_pct = None
                if last_vwap != 0:
                     distance_pct = _safe_float(((last_close - last_vwap) / last_vwap) * 100)
                self.results['vwap_daily'] = {
                    'value': _safe_float(last_vwap),
                    'relation': "above" if last_close > last_vwap else "below" if last_close < last_vwap else "at",
                    'distance_pct': distance_pct
                }
                logger.debug(f"VWAP Giornaliero calcolato: {self.results['vwap_daily']}")
            else:
                 self.results['vwap_daily'] = {'value': None, 'relation': 'unknown', 'distance_pct': None}
                 logger.debug(f"VWAP Giornaliero non calcolato o non valido (LastVWAP: {last_vwap}, LastClose: {last_close}).")

            # 4. Identifica Condizione di Mercato
            self.results['market_condition'] = self.identify_market_condition()
            logger.debug("Condizione di mercato identificata.")

            logger.debug(f"{self.__class__.__name__}.run_analysis() - FINE")
            return self.results

        except Exception as e:
            logger.error(f"Errore generale in StatisticalAnalyzer.run_analysis: {str(e)}", exc_info=True)
            if 'error' not in self.results: # Aggiungi errore generale se non già presente
                self.results['error'] = f"Analisi base fallita: {e}"
            return self.results

# --- END OF FILE statistical_analyzer.py ---