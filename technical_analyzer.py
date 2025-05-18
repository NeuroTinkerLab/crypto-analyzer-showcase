# --- START OF FILE technical_analyzer.py ---
import pandas as pd
import numpy as np
import logging
import re
# Importa config SOLO per le costanti, non altri moduli
try:
    import config # type: ignore
except ImportError:
    # Fallback se config.py non esiste o causa errori
    class ConfigFallback:
        RSI_PERIOD = 14
        MACD_FAST = 12
        MACD_SLOW = 26
        MACD_SIGNAL = 9
        BBANDS_PERIOD = 20
        BBANDS_STDDEV = 2.0
        ATR_PERIOD = 14
        ADX_PERIOD = 14
        SMA_PERIODS = [20, 50, 200]
        EMA_PERIODS = [12, 26, 50]
        # Aggiungiamo fallback per i nuovi parametri se config non carica
        TA_VOLUME_SMA_SHORT = 10
        TA_VOLUME_ZSCORE_PERIOD = 20
        TA_UPDOWN_VOL_PERIOD = 10
        TA_VWAP_ROLLING_PERIOD = 20
        TA_VOLATILITY_PERCENTILE_PERIOD = 100
        TA_ATR_TARGET_MULTIPLIERS = [1.0, 1.5, 2.0] # Esempio
    config = ConfigFallback()
    logging.warning("config.py non trovato o errore import. Uso valori di default per parametri TA.")
# Assicura che i parametri abbiano un fallback anche se config esiste ma manca la chiave
TA_VOLUME_SMA_SHORT = getattr(config, 'TA_VOLUME_SMA_SHORT', 10)
TA_VOLUME_ZSCORE_PERIOD = getattr(config, 'TA_VOLUME_ZSCORE_PERIOD', 20)
TA_UPDOWN_VOL_PERIOD = getattr(config, 'TA_UPDOWN_VOL_PERIOD', 10)
TA_VWAP_ROLLING_PERIOD = getattr(config, 'TA_VWAP_ROLLING_PERIOD', 20)
TA_VOLATILITY_PERCENTILE_PERIOD = getattr(config, 'TA_VOLATILITY_PERCENTILE_PERIOD', 100)
TA_ATR_TARGET_MULTIPLIERS = getattr(config, 'TA_ATR_TARGET_MULTIPLIERS', [1.0, 1.5, 2.0])


from typing import Dict, Any, List, Optional, Union, Callable
import sys # Per controllo numba
import time
import math # Per isnan
import scipy.stats as stats # Importato per percentileofscore

# Importa helper SOLO da statistical_analyzer_helpers
try:
    from statistical_analyzer_helpers import safe_get_last_value, safe_get_value_at_index, _safe_get, _safe_float
    HELPERS_LOADED = True
except ImportError:
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.critical(
        "ERRORE CRITICO: statistical_analyzer_helpers non trovato! "
        "L'analisi tecnica fallirà. Assicurati che il file esista e sia nel PYTHONPATH."
    )
    HELPERS_LOADED = False
    # Definizioni fittizie
    def _safe_get(*args, **kwargs): return None
    def safe_get_value_at_index(*args, **kwargs): return None
    def safe_get_last_value(*args, **kwargs): return None
    def _safe_float(*args, **kwargs): return None

# Importa pandas_ta
try:
    import pandas_ta as ta
except ImportError:
    logging.critical("ERRORE CRITICO: Libreria pandas_ta non trovata. Installala con: pip install pandas-ta")
    ta = None

# Setup logging
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Calcola indicatori tecnici, S/R, trend, Volume Profile (incluso giornaliero/settimanale),
    e feature tecniche combinate. Utilizza pandas_ta per i calcoli degli indicatori.
    Include analisi volume, VWAP rolling, percentili volatilità e target ATR.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inizializza l'analizzatore tecnico.
        """
        if not HELPERS_LOADED:
            raise ImportError("statistical_analyzer_helpers non caricato. Impossibile inizializzare TechnicalAnalyzer.")

        if data is None or data.empty: raise ValueError("Dati non validi per TechnicalAnalyzer.")
        # Validazione indice e timezone
        if not isinstance(data.index, pd.DatetimeIndex):
            try: data.index = pd.to_datetime(data.index, utc=True)
            except Exception as e: raise TypeError(f"Indice non DatetimeIndex in TA: {e}")
        if data.index.tz is None or str(data.index.tz).upper() != 'UTC':
             logger.warning("Indice non UTC in TA. Forzo UTC.")
             try: data.index = data.index.tz_localize('UTC', ambiguous='infer') if data.index.tz is None else data.index.tz_convert('UTC')
             except Exception as tz_err: raise TypeError(f"Errore forzatura UTC in TA: {tz_err}")

        self.data = data.copy() # Lavora su copia

        # Determina timeframe input
        self.input_timeframe_delta: Optional[pd.Timedelta] = None
        if len(self.data.index) > 1:
            time_diffs = self.data.index.to_series().diff().dropna()
            # Use median to be robust against missing candles
            if not time_diffs.empty:
                self.input_timeframe_delta = time_diffs.median()
            else: # Fallback for only two data points
                self.input_timeframe_delta = self.data.index[1] - self.data.index[0]

            # Final check if delta is valid
            if not isinstance(self.input_timeframe_delta, pd.Timedelta) or self.input_timeframe_delta <= pd.Timedelta(0):
                self.input_timeframe_delta = None
                logger.warning("Impossibile determinare un delta timeframe valido.")

        logger.debug(f"Delta timeframe input rilevato: {self.input_timeframe_delta}")

        # Normalizza nomi colonne e verifica/prepara OHLCV
        self.data.columns = [str(col).lower().replace(' ','_') for col in self.data.columns]
        required = ['open', 'high', 'low', 'close']
        missing = [col for col in required if col not in self.data.columns]
        if missing: raise ValueError(f"Colonne OHLC mancanti in TA: {missing}")
        if 'volume' not in self.data.columns:
            logger.warning("Colonna 'volume' mancante in TA. Aggiungo colonna volume=0.")
            self.data['volume'] = 0.0

        # Conversione numerica e dropna
        for col in required + ['volume']:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype(float)

        initial_len = len(self.data)
        self.data.dropna(subset=required, inplace=True)
        rows_dropped = initial_len - len(self.data)
        if rows_dropped > 0: logger.debug(f"TA: Rimossi {rows_dropped} righe con NaN in OHLC.")
        if self.data.empty: raise ValueError("TA: Dati vuoti dopo rimozione NaN OHLC.")

        self.results: Dict[str, Any] = {}
        self._ensure_results_structure()

    def _ensure_results_structure(self):
        """Assicura la struttura base del dizionario results."""
        self.results.setdefault('technical_indicators', {})
        self.results.setdefault('trend_analysis', {})
        self.results.setdefault('support_resistance', {})
        self.results.setdefault('volume_analysis', {}) # NUOVA SEZIONE
        self.results.setdefault('volatility_analysis', {}) # NUOVA SEZIONE (per percentili)
        self.results.setdefault('price_level_targets', {}) # NUOVA SEZIONE (per target ATR)
        self.results.setdefault('volume_profile', {})
        self.results.setdefault('volume_profile_periodic', {'daily': {}, 'weekly': {}})
        self.results.setdefault('combined_features', {})

    # --- Calcolo Indicatori (Helper e Specifici) ---
    def _calculate_indicator(self, indicator_func: Callable, name: str, result_key: str, default_value: Any = None, **kwargs):
        """
        Helper generico per calcolare indicatori con pandas_ta, aggiornare self.data
        e preparare la voce base nel dizionario results.
        """
        # (Codice invariato)
        ti_dict = self.results.setdefault('technical_indicators', {})
        is_dict_output = isinstance(default_value, dict)

        # Initialize or clean the result entry
        if result_key not in ti_dict:
             ti_dict[result_key] = default_value
        elif is_dict_output:
            if isinstance(ti_dict.get(result_key), dict):
                 if isinstance(default_value, dict):
                     for k, v in default_value.items(): ti_dict[result_key].setdefault(k, v)
                 ti_dict[result_key].pop('error', None) # Clean previous error
            else: # Overwrite if type mismatch
                ti_dict[result_key] = default_value
                if isinstance(ti_dict[result_key], dict): ti_dict[result_key].pop('error', None)
        else: # Scalar output
             ti_dict[result_key] = default_value # Reset to default
             ti_dict.pop(f'{result_key}_error', None) # Clean previous scalar error

        if ta is None:
            error_msg = 'pandas_ta missing'
            if is_dict_output and isinstance(ti_dict.get(result_key), dict): ti_dict[result_key]['error'] = error_msg
            else: ti_dict[result_key] = None; ti_dict[f'{result_key}_error'] = error_msg
            return
        if self.data.empty:
             error_msg = 'No data'
             if is_dict_output and isinstance(ti_dict.get(result_key), dict): ti_dict[result_key]['error'] = error_msg
             else: ti_dict[result_key] = None; ti_dict[f'{result_key}_error'] = error_msg
             return

        logger.debug(f"Calcolo {name} con parametri: {list(kwargs.keys())}")
        try:
            # Prepare arguments for pandas_ta function
            relevant_kwargs = {}
            input_cols = ['open', 'high', 'low', 'close', 'volume'] # Standard inputs
            for col in input_cols:
                if col in self.data.columns:
                    relevant_kwargs[col] = self.data[col] # Pass the whole series

            # Pass specific parameters defined in the call
            param_keys = ['length', 'fast', 'slow', 'signal', 'std', 'k', 'd', 'smooth_k', 'af', 'max_af', 'period'] # Common TA params
            for key in param_keys:
                if key in kwargs:
                    relevant_kwargs[key] = kwargs[key]

            # Determine minimum length needed based on parameters (best effort)
            period_arg = 14 # Default reasonable period
            relevant_periods = [v for k, v in relevant_kwargs.items() if k in ['length', 'slow', 'k', 'period'] and isinstance(v, int)]
            if relevant_periods:
                period_arg = max(relevant_periods)

            min_len_needed = max(5, period_arg + 1) # General rule of thumb
            # Specific indicator minimum lengths
            if 'bbands' in name.lower(): min_len_needed = max(min_len_needed, relevant_kwargs.get('length', 20))
            if 'macd' in name.lower(): min_len_needed = max(min_len_needed, relevant_kwargs.get('slow', 26) + relevant_kwargs.get('signal', 9))
            if 'adx' in name.lower(): min_len_needed = max(min_len_needed, 2 * relevant_kwargs.get('length', 14)) # ADX needs more data
            if 'stoch' in name.lower(): min_len_needed = max(min_len_needed, relevant_kwargs.get('k', 14) + relevant_kwargs.get('d', 3))
            if 'psar' in name.lower(): min_len_needed = max(min_len_needed, 10) # PSAR needs some history

            # Check available valid 'close' data points (or other primary series if needed)
            primary_series_col = 'close' # Assume close is primary unless specified otherwise
            if primary_series_col not in self.data.columns or len(self.data[primary_series_col].dropna()) < min_len_needed:
                raise ValueError(f"Dati '{primary_series_col}' validi insufficienti ({len(self.data.get(primary_series_col, pd.Series()).dropna())} < {min_len_needed})")

            # --- CALL PANDAS_TA INDICATOR FUNCTION ---
            indicator_result = indicator_func(**relevant_kwargs)
            # -----------------------------------------

            if indicator_result is None:
                raise ValueError(f"Funzione {name} ha restituito None.")

            # --- APPEND RESULTS TO self.data (Robustly) ---
            logger.debug(f"Tentativo append {name} a self.data...")
            if isinstance(indicator_result, (pd.DataFrame, pd.Series)):
                indicator_params = {k: v for k, v in kwargs.items() if k in param_keys}
                indicator_method_name = indicator_func.__name__ # Get the function name (e.g., 'sma')

                # Check if the method exists directly under df.ta
                if hasattr(self.data.ta, indicator_method_name):
                    try:
                        # Try using the built-in append mechanism
                        getattr(self.data.ta, indicator_method_name)(**indicator_params, append=True)
                        logger.debug(f"Appended {indicator_method_name} using df.ta mechanism.")
                    except Exception as ta_append_err:
                        logger.warning(f"Errore append {indicator_method_name} con df.ta: {ta_append_err}. Tento aggiunta manuale.")
                        # Fallback to manual addition
                        if isinstance(indicator_result, pd.DataFrame):
                            for col in indicator_result.columns:
                                if col not in self.data.columns: self.data[col] = indicator_result[col]
                                else: self.data[col].fillna(indicator_result[col], inplace=True) # Update existing? Or overwrite? Use fillna for now.
                        elif isinstance(indicator_result, pd.Series):
                            col_name = indicator_result.name or result_key # Use series name or result_key as fallback
                            if col_name not in self.data.columns: self.data[col_name] = indicator_result
                            else: self.data[col_name].fillna(indicator_result, inplace=True)
                else:
                    # Method not found under df.ta, add manually
                    logger.warning(f"Metodo 'ta.{indicator_method_name}' non trovato, aggiunta manuale colonne.")
                    if isinstance(indicator_result, pd.DataFrame):
                        for col in indicator_result.columns:
                            if col not in self.data.columns: self.data[col] = indicator_result[col]
                            else: self.data[col].fillna(indicator_result[col], inplace=True)
                    elif isinstance(indicator_result, pd.Series):
                        col_name = indicator_result.name or result_key
                        if col_name not in self.data.columns: self.data[col_name] = indicator_result
                        else: self.data[col_name].fillna(indicator_result, inplace=True)

                logger.debug(f"Colonne DataFrame dopo {name}: {self.data.columns.tolist()}")
            else:
                # This case should ideally not happen if pandas_ta functions are used correctly
                raise TypeError(f"Tipo risultato inatteso da {name}: {type(indicator_result)}")
            # --- END APPEND ---

        except ValueError as ve:
             # Specific error, likely insufficient data or bad parameters
             logger.warning(f"Calcolo {name} fallito (ValueError): {ve}"); error_msg = str(ve)
             if is_dict_output and isinstance(ti_dict.get(result_key), dict):
                 # Set other keys to None on error for dictionary outputs
                 for k in ti_dict[result_key]:
                      if k != 'error': ti_dict[result_key][k] = None
                 ti_dict[result_key]['error'] = error_msg
             else: # Scalar output
                 ti_dict[result_key] = None
                 ti_dict[f'{result_key}_error'] = error_msg
        except Exception as e:
            # General unexpected errors during calculation or appending
            logger.error(f"Errore generico calcolo/append {name}: {e}", exc_info=False); error_msg = f"Calc failed: {e}"
            if is_dict_output and isinstance(ti_dict.get(result_key), dict):
                for k in ti_dict[result_key]:
                     if k != 'error': ti_dict[result_key][k] = None
                ti_dict[result_key]['error'] = error_msg
            else:
                ti_dict[result_key] = None
                ti_dict[f'{result_key}_error'] = error_msg

    # --- Metodi Calcolo Indicatori Specifici (Aggiornati per usare _calculate_indicator) ---
    # (La logica di estrazione e interpretazione rimane invariata, ma il calcolo è delegato)

    def _calculate_moving_averages(self) -> None:
        """Calcola medie mobili SMA ed EMA e rileva crossover."""
        # (Logica invariata)
        if ta is None: logger.error("pandas_ta mancante per medie mobili."); return
        ti_key = 'technical_indicators'; ma_key = 'moving_averages'; self.results[ti_key].setdefault(ma_key, {})
        local_ma_results = {}

        for period in config.SMA_PERIODS:
            self._calculate_indicator(ta.sma, name=f"SMA_{period}", result_key=f"_temp_sma_{period}", length=period)
            sma_col_name = next((col for col in self.data.columns if col.lower() == f"sma_{period}".lower()), None)
            if sma_col_name:
                local_ma_results[f'sma_{period}'] = _safe_float(safe_get_last_value(self.data[sma_col_name]))
            else:
                logger.warning(f"Colonna SMA_{period} (o simile) non trovata dopo calcolo."); local_ma_results[f'sma_{period}'] = None
            self.results[ti_key].pop(f"_temp_sma_{period}", None)
            self.results[ti_key].pop(f"_temp_sma_{period}_error", None)

        for period in config.EMA_PERIODS:
            self._calculate_indicator(ta.ema, name=f"EMA_{period}", result_key=f"_temp_ema_{period}", length=period)
            ema_col_name = next((col for col in self.data.columns if col.lower() == f"ema_{period}".lower()), None)
            if ema_col_name:
                local_ma_results[f'ema_{period}'] = _safe_float(safe_get_last_value(self.data[ema_col_name]))
            else:
                logger.warning(f"Colonna EMA_{period} (o simile) non trovata dopo calcolo."); local_ma_results[f'ema_{period}'] = None
            self.results[ti_key].pop(f"_temp_ema_{period}", None)
            self.results[ti_key].pop(f"_temp_ema_{period}_error", None)

        self.results[ti_key][ma_key] = local_ma_results

        crossover_results = {'golden_cross_recent_5p': None, 'death_cross_recent_5p': None}
        sma50_col_df = next((col for col in self.data.columns if col.lower() == "sma_50"), None)
        sma200_col_df = next((col for col in self.data.columns if col.lower() == "sma_200"), None)

        if sma50_col_df and sma200_col_df:
             sma50_series = self.data[sma50_col_df]; sma200_series = self.data[sma200_col_df]
             if sma50_series.dropna().size >= 2 and sma200_series.dropna().size >= 2:
                 golden = (sma50_series > sma200_series) & (sma50_series.shift(1) <= sma200_series.shift(1))
                 death = (sma50_series < sma200_series) & (sma50_series.shift(1) >= sma200_series.shift(1))
                 lookback = 5
                 if len(golden) >= lookback: crossover_results['golden_cross_recent_5p'] = bool(golden.iloc[-lookback:].any())
                 if len(death) >= lookback: crossover_results['death_cross_recent_5p'] = bool(death.iloc[-lookback:].any())
             else: logger.debug("Serie SMA50/SMA200 non hanno abbastanza punti validi per crossover.")
        else: logger.debug("Colonne SMA50/SMA200 standard non trovate per calcolo crossover.")
        self.results[ti_key]['crossovers'] = crossover_results

    def _calculate_rsi(self, period: int = config.RSI_PERIOD) -> None:
        """Calcola RSI e determina la condizione."""
        # (Logica invariata)
        result_key = 'rsi'; default_value = {'value': None, 'condition': 'Unknown', 'base_thresholds': {'ob': 70, 'os': 30}, 'error': None}
        self._calculate_indicator(ta.rsi, name=f"RSI_{period}", result_key=result_key, length=period, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        rsi_col_name = next((col for col in self.data.columns if col.lower() == f"rsi_{period}".lower()), None)
        latest_rsi = None

        if rsi_col_name:
             latest_rsi = safe_get_last_value(self.data[rsi_col_name]);
             if isinstance(current_result, dict):
                 current_result['value'] = _safe_float(latest_rsi);
                 if latest_rsi is not None: current_result.pop('error', None)
        elif isinstance(current_result, dict) and not current_result.get('error'):
            current_result['error'] = f"Colonna RSI_{period} (o simile) non trovata"

        if isinstance(current_result, dict) and not current_result.get('error'):
            latest_rsi_val = current_result.get('value')
            if latest_rsi_val is not None:
                OB_BASE, OS_BASE = 70, 30; NEAR_OB_BASE, NEAR_OS_BASE = OB_BASE - 5, OS_BASE + 5
                if latest_rsi_val > OB_BASE: current_result['condition'] = "Overbought"
                elif latest_rsi_val < OS_BASE: current_result['condition'] = "Oversold"
                elif latest_rsi_val > NEAR_OB_BASE: current_result['condition'] = "Near Overbought"
                elif latest_rsi_val < NEAR_OS_BASE: current_result['condition'] = "Near Oversold"
                else: current_result['condition'] = "Neutral"
            else: current_result['condition'] = "Unknown (NaN Value)"

    def _calculate_macd(self, fast: int = config.MACD_FAST, slow: int = config.MACD_SLOW, signal: int = config.MACD_SIGNAL) -> None:
        """Calcola MACD, linee e istogramma, e determina condizione."""
        # (Logica invariata)
        result_key = 'macd'; default_value = {'macd_line': None, 'signal_line': None, 'histogram': None, 'condition': 'Unknown', 'error': None}
        self._calculate_indicator(ta.macd, name="MACD", result_key=result_key, fast=fast, slow=slow, signal=signal, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        macd_col = next((c for c in self.data.columns if c.lower() == f"macd_{fast}_{slow}_{signal}".lower()), None)
        hist_col = next((c for c in self.data.columns if c.lower() == f"macdh_{fast}_{slow}_{signal}".lower()), None)
        signal_col = next((c for c in self.data.columns if c.lower() == f"macds_{fast}_{slow}_{signal}".lower()), None)

        values_found = False
        if macd_col: current_result['macd_line'] = _safe_float(safe_get_last_value(self.data[macd_col])); values_found = True
        if hist_col: current_result['histogram'] = _safe_float(safe_get_last_value(self.data[hist_col])); values_found = True
        if signal_col: current_result['signal_line'] = _safe_float(safe_get_last_value(self.data[signal_col])); values_found = True

        if values_found and isinstance(current_result, dict):
            current_result.pop('error', None) # Remove helper error if values were found
        elif not values_found and isinstance(current_result, dict) and not current_result.get('error'):
            current_result['error'] = "Colonne MACD (o simili) non trovate"

        if isinstance(current_result, dict) and not current_result.get('error'):
            latest_histogram = current_result.get('histogram')
            if latest_histogram is not None:
                 tolerance = 1e-9
                 prev_histogram = safe_get_value_at_index(self.data.get(hist_col), index=-2) if hist_col else None
                 if prev_histogram is not None:
                      is_strengthening = False
                      if latest_histogram > tolerance and prev_histogram > tolerance: is_strengthening = latest_histogram > prev_histogram
                      elif latest_histogram < -tolerance and prev_histogram < -tolerance: is_strengthening = latest_histogram < prev_histogram # Corrected logic for bearish strengthening
                      elif latest_histogram > tolerance and prev_histogram <= tolerance: is_strengthening = True # Crossed zero bullish
                      elif latest_histogram < -tolerance and prev_histogram >= -tolerance: is_strengthening = True # Crossed zero bearish
                      strength_label = " (Strengthening)" if is_strengthening else " (Weakening)"
                      if latest_histogram > tolerance: current_result['condition'] = f"Bullish{strength_label}"
                      elif latest_histogram < -tolerance: current_result['condition'] = f"Bearish{strength_label}"
                      else: current_result['condition'] = "Neutral (Zero Cross/Flat)"
                 else: # No history to compare with
                     if latest_histogram > tolerance: current_result['condition'] = "Bullish"
                     elif latest_histogram < -tolerance: current_result['condition'] = "Bearish"
                     else: current_result['condition'] = "Neutral (Zero Cross/Flat)"
            else: current_result['condition'] = "Unknown (NaN Value)"

    def _calculate_bollinger_bands(self, period: int = config.BBANDS_PERIOD, std_dev: float = config.BBANDS_STDDEV) -> None:
        """Calcola Bande di Bollinger e condizioni associate."""
        # (Logica invariata)
        result_key = 'bollinger_bands'; default_value = {'middle': None, 'upper': None, 'lower': None, 'bandwidth': None, 'percent_b': None, 'condition': 'Unknown', 'condition_pb': 'Unknown', 'error': None}
        self._calculate_indicator(ta.bbands, name="BBANDS", result_key=result_key, length=period, std=std_dev, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        std_dev_str = f"{std_dev:.1f}"
        upper_col = next((c for c in self.data.columns if c.lower() == f"bbu_{period}_{std_dev_str}".lower()), None)
        middle_col = next((c for c in self.data.columns if c.lower() == f"bbm_{period}_{std_dev_str}".lower()), None)
        lower_col = next((c for c in self.data.columns if c.lower() == f"bbl_{period}_{std_dev_str}".lower()), None)
        bw_col = next((c for c in self.data.columns if c.lower() == f"bbb_{period}_{std_dev_str}".lower()), None)
        perc_b_col = next((c for c in self.data.columns if c.lower() == f"bbp_{period}_{std_dev_str}".lower()), None)

        values_found = False
        if upper_col: current_result['upper'] = _safe_float(safe_get_last_value(self.data[upper_col])); values_found = True
        if middle_col: current_result['middle'] = _safe_float(safe_get_last_value(self.data[middle_col])); values_found = True
        if lower_col: current_result['lower'] = _safe_float(safe_get_last_value(self.data[lower_col])); values_found = True
        if bw_col: current_result['bandwidth'] = _safe_float(safe_get_last_value(self.data[bw_col])); values_found = True
        if perc_b_col: current_result['percent_b'] = _safe_float(safe_get_last_value(self.data[perc_b_col])); values_found = True

        if values_found and isinstance(current_result, dict):
            current_result.pop('error', None)
        elif not values_found and isinstance(current_result, dict) and not current_result.get('error'):
            current_result['error'] = f"Colonne BBANDS_{period}_{std_dev_str} (o simili) non trovate"

        if isinstance(current_result, dict) and not current_result.get('error'):
            latest_close = safe_get_last_value(self.data['close'])
            latest_upper = current_result.get('upper'); latest_lower = current_result.get('lower'); latest_percent_b = current_result.get('percent_b')
            if latest_close is not None and latest_upper is not None and latest_lower is not None:
                band_width = latest_upper - latest_lower
                tolerance = band_width * 0.02 if band_width > 0 else 1e-6
                if latest_close > (latest_upper - tolerance): current_result['condition'] = "Price Above/Near Upper Band"
                elif latest_close < (latest_lower + tolerance): current_result['condition'] = "Price Below/Near Lower Band"
                else: current_result['condition'] = "Price Within Bands"
            else: current_result['condition'] = "Unknown (NaN Value)"
            if latest_percent_b is not None:
                if latest_percent_b > 1.0: current_result['condition_pb'] = "Overbought (>1)"
                elif latest_percent_b < 0.0: current_result['condition_pb'] = "Oversold (<0)"
                elif latest_percent_b > 0.8: current_result['condition_pb'] = "Near Upper Band (>0.8)"
                elif latest_percent_b < 0.2: current_result['condition_pb'] = "Near Lower Band (<0.2)"
                else: current_result['condition_pb'] = "Mid-Range (0.2-0.8)"
            else: current_result['condition_pb'] = "Unknown (NaN %B)"

    def calculate_atr(self, period=config.ATR_PERIOD) -> Optional[float]:
        """Calcola ATR, lo salva nei results e lo restituisce."""
        # (Logica invariata)
        result_key = 'atr'; default_value = None
        self._calculate_indicator(ta.atr, name=f"ATR_{period}", result_key=result_key, length=period, default_value=default_value)

        ti_dict = self.results.setdefault('technical_indicators', {})
        helper_error = ti_dict.get(f'{result_key}_error')
        final_value = None
        atr_col_name_found = None

        if not helper_error:
            potential_cols = [f'ATR_{period}', f'ATRr_{period}', f'ATRe_{period}']
            for col_base in potential_cols:
                atr_col_name_found = next((col for col in self.data.columns if col.lower() == col_base.lower()), None)
                if atr_col_name_found:
                    logger.debug(f"Trovata colonna ATR: '{atr_col_name_found}'")
                    final_value = _safe_float(safe_get_last_value(self.data[atr_col_name_found]))
                    break
            if atr_col_name_found is None:
                 logger.warning(f"Colonna ATR_{period} (o simile) non trovata nel DataFrame dopo calcolo.")

        ti_dict[result_key] = final_value

        if final_value is not None:
            ti_dict.pop(f'{result_key}_error', None)
        else:
            if not helper_error:
                error_msg = f"Colonna ATR_{period} non trovata o NaN"
                ti_dict[f'{result_key}_error'] = error_msg
                logger.warning(error_msg)

        if final_value is not None and final_value > 0:
            return final_value
        else:
             if final_value is not None:
                  logger.warning(f"ATR calcolato ({final_value}) non è positivo.")
             return None

    def _calculate_stochastic(self, k: int = 14, d: int = 3, smooth_k: int = 3) -> None:
        """Calcola Stocastico, estrae k e d correttamente, e determina condizione."""
        # (Logica invariata)
        result_key = 'stochastic'; default_value = {'k': None, 'd': None, 'condition': 'Unknown', 'error': None}
        self._calculate_indicator(ta.stoch, name="Stochastic", result_key=result_key, k=k, d=d, smooth_k=smooth_k, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        if isinstance(current_result, dict) and current_result.get('error'):
            logger.warning(f"Salto interpretazione Stocastico causa errore calcolo: {current_result['error']}")
            current_result['k'] = None; current_result['d'] = None; current_result['condition'] = "Unknown (Calculation Error)"; return

        k_col_name_generated = f'STOCHk_{k}_{d}_{smooth_k}'; d_col_name_generated = f'STOCHd_{k}_{d}_{smooth_k}'
        k_col_df = next((c for c in self.data.columns if c.lower() == k_col_name_generated.lower()), None)
        d_col_df = next((c for c in self.data.columns if c.lower() == d_col_name_generated.lower()), None)

        latest_k = safe_get_last_value(self.data.get(k_col_df)) if k_col_df else None
        latest_d = safe_get_last_value(self.data.get(d_col_df)) if d_col_df else None

        current_result['k'] = _safe_float(latest_k); current_result['d'] = _safe_float(latest_d)

        keys_to_remove = [key for key in list(current_result.keys()) if key not in ['k', 'd', 'condition', 'error'] and key.lower().startswith(('stochk','stochd'))]
        for key in keys_to_remove: current_result.pop(key, None)

        if current_result['k'] is not None and current_result['d'] is not None:
            prev_k = safe_get_value_at_index(self.data.get(k_col_df), index=-2) if k_col_df else None
            prev_d = safe_get_value_at_index(self.data.get(d_col_df), index=-2) if d_col_df else None
            OB, OS = 80, 20; condition = 'Unknown'
            if current_result['k'] > OB and current_result['d'] > OB: condition = "Overbought"
            elif current_result['k'] < OS and current_result['d'] < OS: condition = "Oversold"
            elif prev_k is not None and prev_d is not None:
                 tolerance = 1e-6
                 if (prev_k <= prev_d + tolerance) and (current_result['k'] > current_result['d'] + tolerance): condition = "Bullish Crossover"
                 elif (prev_k >= prev_d - tolerance) and (current_result['k'] < current_result['d'] - tolerance): condition = "Bearish Crossover"
            if condition == 'Unknown':
                if current_result['k'] > current_result['d']: condition = "K above D"
                elif current_result['k'] < current_result['d']: condition = "K below D"
                else: condition = "K equals D"
            current_result['condition'] = condition
            current_result.pop('error', None)
        else:
            current_result['condition'] = "Unknown (NaN Value)"
            if not current_result.get('error'):
                current_result['error'] = f"Colonne {k_col_name_generated}/{d_col_name_generated} (o simili) non trovate o NaN"

    def _calculate_cci(self, period: int = 20) -> None:
        """Calcola CCI e determina condizione."""
        # (Logica invariata)
        result_key = 'cci'; default_value = {'value': None, 'condition': 'Unknown', 'error': None}
        self._calculate_indicator(ta.cci, name=f"CCI_{period}", result_key=result_key, length=period, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        cci_col_name = next((c for c in self.data.columns if c.lower().startswith(f"cci_{period}")), None)
        latest_cci = None

        if cci_col_name:
            latest_cci = safe_get_last_value(self.data[cci_col_name])
            if isinstance(current_result, dict):
                current_result['value'] = _safe_float(latest_cci);
                if latest_cci is not None: current_result.pop('error', None)
        elif isinstance(current_result, dict) and not current_result.get('error'):
            current_result['error'] = f"Colonna CCI_{period} (o simile) non trovata"

        if isinstance(current_result, dict) and not current_result.get('error'):
            latest_cci_val = current_result.get('value')
            if latest_cci_val is not None:
                OB, OS, NearOB, NearOS, ExtremOB, ExtremOS = 100, -100, 90, -90, 150, -150
                if latest_cci_val > ExtremOB: current_result['condition'] = "Extreme Overbought (>150)"
                elif latest_cci_val < ExtremOS: current_result['condition'] = "Extreme Oversold (<-150)"
                elif latest_cci_val > OB: current_result['condition'] = "Overbought (>100)"
                elif latest_cci_val < OS: current_result['condition'] = "Oversold (<-100)"
                elif latest_cci_val > NearOB: current_result['condition'] = "Near Overbought (>90)"
                elif latest_cci_val < NearOS: current_result['condition'] = "Near Oversold (<-90)"
                elif latest_cci_val > 0: current_result['condition'] = "Above Zero"
                elif latest_cci_val < 0: current_result['condition'] = "Below Zero"
                else: current_result['condition'] = "Neutral (Zero)"
            else: current_result['condition'] = "Unknown (NaN Value)"

    def _calculate_obv(self) -> None:
        """Calcola OBV e determina trend relativo."""
        # (Logica invariata)
        result_key = 'obv'; default_value = {'value': None, 'trend': 'Unknown', 'error': None}
        self._calculate_indicator(ta.obv, name="OBV", result_key=result_key, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        obv_col_name = next((c for c in self.data.columns if c.lower() == "obv"), None)
        latest_obv = None; trend_calculated = False

        if obv_col_name:
            obv_series = self.data[obv_col_name]; latest_obv = safe_get_last_value(obv_series)
            if isinstance(current_result, dict): current_result['value'] = _safe_float(latest_obv)

            min_p = 10
            if obv_series.dropna().size >= min_p:
                obv_sma = obv_series.rolling(window=min_p, min_periods=max(1, min_p//2)).mean();
                last_obv_sma = safe_get_last_value(obv_sma);
                prev_obv_sma = safe_get_value_at_index(obv_sma, index=-2)
                if last_obv_sma is not None and prev_obv_sma is not None:
                    if last_obv_sma > prev_obv_sma: current_result['trend'] = "Rising"
                    elif last_obv_sma < prev_obv_sma: current_result['trend'] = "Falling"
                    else: current_result['trend'] = "Flat"
                    trend_calculated = True
                else: current_result['trend'] = "Unknown (SMA Calc Issue)"
            else: current_result['trend'] = "Unknown (Insufficient Data for Trend)"

            if latest_obv is not None and isinstance(current_result, dict):
                current_result.pop('error', None)
        elif isinstance(current_result, dict) and not current_result.get('error'):
            current_result['error'] = f"Colonna OBV (o simile) non trovata"

    def _calculate_psar(self, af: float = 0.02, max_af: float = 0.2) -> None:
        """Calcola PSAR, estrae valore, direzione e flip correttamente."""
        # (Logica invariata)
        result_key = 'psar'; default_value = {'psar': None, 'direction': 'Unknown', 'flipped_last_candle': None, 'error': None}
        self._calculate_indicator(ta.psar, name="PSAR", result_key=result_key, af=af, max_af=max_af, default_value=default_value)

        current_result = self.results['technical_indicators'].get(result_key, default_value)
        if isinstance(current_result, dict) and current_result.get('error'):
            logger.warning(f"Salto interpretazione PSAR causa errore calcolo: {current_result['error']}")
            current_result['psar'] = None; current_result['direction'] = 'Unknown'; current_result['flipped_last_candle'] = None; return

        actual_long_col = next((col for col in self.data.columns if col.lower().startswith('psarl')), None)
        actual_short_col = next((col for col in self.data.columns if col.lower().startswith('psars')), None)

        current_psar_actual = None; current_direction = 'Unknown'; prev_direction = 'Unknown'; flipped = None

        if actual_long_col and actual_short_col:
            logger.debug(f"Trovate colonne PSAR: long='{actual_long_col}', short='{actual_short_col}'")
            psar_long_series = self.data[actual_long_col]; psar_short_series = self.data[actual_short_col]

            last_psar_long = safe_get_last_value(psar_long_series, default=np.nan)
            last_psar_short = safe_get_last_value(psar_short_series, default=np.nan)

            if pd.notna(last_psar_long):
                current_psar_actual = last_psar_long; current_direction = "Long"
            elif pd.notna(last_psar_short):
                current_psar_actual = last_psar_short; current_direction = "Short"
            else:
                 logger.debug(f"Ultimi valori PSARl/s ({actual_long_col}/{actual_short_col}) sono NaN.")

            if len(psar_long_series.dropna()) > 1 and len(psar_short_series.dropna()) > 1:
                prev_psar_long = safe_get_value_at_index(psar_long_series, index=-2, default=np.nan)
                prev_psar_short = safe_get_value_at_index(psar_short_series, index=-2, default=np.nan)

                if pd.notna(prev_psar_long): prev_direction = "Long"
                elif pd.notna(prev_psar_short): prev_direction = "Short"

                flipped = (prev_direction != 'Unknown' and current_direction != 'Unknown' and prev_direction != current_direction)
            else:
                logger.debug("Dati insufficienti per determinare flip PSAR."); flipped = None

            current_result['psar'] = _safe_float(current_psar_actual)
            current_result['direction'] = current_direction
            current_result['flipped_last_candle'] = flipped
            current_result.pop('error', None)

        else:
             logger.debug(f"Colonne PSAR (psarl*/psars*) non trovate nel DataFrame.")
             if not current_result.get('error'):
                 current_result['error'] = f"PSAR columns not found"
             current_result['psar'] = None; current_result['direction'] = 'Unknown'; current_result['flipped_last_candle'] = None

        keys_to_remove = [key for key in list(current_result.keys()) if key not in ['psar', 'direction', 'flipped_last_candle', 'error'] and key.lower().startswith('psar')]
        for key in keys_to_remove: current_result.pop(key, None)

    def _calculate_adx_detailed(self, period=config.ADX_PERIOD) -> Dict[str, Optional[float]]:
        """Calcola ADX, +DI, -DI. Usato internamente da _analyze_trend."""
        # (Logica invariata)
        result_key = '_temp_adx_detailed'; default_value = {'adx': None, 'plus_di': None, 'minus_di': None, 'error': None}
        self._calculate_indicator(ta.adx, name="ADX", result_key=result_key, length=period, default_value=default_value)

        indicator_output = self.results['technical_indicators'].pop(result_key, default_value)
        helper_error = indicator_output.get('error') if isinstance(indicator_output, dict) else None

        adx_col = next((c for c in self.data.columns if c.lower() == f"adx_{period}".lower()), None)
        dmp_col = next((c for c in self.data.columns if c.lower() == f"dmp_{period}".lower()), None)
        dmn_col = next((c for c in self.data.columns if c.lower() == f"dmn_{period}".lower()), None)

        final_results = {'adx': None, 'plus_di': None, 'minus_di': None, 'error': None}; values_found = False

        if adx_col: final_results['adx'] = _safe_float(safe_get_last_value(self.data[adx_col])); values_found = True
        if dmp_col: final_results['plus_di'] = _safe_float(safe_get_last_value(self.data[dmp_col])); values_found = True
        if dmn_col: final_results['minus_di'] = _safe_float(safe_get_last_value(self.data[dmn_col])); values_found = True

        if values_found:
            final_results.pop('error', None)
        elif helper_error:
            final_results['error'] = helper_error
        else:
            final_results['error'] = f"Colonne ADX_{period}/DMP/DMN (o simili) non trovate"

        return final_results

    # --- NUOVI METODI ---
    def _calculate_volume_analysis(self, sma_period: int = TA_VOLUME_SMA_SHORT, zscore_period: int = TA_VOLUME_ZSCORE_PERIOD, updown_period: int = TA_UPDOWN_VOL_PERIOD) -> None:
        """Calcola analisi volume: SMA, Z-score, VROC, Up/Down Ratio."""
        vol_results = {
            'current_volume': None,
            f'volume_sma_{sma_period}': None,
            'volume_vs_sma_ratio': None,
            f'volume_zscore_{zscore_period}': None,
            'volume_spike_zscore': None, # Boolean based on z-score > threshold (e.g., 2)
            'vroc_1_period_pct': None,
            'vroc_5_period_pct': None,
            f'up_down_volume_ratio_{updown_period}p': None,
            'error': None
        }
        self.results['volume_analysis'] = vol_results

        if 'volume' not in self.data.columns or self.data['volume'].isnull().all():
            vol_results['error'] = "Volume data missing"; return

        volume_series = self.data['volume']
        vol_results['current_volume'] = _safe_float(safe_get_last_value(volume_series))

        try:
            # Volume SMA & Ratio
            if len(volume_series.dropna()) >= sma_period:
                vol_sma = volume_series.rolling(window=sma_period, min_periods=max(1, sma_period//2)).mean()
                last_sma = _safe_float(safe_get_last_value(vol_sma))
                vol_results[f'volume_sma_{sma_period}'] = last_sma
                if vol_results['current_volume'] is not None and last_sma is not None and last_sma > 1e-9:
                    vol_results['volume_vs_sma_ratio'] = _safe_float(vol_results['current_volume'] / last_sma)
            else: logger.debug(f"Dati insuff. per Volume SMA {sma_period}")

            # Volume Z-Score
            if len(volume_series.dropna()) >= zscore_period:
                vol_rolling_mean = volume_series.rolling(window=zscore_period, min_periods=max(1, zscore_period//2)).mean()
                vol_rolling_std = volume_series.rolling(window=zscore_period, min_periods=max(1, zscore_period//2)).std()
                last_mean = safe_get_last_value(vol_rolling_mean)
                last_std = safe_get_last_value(vol_rolling_std)
                if vol_results['current_volume'] is not None and last_mean is not None and last_std is not None and last_std > 1e-9:
                    z_score = (vol_results['current_volume'] - last_mean) / last_std
                    vol_results[f'volume_zscore_{zscore_period}'] = _safe_float(z_score)
                    vol_results['volume_spike_zscore'] = bool(z_score > 2.0) # Soglia esempio 2.0
            else: logger.debug(f"Dati insuff. per Volume Z-Score {zscore_period}")

            # VROC
            if len(volume_series.dropna()) >= 2:
                vroc1 = volume_series.pct_change(periods=1) * 100
                vol_results['vroc_1_period_pct'] = _safe_float(safe_get_last_value(vroc1))
            if len(volume_series.dropna()) >= 6: # Need 5 periods + current
                vroc5 = volume_series.pct_change(periods=5) * 100
                vol_results['vroc_5_period_pct'] = _safe_float(safe_get_last_value(vroc5))

            # Up/Down Volume Ratio
            if len(volume_series.dropna()) >= updown_period and 'close' in self.data.columns and 'open' in self.data.columns:
                recent_data = self.data.iloc[-updown_period:]
                close_gt_open = recent_data['close'] > recent_data['open']
                close_lt_open = recent_data['close'] < recent_data['open']
                up_volume = recent_data.loc[close_gt_open, 'volume'].sum()
                down_volume = recent_data.loc[close_lt_open, 'volume'].sum()
                if pd.notna(down_volume) and down_volume > 1e-9:
                    vol_results[f'up_down_volume_ratio_{updown_period}p'] = _safe_float(up_volume / down_volume)
                elif pd.notna(up_volume) and up_volume > 1e-9:
                    vol_results[f'up_down_volume_ratio_{updown_period}p'] = float('inf') # Infinite if only up volume
                else:
                    vol_results[f'up_down_volume_ratio_{updown_period}p'] = None
            else: logger.debug(f"Dati insuff. per Up/Down Volume Ratio {updown_period}")

            vol_results.pop('error', None)
        except Exception as e:
            logger.error(f"Errore calcolo analisi volume: {e}", exc_info=True)
            vol_results['error'] = str(e)

    def _calculate_vwap_rolling(self, period: int = TA_VWAP_ROLLING_PERIOD) -> None:
        """Calcola VWAP Rolling."""
        vwap_results = {
            f'vwap_rolling_{period}': None,
            f'price_vs_vwap{period}_pct': None,
            'error': None
        }
        # Store results in a specific sub-dictionary if needed, or directly in technical_indicators
        self.results['technical_indicators'][f'vwap_rolling_{period}_analysis'] = vwap_results

        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_cols):
            vwap_results['error'] = "Dati HLCV mancanti"; return
        if len(self.data.dropna(subset=required_cols)) < period:
             vwap_results['error'] = f"Dati insuff. per VWAP rolling {period}"; return

        try:
            df_vwap = self.data.copy()
            tp = (df_vwap['high'] + df_vwap['low'] + df_vwap['close']) / 3
            tp_vol = tp * df_vwap['volume']
            rolling_tp_vol = tp_vol.rolling(window=period, min_periods=max(1, period//2)).sum()
            rolling_volume = df_vwap['volume'].rolling(window=period, min_periods=max(1, period//2)).sum()

            # Calculate VWAP, handle division by zero
            valid_volume = rolling_volume.replace(0, np.nan)
            vwap_series = rolling_tp_vol / valid_volume
            vwap_series = vwap_series.replace([np.inf, -np.inf], np.nan)

            # Add VWAP series to main dataframe (optional, but can be useful)
            vwap_col_name = f"VWAP_{period}"
            self.data[vwap_col_name] = vwap_series

            # Get last value and calculate distance
            last_vwap = _safe_float(safe_get_last_value(vwap_series))
            vwap_results[f'vwap_rolling_{period}'] = last_vwap

            last_close = safe_get_last_value(self.data['close'])
            if last_close is not None and last_vwap is not None and last_vwap != 0:
                dist_pct = ((last_close - last_vwap) / last_vwap) * 100
                vwap_results[f'price_vs_vwap{period}_pct'] = _safe_float(dist_pct)

            vwap_results.pop('error', None)
        except Exception as e:
            logger.error(f"Errore calcolo VWAP rolling {period}: {e}", exc_info=True)
            vwap_results['error'] = str(e)

    def _calculate_volatility_percentiles(self, period: int = TA_VOLATILITY_PERCENTILE_PERIOD) -> None:
        """Calcola i percentili storici per ATR e BB Bandwidth."""
        vol_perc_results = {
            'atr_value': None,
            f'atr_percentile_{period}p': None,
            'bbw_value': None,
            f'bbw_percentile_{period}p': None,
            'error': None
        }
        self.results['volatility_analysis'] = vol_perc_results # Store in new section

        # --- ATR Percentile ---
        # 1. Ottieni il valore ATR corrente dai risultati (calcolato precedentemente)
        atr_val = _safe_get(self.results, ['technical_indicators', 'atr'])
        vol_perc_results['atr_value'] = atr_val # Salva comunque il valore trovato

        # 2. Cerca la colonna ATR nel DataFrame (necessaria per la storia)
        atr_col_name = None
        # Usa il periodo ATR corretto da config
        atr_base_period = config.ATR_PERIOD
        potential_cols = [f'ATR_{atr_base_period}', f'ATRr_{atr_base_period}', f'ATRe_{atr_base_period}'] # Common names from pandas_ta
        for col_base in potential_cols:
            atr_col_name = next((col for col in self.data.columns if col.lower() == col_base.lower()), None)
            if atr_col_name:
                logger.debug(f"Trovata colonna ATR '{atr_col_name}' per calcolo percentile.")
                break # Trovata, esci dal loop

        # 3. Calcola il percentile se la colonna esiste e ci sono dati sufficienti
        if atr_col_name and atr_col_name in self.data.columns:
            atr_series = self.data[atr_col_name].dropna()
            if len(atr_series) >= period:
                try:
                    # Calcola il rank percentile dell'ultimo valore ATR valido
                    last_valid_atr_for_perc = atr_series.iloc[-1] # Valore da confrontare
                    if pd.notna(last_valid_atr_for_perc):
                        historical_atr_window = atr_series.iloc[-period:] # Ultimi N valori
                        # Calcola percentile usando scipy.stats.percentileofscore
                        # 'weak' include il valore stesso nel conteggio <=
                        percentile_score = stats.percentileofscore(historical_atr_window, last_valid_atr_for_perc, kind='weak')
                        vol_perc_results[f'atr_percentile_{period}p'] = _safe_float(percentile_score)
                    else:
                        logger.warning("Ultimo valore ATR è NaN, impossibile calcolare percentile.")

                except ImportError:
                    logger.error("Modulo Scipy non trovato per percentileofscore.")
                    vol_perc_results['error'] = (vol_perc_results.get('error') or "") + " Scipy missing for ATR percentile."
                except Exception as e:
                    logger.warning(f"Errore calcolo percentile ATR {period}: {e}")
                    if not vol_perc_results.get('error'): vol_perc_results['error'] = f"ATR percentile calc failed: {e}"
            else:
                logger.debug(f"Dati ATR insuff. ({len(atr_series)} < {period}) per percentile.")
                if not vol_perc_results.get('error'): vol_perc_results['error'] = "Insufficient ATR data for percentile."
        elif not vol_perc_results.get('error'): # Aggiungi errore solo se non già presente
             logger.warning(f"Colonna ATR (es. ATR_{atr_base_period}) non trovata per calcolo percentile.")
             vol_perc_results['error'] = "ATR column not found for percentile calc."


        # --- Bollinger Bandwidth Percentile ---
        # (Logica simile per BBW, assicurandosi che la colonna esista)
        bbw_val = _safe_get(self.results, ['technical_indicators', 'bollinger_bands', 'bandwidth'])
        vol_perc_results['bbw_value'] = bbw_val
        # Find BBW column name (handles float std dev in name)
        std_dev_str = f"{config.BBANDS_STDDEV:.1f}"
        bbw_col_name = next((c for c in self.data.columns if c.lower() == f"bbb_{config.BBANDS_PERIOD}_{std_dev_str}".lower()), None)

        if bbw_col_name and bbw_col_name in self.data.columns:
            bbw_series = self.data[bbw_col_name].dropna()
            if len(bbw_series) >= period:
                try:
                    last_valid_bbw = bbw_series.iloc[-1]
                    if pd.notna(last_valid_bbw):
                        historical_bbw_window = bbw_series.iloc[-period:]
                        percentile_score_bbw = stats.percentileofscore(historical_bbw_window, last_valid_bbw, kind='weak')
                        vol_perc_results[f'bbw_percentile_{period}p'] = _safe_float(percentile_score_bbw)
                    else:
                        logger.warning("Ultimo valore BBW è NaN, impossibile calcolare percentile.")

                except ImportError:
                    logger.error("Modulo Scipy non trovato per percentileofscore.")
                    vol_perc_results['error'] = (vol_perc_results.get('error') or "") + " Scipy missing for BBW percentile."
                except Exception as e:
                    logger.warning(f"Errore calcolo percentile BBW {period}: {e}")
                    if not vol_perc_results.get('error'): vol_perc_results['error'] = (vol_perc_results.get('error') or "") + f" BBW percentile calc failed: {e}"
            else:
                logger.debug(f"Dati BBW insuff. ({len(bbw_series)} < {period}) per percentile.")
                if not vol_perc_results.get('error'): vol_perc_results['error'] = (vol_perc_results.get('error') or "") + " Insufficient BBW data for percentile."
        elif not vol_perc_results.get('error'):
             logger.warning(f"Colonna BBW (es. BBB_{config.BBANDS_PERIOD}_{std_dev_str}) non trovata per calcolo percentile.")
             vol_perc_results['error'] = (vol_perc_results.get('error') or "") + " BBW column not found for percentile calc."


        # Clean final error if any percentile was calculated
        if vol_perc_results.get(f'atr_percentile_{period}p') is not None or vol_perc_results.get(f'bbw_percentile_{period}p') is not None:
             vol_perc_results.pop('error', None)

    def _calculate_atr_targets(self, multipliers: List[float] = TA_ATR_TARGET_MULTIPLIERS) -> None:
        """Calcola potenziali target basati su multipli dell'ATR."""
        target_results = {}
        self.results['price_level_targets']['atr_based'] = target_results # Store in new section

        current_close = safe_get_last_value(self.data['close'])
        current_atr = _safe_get(self.results, ['technical_indicators', 'atr'])

        if current_close is None or current_atr is None or current_atr <= 0:
            logger.warning("Prezzo o ATR non validi per calcolo target ATR.")
            target_results['error'] = "Invalid price or ATR"
            return

        try:
            for mult in multipliers:
                mult_str = str(mult).replace('.', '_') # e.g., 1.5 -> 1_5
                target_up_key = f'target_{mult_str}x_atr_up'
                target_down_key = f'target_{mult_str}x_atr_down'
                target_results[target_up_key] = _safe_float(current_close + (mult * current_atr))
                target_results[target_down_key] = _safe_float(current_close - (mult * current_atr))
            target_results.pop('error', None)
        except Exception as e:
             logger.error(f"Errore calcolo target ATR: {e}", exc_info=True)
             target_results['error'] = str(e)

    # --- Metodi Esistenti Aggiornati ---

    def _calculate_support_resistance(self, window: int = 20, atr_multiplier: float = 0.5) -> None:
        """
        Calcola supporto e resistenza più vicini basati su minimi/massimi locali e ATR.
        MODIFICATO: Aggiunge distanza in ATR.
        """
        sr_results = {
            'nearest_support': None, 'nearest_resistance': None,
            'support_distance_pct': None, 'resistance_distance_pct': None,
            'support_distance_atr': None, 'resistance_distance_atr': None, # NUOVI CAMPI
            'error': None
        }
        self.results['support_resistance'] = sr_results

        # Ottieni ATR corrente (fondamentale per questo metodo)
        current_atr_val = _safe_get(self.results, ['technical_indicators', 'atr'])
        if current_atr_val is None or current_atr_val <= 0:
            # Tenta ricalcolo se mancante
            logger.debug("ATR non trovato/valido, tento ricalcolo per S/R...")
            current_atr_val = self.calculate_atr() # Chiama metodo che aggiorna anche results
            if current_atr_val is None or current_atr_val <= 0:
                logger.warning("Impossibile ottenere/calcolare ATR valido per S/R.")
                # Non impostare errore qui, l'analisi può continuare senza distanza ATR
                current_atr_val = 0 # Usa 0 per evitare errori sotto, ma le distanze ATR saranno None

        required_cols, min_p = ['high', 'low', 'close'], window
        if not all(c in self.data.columns for c in required_cols) or len(self.data.dropna(subset=required_cols)) < min_p:
            sr_results['error'] = "Insufficient HLC data"; return

        try:
            data_valid_hlc = self.data.dropna(subset=required_cols);
            data_window = data_valid_hlc.iloc[-max(window, min_p):] # Considera almeno 'min_p' punti
            current_close = safe_get_last_value(data_window['close'])

            if current_close is None or current_close <= 0:
                sr_results['error'] = "Invalid current price"; return

            # Usa l'ATR valido o 0
            current_atr = current_atr_val
            atr_filter_threshold = current_atr * atr_multiplier

            # Calcolo Pivot
            roll_window_pivot = max(3, window // 4); highs, lows = data_window['high'], data_window['low']
            try:
                # `center=True` might need adjustment depending on desired pivot behavior
                local_max_indices = highs.index[highs >= highs.rolling(roll_window_pivot, center=True, min_periods=1).max()]
                local_min_indices = lows.index[lows <= lows.rolling(roll_window_pivot, center=True, min_periods=1).min()]
            except ValueError:
                 logger.warning("Dati insuff. per rolling pivot S/R. Uso min/max globali finestra.");
                 local_max_indices = highs.index[highs == highs.max()]
                 local_min_indices = lows.index[lows == lows.min()]

            local_max_prices = data_window.loc[local_max_indices, 'high'].dropna().unique()
            local_min_prices = data_window.loc[local_min_indices, 'low'].dropna().unique()

            # Trova S/R più vicini (con filtro ATR)
            nearest_resistance, min_dist_res = None, float('inf'); nearest_support, min_dist_sup = None, float('inf')
            for res_price in local_max_prices:
                 if res_price > current_close:
                     dist_res_cand = res_price - current_close
                     # Applica filtro ATR
                     if dist_res_cand > atr_filter_threshold and dist_res_cand < min_dist_res:
                         min_dist_res = dist_res_cand; nearest_resistance = res_price
            for sup_price in local_min_prices:
                 if sup_price < current_close:
                     dist_sup_cand = current_close - sup_price
                     # Applica filtro ATR
                     if dist_sup_cand > atr_filter_threshold and dist_sup_cand < min_dist_sup:
                         min_dist_sup = dist_sup_cand; nearest_support = sup_price

            # Salva risultati
            sr_results['nearest_support'] = _safe_float(nearest_support); sr_results['nearest_resistance'] = _safe_float(nearest_resistance)
            if nearest_support is not None:
                sr_results['support_distance_pct'] = _safe_float((min_dist_sup / current_close) * 100)
                # NUOVO: Distanza in ATR
                if current_atr > 1e-9: sr_results['support_distance_atr'] = _safe_float(min_dist_sup / current_atr)
            if nearest_resistance is not None:
                sr_results['resistance_distance_pct'] = _safe_float((min_dist_res / current_close) * 100)
                # NUOVO: Distanza in ATR
                if current_atr > 1e-9: sr_results['resistance_distance_atr'] = _safe_float(min_dist_res / current_atr)

            sr_results.pop('error', None) # Clear error if successful
        except Exception as e:
            logger.error(f"Errore calcolo S/R: {e}", exc_info=True); sr_results['error'] = str(e)

    def _analyze_trend(self) -> None:
        """Analizza il trend combinando ADX e Medie Mobili."""
        # (Logica invariata)
        trend_results = {'trend': 'Unknown', 'momentum': {}, 'adx': {}, 'ma_trend': 'Unknown', 'adx_strength': 'Unknown', 'adx_direction': 'Unknown', 'trend_details': {}}
        self.results['trend_analysis'] = trend_results
        adx_data = self._calculate_adx_detailed(); trend_results['adx'] = adx_data
        adx_value = adx_data.get('adx'); plus_di = adx_data.get('plus_di'); minus_di = adx_data.get('minus_di')
        trend_strength = "Unknown"; trend_direction = "Unknown"
        if adx_value is not None:
            if adx_value < 15: trend_strength = "Very Weak/Absent"
            elif adx_value < 20: trend_strength = "Weak/Developing"
            elif adx_value < 40: trend_strength = "Moderate"
            elif adx_value < 50: trend_strength = "Strong"
            else: trend_strength = "Very Strong"
        if plus_di is not None and minus_di is not None:
            trend_direction = "Bullish" if plus_di > minus_di else "Bearish" if minus_di > plus_di else "Neutral"
        trend_results['adx_strength'] = trend_strength; trend_results['adx_direction'] = trend_direction
        latest_close = safe_get_last_value(self.data.get('close'));
        ma_results = _safe_get(self.results, ['technical_indicators', 'moving_averages'], {})
        sma20 = ma_results.get('sma_20'); sma50 = ma_results.get('sma_50'); sma200 = ma_results.get('sma_200');
        ma_trend = "Unknown"
        if all(v is not None for v in [latest_close, sma20, sma50, sma200]):
            price, s20, s50, s200 = latest_close, sma20, sma50, sma200
            if price > s20 > s50 > s200: ma_trend = "Strong Uptrend (Price > SMA20 > SMA50 > SMA200)"
            elif price < s20 < s50 < s200: ma_trend = "Strong Downtrend (Price < SMA20 < SMA50 < SMA200)"
            elif price > s50 and s50 > s200 and s20 > s50: ma_trend = "Uptrend (Aligned MAs)"
            elif price < s50 and s50 < s200 and s20 < s50: ma_trend = "Downtrend (Aligned MAs)"
            elif price > s50 and price > s200: ma_trend = "Uptrend (Price > SMA50 & SMA200)"
            elif price < s50 and price < s200: ma_trend = "Downtrend (Price < SMA50 & SMA200)"
            elif price > s50: ma_trend = "Weak Uptrend (Price > SMA50)"
            elif price < s50: ma_trend = "Weak Downtrend (Price < SMA50)"
            else: ma_trend = "Sideways/Neutral (MA Context)"
        elif all(v is not None for v in [latest_close, sma50]):
             ma_trend = "Uptrend (vs SMA50)" if latest_close > sma50 else "Downtrend (vs SMA50)"
        trend_results['ma_trend'] = ma_trend
        combined_trend = "Unknown"
        if trend_strength in ("Strong", "Very Strong") and trend_direction != 'Neutral':
            combined_trend = f"{trend_direction} ({trend_strength} ADX)"
        elif ma_trend not in ('Unknown', 'Neutral/Sideways', 'Sideways/Neutral (MA Context)'):
            ma_base_trend = "Bullish" if "Bullish" in ma_trend or "Uptrend" in ma_trend else "Bearish" if "Bearish" in ma_trend or "Downtrend" in ma_trend else "Neutral"
            if trend_strength == "Moderate" and trend_direction == ma_base_trend:
                combined_trend = f"{ma_base_trend} (MA Confirmed by Moderate ADX)"
            else:
                 combined_trend = f"{ma_base_trend} (Based on MA, ADX {trend_strength})"
        elif trend_strength == "Moderate" and trend_direction != 'Neutral':
             combined_trend = f"{trend_direction} (Moderate ADX, MA Neutral/Unknown)"
        else: combined_trend = "Neutral/Sideways"
        trend_results['trend_combined'] = combined_trend; trend_results['trend'] = combined_trend
        momentum_assessment, roc_results = "Unknown", {}
        if 'close' in self.data.columns:
            close_prices = self.data['close']; periods_roc = [5, 10, 20]
            for period in periods_roc:
                if len(close_prices) > period:
                    roc_val = close_prices.pct_change(periods=period).iloc[-1] * 100
                    roc_results[f'roc_{period}p'] = _safe_float(roc_val)
                else: roc_results[f'roc_{period}p'] = None
            roc5, roc10, roc20 = roc_results.get('roc_5p'), roc_results.get('roc_10p'), roc_results.get('roc_20p')
            if all(v is not None for v in [roc5, roc10, roc20]):
                if roc5 > 0 and roc10 > 0 and roc20 > 0: momentum_assessment = "Strong Bullish"
                elif roc5 < 0 and roc10 < 0 and roc20 < 0: momentum_assessment = "Strong Bearish"
                elif roc5 > 0 and roc10 > 0: momentum_assessment = "Bullish (Strengthening)" if roc5 > roc10 else "Bullish (Weakening)"
                elif roc5 < 0 and roc10 < 0: momentum_assessment = "Bearish (Strengthening)" if roc5 < roc10 else "Bearish (Weakening)"
                elif roc5 > 0 and roc10 < 0: momentum_assessment = "Improving (Short-term Turnaround?)"
                elif roc5 < 0 and roc10 > 0: momentum_assessment = "Weakening (Short-term Turnaround?)"
                else: momentum_assessment = "Mixed/Neutral"
            elif roc5 is not None:
                 momentum_assessment = "Bullish (Short-term)" if roc5 > 0 else "Bearish (Short-term)" if roc5 < 0 else "Flat (Short-term)"
        trend_results['momentum'] = {**roc_results, 'assessment': momentum_assessment}
        self._add_trend_details(trend_results)

    def _calculate_trend_slope(self, period=50) -> Optional[float]:
        # (Logica invariata)
        sma_col_df = next((col for col in self.data.columns if col.lower() == f"sma_{period}".lower()), None)
        if sma_col_df is None: return None
        try:
             sma = self.data[sma_col_df].dropna();
             if sma.size < 2: return None
             last_sma = sma.iloc[-1]; prev_sma = sma.iloc[-2];
             last_sma_idx = sma.last_valid_index();
             if last_sma_idx is None: return None
             last_close = safe_get_last_value(self.data.loc[last_sma_idx:, 'close'].dropna())
             if not all(pd.notna(v) for v in [last_sma, prev_sma, last_close]) or last_close is None or last_close <= 0: return None
             slope = last_sma - prev_sma;
             normalized_slope = (slope / last_close) * 100
             return _safe_float(normalized_slope)
        except Exception as e: logger.error(f"Errore slope ({sma_col_df}): {e}"); return None

    def _calculate_trend_duration(self) -> Optional[int]:
        """Calcola la durata dell'attuale trend direzionale basato su DI+/-."""
        # (Logica invariata)
        adx_period = config.ADX_PERIOD
        dmp_col_df = next((c for c in self.data.columns if c.lower() == f"dmp_{adx_period}".lower()), None)
        dmn_col_df = next((c for c in self.data.columns if c.lower() == f"dmn_{adx_period}".lower()), None)

        if dmp_col_df is None or dmn_col_df is None:
            logger.debug("Colonne DI+/- non trovate per duration."); return None

        plus_di, minus_di = self.data.get(dmp_col_df), self.data.get(dmn_col_df)
        if plus_di is None or minus_di is None or plus_di.isnull().all() or minus_di.isnull().all():
            logger.debug("Serie DI+/- vuote o NaN per duration."); return None
        try:
             di_diff = plus_di - minus_di;
             direction_sign = np.where(di_diff > 1e-9, 1, np.where(di_diff < -1e-9, -1, 0))
             direction_sign = pd.Series(direction_sign, index=di_diff.index)

             if direction_sign.empty: return 0
             last_direction = direction_sign.iloc[-1] if not direction_sign.empty else 0
             if last_direction == 0: return 0

             last_different_sign_indices = direction_sign.index[direction_sign != last_direction]

             if last_different_sign_indices.empty:
                 duration = len(direction_sign)
             else:
                 last_change_date = last_different_sign_indices[-1]
                 try:
                     last_change_loc = direction_sign.index.get_loc(last_change_date)
                     duration = len(direction_sign) - (last_change_loc + 1)
                 except KeyError:
                      logger.warning(f"Indice {last_change_date} non trovato in direction_sign.index. Stima durata fallita.")
                      duration = 0

             return int(duration)
        except Exception as e: logger.error(f"Errore calcolo durata trend: {e}", exc_info=True); return 0

    def _add_trend_details(self, trend_results_dict: dict) -> None:
        # (Logica invariata)
        try:
            details = {};
            details['trend_slope_sma20'] = self._calculate_trend_slope(period=20);
            details['trend_slope_sma50'] = self._calculate_trend_slope(period=50)
            details['trend_duration_periods'] = self._calculate_trend_duration()

            latest_close = safe_get_last_value(self.data.get('close'));
            ma_results = _safe_get(self.results, ['technical_indicators', 'moving_averages'], {})

            if latest_close is not None and latest_close > 0:
                for period in config.SMA_PERIODS:
                    key_pct = f'price_vs_sma_{period}_pct';
                    sma_val = ma_results.get(f'sma_{period}');
                    details[key_pct] = ((latest_close - sma_val) / sma_val * 100) if sma_val is not None and sma_val != 0 else None
                for period in config.EMA_PERIODS:
                    key_pct = f'price_vs_ema_{period}_pct';
                    ema_val = ma_results.get(f'ema_{period}');
                    details[key_pct] = ((latest_close - ema_val) / ema_val * 100) if ema_val is not None and ema_val != 0 else None
            else: # Reset if close is invalid
                for period in config.SMA_PERIODS: details[f'price_vs_sma_{period}_pct'] = None
                for period in config.EMA_PERIODS: details[f'price_vs_ema_{period}_pct'] = None

            trend_results_dict['trend_details'] = {k: _safe_float(v) for k, v in details.items()}
        except Exception as e: logger.error(f"Errore dettagli trend: {e}"); trend_results_dict['trend_details'] = {'error': str(e)}

    def _calculate_volume_profile(self, df_input: pd.DataFrame, bins: int = 50, va_percentage: int = 70, name_suffix: str = "") -> dict:
        """Calcola Volume Profile per il DataFrame fornito."""
        # (Logica interna invariata)
        vp_results = {'poc_price': None, 'vah_price': None, 'val_price': None, 'total_volume_in_period': None, 'value_area_percentage': va_percentage, 'price_range_min': None, 'price_range_max': None, 'bins': bins, 'profile': [], 'error': None, 'profile_shape': 'Unknown', 'profile_skewness': None}
        required_cols = ['high', 'low', 'close', 'volume']
        if df_input is None or df_input.empty or not all(c in df_input.columns for c in required_cols) or df_input['volume'].isnull().all() or (df_input['volume'] <= 0).all(): vp_results['error'] = f"Dati HLCV/vol > 0 mancanti VP {name_suffix}"; return vp_results
        try:
            df_vp = df_input[required_cols].dropna().copy(); df_vp = df_vp[df_vp['volume'] > 0]
            if df_vp.empty: vp_results['error'] = f"Nessun dato HLCV valido con vol > 0 VP {name_suffix}"; return vp_results
            df_vp['typical_price'] = (df_vp['high'] + df_vp['low'] + df_vp['close']) / 3; min_price, max_price = df_vp['low'].min(), df_vp['high'].max()
            vp_results['price_range_min'] = _safe_float(min_price); vp_results['price_range_max'] = _safe_float(max_price)
            if vp_results['price_range_min'] is None or vp_results['price_range_max'] is None or vp_results['price_range_max'] <= vp_results['price_range_min']: vp_results['error'] = f"Range prezzo VP {name_suffix} invalido"; return vp_results
            price_range_val = vp_results['price_range_max'] - vp_results['price_range_min']
            if np.isclose(price_range_val, 0): price_bins = np.array([vp_results['price_range_min'] - 1e-6, vp_results['price_range_max'] + 1e-6]); vp_results['bins'] = 1
            else: price_bins = np.linspace(vp_results['price_range_min'], vp_results['price_range_max'], bins + 1)
            if len(np.unique(price_bins)) < 2: vp_results['error'] = f"Bin VP {name_suffix} non validi"; return vp_results
            bin_mids = (price_bins[:-1] + price_bins[1:]) / 2; bin_labels = range(len(bin_mids))
            try:
                df_vp_clean = df_vp.dropna(subset=['typical_price']);
                if df_vp_clean.empty: vp_results['error'] = f"Nessun typical price valido VP {name_suffix}"; return vp_results
                df_vp_clean.loc[:, 'bin_cat'] = pd.cut(df_vp_clean['typical_price'], bins=price_bins, labels=bin_labels, right=False, include_lowest=True)
                df_vp_clean = df_vp_clean.dropna(subset=['bin_cat']);
                if df_vp_clean.empty: vp_results['error'] = f"Nessun bin valido VP {name_suffix}."; return vp_results
                df_vp_clean.loc[:, 'bin_cat'] = df_vp_clean['bin_cat'].astype(pd.Int64Dtype())
            except Exception as e: logger.error(f"Errore bin VP {name_suffix}: {e}"); vp_results['error'] = f"Errore bin: {e}"; return vp_results
            volume_by_bin_index = df_vp_clean.groupby('bin_cat', observed=False)['volume'].sum();
            valid_indices = [idx for idx in volume_by_bin_index.index if pd.notna(idx) and isinstance(idx, int) and 0 <= idx < len(bin_mids)]
            volume_by_price_dict = {bin_mids[idx]: volume_by_bin_index[idx] for idx in valid_indices}
            volume_by_price = pd.Series(volume_by_price_dict).sort_index();
            if volume_by_price.empty: vp_results['error'] = f"Profilo VP {name_suffix} vuoto."; return vp_results
            vp_results['poc_price'] = _safe_float(volume_by_price.idxmax()); total_volume = volume_by_price.sum(); vp_results['total_volume_in_period'] = _safe_float(total_volume)
            if total_volume is None or total_volume <= 0: vp_results['error'] = f"Volume totale nullo/invalido VP {name_suffix}."; return vp_results
            target_va_volume = total_volume * (va_percentage / 100.0); volume_sorted_by_vol = volume_by_price.sort_values(ascending=False)
            cumulative_volume = volume_sorted_by_vol.cumsum(); va_levels_prices = volume_sorted_by_vol[cumulative_volume <= target_va_volume].index
            if not va_levels_prices.empty: vp_results['vah_price'] = _safe_float(va_levels_prices.max()); vp_results['val_price'] = _safe_float(va_levels_prices.min())
            elif vp_results['poc_price'] is not None: vp_results['vah_price'] = vp_results['poc_price']; vp_results['val_price'] = vp_results['poc_price']
            profile_list = [{'price_level': _safe_float(price), 'volume': _safe_float(vol)} for price, vol in volume_by_price.items()]
            vp_results['profile'] = sorted([p for p in profile_list if p['price_level'] is not None and p['volume'] is not None], key=lambda x: x['price_level'])
            if len(vp_results['profile']) >= 3 and all(vp_results.get(k) is not None for k in ['poc_price', 'vah_price', 'val_price']):
                try:
                    prices = np.array([item['price_level'] for item in vp_results['profile']]); volumes = np.array([item['volume'] for item in vp_results['profile']])
                    if np.sum(volumes) <= 1e-9: raise ValueError("Somma volumi troppo piccola per analisi forma")
                    weighted_avg_price = np.average(prices, weights=volumes); weighted_std_dev = None
                    if len(prices) > 1:
                        try: weighted_std_dev = np.sqrt(np.average((prices - weighted_avg_price)**2, weights=volumes))
                        except ZeroDivisionError: weighted_std_dev = 0.0
                    else: weighted_std_dev = 0.0
                    weighted_skewness = None
                    if weighted_std_dev is not None and weighted_std_dev > 1e-9:
                        weighted_skewness = np.average(((prices - weighted_avg_price) / weighted_std_dev)**3, weights=volumes); vp_results['profile_skewness'] = _safe_float(weighted_skewness)
                    else: vp_results['profile_skewness'] = 0.0
                    value_area_range = vp_results['vah_price'] - vp_results['val_price'];
                    poc_rel_pos = (vp_results['poc_price'] - vp_results['val_price']) / value_area_range if value_area_range > 1e-9 else 0.5
                    skew_for_shape = vp_results['profile_skewness'] if vp_results['profile_skewness'] is not None else 0.0
                    if 0.4 <= poc_rel_pos <= 0.6 and abs(skew_for_shape) < 0.3: vp_results['profile_shape'] = 'D-Shape (Balanced)'
                    elif poc_rel_pos > 0.7 and skew_for_shape < -0.2: vp_results['profile_shape'] = 'P-Shape (Rejection Low?)'
                    elif poc_rel_pos < 0.3 and skew_for_shape > 0.2: vp_results['profile_shape'] = 'b-Shape (Rejection High?)'
                    elif abs(skew_for_shape) < 0.4: vp_results['profile_shape'] = 'Balanced (Symmetric)'
                    elif skew_for_shape < -0.4: vp_results['profile_shape'] = 'Skewed Low (Tail Left)'
                    elif skew_for_shape > 0.4: vp_results['profile_shape'] = 'Skewed High (Tail Right)'
                    else: vp_results['profile_shape'] = 'Irregular'
                except Exception as shape_e: logger.warning(f"Errore analisi forma/skew VP {name_suffix}: {shape_e}", exc_info=True); vp_results['profile_shape'] = 'Error'; vp_results['profile_skewness'] = None
            vp_results.pop('error', None) # Clear error if successful
        except Exception as e: logger.error(f"Errore calcolo Volume Profile {name_suffix}: {e}", exc_info=True); vp_results['error'] = str(e);
        return vp_results

    def _calculate_periodic_volume_profile(self, period: str = 'D'):
        # (Logica invariata)
        logger.debug(f"Avvio calcolo VP Periodico: {period}"); target_key = 'daily' if period.upper() == 'D' else 'weekly' if period.upper() == 'W' else None
        if target_key is None: logger.error(f"Periodo VP non supportato: {period}"); return
        self.results['volume_profile_periodic'][target_key] = {'error': 'Calculation not started'}
        if self.data is None or self.data.empty: self.results['volume_profile_periodic'][target_key] = {'error': 'Input data missing'}; return
        if self.input_timeframe_delta is None: self.results['volume_profile_periodic'][target_key] = {'error': 'Cannot determine input timeframe'}; return
        target_delta = pd.Timedelta(days=1) if period.upper() == 'D' else pd.Timedelta(weeks=1); df_to_use = None
        if self.input_timeframe_delta < target_delta:
            logger.debug(f"Resampling dati da {self.input_timeframe_delta} a {period} per VP {target_key}...")
            try:
                 ohlc_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
                 if not isinstance(self.data.index, pd.DatetimeIndex): self.results['volume_profile_periodic'][target_key] = {'error': 'Index non DatetimeIndex'}; return
                 df_resampled = self.data.resample(period).agg(ohlc_dict).dropna(how='all')
                 if df_resampled.empty: self.results['volume_profile_periodic'][target_key] = {'error': f'Dati vuoti dopo resample {period}'}; return
                 df_to_use = df_resampled; logger.debug(f"Dati resamplati a {period}: {len(df_to_use)} righe.")
            except Exception as e: self.results['volume_profile_periodic'][target_key] = {'error': f'Resample a {period} fallito: {e}'}; return
        elif self.input_timeframe_delta == target_delta: logger.debug(f"Uso dati originali ({period}) per VP {target_key}..."); df_to_use = self.data.copy()
        else: self.results['volume_profile_periodic'][target_key] = {'error': f'Input TF > Target TF ({self.input_timeframe_delta} > {target_delta})'}; return
        if df_to_use is not None: vp_periodic_results = self._calculate_volume_profile(df_to_use, name_suffix=target_key); self.results['volume_profile_periodic'][target_key] = vp_periodic_results
        else: self.results['volume_profile_periodic'][target_key] = {'error': 'DataFrame non preparato'}

    def _calculate_combined_features(self) -> None:
        # (Logica invariata)
        logger.debug("Calcolo feature combinate..."); combined_results = {}
        ti = self.results.get('technical_indicators', {}); ma = ti.get('moving_averages', {}); atr = ti.get('atr'); close = safe_get_last_value(self.data.get('close'))
        if close is not None and close > 0:
            for p in config.SMA_PERIODS: sma_val = ma.get(f'sma_{p}');
            if sma_val is not None and sma_val > 0: combined_results[f'dist_close_sma{p}_pct'] = _safe_float(((close - sma_val) / sma_val) * 100)
            for p in config.EMA_PERIODS: ema_val = ma.get(f'ema_{p}');
            if ema_val is not None and ema_val > 0: combined_results[f'dist_close_ema{p}_pct'] = _safe_float(((close - ema_val) / ema_val) * 100)
            sma20 = ma.get('sma_20'); sma50 = ma.get('sma_50'); sma200 = ma.get('sma_200')
            if sma20 is not None and sma50 is not None and sma50 > 0: combined_results['dist_sma20_sma50_pct'] = _safe_float(((sma20 - sma50) / sma50) * 100)
            if sma50 is not None and sma200 is not None and sma200 > 0: combined_results['dist_sma50_sma200_pct'] = _safe_float(((sma50 - sma200) / sma200) * 100)
            macd_hist = _safe_get(ti, ['macd', 'histogram']); rsi_val = _safe_get(ti, ['rsi', 'value']); stoch_k = _safe_get(ti, ['stochastic', 'k']); cci_val = _safe_get(ti, ['cci', 'value'])
            relative_atr = (atr / close) if atr is not None and atr > 0 else None
            if relative_atr is not None and relative_atr > 1e-9:
                 if macd_hist is not None: combined_results['macd_hist_vs_atr_rel'] = _safe_float(macd_hist / relative_atr)
                 if rsi_val is not None: combined_results['rsi_norm_atr_rel'] = _safe_float((rsi_val - 50) / (relative_atr * 100))
                 if stoch_k is not None: combined_results['stoch_k_norm_atr_rel'] = _safe_float((stoch_k - 50) / (relative_atr * 100))
                 if cci_val is not None: combined_results['cci_norm_atr_rel'] = _safe_float(cci_val / (relative_atr * 100 * 10))
            else: logger.debug("ATR relativo nullo/non valido, salto normalizzazione oscillatori.")
        else: logger.warning("Prezzo chiusura non valido, salto calcolo feature combinate.")
        self.results['combined_features'] = {k: v for k, v in combined_results.items() if v is not None}; logger.debug(f"Feature combinate calcolate: {list(self.results['combined_features'].keys())}")


    # --- Orchestrazione Principale (Aggiornata per includere nuove chiamate) ---
    def run_analysis(self) -> Dict[str, Any]:
        """Esegue tutte le analisi tecniche, inclusi volume, volatilità e target."""
        logger.debug(f"{self.__class__.__name__}.run_analysis() - START"); self.results = {}; self._ensure_results_structure()
        if not HELPERS_LOADED: logger.error("TA: helpers non caricato."); return {'error': "statistical_analyzer_helpers not loaded"}
        if self.data is None or self.data.empty: logger.error("TA: Dati vuoti."); return {'error': "Dati vuoti"}
        if len(self.data) < 5: logger.error(f"TA: Dati insufficienti ({len(self.data)})."); return {'error': "Dati insufficienti"}
        if ta is None: logger.error("TA: pandas_ta mancante."); return {'error': "pandas_ta mancante"}
        start_time = time.time()
        try:
            logger.debug("Calcolo indicatori base e ATR...")
            self.calculate_atr(); # Calcola e salva ATR
            self._calculate_moving_averages();
            self._calculate_rsi();
            self._calculate_macd();
            self._calculate_bollinger_bands();
            self._calculate_stochastic();
            self._calculate_cci();
            self._calculate_obv();
            self._calculate_psar()

            logger.debug("Calcolo analisi volume e VWAP rolling...");
            self._calculate_volume_analysis() # NUOVO
            self._calculate_vwap_rolling() # NUOVO

            logger.debug("Calcolo analisi derivate (Trend, S/R, Volatility Percentiles, ATR Targets)...");
            self._analyze_trend();
            self._calculate_support_resistance() # Ora include distanza ATR
            self._calculate_volatility_percentiles() # NUOVO
            self._calculate_atr_targets() # NUOVO

            logger.debug("Calcolo Volume Profile...");
            self.results['volume_profile'] = self._calculate_volume_profile(self.data.copy(), name_suffix="base");
            self._calculate_periodic_volume_profile(period='D');
            self._calculate_periodic_volume_profile(period='W')

            logger.debug("Calcolo feature combinate...");
            self._calculate_combined_features()

            # Pulizia finale errori (invariato)
            ti_final = self.results.get('technical_indicators', {})
            for ind_key, ind_data in list(ti_final.items()):
                is_scalar_error = ind_key.endswith('_error')
                if is_scalar_error:
                    value_key = ind_key.replace('_error', '')
                    if value_key in ti_final and ti_final[value_key] is not None:
                         ti_final.pop(ind_key)
                elif isinstance(ind_data, dict) and 'error' in ind_data:
                     has_value = any(v is not None for k, v in ind_data.items() if k != 'error')
                     if has_value:
                         ind_data.pop('error')
                     elif all(v is None for k,v in ind_data.items() if k != 'error'):
                          logger.warning(f"Indicatore '{ind_key}' ha errore '{ind_data['error']}' e nessun valore valido.")

            exec_time = time.time() - start_time; logger.info(f"Analisi tecnica completata in {exec_time:.2f} secondi.")
        except Exception as e:
            logger.error(f"Errore critico TA.run_analysis: {e}", exc_info=True)
            self.results['error'] = f"Analisi tecnica fallita: {e}"
            self._ensure_results_structure()
            self.results['technical_indicators']['error'] = f"Critical failure: {e}"


        logger.debug(f"{self.__class__.__name__}.run_analysis() - END");
        return self.results

# --- END OF FILE technical_analyzer.py ---