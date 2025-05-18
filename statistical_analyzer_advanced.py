# --- START OF FILE statistical_analyzer_advanced.py ---
# statistical_analyzer_advanced.py
"""
Performs advanced statistical analysis, extending the base StatisticalAnalyzer.
Includes rolling statistics, temporal bias analysis, risk metrics,
stationarity tests, autocorrelation, volatility clustering, GARCH modeling,
normality tests, HOURLY VOLUME analysis, CYCLE ANALYSIS (with duration percentiles),
INTRACANDLE percentage metrics analysis (with persistent historical stats),
timing estimates.
NOTE: Potential Target calculation moved to TradingAdvisor.
"""
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats # Importato anche se già in StatisticalAnalyzer per chiarezza
from scipy.signal import find_peaks
from arch import arch_model # type: ignore
from typing import Dict, Any, Optional, List, Tuple
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, shapiro, skew, kurtosis
import sys
import math
import os
from datetime import datetime, timezone, timedelta, date # Aggiunto timedelta e date

# Import the CORRECT base class
from statistical_analyzer import StatisticalAnalyzer

# Import helpers ONLY from statistical_analyzer_helpers
try:
    from statistical_analyzer_helpers import (
        _safe_float, safe_get_last_value, safe_get_value_at_index, _safe_get,
        calculate_returns, safe_strftime,
        load_historical_stats_from_json,
        save_historical_stats_to_json,
        calculate_single_candle_metrics
    )
except ImportError:
    # Fallback invariati
    logger_critical = logging.getLogger(__name__)
    logger_critical.critical("CRITICAL ERROR: statistical_analyzer_helpers not found! Install or fix path.")
    def _safe_float(value, default=None): return float(value) if pd.notna(value) and not np.isinf(value) else default # type: ignore
    def safe_get_last_value(series, default=None): return series.iloc[-1] if series is not None and not series.empty and pd.notna(series.iloc[-1]) else default # type: ignore
    def safe_get_value_at_index(series, index=-1, default=None): # type: ignore
        try: return float(series[index]) if series is not None and len(series) > abs(index) else default
        except Exception: return default
    def _safe_get(d: Optional[Dict], keys: list, default=None): return default # type: ignore
    def calculate_returns(data: pd.DataFrame) -> pd.DataFrame: return data # type: ignore
    def safe_strftime(dt, fmt, fallback='N/A'): # type: ignore
        try:
            # Gestisce datetime, date, pd.Timestamp, e timestamp numerici (ms)
            if isinstance(dt, (datetime, pd.Timestamp, date)):
                dt_obj = pd.to_datetime(dt)
                if dt_obj.tzinfo is None: dt_obj = dt_obj.tz_localize(timezone.utc)
                elif str(dt_obj.tz).upper() != 'UTC': dt_obj = dt_obj.tz_convert(timezone.utc)
            elif isinstance(dt, (int, float)):
                 # Aggiunto controllo timestamp
                 timestamp_sec = dt / 1000.0
                 if not (datetime(1980, 1, 1).timestamp() < timestamp_sec < datetime(2100, 1, 1).timestamp()):
                      return fallback
                 dt_obj = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
            else: return fallback
            fmt_final = fmt + ("Z" if 'Z' not in fmt and '%Z' not in fmt else "")
            return dt_obj.strftime(fmt_final)
        except Exception: return fallback
    def load_historical_stats_from_json(filename: str) -> Optional[Dict]: return None # type: ignore
    def save_historical_stats_to_json(stats_data: Dict, filename: str) -> bool: return False # type: ignore
    def calculate_single_candle_metrics(o, h, l, c) -> Dict: return {k: None for k in ['range_pct', 'min_max_pct', 'min_close_pct', 'open_max_pct', 'open_min_pct', 'body_pct']} # type: ignore


# Import config per directory e parametri
try:
    # Aggiungi INTRACANDLE_PERCENTILES
    from config import HISTORICAL_STATS_DIR, INTRACANDLE_PERCENTILES, TIMEFRAMES, ORDERED_TIMEFRAMES # Aggiunto TIMEFRAMES/ORDERED
except ImportError:
    HISTORICAL_STATS_DIR = "historical_stats" # Fallback
    INTRACANDLE_PERCENTILES = [10, 25, 50, 75, 90, 95, 99] # Fallback
    TIMEFRAMES = {} # Fallback
    ORDERED_TIMEFRAMES = [] # Fallback
    logging.warning("Import da config fallito in advanced analyzer. Uso fallback.")


# Get logger
logger = logging.getLogger(__name__)

# --- Parametri Cicli ---
CYCLE_PEAK_TROUGH_DISTANCE = 5
CYCLE_PEAK_PROMINENCE_FACTOR = 0.005
CYCLE_MIN_DATA_POINTS = 50
# NUOVO: Periodo per calcolo velocità trend
RECENT_VELOCITY_PERIOD = 10


class StatisticalAnalyzerAdvanced(StatisticalAnalyzer):
    """
    Advanced statistical analysis including rolling stats, bias, risk, GARCH,
    cycles, hourly volume, intracandle metrics, timing estimates, etc.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the advanced statistical analyzer.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and DatetimeIndex UTC.
        """
        logger.debug("StatisticalAnalyzerAdvanced.__init__() - START")
        self.symbol: Optional[str] = None
        self.timeframe: Optional[str] = None
        self.cycle_stats_filename: Optional[str] = None
        self.intracandle_stats_filename: Optional[str] = None

        # Attributi per conservare estremi ciclo precedente
        self._previous_cycle_high_price: Optional[float] = None
        self._previous_cycle_low_price: Optional[float] = None

        self._is_long_timeframe = False

        try:
            # Chiama super().__init__ PRIMA di accedere a self.data
            super().__init__(data)
            if self.data is not None and not self.data.empty:
                 self._ensure_garch_columns()
                 # Calcola subito le metriche intra-candela
                 self._calculate_intracandle_metrics_for_dataframe() # Modificato
            else:
                 logger.warning("Dati non validi dopo super().__init__ in StatisticalAnalyzerAdvanced.")
                 # Non sollevare eccezione, ma run_analysis gestirà dati vuoti

        except (ValueError, TypeError) as e:
            logger.error(f"Error initializing StatisticalAnalyzerAdvanced: {e}")
            raise
        logger.debug("StatisticalAnalyzerAdvanced.__init__() - END")

    def _ensure_garch_columns(self):
        """Ensures 'simple_return_scaled' and 'returns_squared' exist for GARCH."""
        if self.data is None or self.data.empty:
            logger.warning("Cannot ensure GARCH columns: data is missing.")
            return
        if 'simple_return' in self.data.columns:
            if 'simple_return_scaled' not in self.data.columns:
                 logger.debug("Column 'simple_return_scaled' missing. Creating for GARCH.")
                 self.data['simple_return_scaled'] = self.data['simple_return'].fillna(0) * 100000
            if 'returns_squared' not in self.data.columns:
                 logger.debug("Column 'returns_squared' missing. Creating for GARCH.")
                 self.data['returns_squared'] = self.data['simple_return'].fillna(0) ** 2
        else:
            logger.warning("Cannot create GARCH columns as 'simple_return' is missing.")

    def set_symbol_timeframe(self, symbol: str, timeframe: str):
       # Imposta simbolo e timeframe e costruisce i nomi dei file stats.
       # RIGA ERRATA RIMOSSA -> super().set_symbol_timeframe(symbol, timeframe)
        self.symbol = symbol
        self.timeframe = timeframe
        # Determina se è un timeframe lungo
        self._is_long_timeframe = False
        if timeframe:
            try:
                 if timeframe in ORDERED_TIMEFRAMES:
                      # Considera lungo '1d' o superiore
                      daily_index = ORDERED_TIMEFRAMES.index('1d')
                      current_index = ORDERED_TIMEFRAMES.index(timeframe)
                      if current_index >= daily_index:
                          self._is_long_timeframe = True
                          logger.debug(f"Timeframe '{timeframe}' considerato lungo per analisi bias.")
                 else:
                      logger.warning(f"Timeframe '{timeframe}' non trovato in ORDERED_TIMEFRAMES per controllo 'long'.")
                      # Heuristic fallback if not in ordered list
                      if 'd' in timeframe.lower() or 'w' in timeframe.lower() or 'M' in timeframe.lower():
                           self._is_long_timeframe = True
                           logger.debug(f"Timeframe '{timeframe}' (non in lista) considerato lungo per analisi bias (heuristica).")
            except ValueError:
                 logger.error(f"Errore nel trovare timeframe '{timeframe}' o '1d' in ORDERED_TIMEFRAMES.")
            except Exception as e:
                 logger.error(f"Errore imprevisto nel determinare long timeframe: {e}")

        # Costruisce path file storici
        if symbol and timeframe:
            safe_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_')
            self.cycle_stats_filename = os.path.join(
                HISTORICAL_STATS_DIR,
                f"cycle_stats_{safe_symbol}_{timeframe}.json"
            )
            self.intracandle_stats_filename = os.path.join(
                HISTORICAL_STATS_DIR,
                f"candle_stats_{safe_symbol}_{timeframe}.json"
            )
            logger.debug(f"Percorso file statistiche cicli: {self.cycle_stats_filename}")
            logger.debug(f"Percorso file statistiche intra-candela: {self.intracandle_stats_filename}")
        else:
            self.cycle_stats_filename = None
            self.intracandle_stats_filename = None
            logger.warning("Simbolo o timeframe non forniti, impossibile determinare path file storici.")

    def _calculate_intracandle_metrics_for_dataframe(self):
        """
        Calcola le 6 metriche percentuali intra-candela per ogni riga
        del DataFrame self.data e le aggiunge come nuove colonne.
        Non usa più fillna(0.0) indiscriminatamente.
        """
        if self.data is None or self.data.empty:
            logger.warning("Intracandle metrics: Data missing.")
            return

        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_cols):
            logger.warning(f"Intracandle metrics: Missing columns {required_cols}.")
            return

        metric_cols = [
            'ic_range_pct', 'ic_min_max_pct', 'ic_min_close_pct',
            'ic_open_max_pct', 'ic_open_min_pct', 'ic_body_pct'
        ]

        cols_to_calculate = [col for col in metric_cols if col not in self.data.columns]
        if not cols_to_calculate:
            logger.debug("Intracandle metric columns already exist. Skipping recalculation.")
            return

        logger.debug(f"Calculating intracandle metrics for columns: {cols_to_calculate}...")
        start_time = datetime.now()

        def calculate_row_metrics(row):
             o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
             return calculate_single_candle_metrics(o, h, l, c)

        calculated_metrics = self.data.apply(calculate_row_metrics, axis=1)

        for col_name in cols_to_calculate:
            metric_series = calculated_metrics.apply(lambda x: x.get(col_name)).astype(float)
            self.data[col_name] = metric_series
            nan_count = metric_series.isnull().sum()
            if nan_count > 0:
                 logger.debug(f"Intracandle metric '{col_name}': {nan_count} NaN values present after calculation.")

        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Intracandle metrics calculated and added to DataFrame in {elapsed_time:.3f} seconds.")

    def _calculate_rolling_descriptive(self, data_slice: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        new_cols = {}
        if data_slice.empty or len(data_slice) < window:
            logger.debug(f"Skipping rolling descriptive (window {window}): insufficient data.")
            return new_cols
        min_p = max(1, math.ceil(window / 2))

        if 'close' in data_slice.columns:
            close_prices = data_slice['close']
            if len(close_prices.dropna()) >= min_p:
                new_cols[f'rolling_mean_{window}'] = close_prices.rolling(window=window, min_periods=min_p).mean()
                new_cols[f'rolling_median_{window}'] = close_prices.rolling(window=window, min_periods=min_p).median()
                new_cols[f'rolling_std_{window}'] = close_prices.rolling(window=window, min_periods=min_p).std()
                new_cols[f'rolling_max_{window}'] = close_prices.rolling(window=window, min_periods=min_p).max()
                new_cols[f'rolling_min_{window}'] = close_prices.rolling(window=window, min_periods=min_p).min()
                new_cols[f'rolling_quantile_90_{window}'] = close_prices.rolling(window=window, min_periods=min_p).quantile(0.90)
                min_p_sk = max(3, min_p + 1) if window >= 3 else window
                if len(close_prices.dropna()) >= min_p_sk:
                    new_cols[f'rolling_price_skew_{window}'] = close_prices.rolling(window=window, min_periods=min_p_sk).skew()
                    new_cols[f'rolling_price_kurt_{window}'] = close_prices.rolling(window=window, min_periods=min_p_sk).kurt()
                else:
                    new_cols[f'rolling_price_skew_{window}'] = pd.Series(np.nan, index=data_slice.index)
                    new_cols[f'rolling_price_kurt_{window}'] = pd.Series(np.nan, index=data_slice.index)
            else: logger.debug(f"Skipping rolling price stats (window {window}): <{min_p} valid points.")

        if 'volume' in data_slice.columns:
            volume = data_slice['volume']
            min_p_vol = max(1, math.ceil(window / 2))
            if len(volume.dropna()) >= min_p_vol:
                new_cols[f'rolling_volume_mean_{window}'] = volume.rolling(window=window, min_periods=min_p_vol).mean()
                new_cols[f'rolling_volume_std_{window}'] = volume.rolling(window=window, min_periods=min_p_vol).std()
            else: logger.debug(f"Skipping rolling volume stats (window {window}): <{min_p_vol} valid points.")
        return new_cols

    def _calculate_rolling_returns(self, data_slice: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        new_cols = {}
        if data_slice.empty or 'simple_return' not in data_slice.columns:
             logger.debug(f"Skipping rolling returns (window {window}): data/returns missing.")
             return new_cols

        returns = data_slice['simple_return']
        min_p = max(2, math.ceil(window / 2))
        if len(returns.dropna()) < min_p:
             logger.debug(f"Skipping rolling returns (window {window}): <{min_p} valid returns.")
             return new_cols

        new_cols[f'rolling_return_mean_{window}'] = returns.rolling(window=window, min_periods=min_p).mean()
        rolling_std = returns.rolling(window=window, min_periods=min_p).std()
        new_cols[f'rolling_return_std_{window}'] = rolling_std

        min_p_sk = max(3, min_p + 1) if window >= 3 else window
        if len(returns.dropna()) >= min_p_sk:
            new_cols[f'rolling_return_skew_{window}'] = returns.rolling(window=window, min_periods=min_p_sk).skew()
            new_cols[f'rolling_return_kurt_{window}'] = returns.rolling(window=window, min_periods=min_p_sk).kurt()
        else:
            new_cols[f'rolling_return_skew_{window}'] = pd.Series(np.nan, index=data_slice.index)
            new_cols[f'rolling_return_kurt_{window}'] = pd.Series(np.nan, index=data_slice.index)

        rolling_std_safe = rolling_std.replace(0, np.nan)
        new_cols[f'rolling_sharpe_{window}'] = new_cols[f'rolling_return_mean_{window}'] / rolling_std_safe

        downside_returns = returns.where(returns < 0, 0.0)
        rolling_downside_std = downside_returns.rolling(window=window, min_periods=min_p).std()
        rolling_downside_std_safe = rolling_downside_std.replace(0, np.nan)
        new_cols[f'rolling_downside_std_{window}'] = rolling_downside_std
        new_cols[f'rolling_sortino_{window}'] = new_cols[f'rolling_return_mean_{window}'] / rolling_downside_std_safe

        return new_cols

    def _calculate_rolling_rank(self, data_slice: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        new_cols = {}
        if data_slice.empty or 'close' not in data_slice.columns or len(data_slice) < window:
             logger.debug(f"Skipping rolling rank (window {window}): insufficient data.")
             return new_cols
        close_prices = data_slice['close']
        min_p = max(1, math.ceil(window / 2))
        if len(close_prices.dropna()) >= min_p:
            new_cols[f'rank_in_window_{window}'] = close_prices.rolling(window=window, min_periods=min_p).rank(method='average', pct=False)
            new_cols[f'percent_rank_in_window_{window}'] = close_prices.rolling(window=window, min_periods=min_p).rank(pct=True)
        else: logger.debug(f"Skipping rolling rank (window {window}): <{min_p} valid points.")
        return new_cols

    def _calculate_rolling_count(self, data_slice: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        new_cols = {}
        if data_slice.empty or len(data_slice) < window:
             logger.debug(f"Skipping rolling count (window {window}): insufficient data.")
             return new_cols
        min_p = max(1, math.ceil(window / 2))

        if 'close' in data_slice.columns:
            close_prices = data_slice['close']
            if len(close_prices.dropna()) >= min_p:
                try:
                    rolling_idxmax = close_prices.rolling(window=window, min_periods=min_p).apply(np.nanargmax, raw=True, engine='cython')
                    rolling_idxmin = close_prices.rolling(window=window, min_periods=min_p).apply(np.nanargmin, raw=True, engine='cython')
                    new_cols[f'periods_since_high_{window}'] = window - 1 - rolling_idxmax
                    new_cols[f'periods_since_low_{window}'] = window - 1 - rolling_idxmin
                except Exception as e_apply:
                    logger.warning(f"Error calculating rolling argmax/argmin (window {window}, engine=cython): {e_apply}. Fallback a metodo più lento.")
                    try:
                        rolling_idxmax = close_prices.rolling(window=window, min_periods=min_p).apply(np.nanargmax, raw=True)
                        rolling_idxmin = close_prices.rolling(window=window, min_periods=min_p).apply(np.nanargmin, raw=True)
                        new_cols[f'periods_since_high_{window}'] = window - 1 - rolling_idxmax
                        new_cols[f'periods_since_low_{window}'] = window - 1 - rolling_idxmin
                    except Exception as e_fallback:
                         logger.error(f"Errore fallback rolling argmax/argmin (window {window}): {e_fallback}. Imposto a NaN.")
                         new_cols[f'periods_since_high_{window}'] = pd.Series(np.nan, index=data_slice.index)
                         new_cols[f'periods_since_low_{window}'] = pd.Series(np.nan, index=data_slice.index)
            else: logger.debug(f"Skipping periods since H/L (window {window}): <{min_p} valid points.")

        if 'simple_return' in data_slice.columns:
            returns = data_slice['simple_return']
            min_p_ret = max(1, math.ceil(window / 2))
            if len(returns.dropna()) >= min_p_ret:
                 try:
                     new_cols[f'up_periods_in_window_{window}'] = returns.rolling(window=window, min_periods=min_p_ret).apply(lambda x: np.sum(x > 1e-9), raw=True, engine='cython')
                     new_cols[f'down_periods_in_window_{window}'] = returns.rolling(window=window, min_periods=min_p_ret).apply(lambda x: np.sum(x < -1e-9), raw=True, engine='cython')
                 except Exception as e_apply_ret:
                     logger.warning(f"Errore rolling up/down count (window {window}, engine=cython): {e_apply_ret}. Fallback a metodo più lento.")
                     try:
                         new_cols[f'up_periods_in_window_{window}'] = returns.rolling(window=window, min_periods=min_p_ret).apply(lambda x: np.sum(x > 1e-9), raw=True)
                         new_cols[f'down_periods_in_window_{window}'] = returns.rolling(window=window, min_periods=min_p_ret).apply(lambda x: np.sum(x < -1e-9), raw=True)
                     except Exception as e_fallback_ret:
                          logger.error(f"Errore fallback rolling up/down count (window {window}): {e_fallback_ret}. Imposto a NaN.")
                          new_cols[f'up_periods_in_window_{window}'] = pd.Series(np.nan, index=data_slice.index)
                          new_cols[f'down_periods_in_window_{window}'] = pd.Series(np.nan, index=data_slice.index)
            else: logger.debug(f"Skipping up/down periods count (window {window}): <{min_p_ret} valid returns.")
        return new_cols

    def _calculate_non_rolling_counts(self, data_slice: pd.DataFrame) -> Dict[str, pd.Series]:
        new_cols = {}
        default = pd.Series(0, index=data_slice.index)
        if data_slice.empty or 'simple_return' not in data_slice.columns or data_slice['simple_return'].isnull().all():
            logger.debug("Skipping non-rolling count: data/returns missing or all NaN.")
            new_cols['consecutive_up_periods'] = default
            new_cols['consecutive_down_periods'] = default
            return new_cols

        returns = data_slice['simple_return']
        return_sign = np.sign(returns.fillna(0))
        sign_change_group = (return_sign != return_sign.shift(1)).cumsum()
        consecutive_count = return_sign.groupby(sign_change_group).cumcount() + 1
        new_cols['consecutive_up_periods'] = pd.Series(np.where(return_sign > 0, consecutive_count, 0), index=data_slice.index)
        new_cols['consecutive_down_periods'] = pd.Series(np.where(return_sign < 0, consecutive_count, 0), index=data_slice.index)
        return new_cols

    def _calculate_all_rolling_stats(self, windows: Optional[List[int]] = None):
        if self.data is None or self.data.empty:
            logger.warning("Skipping rolling stats calculation: data is missing.")
            return

        if windows is None:
            windows = [5, 10, 20, 30, 50, 60]
            logger.info(f"Calculating rolling stats for default windows: {windows}")
        else:
            logger.info(f"Calculating rolling stats for specified windows: {windows}")

        all_calculated_series: Dict[str, pd.Series] = {}; base_data = self.data
        for window in windows:
            logger.debug(f"  Rolling stats win {window}...");
            all_calculated_series.update(self._calculate_rolling_descriptive(base_data, window));
            all_calculated_series.update(self._calculate_rolling_returns(base_data, window));
            all_calculated_series.update(self._calculate_rolling_rank(base_data, window));
            all_calculated_series.update(self._calculate_rolling_count(base_data, window))

        logger.debug("  Non-rolling counts...");
        all_calculated_series.update(self._calculate_non_rolling_counts(base_data))

        if all_calculated_series:
            logger.debug(f"Adding/Updating {len(all_calculated_series)} rolling/count columns...");
            temp_rolling_df = pd.DataFrame(all_calculated_series, index=base_data.index)

            for col in temp_rolling_df.columns:
                if pd.api.types.is_numeric_dtype(temp_rolling_df[col]):
                    temp_rolling_df[col] = temp_rolling_df[col].replace([np.inf, -np.inf], np.nan)
                else:
                    temp_rolling_df[col] = pd.to_numeric(temp_rolling_df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

            temp_rolling_df.dropna(axis=1, how='all', inplace=True)

            for col in temp_rolling_df.columns:
                 self.data[col] = temp_rolling_df[col]

            logger.debug(f"Aggiornate/Aggiunte {len(temp_rolling_df.columns)} colonne rolling.")
        else:
            logger.debug("No new rolling/count columns calculated.")

    def calculate_volatility(self) -> Dict[str, Any]:
        volatility_stats: Dict[str, Any] = {'historical_std_dev': None, 'rolling_std_dev': {}, 'current_std_dev_30p': None}
        if self.data is None or 'simple_return' not in self.data.columns: return volatility_stats
        returns = self.data['simple_return'].dropna();
        if len(returns) < 2: return volatility_stats

        try:
            volatility_stats['historical_std_dev'] = _safe_float(returns.std())
            rolling_vol_results = {};
            windows_vol_report = [5, 10, 20, 30, 50, 60]
            for window in windows_vol_report:
                rolling_std_col = f'rolling_return_std_{window}';
                std_dev = np.nan
                if rolling_std_col in self.data.columns:
                    std_dev_raw = safe_get_last_value(self.data.get(rolling_std_col), default=np.nan)
                    std_dev = _safe_float(std_dev_raw)
                rolling_vol_results[f'std_dev_{window}p'] = std_dev

            volatility_stats['rolling_std_dev'] = rolling_vol_results
            volatility_stats['current_std_dev_30p'] = rolling_vol_results.get('std_dev_30p', volatility_stats['historical_std_dev'])
            return volatility_stats
        except Exception as e:
            logger.error(f"Error calculating volatility (Advanced): {str(e)}", exc_info=True)
            volatility_stats['historical_std_dev'] = None
            volatility_stats['rolling_std_dev'] = {f'std_dev_{w}p': None for w in windows_vol_report}
            volatility_stats['current_std_dev_30p'] = None
            return volatility_stats

    def calculate_returns_stats(self) -> Dict[str, Any]:
        returns_stats: Dict[str, Any] = {'total_return_period': None, 'avg_period_return': None, 'positive_periods_ratio': None, 'recent_return_5p': None, 'periodic_returns': {}}
        if self.data is None or 'close' not in self.data.columns or self.data['close'].isnull().all() or len(self.data) < 2: return returns_stats
        try:
            returns = self.data['simple_return'].dropna() if 'simple_return' in self.data.columns else pd.Series(dtype=float); close_prices = self.data['close'].dropna()
            if len(close_prices) >= 2: first_price = close_prices.iloc[0]; last_price = close_prices.iloc[-1];
            if first_price != 0 and pd.notna(first_price) and pd.notna(last_price): returns_stats['total_return_period'] = _safe_float((last_price / first_price) - 1)
            if not returns.empty: returns_stats['avg_period_return'] = _safe_float(returns.mean()); returns_stats['positive_periods_ratio'] = _safe_float((returns > 0).mean())
            recent_period = 5; mean_5p_col = f'rolling_return_mean_{recent_period}'; recent_mean_5p = np.nan
            if mean_5p_col in self.data.columns: recent_mean_5p_raw = safe_get_last_value(self.data.get(mean_5p_col), default=np.nan); recent_mean_5p = _safe_float(recent_mean_5p_raw)
            returns_stats['recent_return_5p'] = recent_mean_5p
            last_price = close_prices.iloc[-1] if not close_prices.empty else None
            if last_price is not None:
                periodic_returns_calc = {}
                for periods in [1, 5, 10, 20, 30, 60, 90]:
                    col_name = f'return_{periods}p'
                    if len(close_prices) > periods:
                        prev_price = close_prices.iloc[-periods - 1]
                        if prev_price is not None and pd.notna(prev_price) and prev_price != 0:
                            period_return = (last_price / prev_price) - 1
                            periodic_returns_calc[col_name] = _safe_float(period_return)
                        else: periodic_returns_calc[col_name] = None
                    else: periodic_returns_calc[col_name] = None
                returns_stats['periodic_returns'] = periodic_returns_calc
            return returns_stats
        except Exception as e: logger.error(f"Error calculating return stats (Advanced): {str(e)}", exc_info=True); return {k: v if k != 'periodic_returns' else {} for k,v in returns_stats.items()}

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        risk_metrics: Dict[str, Any] = {'sharpe_ratio_period': None, 'sortino_ratio_period': None, 'max_drawdown': None, 'current_drawdown': None, 'value_at_risk_95': None, 'cvar_95': None, 'rolling_sharpe_30p': None, 'rolling_sortino_30p': None}
        if self.data is None or 'simple_return' not in self.data.columns: return risk_metrics
        returns = self.data['simple_return'].dropna(); min_periods_for_risk = 20
        if len(returns) < min_periods_for_risk: return risk_metrics
        try:
            avg_return = returns.mean(); std_return = returns.std()
            if std_return is not None and pd.notna(std_return) and std_return > 0 and pd.notna(avg_return): risk_metrics['sharpe_ratio_period'] = _safe_float(avg_return / std_return)
            elif pd.notna(avg_return) and avg_return == 0 and pd.notna(std_return) and std_return == 0: risk_metrics['sharpe_ratio_period'] = 0.0
            negative_returns = returns[returns < 0]; downside_deviation = np.nan
            if not negative_returns.empty: downside_deviation = negative_returns.std()
            if pd.notna(downside_deviation) and downside_deviation > 0 and pd.notna(avg_return): risk_metrics['sortino_ratio_period'] = _safe_float(avg_return / downside_deviation)
            elif pd.notna(avg_return) and pd.notna(downside_deviation) and downside_deviation == 0: risk_metrics['sortino_ratio_period'] = float('inf') if avg_return > 0 else 0.0
            cumulative_returns = (1 + returns).cumprod(); running_max = cumulative_returns.cummax(); drawdown = (cumulative_returns / running_max.replace(0, np.nan)) - 1; drawdown = drawdown.fillna(0)
            risk_metrics['max_drawdown'] = _safe_float(drawdown.min()); risk_metrics['current_drawdown'] = _safe_float(drawdown.iloc[-1]) if not drawdown.empty else None
            var_95 = returns.quantile(0.05); risk_metrics['value_at_risk_95'] = _safe_float(var_95)
            if var_95 is not None and pd.notna(var_95): cvar_95 = returns[returns <= var_95].mean(); risk_metrics['cvar_95'] = _safe_float(cvar_95)
            window_risk = 30; sharpe_col = f'rolling_sharpe_{window_risk}'; sortino_col = f'rolling_sortino_{window_risk}'
            if sharpe_col in self.data.columns: risk_metrics['rolling_sharpe_30p'] = _safe_float(safe_get_last_value(self.data.get(sharpe_col)))
            if sortino_col in self.data.columns:
                 sortino_raw = _safe_float(safe_get_last_value(self.data.get(sortino_col)))
                 risk_metrics['rolling_sortino_30p'] = 5.0 if sortino_raw == float('inf') else sortino_raw
            return risk_metrics
        except Exception as e: logger.error(f"Error calculating risk metrics (Advanced): {str(e)}", exc_info=True); return {k: None for k in risk_metrics}

    def _calculate_hourly_bias(self) -> Dict[str, Any]:
        hourly_bias_results = {'hourly_positive_pct': {}, 'error': None}; day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        if self._is_long_timeframe:
            logger.debug("Hourly bias non applicabile per timeframe lungo. Salto.")
            hourly_bias_results['error'] = "Not applicable for long timeframe (e.g., >= 1d)"
            hourly_bias_results['hourly_positive_pct'] = {f"{h:02d}H": None for h in range(24)}
            return hourly_bias_results
        if self.data is None or 'simple_return' not in self.data.columns: hourly_bias_results['error'] = "Simple returns missing"; return hourly_bias_results
        if not isinstance(self.data.index, pd.DatetimeIndex): hourly_bias_results['error'] = "Invalid index type"; return hourly_bias_results
        try:
            returns_clean = self.data['simple_return'].dropna(); min_data_points_hourly = 24 * 5
            if len(returns_clean) < min_data_points_hourly: hourly_bias_results['error'] = f"Insufficient data points ({len(returns_clean)} < {min_data_points_hourly})"; logger.debug(hourly_bias_results['error']); return hourly_bias_results
            positive_pct_by_hour = returns_clean.groupby(returns_clean.index.hour).apply(lambda x: (x > 0).mean() * 100 if not x.empty else np.nan)
            hourly_bias_dict = {};
            for hour in range(24): pct_val = positive_pct_by_hour.get(hour, np.nan); hourly_bias_dict[f"{hour:02d}H"] = _safe_float(pct_val)
            hourly_bias_results['hourly_positive_pct'] = hourly_bias_dict; logger.debug("Hourly bias calculated successfully.")
        except Exception as e: logger.error(f"Error calculating Hourly Bias: {e}", exc_info=True); hourly_bias_results['error'] = str(e); hourly_bias_results['hourly_positive_pct'] = {f"{h:02d}H": None for h in range(24)}
        return hourly_bias_results

    def _calculate_day_hour_bias(self) -> Dict[str, Any]:
        day_hour_bias_results = {'day_hour_positive_pct': {}, 'error': None}; day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        if self._is_long_timeframe:
            logger.debug("Day/Hour bias non applicabile per timeframe lungo. Salto.")
            day_hour_bias_results['error'] = "Not applicable for long timeframe (e.g., >= 1d)"
            day_hour_bias_results['day_hour_positive_pct'] = {day_map.get(d, str(d)): {f"{h:02d}H": None for h in range(24)} for d in range(7)}
            return day_hour_bias_results
        if self.data is None or 'simple_return' not in self.data.columns: day_hour_bias_results['error'] = "Simple returns missing"; return day_hour_bias_results
        if not isinstance(self.data.index, pd.DatetimeIndex): day_hour_bias_results['error'] = "Invalid index type"; return day_hour_bias_results
        try:
            returns_clean = self.data['simple_return'].dropna(); min_data_points_day_hour = 24 * 7 * 4
            if len(returns_clean) < min_data_points_day_hour: day_hour_bias_results['error'] = f"Insufficient data points ({len(returns_clean)} < {min_data_points_day_hour})"; logger.debug(day_hour_bias_results['error']); return day_hour_bias_results
            grouped_stats = returns_clean.groupby([returns_clean.index.dayofweek, returns_clean.index.hour]).apply(lambda x: (x > 0).mean() * 100 if not x.empty else np.nan)
            day_hour_bias_dict = {}
            for day_idx in range(7):
                day_str = day_map.get(day_idx, str(day_idx)); day_hour_bias_dict[day_str] = {}
                for hour_idx in range(24):
                    try: pct_val = grouped_stats.loc[(day_idx, hour_idx)]; day_hour_bias_dict[day_str][f"{hour_idx:02d}H"] = _safe_float(pct_val)
                    except KeyError: day_hour_bias_dict[day_str][f"{hour_idx:02d}H"] = None
                    except Exception as inner_e: logger.warning(f"Error accessing/converting {day_str}-{hour_idx:02d}H bias: {inner_e}"); day_hour_bias_dict[day_str][f"{hour_idx:02d}H"] = None
            day_hour_bias_results['day_hour_positive_pct'] = day_hour_bias_dict; logger.debug("Day/Hour bias calculated successfully.")
        except Exception as e: logger.error(f"Error calculating Day/Hour Bias: {e}", exc_info=True); day_hour_bias_results['error'] = str(e); day_hour_bias_results['day_hour_positive_pct'] = {day_map.get(d, str(d)): {f"{h:02d}H": None for h in range(24)} for d in range(7)}
        return day_hour_bias_results

    def _calculate_daily_bias(self) -> Dict[str, Any]:
        daily_bias_results = {'daily_positive_pct': {}, 'error': None}; day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        if self._is_long_timeframe: # Salta se il timeframe è '1d' o maggiore
            logger.debug("Daily bias non applicabile per timeframe lungo (>= 1d). Salto.")
            daily_bias_results['error'] = "Not applicable for long timeframe (>= 1d)"
            daily_bias_results['daily_positive_pct'] = {day_map.get(d, str(d)): None for d in range(7)}
            return daily_bias_results
        if self.data is None or 'simple_return' not in self.data.columns: daily_bias_results['error'] = "Simple returns missing"; return daily_bias_results
        if not isinstance(self.data.index, pd.DatetimeIndex): daily_bias_results['error'] = "Invalid index type"; return daily_bias_results
        try:
            returns_clean = self.data['simple_return'].dropna(); min_data_points_daily = 7 * 12
            if len(returns_clean) < min_data_points_daily: daily_bias_results['error'] = f"Insufficient data points ({len(returns_clean)} < {min_data_points_daily})"; logger.debug(daily_bias_results['error']); return daily_bias_results
            positive_pct_by_day = returns_clean.groupby(returns_clean.index.dayofweek).apply(lambda x: (x > 0).mean() * 100 if not x.empty else np.nan)
            daily_bias_dict = {}
            for day_idx in range(7): day_str = day_map.get(day_idx, str(day_idx)); pct_val = positive_pct_by_day.get(day_idx, np.nan); daily_bias_dict[day_str] = _safe_float(pct_val)
            daily_bias_results['daily_positive_pct'] = daily_bias_dict; logger.debug("Daily bias calculated successfully.")
        except Exception as e: logger.error(f"Error calculating Daily Bias: {e}", exc_info=True); daily_bias_results['error'] = str(e); daily_bias_results['daily_positive_pct'] = {day_map.get(d, str(d)): None for d in range(7)}
        return daily_bias_results

    def calculate_stationarity(self) -> Dict[str, Any]:
        stationarity_results: Dict[str, Any] = {'statistic': None, 'p_value': None, 'is_stationary': None, 'error': None}
        if self.data is None or 'close' not in self.data.columns or self.data['close'].isnull().all(): stationarity_results['error'] = "Close column missing or empty"; return stationarity_results
        try:
            close_prices = self.data['close'].dropna(); min_adf_points = 20
            if len(close_prices) < min_adf_points: stationarity_results['error'] = f"Insufficient data ({len(close_prices)} < {min_adf_points})"; return stationarity_results
            adf_result = adfuller(close_prices, regression='c', autolag='AIC'); statistic = _safe_float(adf_result[0]); p_value = _safe_float(adf_result[1])
            stationarity_results['statistic'] = statistic; stationarity_results['p_value'] = p_value
            if p_value is not None: stationarity_results['is_stationary'] = bool(p_value < 0.05)
        except Exception as e: logger.error(f"Error calculating stationarity (ADF): {str(e)}", exc_info=True); stationarity_results['error'] = str(e)
        return stationarity_results

    def calculate_autocorrelation(self, lags: int = 20) -> Dict[str, Any]:
        autocorr_results: Dict[str, Any] = {'statistic': None, 'p_value': None, 'is_autocorrelated': None, 'error': None}
        if self.data is None or 'simple_return' not in self.data.columns or self.data['simple_return'].isnull().all(): autocorr_results['error'] = "Simple returns missing or empty"; return autocorr_results
        try:
            returns = self.data['simple_return'].dropna(); effective_lags = min(lags, max(1, len(returns) // 2 - 2)); min_lb_points = 10
            if len(returns) <= effective_lags or effective_lags <= 0 or len(returns) < min_lb_points: autocorr_results['error'] = f"Insufficient data ({len(returns)}) for Ljung-Box with {effective_lags} lags"; return autocorr_results
            lb_test_result = acorr_ljungbox(returns, lags=[effective_lags], return_df=True); statistic = _safe_float(lb_test_result['lb_stat'].iloc[-1]); p_value = _safe_float(lb_test_result['lb_pvalue'].iloc[-1])
            autocorr_results['statistic'] = statistic; autocorr_results['p_value'] = p_value
            if p_value is not None: autocorr_results['is_autocorrelated'] = bool(p_value < 0.05)
        except Exception as e: logger.error(f"Error calculating autocorrelation (Ljung-Box): {str(e)}", exc_info=True); autocorr_results['error'] = str(e)
        return autocorr_results

    def calculate_volatility_clustering(self, lags: int = 12) -> Dict[str, Any]:
        arch_results: Dict[str, Any] = {'statistic': None, 'p_value': None, 'has_volatility_clustering': None, 'error': None}
        if self.data is None or 'simple_return' not in self.data.columns or self.data['simple_return'].isnull().all(): arch_results['error'] = "Simple returns missing or empty"; return arch_results
        try:
            returns = self.data['simple_return'].dropna(); effective_lags = min(lags, max(1, len(returns) - 2)); min_arch_points = 15
            if len(returns) <= effective_lags + 1 or effective_lags <= 0 or len(returns) < min_arch_points: arch_results['error'] = f"Insufficient data ({len(returns)}) for ARCH test with {effective_lags} lags"; return arch_results
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=effective_lags)
            arch_results['statistic'] = _safe_float(lm_stat); arch_results['p_value'] = _safe_float(lm_pvalue)
            if lm_pvalue is not None: arch_results['has_volatility_clustering'] = bool(lm_pvalue < 0.05)
        except Exception as e: logger.error(f"Error calculating volatility clustering (ARCH): {str(e)}", exc_info=True); arch_results['error'] = str(e)
        return arch_results

    def calculate_garch_model(self, p: int = 1, q: int = 1) -> Dict[str, Any]:
        garch_results: Dict[str, Any] = {'garch_converged': None, 'garch_model_type': f'GARCH({p},{q})', 'garch_params': {}, 'garch_loglikelihood': None, 'garch_aic': None, 'garch_bic': None, 'garch_persistence': None, 'error': None}
        required_lags = max(p, q); min_data_points = 50 + required_lags; returns_col = None; use_rescale = False
        if self.data is None: garch_results['error'] = "No data"; return garch_results
        if 'simple_return_scaled' in self.data.columns and not self.data['simple_return_scaled'].isnull().all(): returns_col = 'simple_return_scaled'; use_rescale = False; logger.debug("Using 'simple_return_scaled' for GARCH.")
        elif 'simple_return' in self.data.columns and not self.data['simple_return'].isnull().all(): returns_col = 'simple_return'; use_rescale = True; logger.warning("Using 'simple_return' for GARCH. Consider scaling.")
        else: garch_results['error'] = "No valid return data found"; return garch_results
        try:
            returns_series = self.data[returns_col].dropna()
            if len(returns_series) < min_data_points: garch_results['error'] = f"Insufficient data ({len(returns_series)} < {min_data_points})"; return garch_results
            returns_for_garch = returns_series.values
            model = arch_model(returns_for_garch, vol='Garch', p=p, q=q, mean='Zero', dist='Normal', rescale=use_rescale)
            model_fit = model.fit(disp='off', show_warning=False)
            is_converged = model_fit.convergence_flag == 0; garch_results['garch_converged'] = bool(is_converged)
            if is_converged:
                logger.info(f"GARCH({p},{q}) model estimated."); params = model_fit.params
                garch_results['garch_params'] = {name: _safe_float(val) for name, val in params.items()}; garch_results['garch_loglikelihood'] = _safe_float(model_fit.loglikelihood); garch_results['garch_aic'] = _safe_float(model_fit.aic); garch_results['garch_bic'] = _safe_float(model_fit.bic)
                alpha_coeffs = [_safe_float(params.get(f'alpha[{i}]')) for i in range(1, p + 1)]; beta_coeffs = [_safe_float(params.get(f'beta[{i}]')) for i in range(1, q + 1)]; valid_coeffs = [c for c in alpha_coeffs + beta_coeffs if c is not None]
                if len(valid_coeffs) == p + q: garch_results['garch_persistence'] = float(sum(valid_coeffs))
            else: garch_results['error'] = f"Fit did not converge (flag={model_fit.convergence_flag})"; logger.warning(f"GARCH({p},{q}) model did not converge (flag: {model_fit.convergence_flag}).")
        except Exception as e: logger.error(f"Error calculating GARCH: {str(e)}", exc_info=True); garch_results['error'] = str(e)
        return garch_results

    def test_normality(self, data_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        normality_results: Dict[str, Any] = {'jarque_bera': {}, 'shapiro_wilk': {}, 'conclusion': {}, 'error': None}; series_to_test: Optional[pd.Series] = None
        try:
            if data_series is None:
                if self.data is None or 'simple_return' not in self.data.columns or self.data['simple_return'].isnull().all(): normality_results['error'] = "Return data missing"; return normality_results
                series_to_test = self.data['simple_return'].dropna()
            else: series_to_test = data_series.dropna()
            min_norm_points = 20
            if len(series_to_test) < min_norm_points: normality_results['error'] = f"Insufficient data ({len(series_to_test)} < {min_norm_points})"; return normality_results
            try: # Jarque-Bera
                jb_stat, jb_p = jarque_bera(series_to_test); jb_skew_calc = _safe_float(skew(series_to_test, nan_policy='omit')); jb_kurt_calc = _safe_float(kurtosis(series_to_test, fisher=True, nan_policy='omit'))
                jb_p_safe = _safe_float(jb_p)
                normality_results['jarque_bera'] = {'statistic': _safe_float(jb_stat), 'p_value': jb_p_safe, 'skewness': jb_skew_calc, 'kurtosis': jb_kurt_calc + 3 if jb_kurt_calc is not None else None, 'is_normal': bool(jb_p_safe > 0.05) if jb_p_safe is not None else None}
            except Exception as jb_e: logger.warning(f"Error Jarque-Bera test: {jb_e}"); normality_results['jarque_bera'] = {'error': str(jb_e)}
            try: # Shapiro-Wilk
                data_for_sw = series_to_test; sample_limit = 4999
                if len(series_to_test) > sample_limit: logger.debug(f"Sampling data ({len(series_to_test)} -> {sample_limit}) for Shapiro-Wilk."); data_for_sw = series_to_test.sample(sample_limit, random_state=42)
                if len(data_for_sw.unique()) > 3:
                    sw_stat, sw_p = shapiro(data_for_sw); sw_p_safe = _safe_float(sw_p)
                    normality_results['shapiro_wilk'] = {'statistic': _safe_float(sw_stat), 'p_value': sw_p_safe, 'is_normal': bool(sw_p_safe > 0.05) if sw_p_safe is not None else None}
                else: raise ValueError("Need >= 3 unique data points for Shapiro-Wilk")
            except Exception as sw_e: logger.warning(f"Error Shapiro-Wilk test: {sw_e}"); normality_results['shapiro_wilk'] = {'error': str(sw_e)}
            test_used = 'shapiro_wilk' if len(series_to_test) <= sample_limit and 'error' not in normality_results['shapiro_wilk'] else 'jarque_bera'
            final_is_normal = normality_results.get(test_used, {}).get('is_normal')
            if final_is_normal is None:
                 alternative_test = 'jarque_bera' if test_used == 'shapiro_wilk' else 'shapiro_wilk'
                 final_is_normal = normality_results.get(alternative_test, {}).get('is_normal')
                 if final_is_normal is not None: test_used = alternative_test
            normality_results['conclusion'] = {'is_normal': final_is_normal, 'test_used': test_used}
        except Exception as e: logger.error(f"General error in normality testing: {str(e)}", exc_info=True); normality_results['error'] = str(e)
        return normality_results

    def _run_advanced_statistical_tests(self) -> Dict[str, Any]:
        logger.debug("Running advanced statistical tests...")
        advanced_results = {}
        test_functions = {
            'stationarity': self.calculate_stationarity,
            'autocorrelation': self.calculate_autocorrelation,
            'volatility_clustering': self.calculate_volatility_clustering,
            'garch_model': self.calculate_garch_model,
            'normality': self.test_normality
        }
        for test_name, test_func in test_functions.items():
            try:
                advanced_results[test_name] = test_func()
            except Exception as test_err:
                 logger.error(f"Error running advanced test '{test_name}': {test_err}", exc_info=True)
                 advanced_results[test_name] = {'error': f"Failed: {test_err}"}
        return advanced_results

    def _calculate_hourly_volume_stats(self) -> Dict[str, Any]:
        hourly_vol_results = {
            'hourly_avg_volume': {},
            'last_candle_hour': None,
            'last_candle_volume': None,
            'last_candle_avg_volume_for_hour': None,
            'last_vs_avg_ratio': None,
            'is_significantly_above_average': None,
            'volume_alert_threshold_factor': 2.0,
            'error': None
        }
        if self._is_long_timeframe:
            logger.debug("Hourly volume stats non applicabile per timeframe lungo. Salto.")
            hourly_vol_results['error'] = "Not applicable for long timeframe (e.g., >= 1d)"
            hourly_vol_results['hourly_avg_volume'] = {f"{h:02d}H": None for h in range(24)}
            return hourly_vol_results
        if self.data is None or 'volume' not in self.data.columns:
            hourly_vol_results['error'] = "Volume data missing"; return hourly_vol_results
        if not isinstance(self.data.index, pd.DatetimeIndex):
            hourly_vol_results['error'] = "Invalid index type"; return hourly_vol_results
        try:
            volume_data = self.data['volume'].dropna()
            min_data_points_hourly_vol = 24 * 5
            if len(volume_data) < min_data_points_hourly_vol:
                hourly_vol_results['error'] = f"Insufficient data points ({len(volume_data)} < {min_data_points_hourly_vol})";
                logger.debug(hourly_vol_results['error'])
                return hourly_vol_results
            avg_volume_by_hour = volume_data.groupby(volume_data.index.hour).mean()
            hourly_avg_dict = {}
            for hour in range(24):
                avg_val = avg_volume_by_hour.get(hour, np.nan)
                hourly_avg_dict[f"{hour:02d}H"] = _safe_float(avg_val)
            hourly_vol_results['hourly_avg_volume'] = hourly_avg_dict
            last_candle = self.data.iloc[-1]
            last_timestamp = self.data.index[-1]
            last_hour = last_timestamp.hour
            last_volume = _safe_float(last_candle.get('volume'))
            hourly_vol_results['last_candle_hour'] = last_hour
            hourly_vol_results['last_candle_volume'] = last_volume
            last_hour_avg_volume = hourly_avg_dict.get(f"{last_hour:02d}H")
            hourly_vol_results['last_candle_avg_volume_for_hour'] = last_hour_avg_volume
            if last_volume is not None and last_hour_avg_volume is not None and last_hour_avg_volume > 1e-9:
                ratio = last_volume / last_hour_avg_volume
                hourly_vol_results['last_vs_avg_ratio'] = _safe_float(ratio)
                alert_factor = hourly_vol_results['volume_alert_threshold_factor']
                hourly_vol_results['is_significantly_above_average'] = bool(ratio > alert_factor)
            else:
                hourly_vol_results['is_significantly_above_average'] = False
            logger.debug("Hourly volume stats calculated successfully.")
        except Exception as e:
            logger.error(f"Error calculating Hourly Volume Stats: {e}", exc_info=True)
            hourly_vol_results['error'] = str(e)
            hourly_vol_results['hourly_avg_volume'] = {f"{h:02d}H": None for h in range(24)}
            hourly_vol_results['last_candle_volume'] = None
            hourly_vol_results['last_candle_avg_volume_for_hour'] = None
            hourly_vol_results['last_vs_avg_ratio'] = None
            hourly_vol_results['is_significantly_above_average'] = None
        return hourly_vol_results

    def _calculate_historical_cycle_stats(self, data_for_calc: pd.DataFrame) -> Dict[str, Any]:
        hist_stats = {
            'up_cycles_count': 0, 'down_cycles_count': 0,
            'hist_median_duration_up': None, 'hist_mean_duration_up': None,
            'hist_stddev_duration_up': None, 'hist_skewness_duration_up': None,
            'hist_kurtosis_duration_up': None,
            'hist_p25_duration_up': None, 'hist_p75_duration_up': None,
            'hist_median_duration_down': None, 'hist_mean_duration_down': None,
            'hist_stddev_duration_down': None, 'hist_skewness_duration_down': None,
            'hist_kurtosis_duration_down': None,
            'hist_p25_duration_down': None, 'hist_p75_duration_down': None,
            'hist_duration_ratio_mean': None, 'hist_duration_ratio_median': None,
            'calculation_timestamp': safe_strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
            'data_points_used': len(data_for_calc),
            'error': None
        }
        logger.info(f"Avvio calcolo statistiche storiche cicli (dati: {len(data_for_calc)})...")
        if data_for_calc is None or data_for_calc.empty: hist_stats['error'] = "No data provided"; return hist_stats
        required_cols = ['high', 'low', 'close']
        if not all(c in data_for_calc.columns for c in required_cols): hist_stats['error'] = f"Missing columns: {required_cols}"; return hist_stats
        if len(data_for_calc) < CYCLE_MIN_DATA_POINTS: hist_stats['error'] = f"Insufficient points ({len(data_for_calc)} < {CYCLE_MIN_DATA_POINTS})"; return hist_stats
        if not isinstance(data_for_calc.index, pd.DatetimeIndex): hist_stats['error'] = "Not DatetimeIndex"; return hist_stats
        try:
            highs = data_for_calc['high']; lows = data_for_calc['low']; data_index = data_for_calc.index
            avg_price = data_for_calc['close'].mean()
            min_prominence = avg_price * CYCLE_PEAK_PROMINENCE_FACTOR
            min_distance = CYCLE_PEAK_TROUGH_DISTANCE
            peak_indices_loc, _ = find_peaks(highs, distance=min_distance, prominence=min_prominence)
            trough_indices_loc, _ = find_peaks(-lows, distance=min_distance, prominence=min_prominence)
            if len(peak_indices_loc) == 0 or len(trough_indices_loc) == 0: hist_stats['error'] = "No peaks/troughs"; return hist_stats
            peaks = pd.Series(highs.iloc[peak_indices_loc].values, index=data_index[peak_indices_loc], name='Peak')
            troughs = pd.Series(lows.iloc[trough_indices_loc].values, index=data_index[trough_indices_loc], name='Trough')
            extrema = pd.concat([peaks, troughs]).sort_index()
            up_durations = []; down_durations = []; last_type = None; last_index_loc = -1
            for dt, price in extrema.items():
                current_index_loc = data_for_calc.index.get_loc(dt)
                current_type = 'peak' if not pd.isna(peaks.get(dt)) else 'trough'
                if last_type is not None and current_type != last_type:
                    duration = current_index_loc - last_index_loc
                    if duration > 0:
                        if last_type == 'trough' and current_type == 'peak': up_durations.append(duration)
                        elif last_type == 'peak' and current_type == 'trough': down_durations.append(duration)
                last_type = current_type; last_index_loc = current_index_loc
            if up_durations:
                up_dur_series = pd.Series(up_durations); hist_stats['up_cycles_count'] = len(up_dur_series)
                hist_stats['hist_median_duration_up'] = _safe_float(up_dur_series.median())
                hist_stats['hist_mean_duration_up'] = _safe_float(up_dur_series.mean())
                hist_stats['hist_stddev_duration_up'] = _safe_float(up_dur_series.std())
                hist_stats['hist_skewness_duration_up'] = _safe_float(up_dur_series.skew())
                hist_stats['hist_kurtosis_duration_up'] = _safe_float(up_dur_series.kurt())
                hist_stats['hist_p25_duration_up'] = _safe_float(up_dur_series.quantile(0.25))
                hist_stats['hist_p75_duration_up'] = _safe_float(up_dur_series.quantile(0.75))
            if down_durations:
                down_dur_series = pd.Series(down_durations); hist_stats['down_cycles_count'] = len(down_dur_series)
                hist_stats['hist_median_duration_down'] = _safe_float(down_dur_series.median())
                hist_stats['hist_mean_duration_down'] = _safe_float(down_dur_series.mean())
                hist_stats['hist_stddev_duration_down'] = _safe_float(down_dur_series.std())
                hist_stats['hist_skewness_duration_down'] = _safe_float(down_dur_series.skew())
                hist_stats['hist_kurtosis_duration_down'] = _safe_float(down_dur_series.kurt())
                hist_stats['hist_p25_duration_down'] = _safe_float(down_dur_series.quantile(0.25))
                hist_stats['hist_p75_duration_down'] = _safe_float(down_dur_series.quantile(0.75))
            mean_up = hist_stats['hist_mean_duration_up']; mean_down = hist_stats['hist_mean_duration_down']
            median_up = hist_stats['hist_median_duration_up']; median_down = hist_stats['hist_median_duration_down']
            if mean_up is not None and mean_down is not None and mean_down != 0: hist_stats['hist_duration_ratio_mean'] = _safe_float(mean_up / mean_down)
            if median_up is not None and median_down is not None and median_down != 0: hist_stats['hist_duration_ratio_median'] = _safe_float(median_up / median_down)
            hist_stats.pop('error', None); logger.info("Calcolo statistiche storiche cicli completato.")
        except ImportError: hist_stats['error'] = "Modulo scipy non trovato."; logger.critical(hist_stats['error'])
        except Exception as e: logger.error(f"Errore calcolo stats storiche cicli: {e}", exc_info=True); hist_stats['error'] = f"Calculation failed: {e}"
        return hist_stats

    def _calculate_cycle_analysis(self) -> Dict[str, Any]:
        cycle_results: Dict[str, Any] = {
            'current_cycle_start_time': None, 'current_cycle_start_price': None, 'current_cycle_type': None, 'current_cycle_duration': None,
            'current_cycle_price_trend': 'Unknown',
            'historical_durations': {'error': "Not loaded or calculated yet"},
            'derived_duration_indicators': {}, 'error': None
        }
        derived_sub_results = cycle_results['derived_duration_indicators']

        if self.data is None or self.data.empty: cycle_results['error'] = "No data"; return cycle_results
        required_cols = ['high', 'low', 'close']; min_points = CYCLE_MIN_DATA_POINTS
        if not all(c in self.data.columns for c in required_cols): cycle_results['error'] = f"Missing: {required_cols}"; return cycle_results
        if len(self.data) < min_points: cycle_results['error'] = f"Insufficient points ({len(self.data)} < {min_points})"; return cycle_results
        if not isinstance(self.data.index, pd.DatetimeIndex): cycle_results['error'] = "Not DatetimeIndex"; return cycle_results

        loaded_historical_stats = None
        if self.cycle_stats_filename:
            loaded_historical_stats = load_historical_stats_from_json(self.cycle_stats_filename)
            if loaded_historical_stats: logger.info(f"Stats cicli caricate da {self.cycle_stats_filename}"); cycle_results['historical_durations'] = loaded_historical_stats
            else: logger.warning(f"Impossibile caricare stats cicli da {self.cycle_stats_filename}. Ricalcolo (lento)..."); cycle_results['historical_durations']['error'] = "Failed load"
        else: logger.warning("Path stats storiche cicli non definito. Ricalcolo..."); cycle_results['historical_durations']['error'] = "Filename not set"

        try:
            highs = self.data['high']; lows = self.data['low']; closes = self.data['close']; data_index = self.data.index
            avg_price = closes.mean(); min_prominence = avg_price * CYCLE_PEAK_PROMINENCE_FACTOR; min_distance = CYCLE_PEAK_TROUGH_DISTANCE
            peak_indices_loc, _ = find_peaks(highs, distance=min_distance, prominence=min_prominence)
            trough_indices_loc, _ = find_peaks(-lows, distance=min_distance, prominence=min_prominence)

            if not loaded_historical_stats or loaded_historical_stats.get('error'):
                logger.warning("Eseguo calcolo fallback stats storiche cicli (lento)...")
                fallback_hist_stats = self._calculate_historical_cycle_stats(self.data.copy())
                cycle_results['historical_durations'] = fallback_hist_stats
                if self.cycle_stats_filename and not fallback_hist_stats.get('error'):
                    save_ok = save_historical_stats_to_json(fallback_hist_stats, self.cycle_stats_filename)
                    if save_ok: logger.info(f"Stats storiche cicli (fallback) salvate in {self.cycle_stats_filename}")
                    else: logger.error(f"Fallito salvataggio stats storiche cicli (fallback) in {self.cycle_stats_filename}")
                loaded_historical_stats = fallback_hist_stats

            last_type = None; last_index_loc = -1; last_dt = None
            if len(peak_indices_loc) > 0 or len(trough_indices_loc) > 0:
                peaks = pd.Series(highs.iloc[peak_indices_loc].values, index=data_index[peak_indices_loc], name='Peak')
                troughs = pd.Series(lows.iloc[trough_indices_loc].values, index=data_index[trough_indices_loc], name='Trough')
                extrema = pd.concat([peaks, troughs]).sort_index()
                if not extrema.empty:
                    last_extreme_dt = extrema.index[-1]; last_index_loc = self.data.index.get_loc(last_extreme_dt)
                    last_type = 'peak' if pd.notna(peaks.get(last_extreme_dt)) else 'trough'; last_dt = last_extreme_dt
                else: logger.warning("Serie extrema vuota nonostante picchi/minimi trovati?")
            else: logger.warning("Nessun picco/minimo per analisi ciclo corrente."); cycle_results['error'] = "No peaks/troughs found"

            if last_type is not None and last_index_loc >= 0 and last_dt is not None:
                cycle_results['current_cycle_start_time'] = safe_strftime(last_dt, '%Y-%m-%d %H:%M:%S')
                start_price = closes.loc[last_dt] if last_dt in closes.index else None
                cycle_results['current_cycle_start_price'] = _safe_float(start_price)
                cycle_results['current_cycle_type'] = 'up' if last_type == 'trough' else 'down'
                cycle_results['current_cycle_duration'] = len(self.data) - 1 - last_index_loc
                current_price = safe_get_last_value(closes)
                if start_price is not None and current_price is not None:
                     cycle_results['current_cycle_price_trend'] = "Bullish (Price above start)" if current_price > start_price else "Bearish (Price below start)" if current_price < start_price else "Flat (Price at start)"
            else: cycle_results['current_cycle_type'] = 'unknown'; cycle_results['current_cycle_duration'] = None; cycle_results['current_cycle_start_time'] = None; cycle_results['current_cycle_start_price'] = None

            hist_stats_for_derived = cycle_results.get('historical_durations', {})
            if hist_stats_for_derived and not hist_stats_for_derived.get('error'):
                median_up = hist_stats_for_derived.get('hist_median_duration_up'); median_down = hist_stats_for_derived.get('hist_median_duration_down')
                p25_up = hist_stats_for_derived.get('hist_p25_duration_up'); p75_up = hist_stats_for_derived.get('hist_p75_duration_up')
                p25_down = hist_stats_for_derived.get('hist_p25_duration_down'); p75_down = hist_stats_for_derived.get('hist_p75_duration_down')
                mean_up = hist_stats_for_derived.get('hist_mean_duration_up'); mean_down = hist_stats_for_derived.get('hist_mean_duration_down')
                stddev_up = hist_stats_for_derived.get('hist_stddev_duration_up'); stddev_down = hist_stats_for_derived.get('hist_stddev_duration_down')
                current_duration = cycle_results.get('current_cycle_duration')

                if current_duration is not None and current_duration > 0:
                    current_type = cycle_results.get('current_cycle_type')
                    if current_type == 'up':
                        if median_up is not None and median_up > 0: derived_sub_results['current_duration_vs_median_up_ratio'] = _safe_float(current_duration / median_up)
                        if mean_up is not None and stddev_up is not None and stddev_up > 0: derived_sub_results['current_duration_in_stddev_up'] = _safe_float((current_duration - mean_up) / stddev_up)
                        if p25_up is not None and p25_up > 0: derived_sub_results['current_duration_vs_p25_up_ratio'] = _safe_float(current_duration / p25_up)
                        if p75_up is not None and p75_up > 0: derived_sub_results['current_duration_vs_p75_up_ratio'] = _safe_float(current_duration / p75_up)
                        if median_up is not None: derived_sub_results['time_to_median_up_target'] = _safe_float(median_up - current_duration)
                    elif current_type == 'down':
                        if median_down is not None and median_down > 0: derived_sub_results['current_duration_vs_median_down_ratio'] = _safe_float(current_duration / median_down)
                        if mean_down is not None and stddev_down is not None and stddev_down > 0: derived_sub_results['current_duration_in_stddev_down'] = _safe_float((current_duration - mean_down) / stddev_down)
                        if p25_down is not None and p25_down > 0: derived_sub_results['current_duration_vs_p25_down_ratio'] = _safe_float(current_duration / p25_down)
                        if p75_down is not None and p75_down > 0: derived_sub_results['current_duration_vs_p75_down_ratio'] = _safe_float(current_duration / p75_down)
                        if median_down is not None: derived_sub_results['time_to_median_down_target'] = _safe_float(median_down - current_duration)
            else: logger.warning("Stats storiche cicli non disponibili per indicatori derivati.")
            cycle_results.pop('error', None)
        except ImportError: cycle_results['error'] = "scipy mancante."; logger.critical(cycle_results['error']); cycle_results['historical_durations'] = {'error': cycle_results['error']}; cycle_results['derived_duration_indicators'] = {}
        except Exception as e:
            logger.error(f"Errore analisi cicli: {e}", exc_info=True); cycle_results['error'] = f"Cycle analysis failed: {e}"
            cycle_results['historical_durations'] = {'error': str(e)}
            cycle_results['derived_duration_indicators'] = {}; cycle_results['current_cycle_duration'] = None; cycle_results['current_cycle_type'] = None
        return cycle_results

    def _calculate_historical_intracandle_stats(self, data_for_calc: pd.DataFrame) -> Dict[str, Any]:
        hist_stats = {
            'stats_by_metric': {},
            'calculation_timestamp': safe_strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
            'data_points_used': len(data_for_calc),
            'error': None
        }
        logger.info(f"Avvio calcolo statistiche storiche intra-candela (dati: {len(data_for_calc)})...")
        if data_for_calc is None or data_for_calc.empty: hist_stats['error'] = "No data provided"; return hist_stats
        metric_cols = ['ic_range_pct', 'ic_min_max_pct', 'ic_min_close_pct', 'ic_open_max_pct', 'ic_open_min_pct', 'ic_body_pct']
        if not all(col in data_for_calc.columns for col in metric_cols):
            logger.warning("Colonne metriche intra-candela mancanti per storico. Tentativo di calcolo...")
            temp_df = data_for_calc.copy()
            temp_analyzer = StatisticalAnalyzerAdvanced(temp_df)
            temp_analyzer._calculate_intracandle_metrics_for_dataframe()
            data_to_use = temp_analyzer.data
            if not all(col in data_to_use.columns for col in metric_cols):
                 hist_stats['error'] = "Failed to calculate intracandle metric columns for history."; return hist_stats
        else: data_to_use = data_for_calc
        for metric_col in metric_cols:
            metric_series = data_to_use[metric_col].dropna()
            if metric_series.empty:
                logger.warning(f"Nessun dato valido per la metrica storica: {metric_col}"); hist_stats['stats_by_metric'][metric_col] = {'error': 'No valid data'}; continue
            stats_for_metric: Dict[str, Optional[float]] = {}
            try:
                stats_for_metric['mean'] = _safe_float(metric_series.mean())
                stats_for_metric['median'] = _safe_float(metric_series.median())
                stats_for_metric['std_dev'] = _safe_float(metric_series.std())
                stats_for_metric['min'] = _safe_float(metric_series.min())
                stats_for_metric['max'] = _safe_float(metric_series.max())
                percentiles_to_calc = [p / 100.0 for p in INTRACANDLE_PERCENTILES]
                calculated_percentiles = metric_series.quantile(percentiles_to_calc)
                for p_val, p_label in zip(percentiles_to_calc, INTRACANDLE_PERCENTILES):
                    stats_for_metric[f'p{p_label}'] = _safe_float(calculated_percentiles.get(p_val))
                hist_stats['stats_by_metric'][metric_col] = stats_for_metric
            except Exception as e:
                logger.error(f"Errore calcolo statistiche per {metric_col}: {e}", exc_info=True); hist_stats['stats_by_metric'][metric_col] = {'error': f"Calculation failed: {e}"}
        hist_stats.pop('error', None); logger.info("Calcolo statistiche storiche intra-candela completato.")
        return hist_stats

    def _analyze_current_candle_intracandle_stats(self) -> Dict[str, Any]:
        analysis_results = {
            'last_closed_candle_metrics': {},
            'historical_comparison': {},
            'historical_stats_summary': {'error': "Not loaded or calculated yet"},
            'last_candle_timestamp_used': None, # Aggiunto per tracciabilità
            'error': None
        }
        metric_cols = [
            'ic_range_pct', 'ic_min_max_pct', 'ic_min_close_pct',
            'ic_open_max_pct', 'ic_open_min_pct', 'ic_body_pct'
        ]

        if self.data is None or self.data.empty:
            analysis_results['error'] = "No data"; return analysis_results
        if len(self.data) < 1:
            analysis_results['error'] = "Insufficient data (need >= 1 candle)"; return analysis_results

        # 1. Assicura che le colonne esistano
        if not all(col in self.data.columns for col in metric_cols):
             logger.warning("Colonne metriche intra-candela mancanti per analisi corrente. Tentativo ricalcolo...")
             self._calculate_intracandle_metrics_for_dataframe()
             if not all(col in self.data.columns for col in metric_cols):
                 analysis_results['error'] = "Failed to ensure intracandle columns exist."; return analysis_results

        # Ricerca candela valida
        last_valid_candle_metrics = None
        last_valid_candle_timestamp = None
        max_lookback = 5
        data_to_search = self.data.sort_index(ascending=True)

        for i in range(1, min(max_lookback + 1, len(data_to_search) + 1)):
            try:
                candidate_candle_raw = data_to_search.iloc[-i]
                are_metrics_valid = not candidate_candle_raw[metric_cols].isnull().any()
                temp_metrics = {}

                if are_metrics_valid:
                    for col in metric_cols:
                        temp_metrics[col] = _safe_float(candidate_candle_raw.get(col))
                    last_valid_candle_metrics = temp_metrics
                    last_valid_candle_timestamp = candidate_candle_raw.name
                    logger.info(f"Intracandle: Usata candela valida a ritroso di {i-1} periodi (Timestamp: {safe_strftime(last_valid_candle_timestamp)}).")
                    break
                else:
                     logger.debug(f"Intracandle: Candela a ritroso di {i-1} periodi non valida (contiene NaN).")
            except IndexError:
                logger.warning(f"Intracandle: Indice -{i} fuori range durante ricerca candela valida.")
                break

        if last_valid_candle_metrics is None:
            logger.error(f"Intracandle: Nessuna candela valida (senza NaN) trovata nelle ultime {max_lookback} candele.")
            analysis_results['error'] = f"No valid (non-NaN) candle metrics found in last {max_lookback} periods."
            analysis_results['last_closed_candle_metrics'] = {col: None for col in metric_cols}
            analysis_results['historical_comparison'] = {col: {'value': None, 'percentile_rank': 'N/A'} for col in metric_cols}
            return analysis_results
        else:
            analysis_results['last_closed_candle_metrics'] = last_valid_candle_metrics
            analysis_results['last_candle_timestamp_used'] = safe_strftime(last_valid_candle_timestamp)

        # 2. Carica (o calcola fallback) statistiche storiche
        loaded_historical_stats = None
        if self.intracandle_stats_filename:
            loaded_historical_stats = load_historical_stats_from_json(self.intracandle_stats_filename)
            if loaded_historical_stats:
                logger.info(f"Stats intra-candela caricate da {self.intracandle_stats_filename}")
                analysis_results['historical_stats_summary'] = loaded_historical_stats
            else:
                logger.warning(f"Impossibile caricare stats intra-candela da {self.intracandle_stats_filename}. Ricalcolo fallback (lento)...")
                analysis_results['historical_stats_summary']['error'] = "Failed load, attempting fallback calculation."
        else:
            logger.warning("Path stats storiche intra-candela non definito. Ricalcolo fallback...")
            analysis_results['historical_stats_summary']['error'] = "Filename not set, attempting fallback calculation."

        if not loaded_historical_stats or loaded_historical_stats.get('error'):
             try:
                logger.warning("Eseguo calcolo fallback stats storiche intra-candela (lento)...")
                fallback_hist_stats = self._calculate_historical_intracandle_stats(self.data.copy())
                analysis_results['historical_stats_summary'] = fallback_hist_stats
                if self.intracandle_stats_filename and not fallback_hist_stats.get('error'):
                    save_ok = save_historical_stats_to_json(fallback_hist_stats, self.intracandle_stats_filename)
                    if save_ok: logger.info(f"Stats storiche intra-candela (fallback) salvate in {self.intracandle_stats_filename}")
                    else: logger.error(f"Fallito salvataggio stats storiche intra-candela (fallback) in {self.intracandle_stats_filename}")
                loaded_historical_stats = fallback_hist_stats
             except Exception as fallback_err:
                 logger.error(f"Errore durante calcolo fallback stats storiche intra-candela: {fallback_err}")
                 analysis_results['historical_stats_summary']['error'] = f"Fallback calculation failed: {fallback_err}"
                 analysis_results['error'] = (analysis_results.get('error') or "") + " Historical stats calculation failed."
                 loaded_historical_stats = None

        # 3. Confronta con storico
        comparison_results = {}
        hist_stats_data = loaded_historical_stats.get('stats_by_metric') if loaded_historical_stats else None

        if hist_stats_data and not analysis_results.get('error'):
            for metric_col, last_value in last_valid_candle_metrics.items():
                comparison_results[metric_col] = {'value': last_value, 'percentile_rank': 'N/A'}
                if last_value is None: continue

                metric_hist = hist_stats_data.get(metric_col)
                if not metric_hist or metric_hist.get('error'): continue

                percentiles_sorted = sorted([p for p in INTRACANDLE_PERCENTILES])
                percentile_rank = 0
                found_rank = False
                for p in percentiles_sorted:
                    p_key = f'p{p}'
                    percentile_value = metric_hist.get(p_key)
                    if percentile_value is None: continue
                    if last_value <= percentile_value + 1e-9:
                         percentile_rank = p
                         found_rank = True
                         break
                if not found_rank: percentile_rank = 100
                comparison_results[metric_col]['percentile_rank'] = percentile_rank
            analysis_results['historical_comparison'] = comparison_results
        elif not hist_stats_data:
            logger.warning("Dati storici ('stats_by_metric') non trovati per confronto intracandle.")
            analysis_results['historical_comparison'] = {'error': 'Historical stats data missing'}
            if not analysis_results.get('error'): analysis_results['error'] = (analysis_results.get('error') or "") + " Hist stats missing for comparison."
        else:
             analysis_results['historical_comparison'] = {'error': 'Failed to find valid recent candle metrics'}

        if not analysis_results.get('error') and \
           not _safe_get(analysis_results, ['historical_stats_summary', 'error']) and \
           not _safe_get(analysis_results, ['historical_comparison', 'error']):
             analysis_results.pop('error', None)

        return analysis_results

    def update_historical_stats(self) -> Tuple[bool, bool]:
        if self.data is None or self.data.empty:
            logger.error("Cannot update historical stats: Data is missing.")
            return False, False
        if not self.symbol or not self.timeframe:
            logger.error("Cannot update historical stats: Symbol or Timeframe not set.")
            return False, False

        success_cycles = False
        logger.info(f"*** STARTING HISTORICAL CYCLE STATS UPDATE for {self.symbol} ({self.timeframe}) ***")
        update_start_time_cycles = datetime.now()
        try:
            historical_cycle_stats = self._calculate_historical_cycle_stats(self.data.copy())
            if not historical_cycle_stats or historical_cycle_stats.get('error'):
                 err_msg = historical_cycle_stats.get('error', 'Unknown') if historical_cycle_stats else 'None';
                 logger.error(f"Failed to calculate historical cycle stats for {self.symbol} ({self.timeframe}): {err_msg}")
            elif self.cycle_stats_filename:
                logger.info(f"Saving historical cycle stats to: {self.cycle_stats_filename}")
                save_success = save_historical_stats_to_json(historical_cycle_stats, self.cycle_stats_filename)
                if save_success:
                    logger.info(f"*** CYCLE STATS UPDATE for {self.symbol} ({self.timeframe}) COMPLETED ***"); success_cycles = True
                else: logger.error(f"Failed to save cycle stats for {self.symbol} ({self.timeframe}).")
            else: logger.error(f"Cycle stats filename not defined for {self.symbol} ({self.timeframe}).")
        except Exception as e: logger.error(f"Unhandled error during cycle stats update for {self.symbol} ({self.timeframe}): {e}", exc_info=True)
        update_end_time_cycles = datetime.now()
        logger.info(f"Cycle stats update duration: {(update_end_time_cycles - update_start_time_cycles).total_seconds():.2f} sec.")

        success_intracandle = False
        logger.info(f"*** STARTING HISTORICAL INTRACANDLE STATS UPDATE for {self.symbol} ({self.timeframe}) ***")
        update_start_time_ic = datetime.now()
        try:
            self._calculate_intracandle_metrics_for_dataframe()
            historical_intracandle_stats = self._calculate_historical_intracandle_stats(self.data.copy())
            if not historical_intracandle_stats or historical_intracandle_stats.get('error'):
                err_msg = historical_intracandle_stats.get('error', 'Unknown') if historical_intracandle_stats else 'None';
                logger.error(f"Failed to calculate historical intracandle stats for {self.symbol} ({self.timeframe}): {err_msg}")
            elif self.intracandle_stats_filename:
                logger.info(f"Saving historical intracandle stats to: {self.intracandle_stats_filename}")
                save_success_ic = save_historical_stats_to_json(historical_intracandle_stats, self.intracandle_stats_filename)
                if save_success_ic:
                    logger.info(f"*** INTRACANDLE STATS UPDATE for {self.symbol} ({self.timeframe}) COMPLETED ***"); success_intracandle = True
                else: logger.error(f"Failed to save intracandle stats for {self.symbol} ({self.timeframe}).")
            else: logger.error(f"Intracandle stats filename not defined for {self.symbol} ({self.timeframe}).")
        except Exception as e: logger.error(f"Unhandled error during intracandle stats update for {self.symbol} ({self.timeframe}): {e}", exc_info=True)
        update_end_time_ic = datetime.now()
        logger.info(f"Intracandle stats update duration: {(update_end_time_ic - update_start_time_ic).total_seconds():.2f} sec.")

        return success_cycles, success_intracandle

    def _calculate_recent_trend_velocity(self, period: int = RECENT_VELOCITY_PERIOD) -> Dict[str, Optional[float]]:
        velocity_results = {
            f'avg_price_change_per_period_{period}p': None,
            'avg_price_change_per_period_normalized_atr': None,
            'error': None
        }
        if self.data is None or 'close' not in self.data.columns or len(self.data) < period + 1:
            velocity_results['error'] = "Insufficient data"; return velocity_results

        try:
            close_prices = self.data['close']
            price_changes = close_prices.diff().iloc[-period:]
            if price_changes.empty:
                 velocity_results['error'] = "No price changes in period"; return velocity_results

            avg_change = price_changes.mean()
            velocity_results[f'avg_price_change_per_period_{period}p'] = _safe_float(avg_change)

            current_atr = _safe_get(self.results, ['technical_analysis', 'technical_indicators', 'atr'])
            if current_atr is None:
                 atr_col_name = next((col for col in self.data.columns if col.lower().startswith('atr_')), None)
                 if atr_col_name:
                      current_atr = _safe_float(safe_get_last_value(self.data[atr_col_name]))

            if current_atr is not None and current_atr > 1e-9 and avg_change is not None and pd.notna(avg_change):
                normalized_velocity = avg_change / current_atr
                velocity_results['avg_price_change_per_period_normalized_atr'] = _safe_float(normalized_velocity)

            velocity_results.pop('error', None)
        except Exception as e:
            logger.error(f"Errore calcolo velocità trend: {e}", exc_info=True)
            velocity_results['error'] = str(e)
        return velocity_results

    def _calculate_time_to_level_estimates(self, avg_movement_factor: float = 0.75) -> Dict[str, Optional[float]]:
        time_est_results = {
            'time_to_resistance_est_periods': None,
            'time_to_support_est_periods': None,
            'error': None
        }
        if self.data is None or self.data.empty:
             time_est_results['error'] = "No data"; return time_est_results

        try:
            sr_data = _safe_get(self.results, ['technical_analysis', 'support_resistance'], {})
            current_atr = _safe_get(self.results, ['technical_analysis', 'technical_indicators', 'atr'])

            dist_res_atr = sr_data.get('resistance_distance_atr')
            dist_sup_atr = sr_data.get('support_distance_atr')

            if current_atr is None or current_atr <= 1e-9:
                 time_est_results['error'] = "ATR non valido"; return time_est_results

            if dist_res_atr is not None and dist_res_atr > 0:
                time_to_res = dist_res_atr / avg_movement_factor
                time_est_results['time_to_resistance_est_periods'] = _safe_float(time_to_res)

            if dist_sup_atr is not None and dist_sup_atr > 0:
                time_to_sup = dist_sup_atr / avg_movement_factor
                time_est_results['time_to_support_est_periods'] = _safe_float(time_to_sup)

            time_est_results.pop('error', None)

        except Exception as e:
            logger.error(f"Errore calcolo stime tempo: {e}", exc_info=True)
            time_est_results['error'] = str(e)
        return time_est_results

    def _find_previous_cycle_extremes(self) -> Tuple[Optional[float], Optional[float]]:
        if self._previous_cycle_high_price is not None and self._previous_cycle_low_price is not None:
             return self._previous_cycle_high_price, self._previous_cycle_low_price

        prev_high, prev_low = None, None
        if self.data is None or self.data.empty or len(self.data) < CYCLE_MIN_DATA_POINTS * 2:
            logger.debug("Dati insufficienti per trovare estremi ciclo precedente.")
            return prev_high, prev_low
        try:
            highs = self.data['high']; lows = self.data['low']; data_index = self.data.index
            avg_price = self.data['close'].mean(); min_prominence = avg_price * CYCLE_PEAK_PROMINENCE_FACTOR; min_distance = CYCLE_PEAK_TROUGH_DISTANCE
            peak_indices_loc, _ = find_peaks(highs, distance=min_distance, prominence=min_prominence)
            trough_indices_loc, _ = find_peaks(-lows, distance=min_distance, prominence=min_prominence)

            if len(peak_indices_loc) < 2 and len(trough_indices_loc) < 2:
                logger.debug("Meno di 2 picchi/minimi trovati per estremi ciclo precedente.")
                return prev_high, prev_low

            peaks = pd.Series(highs.iloc[peak_indices_loc].values, index=data_index[peak_indices_loc], name='Peak')
            troughs = pd.Series(lows.iloc[trough_indices_loc].values, index=data_index[trough_indices_loc], name='Trough')
            extrema = pd.concat([peaks, troughs]).sort_index()

            if len(extrema) >= 3:
                penultimate_extreme_dt = extrema.index[-2]
                penultimate_type = 'peak' if pd.notna(peaks.get(penultimate_extreme_dt)) else 'trough'

                if penultimate_type == 'peak' and len(peak_indices_loc) >= 2:
                    prev_high = _safe_float(peaks.loc[penultimate_extreme_dt])
                    valid_troughs_before = troughs[troughs.index < penultimate_extreme_dt]
                    if not valid_troughs_before.empty:
                         prev_low = _safe_float(valid_troughs_before.iloc[-1])

                elif penultimate_type == 'trough' and len(trough_indices_loc) >= 2:
                    prev_low = _safe_float(troughs.loc[penultimate_extreme_dt])
                    valid_peaks_before = peaks[peaks.index < penultimate_extreme_dt]
                    if not valid_peaks_before.empty:
                         prev_high = _safe_float(valid_peaks_before.iloc[-1])

            self._previous_cycle_high_price = prev_high
            self._previous_cycle_low_price = prev_low

        except ImportError: logger.critical("Scipy mancante per find_peaks.")
        except Exception as e: logger.error(f"Errore ricerca estremi ciclo precedente: {e}", exc_info=True)
        return prev_high, prev_low

    def run_analysis(self) -> Dict[str, Any]:
        """
        Runs ALL analyses: Base, Technical, Rolling, Advanced tests, Bias,
        Risk/Vol/Ret details, CYCLES, HOURLY VOLUME, INTRACANDLE metrics,
        TIMING ESTIMATES.
        Structures the output for the final report. Historical stats are loaded if available.
        NOTE: Potential Target calculation is now handled externally in TradingAdvisor.
        Handles non-critical errors (like temporal bias on long TFs) gracefully.
        """
        logger.debug("StatisticalAnalyzerAdvanced.run_analysis() - START")

        # Usa self.results dalla classe base (che chiama TA) come punto di partenza
        try:
            base_results = super().run_analysis()
            if not base_results or isinstance(base_results.get('error'), str):
                err_msg = base_results.get('error', 'Unknown error') if base_results else 'Base analysis failed'
                logger.error(f"Base analysis failed: {err_msg}. Stopping advanced analysis.")
                self.results = {
                    'statistical_analysis': {'error': err_msg},
                    'technical_analysis': {'error': err_msg},
                    'advanced_statistical_analysis': {'error': err_msg},
                    'cycle_analysis': {'error': err_msg},
                    'intracandle_analysis': {'error': err_msg},
                    'timing_estimates': {'error': err_msg},
                    'error': err_msg
                 }
                return self.results
            self.results = base_results # Inizializza con i risultati base e TA
        except Exception as base_err:
            logger.error(f"Critical error during super().run_analysis(): {base_err}", exc_info=True)
            return {'error': f'Critical failure in base analysis: {base_err}'}

        # Inizializza/Assicura sezioni avanzate
        self.results.setdefault('advanced_statistical_analysis', {})
        self.results.setdefault('cycle_analysis', {})
        self.results.setdefault('intracandle_analysis', {})
        self.results.setdefault('timing_estimates', {})
        self.results.setdefault('hourly_volume_alert', {'error': 'Not run yet'})

        # --- Calcolo Rolling Stats ---
        try:
            windows_to_calculate = [5, 10, 20, 30, 50, 60]
            self._calculate_all_rolling_stats(windows=windows_to_calculate)
            rolling_stats_results = {}
            windows_to_report = windows_to_calculate
            for window in windows_to_report:
                window_key = f'window_{window}'; window_results_dict = {}; has_window_data = False
                prefixes = [
                    'rolling_mean_', 'rolling_median_', 'rolling_std_', 'rolling_max_', 'rolling_min_', 'rolling_quantile_90_',
                    'rolling_volume_mean_', 'rolling_volume_std_', 'rolling_price_skew_', 'rolling_price_kurt_',
                    'rolling_return_mean_', 'rolling_return_std_', 'rolling_return_skew_', 'rolling_return_kurt_',
                    'rolling_sharpe_', 'rolling_downside_std_', 'rolling_sortino_', 'rank_in_window_',
                    'percent_rank_in_window_', 'periods_since_high_', 'periods_since_low_',
                    'up_periods_in_window_', 'down_periods_in_window_' ]
                for prefix in prefixes:
                    col_name = f"{prefix}{window}"; metric_name = prefix.replace('rolling_', '').replace(f'_{window}', '')
                    if col_name in self.data.columns:
                        value = safe_get_last_value(self.data.get(col_name), default=None)
                        float_val = _safe_float(value)
                        if float_val is not None:
                            window_results_dict[metric_name] = float_val
                            has_window_data = True
                    else:
                         logger.debug(f"Colonna rolling mancante per report: {col_name}")

                if has_window_data: rolling_stats_results[window_key] = window_results_dict
            self.results.setdefault('advanced_statistical_analysis', {})['rolling_stats'] = rolling_stats_results
        except Exception as roll_err:
            logger.error(f"Error calculating/extracting rolling stats: {roll_err}", exc_info=True)
            self.results.setdefault('advanced_statistical_analysis', {})['rolling_stats'] = {'error': str(roll_err)}

        # --- Calcolo Consecutive Stats ---
        try:
             consecutive_stats = {}
             up_col, down_col = 'consecutive_up_periods', 'consecutive_down_periods'
             last_up = safe_get_last_value(self.data.get(up_col), default=0) if up_col in self.data.columns else 0
             last_down = safe_get_last_value(self.data.get(down_col), default=0) if down_col in self.data.columns else 0
             consecutive_stats['consecutive_up'] = int(last_up) if pd.notna(last_up) else 0
             consecutive_stats['consecutive_down'] = int(last_down) if pd.notna(last_down) else 0
             self.results.setdefault('advanced_statistical_analysis', {})['consecutive_stats'] = consecutive_stats
        except Exception as cons_err: logger.error(f"Error extracting consecutive stats: {cons_err}", exc_info=True); self.results.setdefault('advanced_statistical_analysis', {})['consecutive_stats'] = {'error': str(cons_err)}

        # --- Test Statistici Avanzati ---
        adv_test_results = self._run_advanced_statistical_tests()
        self.results.setdefault('advanced_statistical_analysis', {}).update(adv_test_results)

        # --- Bias Temporali ---
        try:
              logger.debug("Calculating temporal biases...")
              hourly_bias = self._calculate_hourly_bias(); day_hour_bias = self._calculate_day_hour_bias(); daily_bias = self._calculate_daily_bias()
              bias_errors = []
              if hourly_bias.get('error'): bias_errors.append(f"Hourly: {hourly_bias['error']}")
              if day_hour_bias.get('error'): bias_errors.append(f"Day/Hour: {day_hour_bias['error']}")
              if daily_bias.get('error'): bias_errors.append(f"Daily: {daily_bias['error']}")
              bias_error_str = "; ".join(bias_errors) if bias_errors else None
              self.results.setdefault('advanced_statistical_analysis', {})['temporal_bias'] = {
                  'hourly_bias_analysis': hourly_bias, 'day_hour_bias_analysis': day_hour_bias, 'daily_bias_analysis': daily_bias,
                  'error': bias_error_str
              }
        except Exception as bias_err:
             logger.error(f"Error calculating temporal biases: {bias_err}", exc_info=True)
             self.results.setdefault('advanced_statistical_analysis', {})['temporal_bias'] = {'error': str(bias_err)}

        # --- Metriche Dettagliate Volatilità, Rendimenti, Rischio ---
        try:
             logger.debug("Calculating detailed volatility, returns, risk...")
             self.results.setdefault('advanced_statistical_analysis', {})['volatility'] = self.calculate_volatility()
             self.results.setdefault('advanced_statistical_analysis', {})['returns'] = self.calculate_returns_stats()
             self.results.setdefault('advanced_statistical_analysis', {})['risk'] = self.calculate_risk_metrics()
        except Exception as final_metrics_err: logger.error(f"Error calculating final metrics (vol, ret, risk): {final_metrics_err}", exc_info=True); self.results.setdefault('advanced_statistical_analysis', {})['final_metrics_error'] = str(final_metrics_err)

        # --- Analisi Volume Orario ---
        hourly_vol_results = None
        try:
            logger.debug("Calculating hourly volume analysis...")
            hourly_vol_results = self._calculate_hourly_volume_stats()
            self.results.setdefault('advanced_statistical_analysis', {})['hourly_volume_analysis'] = hourly_vol_results
        except Exception as hourly_vol_err:
            logger.error(f"Error calculating hourly volume analysis: {hourly_vol_err}", exc_info=True)
            hourly_vol_results = {'error': f"Failed: {hourly_vol_err}"}
            self.results.setdefault('advanced_statistical_analysis', {})['hourly_volume_analysis'] = hourly_vol_results

        # Popola hourly_volume_alert
        try:
             self.results['hourly_volume_alert'] = {
                 'is_significantly_above_average': hourly_vol_results.get('is_significantly_above_average') if hourly_vol_results else None,
                 'last_volume': hourly_vol_results.get('last_candle_volume') if hourly_vol_results else None,
                 'average_volume_for_hour': hourly_vol_results.get('last_candle_avg_volume_for_hour') if hourly_vol_results else None,
                 'ratio_vs_average': hourly_vol_results.get('last_vs_avg_ratio') if hourly_vol_results else None,
                 'threshold_factor': hourly_vol_results.get('volume_alert_threshold_factor') if hourly_vol_results else None,
                 'error': hourly_vol_results.get('error') if hourly_vol_results else "Calculation failed"
             }
        except Exception as alert_pop_err:
             logger.error(f"Errore nel popolare hourly_volume_alert: {alert_pop_err}")
             self.results['hourly_volume_alert'] = {'error': f'Population failed: {alert_pop_err}'}

        # --- Analisi Cicli ---
        try:
            logger.debug("Calculating cycle analysis (loading historical stats if available)...")
            self.results['cycle_analysis'] = self._calculate_cycle_analysis()
        except Exception as cycle_err:
            logger.error(f"Error calculating cycle analysis: {cycle_err}", exc_info=True)
            self.results['cycle_analysis'] = {'error': f"Cycle analysis failed: {cycle_err}"}

        # --- Analisi Intra-Candela ---
        try:
             logger.debug("Calculating intracandle analysis (loading historical stats if available)...")
             self.results['intracandle_analysis'] = self._analyze_current_candle_intracandle_stats()
             if isinstance(self.results['intracandle_analysis'], dict) and self.results['intracandle_analysis'].get('error'):
                  logger.error(f"Intracandle analysis failed: {self.results['intracandle_analysis']['error']}")
        except Exception as ic_err:
            logger.error(f"Error calculating intracandle analysis: {ic_err}", exc_info=True)
            self.results['intracandle_analysis'] = {'error': f"Intracandle analysis failed: {ic_err}"}

        # --- Calcolo Stime Temporali ---
        try:
            logger.debug("Calculating timing estimates...")
            timing_results = {
                'recent_trend_velocity': self._calculate_recent_trend_velocity(),
                'time_to_level_atr_based': self._calculate_time_to_level_estimates()
            }
            cycle_duration_analysis = _safe_get(self.results, ['cycle_analysis', 'derived_duration_indicators'], {})
            hist_duration_data = _safe_get(self.results, ['cycle_analysis', 'historical_durations'], {})
            timing_results['cycle_duration_analysis'] = {
                'current_duration_vs_p25_up_ratio': cycle_duration_analysis.get('current_duration_vs_p25_up_ratio'),
                'current_duration_vs_p75_up_ratio': cycle_duration_analysis.get('current_duration_vs_p75_up_ratio'),
                'hist_p25_duration_up': hist_duration_data.get('hist_p25_duration_up'),
                'hist_p75_duration_up': hist_duration_data.get('hist_p75_duration_up'),
                'current_duration_vs_p25_down_ratio': cycle_duration_analysis.get('current_duration_vs_p25_down_ratio'),
                'current_duration_vs_p75_down_ratio': cycle_duration_analysis.get('current_duration_vs_p75_down_ratio'),
                'hist_p25_duration_down': hist_duration_data.get('hist_p25_duration_down'),
                'hist_p75_duration_down': hist_duration_data.get('hist_p75_duration_down')
            }
            self.results['timing_estimates'] = timing_results
        except Exception as timing_err:
            logger.error(f"Error calculating timing estimates: {timing_err}", exc_info=True)
            self.results['timing_estimates'] = {'error': f"Failed: {timing_err}"}

        # --- Controllo Errori Finale Rivisto ---
        critical_errors_found = False
        final_errors = []
        sections_to_check = [
            'statistical_analysis', 'technical_analysis', 'advanced_statistical_analysis',
            'cycle_analysis', 'intracandle_analysis', 'timing_estimates',
            'patterns', 'fibonacci', 'multi_timeframe_analysis', 'dwm_ohlc_data',
            'hourly_volume_alert', 'deribit_data', 'level_proximity', 'entry_exit_refinement'
        ]
        non_critical_keys_long_tf = [
            ('advanced_statistical_analysis', 'temporal_bias'),
            ('advanced_statistical_analysis', 'hourly_volume_analysis'),
            ('hourly_volume_alert', None)
        ]
        for section in sections_to_check:
            data = self.results.get(section)
            if isinstance(data, dict) and data.get('error') and "Not run yet" not in str(data.get('error')):
                is_non_critical = self._is_long_timeframe and (section, None) in non_critical_keys_long_tf
                final_errors.append(f"Section '{section}': {data['error']}")
                if not is_non_critical: critical_errors_found = True
            elif section == 'deribit_data' and isinstance(data, dict) and data.get('fetch_error'):
                final_errors.append(f"Section 'deribit_data': {data['fetch_error']}")
                critical_errors_found = True
            elif isinstance(data, dict):
                 for sub_key, sub_data in data.items():
                      if sub_key == 'error': continue
                      if isinstance(sub_data, dict) and sub_data.get('error') and "Not run yet" not in str(sub_data.get('error')):
                            is_non_critical = self._is_long_timeframe and (section, sub_key) in non_critical_keys_long_tf
                            final_errors.append(f"Subsection '{section}.{sub_key}': {sub_data['error']}")
                            if not is_non_critical: critical_errors_found = True

        if final_errors:
            unique_errors = sorted(list(set(final_errors)))
            error_summary = f"Analysis completed with {len(unique_errors)} issue(s)."
            logger.warning(f"{error_summary} Details: {'; '.join(unique_errors)}")
            if critical_errors_found:
               self.results['error'] = f"{error_summary} (Critical errors present)"
            else:
                self.results.pop('error', None)
                self.results['analysis_warnings'] = unique_errors
        else:
            self.results.pop('error', None)
            self.results.pop('analysis_warnings', None)

        logger.debug("StatisticalAnalyzerAdvanced.run_analysis() - END")
        return self.results

    def update_historical_stats(self) -> Tuple[bool, bool]:
        if self.data is None or self.data.empty:
            logger.error("Cannot update historical stats: Data is missing.")
            return False, False
        if not self.symbol or not self.timeframe:
            logger.error("Cannot update historical stats: Symbol or Timeframe not set.")
            return False, False

        success_cycles = False
        logger.info(f"*** STARTING HISTORICAL CYCLE STATS UPDATE for {self.symbol} ({self.timeframe}) ***")
        update_start_time_cycles = datetime.now()
        try:
            historical_cycle_stats = self._calculate_historical_cycle_stats(self.data.copy())
            if not historical_cycle_stats or historical_cycle_stats.get('error'):
                 err_msg = historical_cycle_stats.get('error', 'Unknown') if historical_cycle_stats else 'None';
                 logger.error(f"Failed to calculate historical cycle stats for {self.symbol} ({self.timeframe}): {err_msg}")
            elif self.cycle_stats_filename:
                logger.info(f"Saving historical cycle stats to: {self.cycle_stats_filename}")
                save_success = save_historical_stats_to_json(historical_cycle_stats, self.cycle_stats_filename)
                if save_success:
                    logger.info(f"*** CYCLE STATS UPDATE for {self.symbol} ({self.timeframe}) COMPLETED ***"); success_cycles = True
                else: logger.error(f"Failed to save cycle stats for {self.symbol} ({self.timeframe}).")
            else: logger.error(f"Cycle stats filename not defined for {self.symbol} ({self.timeframe}).")
        except Exception as e: logger.error(f"Unhandled error during cycle stats update for {self.symbol} ({self.timeframe}): {e}", exc_info=True)
        update_end_time_cycles = datetime.now()
        logger.info(f"Cycle stats update duration: {(update_end_time_cycles - update_start_time_cycles).total_seconds():.2f} sec.")

        success_intracandle = False
        logger.info(f"*** STARTING HISTORICAL INTRACANDLE STATS UPDATE for {self.symbol} ({self.timeframe}) ***")
        update_start_time_ic = datetime.now()
        try:
            self._calculate_intracandle_metrics_for_dataframe()
            historical_intracandle_stats = self._calculate_historical_intracandle_stats(self.data.copy())
            if not historical_intracandle_stats or historical_intracandle_stats.get('error'):
                err_msg = historical_intracandle_stats.get('error', 'Unknown') if historical_intracandle_stats else 'None';
                logger.error(f"Failed to calculate historical intracandle stats for {self.symbol} ({self.timeframe}): {err_msg}")
            elif self.intracandle_stats_filename:
                logger.info(f"Saving historical intracandle stats to: {self.intracandle_stats_filename}")
                save_success_ic = save_historical_stats_to_json(historical_intracandle_stats, self.intracandle_stats_filename)
                if save_success_ic:
                    logger.info(f"*** INTRACANDLE STATS UPDATE for {self.symbol} ({self.timeframe}) COMPLETED ***"); success_intracandle = True
                else: logger.error(f"Failed to save intracandle stats for {self.symbol} ({self.timeframe}).")
            else: logger.error(f"Intracandle stats filename not defined for {self.symbol} ({self.timeframe}).")
        except Exception as e: logger.error(f"Unhandled error during intracandle stats update for {self.symbol} ({self.timeframe}): {e}", exc_info=True)
        update_end_time_ic = datetime.now()
        logger.info(f"Intracandle stats update duration: {(update_end_time_ic - update_start_time_ic).total_seconds():.2f} sec.")

        return success_cycles, success_intracandle

# --- END OF FILE statistical_analyzer_advanced.py ---
