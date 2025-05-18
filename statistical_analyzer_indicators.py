import pandas as pd
import numpy as np
import logging
from statistical_analyzer import StatisticalAnalyzer
from typing import Dict, List, Any, Tuple, Optional, Union
from statistical_analyzer_helpers import safe_get_last_value  # Add this import at the top

# Setup logging
logger = logging.getLogger(__name__)

class IndicatorAnalyzer(StatisticalAnalyzer):
    """
    Classe per il calcolo degli indicatori tecnici.
    Estende StatisticalAnalyzer.
    """

    def __init__(self, data):
        """
        Inizializza l'analizzatore di indicatori.

        Args:
            data: DataFrame con dati OHLCV
        """
        super().__init__(data)

    def safe_get_last_value(self, series, index=-1, default=0.0):
        """
        Safely extract a value from a pandas Series or numpy array.

        Args:
            series: The pandas Series or numpy array
            index: The index to extract (default: -1 for last element)
            default: Default value to return if extraction fails (default: 0.0)

        Returns:
            float: The extracted value or default value
        """
        try:
            if series is None or len(series) == 0:
                return default

            value = series.iloc[index] if hasattr(series, 'iloc') else series[index]

            # Check for NaN, inf, or None
            if pd.isna(value) or np.isinf(value) or value is None:
                return default

            return float(value)  # Ensure return type is float
        except Exception as e:
            logger.warning(f"Error extracting value from series: {str(e)}")
            return default

    def calculate_rsi(self, period=14):
        """
        Calcola il Relative Strength Index (RSI).

        Args:
            period: Periodo per il calcolo dell'RSI

        Returns:
            float: Valore dell'RSI
        """
        if len(self.data) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need at least {period + 1} points.")
            return 50.0  # Default value as float

        try:
            # Get price series
            prices = self.data['close']

            # Calculate price changes
            delta = prices.diff()

            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Calculate RS
            rs = self.safe_get_last_value(avg_gain) / max(self.safe_get_last_value(avg_loss), 0.001)  # Avoid division by zero

            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))

            # Ensure result is a float between 0 and 100
            return min(max(float(rsi), 0.0), 100.0)
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return 50.0  # Default value on error

    def calculate_macd(self, fast_period=12, slow_period=26, signal_period=9):
        """
        Calcola il Moving Average Convergence Divergence (MACD).

        Args:
            fast_period: Periodo per la media mobile veloce
            slow_period: Periodo per la media mobile lenta
            signal_period: Periodo per la linea di segnale

        Returns:
            Dict: Valori del MACD
        """
        # Default return dictionary
        default_result = {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}

        if len(self.data) < slow_period + signal_period:
            logger.warning(f"Insufficient data for MACD calculation. Need at least {slow_period + signal_period} points.")
            return default_result

        try:
            # Get price series
            prices = self.data['close']

            # Calculate EMAs
            fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

            # Calculate MACD line
            macd_line = fast_ema - slow_ema

            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # Calculate histogram
            histogram = macd_line - signal_line

            # Get the last values
            macd_value = self.safe_get_last_value(macd_line)
            signal_value = self.safe_get_last_value(signal_line)
            histogram_value = self.safe_get_last_value(histogram)

            return {
                'macd': float(macd_value),
                'signal': float(signal_value),
                'histogram': float(histogram_value)
            }
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return default_result

    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """
        Calcola le Bollinger Bands.

        Args:
            period: Periodo per la media mobile
            std_dev: Numero di deviazioni standard

        Returns:
            Dict: Valori delle Bollinger Bands
        """
        # Default return dictionary
        default_result = {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}

        if len(self.data) < period:
            logger.warning(f"Insufficient data for Bollinger Bands calculation. Need at least {period} points.")
            return default_result

        try:
            # Get price series
            prices = self.data['close']

            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()

            # Calculate standard deviation
            std = prices.rolling(window=period).std()

            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)

            # Get the last values
            upper_value = self.safe_get_last_value(upper_band)
            middle_value = self.safe_get_last_value(middle_band)
            lower_value = self.safe_get_last_value(lower_band)

            return {
                'upper': float(upper_value),
                'middle': float(middle_value),
                'lower': float(lower_value)
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return default_result

    def calculate_stochastic(self, k_period=14, d_period=3):
        """
        Calcola l'oscillatore stocastico.

        Args:
            k_period: Periodo per %K
            d_period: Periodo per %D

        Returns:
            Dict: Valori dell'oscillatore stocastico
        """
        if len(self.data) < k_period:
            return {'k': 50.0, 'd': 50.0}

        # Calcola il minimo e il massimo nel periodo
        low_min = self.data['low'].rolling(window=k_period).min()
        high_max = self.data['high'].rolling(window=k_period).max()

        # Calcola %K
        k = 100 * ((self.data['close'] - low_min) / (high_max - low_min))

        # Calcola %D (media mobile di %K)
        d = k.rolling(window=d_period).mean()

        # Use safe_get_last_value and handle None values
        k_value = self.safe_get_last_value(k)
        d_value = self.safe_get_last_value(d)

        # Create result dictionary with proper null handling
        result = {}
        result['k'] = 50.0 if k_value is None or np.isnan(k_value) else float(k_value)
        result['d'] = 50.0 if d_value is None or np.isnan(d_value) else float(d_value)

        return result

    # For the calculate_moving_averages method
    def calculate_moving_averages(self, periods=[20, 50, 200]):
        """
        Calcola le medie mobili per diversi periodi.

        Args:
            periods: Lista dei periodi per le medie mobili

        Returns:
            Dict: Valori delle medie mobili
        """
        result = {}

        for period in periods:
            if len(self.data) < period:
                result[f'ma_{period}'] = 0.0  # Changed None to 0
                result[f'ema_{period}'] = 0.0  # Changed None to 0
                continue

            ma = self.data['close'].rolling(window=period).mean()
            ma_value = self.safe_get_last_value(ma)
            result[f'ma_{period}'] = 0.0 if ma_value is None or np.isnan(ma_value) else float(ma_value)

            # Calcola anche la media mobile esponenziale
            ema = self.data['close'].ewm(span=period, adjust=False).mean()
            ema_value = self.safe_get_last_value(ema)
            result[f'ema_{period}'] = 0.0 if ema_value is None or np.isnan(ema_value) else float(ema_value)

        return result

    # For the calculate_ichimoku method
    def calculate_ichimoku(self):
        """
        Calcola l'indicatore Ichimoku Cloud.

        Returns:
            Dict: Componenti dell'Ichimoku Cloud
        """
        if len(self.data) < 52:  # Minimo per calcolare il Senkou Span B
            return {
                'tenkan_sen': 0.0,
                'kijun_sen': 0.0,
                'senkou_span_a': 0.0,
                'senkou_span_b': 0.0,
                'chikou_span': 0.0
            }

        # Calcola Tenkan-sen (Conversion Line)
        period9_high = self.data['high'].rolling(window=9).max()
        period9_low = self.data['low'].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2

        # Calcola Kijun-sen (Base Line)
        period26_high = self.data['high'].rolling(window=26).max()
        period26_low = self.data['low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2

        # Calcola Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Calcola Senkou Span B (Leading Span B)
        period52_high = self.data['high'].rolling(window=52).max()
        period52_low = self.data['low'].rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

        # Calcola Chikou Span (Lagging Span)
        chikou_span = self.data['close'].shift(-26)

        # Fix: Use safe_get_last_value and handle None values properly
        tenkan_sen_value = self.safe_get_last_value(tenkan_sen)
        kijun_sen_value = self.safe_get_last_value(kijun_sen)
        senkou_span_a_value = self.safe_get_last_value(senkou_span_a)
        senkou_span_b_value = self.safe_get_last_value(senkou_span_b)
        chikou_span_value = self.safe_get_last_value(chikou_span)

        # Create result dictionary with proper null handling
        result = {}
        result['tenkan_sen'] = 0.0 if tenkan_sen_value is None or np.isnan(tenkan_sen_value) else float(tenkan_sen_value)
        result['kijun_sen'] = 0.0 if kijun_sen_value is None or np.isnan(kijun_sen_value) else float(kijun_sen_value)
        result['senkou_span_a'] = 0.0 if senkou_span_a_value is None or np.isnan(senkou_span_a_value) else float(senkou_span_a_value)
        result['senkou_span_b'] = 0.0 if senkou_span_b_value is None or np.isnan(senkou_span_b_value) else float(senkou_span_b_value)
        result['chikou_span'] = 0.0 if chikou_span_value is None or np.isnan(chikou_span_value) else float(chikou_span_value)

        return result

    # For the calculate_atr method
    def calculate_atr(self, period=14):
        """
        Calcola l'Average True Range (ATR).

        Args:
            period: Periodo per il calcolo dell'ATR

        Returns:
            float: Valore dell'ATR
        """
        if len(self.data) < period + 1:
            return 0.0

        # Calcola il True Range
        high_low = self.data['high'] - self.data['low']
        high_close = abs(self.data['high'] - self.data['close'].shift(1))
        low_close = abs(self.data['low'] - self.data['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calcola l'ATR
        atr = true_range.rolling(window=period).mean()

        # Fix: Use safe_get_last_value and handle None values properly
        atr_value = self.safe_get_last_value(atr)
        return 0.0 if atr_value is None or np.isnan(atr_value) else float(atr_value)

    def calculate_adx(self, period=14):
        """
        Calcola l'Average Directional Index (ADX).

        Args:
            period: Periodo per il calcolo dell'ADX

        Returns:
            float: Valore dell'ADX
        """
        result = self._calculate_adx_detailed(period)

        # Return just the ADX value as a float to match the parent class return type
        if isinstance(result, dict) and 'adx' in result:
            return float(result['adx'])
        return 0.0

    # For the _calculate_adx_detailed method
    def _calculate_adx_detailed(self, period=14) -> Dict[str, float]:
        """
        Implementazione dettagliata del calcolo dell'ADX che restituisce un dizionario.

        Args:
            period: Periodo per il calcolo dell'ADX

        Returns:
            Dict: Valori dell'ADX e delle linee direzionali
        """
        if len(self.data) < 2 * period:
            return {'adx': 0.0, 'di_plus': 0.0, 'di_minus': 0.0}

        df = self.data.copy()

        # Calcola il True Range
        df['high-low'] = df['high'] - df['low']
        df['high-prev_close'] = abs(df['high'] - df['close'].shift(1))
        df['low-prev_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)

        # Calcola +DM e -DM
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']

        df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        # Calcola +DI e -DI
        df['+di'] = 100 * (df['+dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())
        df['-di'] = 100 * (df['-dm'].rolling(window=period).mean() / df['tr'].rolling(window=period).mean())

        # Calcola DX
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])

        # Calcola ADX
        df['adx'] = df['dx'].rolling(window=period).mean()

        # Fix: Simplify the conditional checks and handle None values properly
        adx_value = self.safe_get_last_value(df['adx'])
        di_plus_value = self.safe_get_last_value(df['+di'])
        di_minus_value = self.safe_get_last_value(df['-di'])

        # Create result dictionary with proper null handling
        result = {}
        result['adx'] = 0.0 if adx_value is None or np.isnan(adx_value) else float(adx_value)
        result['di_plus'] = 0.0 if di_plus_value is None or np.isnan(di_plus_value) else float(di_plus_value)
        result['di_minus'] = 0.0 if di_minus_value is None or np.isnan(di_minus_value) else float(di_minus_value)

        return result

    def calculate_all_indicators(self):
        """
        Calcola tutti gli indicatori tecnici disponibili.

        Returns:
            Dict: Risultati del calcolo degli indicatori
        """
        indicators = {}

        # Calcola RSI
        indicators['rsi'] = self.calculate_rsi()

        # Calcola MACD
        indicators['macd'] = self.calculate_macd()

        # Calcola Bollinger Bands
        indicators['bollinger'] = self.calculate_bollinger_bands()

        # Calcola Stochastic Oscillator
        indicators['stochastic'] = self.calculate_stochastic()

        # Calcola Moving Averages
        indicators['moving_averages'] = self.calculate_moving_averages()

        # Calcola Ichimoku Cloud
        indicators['ichimoku'] = self.calculate_ichimoku()

        # Calcola ATR
        indicators['atr'] = self.calculate_atr()

        # Calcola ADX
        indicators['adx'] = self.calculate_adx()

        return indicators