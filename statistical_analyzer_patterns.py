# --- START OF FILE statistical_analyzer_patterns.py ---
# statistical_analyzer_patterns.py
"""
Modulo per l'analisi dei pattern di candele giapponesi,
includendo valutazione di qualità, volume e contesto trend.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List, Callable, Any
import sys # Per controllo numba (se usato in futuro)
import time # Aggiunto per logging tempo esecuzione

# Import helper
try:
    from statistical_analyzer_helpers import safe_get_last_value, _safe_float
except ImportError:
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.warning("statistical_analyzer_helpers non trovato in patterns.")
    def safe_get_last_value(series, default=None): return default # type: ignore
    def _safe_float(value, default=None):
        try: return float(value) if pd.notna(value) and not np.isinf(value) else default
        except: return default

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    Classe per l'analisi dei pattern di candele giapponesi.
    Include valutazione di qualità, volume e contesto.
    Calcola internamente SMA, Volume SMA e ATR necessari.
    """
    # Soglie e Fattori Configurabili
    PRICE_EQUALITY_TOLERANCE_FACTOR = 0.0005 # Tolleranza per uguaglianza prezzi (0.05%)
    VOLUME_CONFIRMATION_FACTOR = 1.5 # Volume > 1.5x media per conferma
    QUALITY_ATR_BODY_STRONG = 0.6 # Corpo > 60% ATR per pattern forte
    QUALITY_ATR_BODY_MEDIUM = 0.3 # Corpo > 30% ATR per pattern medio
    QUALITY_ATR_SHADOW_LONG = 1.5 # Ombra lunga > 1.5x ATR (es. Hammer)
    QUALITY_ATR_SHADOW_SHORT = 0.5 # Ombra corta < 0.5x ATR (es. Doji)

    # Periodi per indicatori interni
    SMA_CONTEXT_PERIOD = 20
    VOLUME_SMA_PERIOD = 20
    ATR_QUALITY_PERIOD = 14

    def __init__(self, data: Optional[pd.DataFrame]):
        """
        Inizializza l'analizzatore di pattern.

        Args:
            data (Optional[pd.DataFrame]): DataFrame con dati OHLCV e DatetimeIndex UTC.
                                           Deve contenere colonne 'open', 'high', 'low', 'close', 'volume'.
        """
        if data is None or data.empty:
            logger.warning("PatternAnalyzer: Dati non forniti o vuoti.")
            self.data = pd.DataFrame(); self.last_atr = None; return
        if not isinstance(data.index, pd.DatetimeIndex):
            try: data.index = pd.to_datetime(data.index, utc=True)
            except Exception as e: logger.error(f"PatternAnalyzer: Indice non DatetimeIndex: {e}"); self.data = pd.DataFrame(); return
        if data.index.tz is None or str(data.index.tz).upper() != 'UTC':
            logger.warning(f"PatternAnalyzer: Indice non UTC ({data.index.tz}). Forzo UTC.")
            try: data.index = data.index.tz_localize('UTC', ambiguous='infer') if data.index.tz is None else data.index.tz_convert('UTC')
            except Exception as tz_err: logger.error(f"PatternAnalyzer: Errore forzatura UTC: {tz_err}"); self.data = pd.DataFrame(); return
        self.data = data.copy()
        self.data.columns = [str(col).lower().replace(' ','_') for col in self.data.columns]
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in self.data.columns];
        if missing_cols: logger.error(f"PatternAnalyzer: Dati OHLCV incompleti (mancano: {missing_cols})."); self.data = pd.DataFrame(); self.last_atr = None; return
        try:
            for col in required_columns: self.data[col] = pd.to_numeric(self.data[col], errors='coerce').astype(float)
            self.data.dropna(subset=required_columns, inplace=True)
        except Exception as e: logger.error(f"PatternAnalyzer: Errore conversione numerica OHLCV: {e}"); self.data = pd.DataFrame(); self.last_atr = None; return
        if self.data.empty: logger.warning("PatternAnalyzer: Dati vuoti dopo pulizia OHLCV."); self.last_atr = None; return
        self._calculate_internal_indicators()


    def _calculate_internal_indicators(self):
        """Calcola indicatori interni necessari: SMA, Volume SMA, ATR."""
        if self.data.empty: self.last_atr = None; return
        # SMA
        sma_col = f'sma_{self.SMA_CONTEXT_PERIOD}'
        if 'close' in self.data.columns and len(self.data) >= self.SMA_CONTEXT_PERIOD: self.data[sma_col] = self.data['close'].rolling(window=self.SMA_CONTEXT_PERIOD).mean()
        else: self.data[sma_col] = np.nan; logger.debug(f"Dati insuff. per {sma_col}.")
        # Volume SMA
        vol_sma_col = f'volume_sma_{self.VOLUME_SMA_PERIOD}'
        if 'volume' in self.data.columns and len(self.data) >= self.VOLUME_SMA_PERIOD: self.data[vol_sma_col] = self.data['volume'].rolling(window=self.VOLUME_SMA_PERIOD).mean()
        else: self.data[vol_sma_col] = np.nan; logger.debug(f"Dati insuff. per {vol_sma_col}.")
        # ATR
        atr_col = f'atr_{self.ATR_QUALITY_PERIOD}'; required_atr_cols = ['high', 'low', 'close']
        if all(c in self.data.columns for c in required_atr_cols) and len(self.data) >= self.ATR_QUALITY_PERIOD + 1:
            try:
                h_l = self.data['high'] - self.data['low']; h_pc = abs(self.data['high'] - self.data['close'].shift(1)); l_pc = abs(self.data['low'] - self.data['close'].shift(1))
                tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1, skipna=False)
                # --- CORREZIONE fillna ---
                tr = tr.bfill() # Usa .bfill() invece di fillna(method='bfill')
                # --- FINE CORREZIONE ---
                tr = tr.fillna(0)
                self.data[atr_col] = tr.ewm(span=self.ATR_QUALITY_PERIOD, adjust=False).mean()
                self.last_atr = safe_get_last_value(self.data[atr_col], default=None)
            except Exception as e: logger.warning(f"Errore calcolo ATR interno: {e}"); self.data[atr_col] = np.nan; self.last_atr = None
        else: self.data[atr_col] = np.nan; self.last_atr = None; logger.debug(f"Dati insuff. per {atr_col}.")
        logger.debug(f"Ultimo ATR calcolato internamente: {self.last_atr}")

    # --- Funzioni Helper ---
    def _get_last_n_candles(self, n: int) -> Optional[pd.DataFrame]:
        """Ottiene le ultime N candele se disponibili."""
        if len(self.data) < n:
            logger.debug(f"Richieste {n} candele, disponibili {len(self.data)}.")
            return None
        return self.data.iloc[-n:].copy()

    def _are_prices_equal(self, p1: Optional[float], p2: Optional[float]) -> bool:
        """Verifica se due prezzi sono 'uguali' usando una tolleranza relativa."""
        if p1 is None or p2 is None or pd.isna(p1) or pd.isna(p2): return False
        denominator = max(abs(p1), abs(p2), 1e-9)
        tolerance_abs = denominator * self.PRICE_EQUALITY_TOLERANCE_FACTOR
        return abs(p1 - p2) <= tolerance_abs

    def _assess_quality_volume_context(
        self,
        candles_df: pd.DataFrame,
        key_candle_index: int,
        expected_context: Optional[str] = None,
        check_volume: bool = True,
        quality_logic: Optional[Callable[[pd.Series, float], str]] = None
    ) -> Tuple[str, bool, bool]:
        """Valuta qualità, volume e contesto per un pattern."""
        quality = 'low'; volume_ok = False; context_ok = (expected_context is None)
        if candles_df is None or candles_df.empty or abs(key_candle_index) > len(candles_df): logger.warning("Input invalido per _assess_quality_volume_context."); return quality, volume_ok, context_ok
        try:
            key_candle = candles_df.iloc[key_candle_index]; atr_val = self.last_atr
            if atr_val is None or pd.isna(atr_val) or atr_val <= 1e-9: atr_val = abs(key_candle['high'] - key_candle['low']); atr_val = max(atr_val, abs(key_candle['close'] * 0.001)); logger.debug(f"Usato ATR fallback: {atr_val}")
            if quality_logic and atr_val > 1e-9: quality = quality_logic(key_candle, atr_val)
            if check_volume:
                 vol = key_candle.get('volume'); vol_sma_col = f'volume_sma_{self.VOLUME_SMA_PERIOD}'; vol_sma = key_candle.get(vol_sma_col)
                 if vol is not None and vol_sma is not None and vol_sma > 0: volume_ok = (vol >= vol_sma * self.VOLUME_CONFIRMATION_FACTOR)
                 elif vol is not None and vol > 0: volume_ok = True; logger.debug("Volume SMA non disponibile, considerato OK se volume > 0.")
            if expected_context:
                 sma_col = f'sma_{self.SMA_CONTEXT_PERIOD}'; first_pattern_candle_orig_idx = candles_df.index[0]
                 try:
                     first_pattern_candle_loc = self.data.index.get_loc(first_pattern_candle_orig_idx); context_candle_prev_loc = first_pattern_candle_loc - 1
                     if context_candle_prev_loc >= 0:
                         prev_candle = self.data.iloc[context_candle_prev_loc]; sma_prev = prev_candle.get(sma_col); close_prev = prev_candle.get('close')
                         if sma_prev is not None and pd.notna(sma_prev) and close_prev is not None and pd.notna(close_prev):
                             if expected_context == 'downtrend' and close_prev < sma_prev: context_ok = True
                             elif expected_context == 'uptrend' and close_prev > sma_prev: context_ok = True
                             else: context_ok = False
                         else: context_ok = False
                     else: context_ok = False
                 except KeyError: logger.warning(f"Indice {first_pattern_candle_orig_idx} non trovato nel DataFrame originale per contesto."); context_ok = False
        except IndexError: logger.warning(f"Indice {key_candle_index} fuori range in _assess_quality_volume_context (len: {len(candles_df)}).");
        except Exception as e: logger.error(f"Errore in _assess_quality_volume_context: {e}", exc_info=True)
        return quality, volume_ok, context_ok


    # --- Implementazioni Pattern Specifici ---
    # (Logica interna _is_* invariata rispetto alla versione precedente corretta)
    def _is_bullish_engulfing(self) -> Dict[str, Any]:
        pattern_name = 'bullish_engulfing'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2);
        if last_2 is None: return result
        try:
            c0, o0 = last_2['close'].iloc[0], last_2['open'].iloc[0]; c1, o1 = last_2['close'].iloc[1], last_2['open'].iloc[1]
            first_bearish = c0 < o0; second_bullish = c1 > o1; engulfing = (o1 <= c0) and (c1 >= o0)
            detected = first_bearish and second_bullish and engulfing; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_hammer(self) -> Dict[str, Any]:
        pattern_name = 'hammer'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_candle_df = self._get_last_n_candles(1)
        if last_candle_df is None: return result
        try:
            last_candle = last_candle_df.iloc[0]; o, h, l, c = last_candle['open'], last_candle['high'], last_candle['low'], last_candle['close']
            body = abs(o - c); body = max(body, c * 0.0001); lower_shadow = min(o, c) - l; upper_shadow = h - max(o, c)
            is_hammer_shape = (lower_shadow >= 2 * body) and (upper_shadow <= body * 1.1); result['detected'] = is_hammer_shape
            if is_hammer_shape:
                def quality_logic(candle, atr): body_i = abs(candle['open'] - candle['close']); body_i = max(body_i, atr * 0.01); lower_shadow_i = min(candle['open'], candle['close']) - candle['low']; return 'high' if lower_shadow_i > atr * self.QUALITY_ATR_SHADOW_LONG and body_i < atr * self.QUALITY_ATR_SHADOW_SHORT else 'medium' if lower_shadow_i > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_candle_df, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_morning_star(self) -> Dict[str, Any]:
        pattern_name = 'morning_star'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            o0, c0 = last_3['open'].iloc[0], last_3['close'].iloc[0]; o1, c1 = last_3['open'].iloc[1], last_3['close'].iloc[1]; o2, c2 = last_3['open'].iloc[2], last_3['close'].iloc[2]
            body0 = abs(o0-c0); body1 = abs(o1-c1); body2 = abs(o2-c2); first_bearish_long = (c0 < o0) and (body0 > (o0 * 0.01))
            second_small_body = body1 < (max(o1,c1) * 0.005) or body1 < body0 * 0.3; gap_down_body = max(o1, c1) < min(o0, c0); gap_up_body = min(o2, c2) > max(o1, c1)
            third_bullish_long = (c2 > o2) and (body2 > (o2 * 0.01)); closes_in_first_body = c2 > (o0 + c0) / 2
            detected = (first_bearish_long and second_small_body and gap_down_body and gap_up_body and third_bullish_long and closes_in_first_body)
            result['detected'] = detected
            if detected:
                 def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                 result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_bearish_engulfing(self) -> Dict[str, Any]:
        pattern_name = 'bearish_engulfing'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2);
        if last_2 is None: return result
        try:
            c0, o0 = last_2['close'].iloc[0], last_2['open'].iloc[0]; c1, o1 = last_2['close'].iloc[1], last_2['open'].iloc[1]
            first_bullish = c0 > o0; second_bearish = c1 < o1; engulfing = (o1 >= c0) and (c1 <= o0)
            detected = first_bullish and second_bearish and engulfing; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_hanging_man(self) -> Dict[str, Any]:
        pattern_name = 'hanging_man'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_candle_df = self._get_last_n_candles(1)
        if last_candle_df is None: return result
        try:
            last_candle = last_candle_df.iloc[0]; o, h, l, c = last_candle['open'], last_candle['high'], last_candle['low'], last_candle['close']
            body = abs(o - c); body = max(body, c * 0.0001); lower_shadow = min(o, c) - l; upper_shadow = h - max(o, c)
            is_hanging_shape = (lower_shadow >= 2 * body) and (upper_shadow <= body * 1.1); result['detected'] = is_hanging_shape
            if is_hanging_shape:
                def quality_logic(candle, atr): body_i = abs(candle['open'] - candle['close']); body_i = max(body_i, atr * 0.01); lower_shadow_i = min(candle['open'], candle['close']) - candle['low']; return 'high' if lower_shadow_i > atr * self.QUALITY_ATR_SHADOW_LONG and body_i < atr * self.QUALITY_ATR_SHADOW_SHORT else 'medium' if lower_shadow_i > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_candle_df, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_evening_star(self) -> Dict[str, Any]:
        pattern_name = 'evening_star'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            o0, c0 = last_3['open'].iloc[0], last_3['close'].iloc[0]; o1, c1 = last_3['open'].iloc[1], last_3['close'].iloc[1]; o2, c2 = last_3['open'].iloc[2], last_3['close'].iloc[2]
            body0 = abs(o0-c0); body1 = abs(o1-c1); body2 = abs(o2-c2); first_bullish_long = (c0 > o0) and (body0 > (o0 * 0.01))
            second_small_body = body1 < (max(o1,c1) * 0.005) or body1 < body0 * 0.3; gap_up_body = min(o1, c1) > max(o0, c0); gap_down_body = max(o2, c2) < min(o1, c1)
            third_bearish_long = (c2 < o2) and (body2 > (o2 * 0.01)); closes_in_first_body = c2 < (o0 + c0) / 2
            detected = (first_bullish_long and second_small_body and gap_up_body and gap_down_body and third_bearish_long and closes_in_first_body)
            result['detected'] = detected
            if detected:
                 def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                 result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_shooting_star(self) -> Dict[str, Any]:
        pattern_name = 'shooting_star'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_candle_df = self._get_last_n_candles(1)
        if last_candle_df is None: return result
        try:
            last_candle = last_candle_df.iloc[0]; o, h, l, c = last_candle['open'], last_candle['high'], last_candle['low'], last_candle['close']
            body = abs(o - c); body = max(body, c * 0.0001); lower_shadow = min(o, c) - l; upper_shadow = h - max(o, c)
            is_shooting_shape = (upper_shadow >= 2 * body) and (lower_shadow <= body * 1.1); result['detected'] = is_shooting_shape
            if is_shooting_shape:
                def quality_logic(candle, atr): body_i = abs(candle['open'] - candle['close']); body_i = max(body_i, atr * 0.01); upper_shadow_i = candle['high'] - max(candle['open'], candle['close']); return 'high' if upper_shadow_i > atr * self.QUALITY_ATR_SHADOW_LONG and body_i < atr * self.QUALITY_ATR_SHADOW_SHORT else 'medium' if upper_shadow_i > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_candle_df, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_three_black_crows(self) -> Dict[str, Any]:
        pattern_name = 'three_black_crows'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            bodies, opens, closes, lows = [], [], [], []
            all_bearish = True
            for i in range(3):
                o, c, l = last_3['open'].iloc[i], last_3['close'].iloc[i], last_3['low'].iloc[i]
                if c >= o:
                    all_bearish = False
                    break
                bodies.append(abs(o-c))
                opens.append(o)
                closes.append(c)
                lows.append(l)
            if not all_bearish: return result
            progressive_lower = True; opens_in_body = True; close_near_low = True; min_body_factor = 0.005
            long_bodies = all(b > (c * min_body_factor) for b, c in zip(bodies, closes))
            if not long_bodies: return result
            for i in range(1, 3):
                if not (opens[i] < opens[i-1] and opens[i] > closes[i-1]): opens_in_body = False; break
                if not (closes[i] < closes[i-1]): progressive_lower = False; break
                if bodies[i] > 0 and (closes[i] - lows[i]) / bodies[i] > 0.25: close_near_low = False;
            detected = progressive_lower and opens_in_body; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr):
                    body = abs(candle['close'] - candle['open'])
                    q = 'low'
                    if body > atr * self.QUALITY_ATR_BODY_MEDIUM:
                        q = 'medium'
                    if body > atr * self.QUALITY_ATR_BODY_STRONG:
                        q = 'high'
                    if not close_near_low:
                        q = 'medium' if q == 'high' else 'low'
                    return q
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_doji(self) -> Dict[str, Any]:
        pattern_name = 'doji'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': True}; last_candle_df = self._get_last_n_candles(1)
        if last_candle_df is None: return result
        try:
            last_candle = last_candle_df.iloc[0]; o, c, h, l = last_candle['open'], last_candle['close'], last_candle['high'], last_candle['low']
            price_range = h - l; price = max(o, c, 1e-9); body_size = abs(o - c)
            is_very_small_body = body_size < (price * 0.001) or (price_range > 1e-9 and body_size < price_range * 0.05); result['detected'] = is_very_small_body
            if is_very_small_body:
                def quality_logic(candle, atr): low_shadow = min(candle['open'], candle['close']) - candle['low']; up_shadow = candle['high'] - max(candle['open'], candle['close']); return 'high' if low_shadow > atr * self.QUALITY_ATR_SHADOW_SHORT or up_shadow > atr * self.QUALITY_ATR_SHADOW_SHORT else 'medium'
                result['quality'], _, _ = self._assess_quality_volume_context(last_candle_df, -1, None, False, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_harami(self) -> Dict[str, Any]:
        pattern_name = 'harami'; result = {'detected': False, 'type': 'neutral', 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2)
        if last_2 is None: return result
        try:
            o0, c0 = last_2['open'].iloc[0], last_2['close'].iloc[0]; o1, c1 = last_2['open'].iloc[1], last_2['close'].iloc[1]
            first_body = abs(o0 - c0); second_body = abs(o1 - c1)
            if first_body < (o0 * 0.001): return result
            is_small_second = second_body < (first_body * 0.6); is_inside = (max(o1, c1) <= max(o0, c0)) and (min(o1, c1) >= min(o0, c0))
            detected = is_small_second and is_inside; result['detected'] = detected
            if detected:
                harami_type = 'neutral'; expected_context = None
                if c0 < o0 and c1 > o1: harami_type = 'bullish'; expected_context = 'downtrend'
                elif c0 > o0 and c1 < o1: harami_type = 'bearish'; expected_context = 'uptrend'
                result['type'] = harami_type
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body < atr * 0.1 else 'medium' if body < atr * 0.3 else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, expected_context, False, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_three_inside_up(self) -> Dict[str, Any]:
        pattern_name = 'three_inside_up'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            temp_data_prev = self.data.iloc[:-1] if len(self.data) >= 3 else pd.DataFrame();
            if temp_data_prev.empty: return result
            temp_analyzer = PatternAnalyzer(temp_data_prev); harami_result_prev = temp_analyzer._is_harami()
            is_bullish_harami_prev = harami_result_prev.get('detected', False) and harami_result_prev.get('type') == 'bullish'
            if not is_bullish_harami_prev: return result
            c1 = last_3['close'].iloc[1]; o2, c2 = last_3['open'].iloc[2], last_3['close'].iloc[2]
            third_confirms = (c2 > o2) and (c2 > c1); result['detected'] = third_confirms
            if third_confirms:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_three_inside_down(self) -> Dict[str, Any]:
        pattern_name = 'three_inside_down'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            temp_data_prev = self.data.iloc[:-1] if len(self.data) >= 3 else pd.DataFrame();
            if temp_data_prev.empty: return result
            temp_analyzer = PatternAnalyzer(temp_data_prev); harami_result_prev = temp_analyzer._is_harami()
            is_bearish_harami_prev = harami_result_prev.get('detected', False) and harami_result_prev.get('type') == 'bearish'
            if not is_bearish_harami_prev: return result
            c1 = last_3['close'].iloc[1]; o2, c2 = last_3['open'].iloc[2], last_3['close'].iloc[2]
            third_confirms = (c2 < o2) and (c2 < c1); result['detected'] = third_confirms
            if third_confirms:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_tweezer_bottoms(self) -> Dict[str, Any]:
        pattern_name = 'tweezer_bottoms'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2)
        if last_2 is None: return result
        try:
            l0 = last_2['low'].iloc[0]; l1 = last_2['low'].iloc[1]; equal_lows = self._are_prices_equal(l0, l1); result['detected'] = equal_lows
            if equal_lows:
                 def quality_logic(candle, atr): return 'high' if candle['close'] > candle['open'] else 'medium'
                 result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_tweezer_tops(self) -> Dict[str, Any]:
        pattern_name = 'tweezer_tops'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2)
        if last_2 is None: return result
        try:
            h0 = last_2['high'].iloc[0]; h1 = last_2['high'].iloc[1]; equal_highs = self._are_prices_equal(h0, h1); result['detected'] = equal_highs
            if equal_highs:
                 def quality_logic(candle, atr): return 'high' if candle['close'] < candle['open'] else 'medium'
                 result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_inverted_hammer(self) -> Dict[str, Any]:
        pattern_name = 'inverted_hammer'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_candle_df = self._get_last_n_candles(1)
        if last_candle_df is None: return result
        try:
            last_candle = last_candle_df.iloc[0]; o, h, l, c = last_candle['open'], last_candle['high'], last_candle['low'], last_candle['close']
            body = abs(o - c); body = max(body, c * 0.0001); lower_shadow = min(o, c) - l; upper_shadow = h - max(o, c)
            is_inverted_shape = (upper_shadow >= 2 * body) and (lower_shadow <= body * 1.1); result['detected'] = is_inverted_shape
            if is_inverted_shape:
                def quality_logic(candle, atr): body_i = abs(candle['open'] - candle['close']); body_i = max(body_i, atr * 0.01); upper_shadow_i = candle['high'] - max(candle['open'], candle['close']); return 'high' if upper_shadow_i > atr * self.QUALITY_ATR_SHADOW_LONG and body_i < atr * self.QUALITY_ATR_SHADOW_SHORT else 'medium' if upper_shadow_i > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_candle_df, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_piercing_line(self) -> Dict[str, Any]:
        pattern_name = 'piercing_line'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2)
        if last_2 is None: return result
        try:
            o0, c0 = last_2['open'].iloc[0], last_2['close'].iloc[0]; o1, c1 = last_2['open'].iloc[1], last_2['close'].iloc[1]
            first_bearish = c0 < o0; second_bullish = c1 > o1; gap_down_open = o1 < c0; closes_above_midpoint = c1 > (o0 + c0) / 2; closes_below_open = c1 < o0
            detected = first_bearish and second_bullish and gap_down_open and closes_above_midpoint and closes_below_open; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_dark_cloud_cover(self) -> Dict[str, Any]:
        pattern_name = 'dark_cloud_cover'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_2 = self._get_last_n_candles(2)
        if last_2 is None: return result
        try:
            o0, c0 = last_2['open'].iloc[0], last_2['close'].iloc[0]; o1, c1 = last_2['open'].iloc[1], last_2['close'].iloc[1]
            first_bullish = c0 > o0; second_bearish = c1 < o1; gap_up_open = o1 > c0; closes_below_midpoint = c1 < (o0 + c0) / 2; closes_above_open = c1 > o0
            detected = first_bullish and second_bearish and gap_up_open and closes_below_midpoint and closes_above_open; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_2, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result

    # --- METODO REINTEGRATO ---
    def _is_bull_flag(self) -> Dict[str, Any]:
        """Verifica pattern Bull Flag (logica semplificata)."""
        pattern_name = 'bull_flag'
        result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False} # Default
        lookback = 15 # Periodo per cercare il flag
        data_slice = self._get_last_n_candles(lookback)
        if data_slice is None: return result
        try:
            # Logica semplificata: cerca un rally seguito da consolidamento
            low_point_idx_loc = data_slice['low'].idxmin() # Trova minimo nel lookback
            # Verifica se idxmin ha restituito un indice valido (potrebbe essere NaT se tutto NaN)
            if pd.isna(low_point_idx_loc):
                logger.debug(f"{pattern_name}: Impossibile trovare il minimo nel lookback.")
                return result

            data_since_low = data_slice.loc[low_point_idx_loc:]
            if len(data_since_low) < 5: return result # Necessita di qualche candela dopo il minimo

            initial_rally = data_since_low['high'].iloc[-1] - data_since_low['low'].iloc[0]
            consolidation_period = 5 # Ultime 5 candele come consolidamento
            if len(data_since_low) < consolidation_period + 2: return result # Abbastanza dati per consolidamento + breakout?
            consolidation_data = data_since_low.iloc[-consolidation_period:]
            consolidation_range = consolidation_data['high'].max() - consolidation_data['low'].min()

            # Condizioni Euristiche
            # Assicura che il prezzo di riferimento per il rally sia valido
            start_price_rally = data_since_low['close'].iloc[0]
            if pd.isna(start_price_rally) or start_price_rally <= 0:
                 is_significant_rally = False # Non possiamo valutare il rally
                 logger.debug(f"{pattern_name}: Prezzo iniziale per rally non valido.")
            else:
                 is_significant_rally = initial_rally > (start_price_rally * 0.05) # Rally almeno 5%?

            is_tight_consolidation = consolidation_range < (initial_rally * 0.5) if initial_rally > 0 else True # Consolidamento < 50% rally

            result['detected'] = is_significant_rally and is_tight_consolidation
            if result['detected']:
                 result['quality'] = 'medium' # Qualità media di default se rilevato
                 result['volume_ok'] = True # Non controllato qui
                 result['context_ok'] = True # Contesto non controllato qui
                 logger.debug(f"{pattern_name} rilevato (semplificato).")
        except Exception as e:
             logger.warning(f"Errore durante il rilevamento di {pattern_name}: {e}", exc_info=False) # Logga solo messaggio errore
        return result
    # --- FINE METODO REINTEGRATO ---

    def _is_rising_three_methods(self) -> Dict[str, Any]:
        pattern_name = 'rising_three_methods'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': True}; last_5 = self._get_last_n_candles(5)
        if last_5 is None: return result
        try:
            o0, c0, h0, l0 = last_5.iloc[0][['open', 'close', 'high', 'low']]; first_long_bullish = (c0 > o0) and (abs(c0 - o0) > (o0 * 0.015));
            if not first_long_bullish: return result
            in_range = True;
            for i in range(1, 4):
                hi, li = last_5.iloc[i][['high', 'low']]
                if not ((hi < h0) and (li > l0)):
                    in_range = False
                    break
            if not in_range: return result
            o4, c4 = last_5.iloc[4][['open', 'close']]; fifth_long_bullish_closes_higher = (c4 > o4) and (abs(c4-o4) > (o4 * 0.015)) and (c4 > c0); result['detected'] = fifth_long_bullish_closes_higher
            if fifth_long_bullish_closes_higher:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], _ = self._assess_quality_volume_context(last_5, -1, None, True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_falling_three_methods(self) -> Dict[str, Any]:
        pattern_name = 'falling_three_methods'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': True}; last_5 = self._get_last_n_candles(5)
        if last_5 is None: return result
        try:
            o0, c0, h0, l0 = last_5.iloc[0][['open', 'close', 'high', 'low']]; first_long_bearish = (c0 < o0) and (abs(c0 - o0) > (o0 * 0.015));
            if not first_long_bearish: return result
            in_range = True
            for i in range(1, 4):
                hi, li = last_5.iloc[i][['high', 'low']]
                if not ((hi < h0) and (li > l0)):
                    in_range = False
                    break
            if not in_range: return result
            o4, c4 = last_5.iloc[4][['open', 'close']]; fifth_long_bearish_closes_lower = (c4 < o4) and (abs(c4-o4) > (o4 * 0.015)) and (c4 < c0); result['detected'] = fifth_long_bearish_closes_lower
            if fifth_long_bearish_closes_lower:
                 def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                 result['quality'], result['volume_ok'], _ = self._assess_quality_volume_context(last_5, -1, None, True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_morning_doji_star(self) -> Dict[str, Any]:
        pattern_name = 'morning_doji_star'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            o0, c0 = last_3['open'].iloc[0], last_3['close'].iloc[0]; o2, c2 = last_3['open'].iloc[2], last_3['close'].iloc[2]; body0 = abs(o0-c0); body2 = abs(o2-c2)
            first_bearish_long = (c0 < o0) and (body0 > (o0 * 0.01)); temp_data_mid = self.data.iloc[-2:-1] if len(self.data) >= 2 else pd.DataFrame(); second_is_doji = False; second_doji_quality = 'low'
            if not temp_data_mid.empty: temp_analyzer_mid = PatternAnalyzer(temp_data_mid); second_doji_result = temp_analyzer_mid._is_doji(); second_is_doji = second_doji_result.get('detected', False); second_doji_quality = second_doji_result.get('quality', 'low')
            third_bullish_long = (c2 > o2) and (body2 > (o2 * 0.01)); closes_in_first_body = c2 > (o0 + c0) / 2
            detected = first_bearish_long and second_is_doji and third_bullish_long and closes_in_first_body; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG and second_doji_quality == 'high' else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'downtrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_evening_doji_star(self) -> Dict[str, Any]:
        pattern_name = 'evening_doji_star'; result = {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False}; last_3 = self._get_last_n_candles(3)
        if last_3 is None: return result
        try:
            o0, c0 = last_3['open'].iloc[0], last_3['close'].iloc[0]; o2, c2 = last_3['open'].iloc[2], last_3['close'].iloc[2]; body0 = abs(o0-c0); body2 = abs(o2-c2)
            first_bullish_long = (c0 > o0) and (body0 > (o0 * 0.01)); temp_data_mid = self.data.iloc[-2:-1] if len(self.data) >= 2 else pd.DataFrame(); second_is_doji = False; second_doji_quality = 'low'
            if not temp_data_mid.empty: temp_analyzer_mid = PatternAnalyzer(temp_data_mid); second_doji_result = temp_analyzer_mid._is_doji(); second_is_doji = second_doji_result.get('detected', False); second_doji_quality = second_doji_result.get('quality', 'low')
            third_bearish_long = (c2 < o2) and (body2 > (o2 * 0.01)); closes_in_first_body = c2 < (o0 + c0) / 2
            detected = first_bullish_long and second_is_doji and third_bearish_long and closes_in_first_body; result['detected'] = detected
            if detected:
                def quality_logic(candle, atr): body = abs(candle['close'] - candle['open']); return 'high' if body > atr * self.QUALITY_ATR_BODY_STRONG and second_doji_quality == 'high' else 'medium' if body > atr * self.QUALITY_ATR_BODY_MEDIUM else 'low'
                result['quality'], result['volume_ok'], result['context_ok'] = self._assess_quality_volume_context(last_3, -1, 'uptrend', True, quality_logic)
        except Exception as e: logger.warning(f"Errore {pattern_name}: {e}")
        return result
    def _is_three_white_soldiers(self) -> Dict[str, Any]: return {'detected': False, 'quality': 'low', 'volume_ok': False, 'context_ok': False} # Placeholder

    # --- Funzione Principale (detect_all_patterns) ---
    def detect_all_patterns(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Rileva tutti i pattern di candele implementati."""
        # (Logica detect_all_patterns INVARIATA rispetto alla versione precedente corretta)
        if self.data.empty: logger.warning("detect_all_patterns: No data."); return {'bullish': {}, 'bearish': {}, 'continuation': {}, 'neutral': {'error': 'No data'}}
        start_time = time.time()
        patterns: Dict[str, Dict[str, Dict[str, Any]]] = {'bullish': {}, 'bearish': {}, 'continuation': {}, 'neutral': {}}
        pattern_functions = {
            'bullish': [('engulfing', self._is_bullish_engulfing), ('hammer', self._is_hammer), ('morning_star', self._is_morning_star), ('morning_doji_star', self._is_morning_doji_star), ('three_white_soldiers', self._is_three_white_soldiers), ('three_inside_up', self._is_three_inside_up), ('tweezer_bottoms', self._is_tweezer_bottoms), ('inverted_hammer', self._is_inverted_hammer), ('piercing_line', self._is_piercing_line)],
            'bearish': [('engulfing', self._is_bearish_engulfing), ('hanging_man', self._is_hanging_man), ('evening_star', self._is_evening_star), ('evening_doji_star', self._is_evening_doji_star), ('shooting_star', self._is_shooting_star), ('three_black_crows', self._is_three_black_crows), ('three_inside_down', self._is_three_inside_down), ('tweezer_tops', self._is_tweezer_tops), ('dark_cloud_cover', self._is_dark_cloud_cover)],
            'continuation': [('bull_flag', self._is_bull_flag), ('rising_three_methods', self._is_rising_three_methods), ('falling_three_methods', self._is_falling_three_methods)],
            'neutral': [('doji', self._is_doji)]
        }
        for category, func_list in pattern_functions.items():
            for name, func in func_list:
                try: patterns[category][name] = func()
                except Exception as e: logger.error(f"Errore pattern '{name}' ({category}): {e}", exc_info=False); patterns[category][name] = {'detected': False, 'error': str(e)}
        try:
            harami_result = self._is_harami()
            if harami_result.get('detected'):
                category_target = harami_result.get('type', 'neutral'); category_target = category_target if category_target in patterns else 'neutral'
                patterns[category_target]['harami'] = harami_result
        except Exception as e: logger.error(f"Errore pattern 'harami': {e}", exc_info=False)
        exec_time = time.time() - start_time; detected_count = sum(1 for cat_dict in patterns.values() for p_details in cat_dict.values() if p_details.get('detected'))
        logger.info(f"Analisi Pattern completata in {exec_time:.3f} sec. Rilevati: {detected_count} pattern.")
        return patterns

# --- END OF FILE statistical_analyzer_patterns.py ---

