# --- START OF FILE fibonacci_calculator.py ---
# fibonacci_calculator.py
"""
Modulo per il calcolo dei livelli di ritracciamento/estensione di Fibonacci
e delle zone temporali di Fibonacci.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union # Aggiunti tipi
from datetime import datetime, timezone, timedelta # Aggiunti tipi

# Importa helper (anche se non strettamente usati qui, buona pratica averli disponibili)
try:
    from statistical_analyzer_helpers import _safe_float, safe_strftime
except ImportError:
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.warning("statistical_analyzer_helpers non trovato in fibonacci_calculator. Uso fallback.")
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try: return float(value) if pd.notna(value) and not np.isinf(value) else default
        except: return default
    # Fallback locale per safe_strftime se l'import fallisce
    def safe_strftime(date_input: Optional[Union[datetime, pd.Timestamp, int, float]], fmt: str = "%Y-%m-%d %H:%M:%S", fallback: str = "N/A") -> str:
        if date_input is None or pd.isna(date_input): return fallback
        dt_object: Optional[datetime] = None
        try:
            if isinstance(date_input, (datetime, pd.Timestamp)):
                dt_object = pd.to_datetime(date_input)
                if dt_object.tzinfo is None: dt_object = dt_object.tz_localize(timezone.utc)
                elif str(dt_object.tz).upper() != 'UTC': dt_object = dt_object.tz_convert(timezone.utc)
            elif isinstance(date_input, (int, float)): dt_object = datetime.fromtimestamp(date_input / 1000.0, tz=timezone.utc)
            else: return fallback
            if dt_object:
                fmt_final = fmt + ("Z" if 'Z' not in fmt and '%Z' not in fmt else "")
                return dt_object.strftime(fmt_final)
            else: return fallback
        except Exception: return fallback


logger = logging.getLogger(__name__)

# --- Funzione Core Livelli Prezzo (invariata) ---
def calculate_fibonacci_levels(
    high: Optional[float],
    low: Optional[float],
    up_trend: bool # Richiede esplicitamente il trend
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Calcola i livelli di ritracciamento ed estensione di Fibonacci.

    Args:
        high (Optional[float]): Prezzo massimo nel periodo considerato.
        low (Optional[float]): Prezzo minimo nel periodo considerato.
        up_trend (bool): True se il trend principale è considerato rialzista, False se ribassista.

    Returns:
        dict: Dizionario con chiavi 'retracements' e 'extensions'.
              Ogni chiave contiene un dizionario di livelli {livello_str: prezzo_float | None}.
              Restituisce dizionari vuoti se high o low non sono validi o non permettono il calcolo.
    """
    retracements: Dict[str, Optional[float]] = {}
    extensions: Dict[str, Optional[float]] = {}
    result = {'retracements': retracements, 'extensions': extensions}

    # Validazione input
    high_f = _safe_float(high)
    low_f = _safe_float(low)

    if high_f is None or low_f is None:
        logger.warning("High o Low non validi (None, NaN, Inf). Impossibile calcolare livelli Fibonacci.")
        return result
    if high_f <= low_f:
        logger.warning(f"High ({high_f}) deve essere maggiore di Low ({low_f}). Impossibile calcolare livelli Fibonacci.")
        return result

    try:
        price_range = high_f - low_f
        if price_range <= 1e-9: # Range troppo piccolo
             logger.warning(f"Range prezzo ({price_range}) troppo piccolo. Impossibile calcolare livelli Fibonacci.")
             return result

        # Rapporti Fibo comuni
        fibo_ratios_retracement = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fibo_ratios_extension = [0.0, 1.0, 1.272, 1.618, 2.0, 2.618, 3.618, 4.236] # Aggiunti livelli estensione comuni

        # Calcolo RITRACCIAMENTI
        for ratio in fibo_ratios_retracement:
            level_name = f"{ratio:.3f}"
            if up_trend: # Durante un uptrend, i ritracciamenti sono SOTTO l'high
                level_val = high_f - price_range * ratio
            else: # Durante un downtrend, i ritracciamenti sono SOPRA il low
                level_val = low_f + price_range * ratio
            retracements[level_name] = _safe_float(level_val) # Salva come float o None

        # Calcolo ESTENSIONI (proiezioni oltre il range iniziale)
        for ratio in fibo_ratios_extension:
            level_name = f"{ratio:.3f}"
            if up_trend: # Durante un uptrend, le estensioni si proiettano SOPRA l'high
                # 0.0 = low, 1.0 = high
                if ratio == 0.0: level_val = low_f
                elif ratio == 1.0: level_val = high_f
                else: level_val = high_f + price_range * (ratio - 1.0) # Proietta oltre l'high
            else: # Durante un downtrend, le estensioni si proiettano SOTTO il low
                # 0.0 = high, 1.0 = low
                if ratio == 0.0: level_val = high_f
                elif ratio == 1.0: level_val = low_f
                else: level_val = low_f - price_range * (ratio - 1.0) # Proietta oltre il low
            extensions[level_name] = _safe_float(level_val)

    except Exception as e:
        logger.error(f"Errore nel calcolo dei livelli Fibonacci: {e}", exc_info=True)
        # Resetta i dizionari in caso di errore
        result['retracements'] = {f"{r:.3f}": None for r in fibo_ratios_retracement}
        result['extensions'] = {f"{r:.3f}": None for r in fibo_ratios_extension}

    return result

# --- Funzioni Wrapper Livelli Prezzo (invariate) ---
def get_fibonacci_retracements(
    data: pd.DataFrame,
    period: int = 60,
    up_trend: Optional[bool] = None # Trend può essere fornito o dedotto
) -> Dict[str, Optional[float]]:
    """
    Calcola i livelli di ritracciamento di Fibonacci per un periodo dato.

    Args:
        data (pd.DataFrame): DataFrame con colonne 'high', 'low', 'close' e DatetimeIndex UTC.
        period (int): Numero di periodi recenti su cui calcolare High/Low.
        up_trend (Optional[bool]): Direzione del trend principale. Se None, viene dedotto.

    Returns:
        Dict[str, Optional[float]]: Dizionario {livello_str: prezzo_float | None}.
                                    Restituisce dizionario vuoto se i dati sono insufficienti.
    """
    # Validazione input dati
    required_cols = ['high', 'low', 'close']
    if not isinstance(data, pd.DataFrame) or data.empty:
         logger.warning("Fibo Ret: Input 'data' non è un DataFrame valido o è vuoto.")
         return {}
    if not all(col in data.columns for col in required_cols):
        logger.warning(f"Fibo Ret: Colonne {required_cols} mancanti nei dati.")
        return {}
    if len(data) < period:
        logger.warning(f"Fibo Ret: Dati insufficienti ({len(data)}) per periodo {period}.")
        return {}
    if not isinstance(data.index, pd.DatetimeIndex):
         logger.warning("Fibo Ret: Indice non è DatetimeIndex.")
         return {}

    try:
        # Seleziona dati del periodo e trova High/Low
        period_data = data.iloc[-period:]
        high = period_data['high'].max()
        low = period_data['low'].min()

        # Determina trend se non fornito (con fallback e warning)
        if up_trend is None:
            logger.warning("Trend non fornito a get_fibonacci_retracements. Uso fallback TF corrente.")
            try:
                first_price = period_data['close'].iloc[0]
                last_price = period_data['close'].iloc[-1]
                if pd.isna(first_price) or pd.isna(last_price):
                     logger.warning("Fallback Fibo trend: prezzi NaN. Default a True (uptrend).")
                     up_trend = True
                else: up_trend = last_price > first_price
            except IndexError:
                 logger.warning("Fallback Fibo trend: IndexError. Default a True (uptrend).")
                 up_trend = True

        # Calcola livelli usando la funzione core
        levels = calculate_fibonacci_levels(high, low, up_trend=up_trend)
        return levels.get('retracements', {}) # Restituisci solo i ritracciamenti

    except Exception as e:
        logger.error(f"Errore in get_fibonacci_retracements: {e}", exc_info=True)
        return {} # Restituisci dizionario vuoto in caso di errore

def get_fibonacci_extensions(
    data: pd.DataFrame,
    period: int = 60,
    up_trend: Optional[bool] = None # Trend può essere fornito o dedotto
) -> Dict[str, Optional[float]]:
    """
    Calcola i livelli di estensione di Fibonacci per un periodo dato.
    (Logica quasi identica a get_fibonacci_retracements, ma restituisce 'extensions')
    """
    # Validazione input dati (identica a retracements)
    required_cols = ['high', 'low', 'close']
    if not isinstance(data, pd.DataFrame) or data.empty:
         logger.warning("Fibo Ext: Input 'data' non è un DataFrame valido o è vuoto.")
         return {}
    if not all(col in data.columns for col in required_cols):
        logger.warning(f"Fibo Ext: Colonne {required_cols} mancanti nei dati.")
        return {}
    if len(data) < period:
        logger.warning(f"Fibo Ext: Dati insufficienti ({len(data)}) per periodo {period}.")
        return {}
    if not isinstance(data.index, pd.DatetimeIndex):
         logger.warning("Fibo Ext: Indice non è DatetimeIndex.")
         return {}

    try:
        period_data = data.iloc[-period:]
        high = period_data['high'].max()
        low = period_data['low'].min()

        if up_trend is None:
            logger.warning("Trend non fornito a get_fibonacci_extensions. Uso fallback TF corrente.")
            try:
                first_price = period_data['close'].iloc[0]
                last_price = period_data['close'].iloc[-1]
                if pd.isna(first_price) or pd.isna(last_price): up_trend = True
                else: up_trend = last_price > first_price
            except IndexError: up_trend = True

        levels = calculate_fibonacci_levels(high, low, up_trend=up_trend)
        return levels.get('extensions', {}) # Restituisci solo le estensioni

    except Exception as e:
        logger.error(f"Errore in get_fibonacci_extensions: {e}", exc_info=True)
        return {}

# --- NUOVA FUNZIONE: Fibonacci Time Zones ---
def calculate_fibonacci_time_zones(
    data: pd.DataFrame,
    period: int = 60,
    start_from: str = 'low' # 'low' o 'high'
) -> Dict[str, Any]:
    """
    Calcola le zone temporali di Fibonacci proiettate da un minimo o massimo recente.

    Args:
        data (pd.DataFrame): DataFrame con colonne 'high', 'low', 'close' e DatetimeIndex UTC.
        period (int): Numero di periodi recenti su cui cercare il punto di partenza (min/max).
        start_from (str): 'low' per iniziare dal minimo, 'high' per iniziare dal massimo.

    Returns:
        Dict[str, Any]: Dizionario contenente il punto di partenza e le date/timestamp
                        delle zone temporali proiettate. Include 'error' se fallisce.
    """
    results: Dict[str, Any] = {
        'start_point_type': start_from,
        'start_point_timestamp': None,
        'start_point_price': None,
        'start_point_index_loc': None,
        'projected_zones_utc': [],
        'error': None
    }

    # Validazione input
    required_cols = ['high', 'low', 'close']
    if not isinstance(data, pd.DataFrame) or data.empty:
        results['error'] = "Input 'data' non è un DataFrame valido o è vuoto."; return results
    if not all(col in data.columns for col in required_cols):
        results['error'] = f"Colonne {required_cols} mancanti."; return results
    if len(data) < period:
        results['error'] = f"Dati insufficienti ({len(data)}) per periodo {period}."; return results
    if not isinstance(data.index, pd.DatetimeIndex):
        results['error'] = "Indice non è DatetimeIndex."; return results
    if data.index.tz is None or str(data.index.tz).upper() != 'UTC':
        logger.warning("Fibo Time: Indice non UTC. Verrà forzato UTC per i calcoli.")
        try:
            data = data.copy() # Evita SettingWithCopyWarning
            if data.index.tz is None: data.index = data.index.tz_localize('UTC')
            else: data.index = data.index.tz_convert('UTC')
        except Exception as tz_err:
            results['error'] = f"Errore forzatura UTC: {tz_err}"; return results

    try:
        # 1. Identifica punto di partenza
        period_data = data.iloc[-period:]
        start_point_dt: Optional[pd.Timestamp] = None
        start_point_price: Optional[float] = None
        start_point_index_loc: Optional[int] = None

        if start_from == 'low':
            start_point_dt = period_data['low'].idxmin()
            if pd.notna(start_point_dt):
                start_point_price = _safe_float(period_data.loc[start_point_dt, 'low'])
        elif start_from == 'high':
            start_point_dt = period_data['high'].idxmax()
            if pd.notna(start_point_dt):
                start_point_price = _safe_float(period_data.loc[start_point_dt, 'high'])
        else:
            results['error'] = "Valore 'start_from' non valido (usa 'low' o 'high')."; return results

        if start_point_dt is None or start_point_price is None:
            results['error'] = f"Impossibile trovare punto di partenza '{start_from}' nel periodo."; return results

        # 2. Ottieni indice numerico del punto di partenza nel DataFrame *originale*
        try:
            # Usare get_loc sull'indice originale è più sicuro
            start_point_index_loc = data.index.get_loc(start_point_dt)
            results['start_point_timestamp'] = safe_strftime(start_point_dt)
            results['start_point_price'] = start_point_price
            results['start_point_index_loc'] = start_point_index_loc
        except KeyError:
            results['error'] = f"Timestamp di partenza {start_point_dt} non trovato nell'indice originale."; return results
        except Exception as e_loc:
             results['error'] = f"Errore ottenimento locazione indice: {e_loc}"; return results

        # 3. Proietta i periodi futuri
        fibo_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233] # Limite ragionevole
        projected_zones: List[str] = []
        last_data_timestamp = data.index[-1]
        time_delta_per_period: Optional[timedelta] = None

        # Stima il delta temporale tra le candele
        if len(data.index) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            if not time_diffs.empty:
                time_delta_per_period = time_diffs.median() # Usa mediana per robustezza
            else: # Fallback per 2 punti
                time_delta_per_period = data.index[1] - data.index[0]
        if time_delta_per_period is None or time_delta_per_period <= timedelta(0):
             results['error'] = "Impossibile determinare il delta temporale tra le candele."; return results

        for fib_num in fibo_sequence:
            projected_index_loc = start_point_index_loc + fib_num
            projected_dt: Optional[pd.Timestamp] = None

            # Se l'indice è all'interno dei dati esistenti, prendi il timestamp da lì
            if 0 <= projected_index_loc < len(data.index):
                projected_dt = data.index[projected_index_loc]
            # Altrimenti, proietta nel futuro basandoti sul delta temporale
            else:
                periods_beyond_data = projected_index_loc - (len(data.index) - 1)
                projected_dt = last_data_timestamp + (periods_beyond_data * time_delta_per_period)

            if projected_dt:
                projected_zones.append(safe_strftime(projected_dt))

        results['projected_zones_utc'] = projected_zones
        results.pop('error', None) # Rimuovi errore se tutto ok

    except Exception as e:
        logger.error(f"Errore in calculate_fibonacci_time_zones: {e}", exc_info=True)
        results['error'] = f"Calcolo fallito: {e}"
        results['projected_zones_utc'] = [] # Resetta zone in caso di errore

    return results
# --- FINE NUOVA FUNZIONE ---

# --- END OF FILE fibonacci_calculator.py ---