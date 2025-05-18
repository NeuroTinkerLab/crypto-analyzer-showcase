# --- START OF FILE statistical_analyzer_helpers.py ---
# statistical_analyzer_helpers.py
"""
Modulo contenente funzioni helper per l'analisi statistica,
la manipolazione sicura dei dati, la formattazione e il salvataggio
dei risultati in formati (JSON, TXT) adatti all'elaborazione LLM.
Include anche funzioni per salvare/caricare statistiche storiche persistenti
(cicli, intra-candela, dati Deribit aggregati).
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import operator
from datetime import datetime, date, timezone # Assicurati che timezone sia importato

# Setup logging
logger = logging.getLogger(__name__)

# --- Helper Generico Safe Get ---
def _safe_get(data: Optional[Dict], keys: List[Any], default: Any = None) -> Any:
    """Recupera in modo sicuro un valore da un dizionario nidificato."""
    if data is None: return default
    temp_data = data
    for key in keys:
        if isinstance(temp_data, dict):
            temp_data = temp_data.get(key)
        elif isinstance(key, int) and isinstance(temp_data, (list, tuple)):
             if -len(temp_data) <= key < len(temp_data):
                 try: temp_data = temp_data[key]
                 except IndexError: return default
             else: return default # Indice fuori range
        else:
            return default # Tipo non supportato o chiave non trovata
        if temp_data is None: return default # Esce se un livello è None

    # Controllo finale per NaN/Inf, anche per tipi numpy
    if isinstance(temp_data, (float, int, np.number)) and (pd.isna(temp_data) or np.isinf(temp_data)):
        return default
    return temp_data


# --- Helper Conversione Sicura a Float ---
def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Helper per convertire un valore in float in modo sicuro, gestendo None, NaN, Inf."""
    if value is None or pd.isna(value): return default
    # --- MODIFICA: Controlla Inf esplicitamente PRIMA della conversione ---
    if isinstance(value, (float, np.floating)) and np.isinf(value):
        # logger.debug(f"Valore Inf rilevato: {value}. Restituisco default.")
        return default
    # --- FINE MODIFICA ---
    try:
        # Gestisce anche tipi numerici numpy
        if isinstance(value, np.number):
             # Controlla di nuovo NaN/Inf dopo la potenziale conversione numpy
             float_val = float(value)
             if np.isnan(float_val) or np.isinf(float_val):
                 # logger.debug(f"Valore Numpy NaN/Inf rilevato: {value}. Restituisco default.")
                 return default
             return float_val
        # Conversione standard
        float_val = float(value)
        # Controllo finale post-conversione
        if np.isnan(float_val) or np.isinf(float_val):
             # logger.debug(f"Valore NaN/Inf rilevato post-conversione: {value}. Restituisco default.")
             return default
        return float_val
    except (ValueError, TypeError):
        # logger.debug(f"Impossibile convertire '{value}' (tipo: {type(value)}) in float.", exc_info=False) # Log troppo verboso
        return default
    except Exception as e:
        logger.error(f"Errore imprevisto convertendo '{value}' in float: {e}", exc_info=True)
        return default


# --- Formatting Helpers ---

def safe_strftime(date_input: Optional[Union[datetime, date, pd.Timestamp, int, float]], fmt: str = "%Y-%m-%d %H:%M:%S", fallback: str = "N/A") -> str:
    """
    Formatta in modo sicuro un oggetto data/datetime o un timestamp numerico (ms) in stringa UTC.
    """
    if date_input is None or pd.isna(date_input):
        return fallback

    dt_object: Optional[datetime] = None

    try:
        if isinstance(date_input, (datetime, date, pd.Timestamp)):
            dt_object = pd.to_datetime(date_input) # Converte anche date in datetime
            # Assicura che sia timezone-aware (UTC) se possibile
            if dt_object.tzinfo is None:
                 try: dt_object = dt_object.tz_localize(timezone.utc) # Assume UTC se naive
                 except Exception: pass # Ignora errori di localizzazione (es. già localizzato)
            elif str(dt_object.tz).upper() != 'UTC':
                 try: dt_object = dt_object.tz_convert(timezone.utc)
                 except Exception: pass # Ignora errori di conversione
        elif isinstance(date_input, (int, float)):
            # Assume input numerico sia timestamp in MILLISECONDI
            timestamp_sec = date_input / 1000.0
            # Aggiungi controllo per timestamp irrealistici (es. anno < 1980 o > 2100)
            if not (datetime(1980, 1, 1).timestamp() < timestamp_sec < datetime(2100, 1, 1).timestamp()):
                 # logger.warning(f"Timestamp {date_input} ({timestamp_sec}s) fuori dal range realistico. Restituisco fallback.") # Log troppo verboso
                 return fallback
            dt_object = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
        else:
            # Tipo non riconosciuto
            # logger.debug(f"Tipo input non gestito da safe_strftime: {type(date_input)}") # Log troppo verboso
            return fallback

        # Formattazione finale
        if dt_object:
            # Aggiungi 'Z' per indicare UTC esplicitamente se non già nel formato
            fmt_final = fmt
            if 'Z' not in fmt and '%Z' not in fmt:
                fmt_final += "Z"
            return dt_object.strftime(fmt_final)
        else:
            return fallback # Se la conversione fallisce

    except (ValueError, OverflowError, OSError, TypeError, AttributeError) as e:
         # Errori comuni nella conversione/formattazione
         logger.warning(f"Errore nella formattazione di '{date_input}': {e}")
         return fallback
    except Exception as e:
        # Errori imprevisti
        logger.error(f"Errore imprevisto in safe_strftime per '{date_input}': {e}", exc_info=True)
        return fallback


def format_percentage(value: Optional[Union[float, int, np.number]], decimal_places: int = 2, fallback: str = "N/A") -> str:
    """Formatta un valore float come stringa percentuale."""
    val_f = _safe_float(value)
    if val_f is None: return fallback
    try: return f"{val_f * 100:.{decimal_places}f}%"
    except (ValueError, TypeError) as e: logger.warning(f"Errore formattazione percentuale {value}: {e}"); return fallback

def format_number(value: Optional[Union[int, float, np.number]], decimal_places: int = 2, fallback: str = "N/A") -> str:
    """Formatta un numero con N cifre decimali."""
    val_f = _safe_float(value)
    if val_f is None: return fallback
    try: return f"{val_f:.{decimal_places}f}"
    except (ValueError, TypeError) as e: logger.warning(f"Errore formattazione numero {value}: {e}"); return fallback


# --- Safe Operation Helpers ---
def safe_compare(a: Any, b: Any, op: Callable = operator.eq) -> Optional[bool]:
    """Confronta in modo sicuro due valori, gestendo None e NaN."""
    if a is None or b is None or pd.isna(a) or pd.isna(b): return None
    try:
        num_types = (int, float, np.number)
        # Consenti confronto tra tipi numerici diversi
        if not (isinstance(a, type(b)) or isinstance(b, type(a)) or (isinstance(a, num_types) and isinstance(b, num_types))):
            return None # Non confrontare tipi incompatibili (es. stringa con numero)
        return bool(op(a, b))
    except (TypeError, ValueError) as e: logger.debug(f"Errore confronto (TypeError/ValueError) tra {a} ({type(a)}) e {b} ({type(b)}): {e}"); return None
    except Exception as e: logger.error(f"Errore imprevisto confronto: {e}", exc_info=True); return None

def safe_lt_compare(a: Any, b: Any) -> Optional[bool]:
    """Confronta in modo sicuro a < b."""
    return safe_compare(a, b, operator.lt)

def safe_get_value_at_index(
    data: Optional[Union[pd.Series, np.ndarray, list, tuple]],
    index: int = -1,
    default: Any = None
) -> Any:
    """
    Ottiene in modo sicuro un valore a un indice specificato.
    Restituisce default se l'indice è fuori range, l'input è None/vuoto, o il valore è NA/None/Inf.
    Prova a restituire un float se possibile.
    """
    if data is None: return default
    try:
        # Controlla se l'oggetto ha una lunghezza e se non è vuoto
        if not hasattr(data, '__len__') or len(data) == 0: return default
        n = len(data)
        # Calcola l'indice effettivo (gestisce indici negativi)
        effective_index = index if index >= 0 else n + index
        # Verifica se l'indice effettivo è nei limiti
        if 0 <= effective_index < n:
            # Estrai il valore in base al tipo di dato
            if isinstance(data, (np.ndarray, list, tuple)): value = data[effective_index]
            elif isinstance(data, pd.Series): value = data.iloc[effective_index]
            else: logger.warning(f"Tipo dati non supportato in safe_get_value_at_index: {type(data)}"); return default

            # Controlla se il valore è None o NaN/Inf
            if value is None or pd.isna(value) or (isinstance(value, (float, np.floating)) and np.isinf(value)):
                return default

            # Tenta di convertire in float se possibile, altrimenti restituisce il valore originale
            val_float = _safe_float(value, default=None)
            return val_float if val_float is not None else value
        else:
            # Indice fuori dai limiti
            return default
    except (IndexError, AttributeError, TypeError) as e:
        logger.debug(f"Errore accesso indice {index} per tipo {type(data)}: {e}")
        return default
    except Exception as e:
        logger.error(f"Errore imprevisto in safe_get_value_at_index: {e}", exc_info=True)
        return default

def safe_get_last_value(series: Optional[pd.Series], default: Any = None) -> Any:
    """Ottiene l'ultimo valore valido da una pandas Series."""
    return safe_get_value_at_index(series, index=-1, default=default)


# --- Data Calculation Helpers ---
# (calculate_returns, calculate_volatility_metrics, detect_outliers, calculate_drawdowns rimangono invariate)
def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calcola rendimenti semplici e logaritmici."""
    if not isinstance(data, pd.DataFrame): logger.error("Input per calculate_returns non è un DataFrame."); return pd.DataFrame()
    if 'close' not in data.columns: logger.warning("Colonna 'close' mancante per calcolare i rendimenti."); return data
    df = data.copy()
    try:
        # Rimuovi colonne esistenti per evitare duplicati se chiamata più volte
        df.drop(columns=['simple_return', 'log_return'], errors='ignore', inplace=True)
        # Usa .replace per evitare divisione per zero e .diff()
        close_valid = df['close'].replace(0, np.nan)
        df['simple_return'] = close_valid.pct_change().replace([np.inf, -np.inf], np.nan)
        df['log_return'] = np.log(close_valid / close_valid.shift(1)).replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        logger.error(f"Errore calcolo rendimenti: {e}", exc_info=True)
        df['simple_return'] = np.nan
        df['log_return'] = np.nan
    return df

def calculate_volatility_metrics(data: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    """Calcola deviazioni standard rolling dei rendimenti."""
    if not isinstance(data, pd.DataFrame): logger.error("Input per calculate_volatility_metrics non è un DataFrame."); return pd.DataFrame()
    df = data.copy()
    if 'simple_return' not in df.columns:
        logger.warning("Calcolo rendimenti prima della volatilità.")
        df = calculate_returns(df)
    if 'simple_return' not in df.columns: return df # Non può procedere se i rendimenti mancano
    try:
        for window in windows:
            col_name_std = f'{window}p_std_dev'
            # Assicurati che ci siano abbastanza punti non-NaN per calcolare la rolling std
            if len(df['simple_return'].dropna()) >= window:
                df[col_name_std] = df['simple_return'].rolling(window=window, min_periods=max(1, window//2)).std()
            else:
                df[col_name_std] = np.nan # Imposta a NaN se non ci sono abbastanza dati
    except Exception as e:
        logger.error(f"Errore calcolo metriche volatilità: {e}", exc_info=True)
    return df

def detect_outliers(data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> Optional[pd.Series]:
    """Rileva outliers in una Series usando IQR (default), Z-score o MAD."""
    if not isinstance(data, pd.Series): logger.error("Input per detect_outliers non è una pandas Series."); return None
    if data.empty or data.isnull().all(): logger.warning("Serie vuota o tutta NaN per detect_outliers."); return pd.Series(False, index=data.index)
    try:
        data_clean = data.dropna()
        if data_clean.empty: return pd.Series(False, index=data.index) # Nessun dato valido
        outliers = pd.Series(False, index=data.index) # Inizializza a False per tutti

        if method == 'zscore':
            mean, std = data_clean.mean(), data_clean.std()
            if std is None or pd.isna(std) or std == 0: return outliers # Non calcolabile
            z_scores = (data - mean) / std
            outliers.loc[data.index] = (abs(z_scores) > threshold) # Applica a tutti gli indici originali
        elif method == 'iqr':
            q1, q3 = data_clean.quantile(0.25), data_clean.quantile(0.75)
            iqr_val = q3 - q1
            if iqr_val == 0: return outliers # Non calcolabile se IQR è zero
            lower_bound, upper_bound = q1 - (threshold * iqr_val), q3 + (threshold * iqr_val)
            outliers.loc[data.index] = (data < lower_bound) | (data > upper_bound)
        elif method == 'mad': # Median Absolute Deviation
            median = data_clean.median()
            mad_val = (abs(data_clean - median)).median()
            if mad_val == 0: return outliers # Non calcolabile
            # Scala MAD per stimare std dev (per distribuzione normale)
            mad_normalized = mad_val / 0.67449
            if mad_normalized == 0: return outliers
            modified_z = (data - median) / mad_normalized
            outliers.loc[data.index] = (abs(modified_z) > threshold)
        else:
            logger.warning(f"Metodo outlier non riconosciuto: {method}. Uso IQR.")
            return detect_outliers(data, method='iqr', threshold=threshold) # Usa soglia passata

        outliers[data.isna()] = False # Gli NA non sono outliers
        return outliers
    except Exception as e:
        logger.error(f"Errore rilevamento outliers (metodo {method}): {e}", exc_info=True)
        return pd.Series(False, index=data.index) # Ritorna False in caso di errore

def calculate_drawdowns(returns: pd.Series) -> List[Dict[str, Any]]:
    """Calcola i periodi di drawdown da una serie di rendimenti."""
    if not isinstance(returns, pd.Series) or returns.empty: return []
    if not isinstance(returns.index, pd.DatetimeIndex): logger.error("calculate_drawdowns richiede DatetimeIndex."); return []

    drawdown_periods = []
    try:
        # Calcola rendimenti cumulativi e massimo corrente
        cumulative_returns = (1 + returns.fillna(0)).cumprod()
        running_max = cumulative_returns.cummax()

        # Calcola drawdown come deviazione percentuale dal massimo corrente
        drawdowns = (cumulative_returns / running_max.replace(0, np.nan)) - 1
        drawdowns = drawdowns.fillna(0) # Riempi eventuali NaN iniziali con 0

        # Identifica i periodi in cui si è in drawdown (escludendo valori molto vicini a zero)
        in_drawdown = (drawdowns < -1e-9) # Usa soglia piccola per evitare floating point issues
        if not in_drawdown.any(): return [] # Nessun drawdown significativo

        # Trova punti di inizio e fine dei drawdown
        diff = in_drawdown.astype(int).diff().fillna(0)
        start_indices = diff[diff == 1].index
        end_indices = diff[diff == -1].index

        # Gestisci casi limite (drawdown inizia all'inizio o finisce alla fine)
        if in_drawdown.iloc[0]:
            start_indices = start_indices.insert(0, in_drawdown.index[0])
        if in_drawdown.iloc[-1]:
            last_start = start_indices[-1] if not start_indices.empty else in_drawdown.index[0]
            valid_end_candidates = end_indices[end_indices >= last_start]
            if valid_end_candidates.empty or end_indices.empty or end_indices[-1] < last_start:
                 end_indices = end_indices.insert(len(end_indices), in_drawdown.index[-1])


        # Itera sugli inizi di drawdown per trovare dettagli
        processed_end_locs = -1
        for start_date in start_indices:
            possible_ends_after_start = end_indices[end_indices > start_date]
            if processed_end_locs >= 0 and not end_indices.empty:
                possible_ends_after_start = possible_ends_after_start[possible_ends_after_start > end_indices[processed_end_locs]]

            if possible_ends_after_start.empty:
                end_date = drawdowns.index[-1]
            else:
                end_date = possible_ends_after_start[0]
                try:
                    processed_end_locs = np.where(end_indices == end_date)[0][0]
                except IndexError:
                     logger.warning(f"Impossibile trovare la posizione di end_date {end_date} in end_indices.")
                     processed_end_locs = len(end_indices) - 1 # Assumi sia l'ultimo

            # Analizza l'episodio di drawdown
            drawdown_episode = drawdowns.loc[start_date:end_date]
            if drawdown_episode.empty: continue

            max_dd_value = drawdown_episode.min()
            trough_date = drawdown_episode.idxmin()

            # Trova il picco precedente
            peak_date = running_max.loc[:start_date].idxmax()
            peak_value = running_max.loc[peak_date] # Valore cumulativo al picco

            # Trova la data di recupero (se esiste)
            recovery_date = None
            post_trough_cum_returns = cumulative_returns.loc[trough_date:]
            recovery_candidates = post_trough_cum_returns[post_trough_cum_returns >= peak_value]
            if not recovery_candidates.empty:
                recovery_date = recovery_candidates.index[0]

            # Calcola durate
            duration_trough_td = trough_date - peak_date
            duration_trough = duration_trough_td.days if pd.notna(duration_trough_td) else None
            recovery_duration_td = (recovery_date - trough_date) if recovery_date is not None else None
            recovery_duration = recovery_duration_td.days if pd.notna(recovery_duration_td) else None

            drawdown_periods.append({
                'peak_date': safe_strftime(peak_date, "%Y-%m-%d"),
                'trough_date': safe_strftime(trough_date, "%Y-%m-%d"),
                'start_date': safe_strftime(start_date, "%Y-%m-%d"),
                'end_date': safe_strftime(end_date, "%Y-%m-%d"), # Data fine drawdown (quando < 0 finisce)
                'recovery_date': safe_strftime(recovery_date, "%Y-%m-%d") if recovery_date else None, # Data recupero picco
                'max_drawdown': _safe_float(max_dd_value),
                'duration_to_trough_days': duration_trough,
                'recovery_days': recovery_duration # Giorni da trough a recupero
            })

        # Ordina per drawdown massimo (più negativo prima)
        drawdown_periods.sort(key=lambda x: x.get('max_drawdown', 0) or 0)
    except Exception as e:
        logger.error(f"Errore calcolo drawdowns: {e}", exc_info=True)
        return [] # Restituisce lista vuota in caso di errore
    return drawdown_periods


# --- MODIFICA QUI: Calcolo Metriche Intra-Candela ---
def calculate_single_candle_metrics(
    open_p: Optional[float],
    high_p: Optional[float],
    low_p: Optional[float],
    close_p: Optional[float]
) -> Dict[str, Optional[float]]:
    """
    Calcola le metriche percentuali per una singola candela OHLC.
    Reso più robusto alla divisione per zero o valori di open molto piccoli.

    Args:
        open_p: Prezzo di apertura.
        high_p: Prezzo massimo.
        low_p: Prezzo minimo.
        close_p: Prezzo di chiusura.

    Returns:
        Dict[str, Optional[float]]: Dizionario contenente le 6 metriche percentuali.
                                    I valori sono None se non calcolabili.
    """
    metrics = {
        'ic_range_pct': None, 'ic_min_max_pct': None, 'ic_min_close_pct': None,
        'ic_open_max_pct': None, 'ic_open_min_pct': None, 'ic_body_pct': None
    }

    # Verifica input validi (inclusi NaN)
    if None in [open_p, high_p, low_p, close_p] or \
       any(pd.isna(p) for p in [open_p, high_p, low_p, close_p]):
        return metrics

    # Verifica coerenza OHLC base
    # Aggiunta tolleranza minima per confronti float
    epsilon = 1e-9
    if low_p > high_p + epsilon or \
       open_p > high_p + epsilon or open_p < low_p - epsilon or \
       close_p > high_p + epsilon or close_p < low_p - epsilon:
        logger.warning(f"Dati OHLC incoerenti: O={open_p}, H={high_p}, L={low_p}, C={close_p}. Metriche saranno None.")
        return metrics

    # --- Gestione Robusta Divisione per Zero/Open Piccolo ---
    # Usa una soglia assoluta minima per il denominatore
    min_abs_denominator_threshold = 1e-9 # Evita divisione per zero esatta

    if abs(open_p) < min_abs_denominator_threshold:
        logger.debug(f"Open price {open_p} troppo vicino a zero. Metriche % saranno None.")
        return metrics # Restituisce None se open è troppo piccolo

    # --- Calcolo con Controllo Post-Calcolo ---
    try:
        # Calcola i valori intermedi
        range_val = high_p - low_p
        min_close_val = close_p - low_p
        open_max_val = high_p - open_p
        open_min_val = open_p - low_p
        body_val = abs(close_p - open_p)

        # Calcola le percentuali e controlla NaN/inf dopo ogni calcolo
        metrics['ic_range_pct'] = _safe_float((range_val / open_p) * 100)
        metrics['ic_min_max_pct'] = metrics['ic_range_pct'] # Alias

        metrics['ic_min_close_pct'] = _safe_float((min_close_val / open_p) * 100)
        metrics['ic_open_max_pct'] = _safe_float((open_max_val / open_p) * 100)
        metrics['ic_open_min_pct'] = _safe_float((open_min_val / open_p) * 100)
        metrics['ic_body_pct'] = _safe_float((body_val / open_p) * 100)

        # Controllo finale esplicito per NaN/inf (anche se _safe_float dovrebbe già gestirli)
        for k, v in metrics.items():
            if v is not None and (np.isnan(v) or np.isinf(v)):
                logger.warning(f"Risultato NaN/inf per metrica '{k}' (O={open_p}, H={high_p}, L={low_p}, C={close_p}). Imposto a None.")
                metrics[k] = None

    except ZeroDivisionError:
         # Questo non dovrebbe accadere grazie al controllo su open_p, ma per sicurezza
         logger.error(f"Errore ZeroDivisionError inatteso (O={open_p}). Metriche saranno None.")
         return {k: None for k in metrics}
    except Exception as e:
        logger.error(f"Errore nel calcolo metriche candela (O:{open_p}, H:{high_p}, L:{low_p}, C:{close_p}): {e}", exc_info=False)
        return {k: None for k in metrics}

    return metrics
# --- FINE MODIFICA ---


# --- JSON Serialization Helper ---
def convert_to_json_serializable(obj: Any) -> Any:
    """Converte ricorsivamente un oggetto in un formato serializzabile JSON."""
    # Gestione Tipi Base e Speciali
    if obj is None or obj is pd.NA: return None
    if isinstance(obj, (str, bool)): return obj
    if isinstance(obj, (int, np.integer)): return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    # Usa safe_strftime per date/timestamps
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return safe_strftime(obj, fmt="%Y-%m-%dT%H:%M:%SZ", fallback=str(obj))

    # Gestione Strutture Dati Comuni
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            try:
                serial_k = convert_to_json_serializable(k)
                if not isinstance(serial_k, (str, int, float, bool)) and serial_k is not None:
                    # logger.warning(f"Chiave dizionario tipo {type(k)} convertita in stringa: '{str(k)}'.") # Log troppo verboso
                    serial_k = str(k)
                new_dict[serial_k] = convert_to_json_serializable(v)
            except Exception as e_dict:
                logger.warning(f"Errore serializzazione elemento dict (chiave '{k}'): {e_dict}. Salto elemento.")
        return new_dict
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_json_serializable(item) for item in obj]

    # Gestione Tipi Specifici Librerie (Pandas/Numpy)
    if isinstance(obj, (np.ndarray, pd.Index)):
        return [convert_to_json_serializable(item) for item in obj.tolist()]
    if isinstance(obj, pd.Series):
        try:
            # Verifica se l'indice è semplice (numerico o stringa) e unico
            is_simple_index = pd.api.types.is_numeric_dtype(obj.index.dtype) or pd.api.types.is_string_dtype(obj.index.dtype)
            if is_simple_index and obj.index.is_unique:
                 # Converte in dizionario se l'indice è semplice e unico
                 valid_items = {convert_to_json_serializable(k): convert_to_json_serializable(v)
                                for k, v in obj.items() if pd.notna(k) and k is not None} # Usa .items() per Series
                 return valid_items
            else: # Altrimenti, converti in lista
                return [convert_to_json_serializable(item) for item in obj.tolist()]
        except Exception as e_series:
            logger.warning(f"Errore conversione Series JSON: {e_series}. Uso tolist().")
            return [convert_to_json_serializable(item) for item in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        try:
            # Orient='records' è generalmente buono per LLM
            df_serializable = obj.replace([np.inf, -np.inf], None) # Sostituisci Inf con None prima di serializzare
            # Converte l'indice se non è di tipo base prima di to_dict
            if not pd.api.types.is_numeric_dtype(df_serializable.index.dtype) and \
               not pd.api.types.is_string_dtype(df_serializable.index.dtype) and \
               not isinstance(df_serializable.index, pd.DatetimeIndex):
                 df_serializable.index = df_serializable.index.astype(str)
            # Resetta l'indice se è DatetimeIndex per includerlo nei record
            if isinstance(df_serializable.index, pd.DatetimeIndex):
                 df_serializable = df_serializable.reset_index()
                 # Rinomina la colonna indice se necessario (es. 'index' o 'timestamp')
                 if 'index' in df_serializable.columns:
                      df_serializable = df_serializable.rename(columns={'index': 'timestamp_index'})

            return [convert_to_json_serializable(row) for row in df_serializable.to_dict(orient='records')]
        except Exception as e_df:
            logger.warning(f"Errore conversione DataFrame JSON: {e_df}.")
            return "Error: DataFrame non serializzabile"

    # Fallback Generale
    try:
        # Prova a convertire tipi numpy specifici che potrebbero non essere gestiti sopra
        if hasattr(obj, 'item') and callable(obj.item):
             # item() converte tipi scalari numpy (es. np.float64) in tipi Python nativi
            return convert_to_json_serializable(obj.item())
        return str(obj) # Ultimo tentativo
    except Exception as e_fallback:
        logger.warning(f"Oggetto tipo {type(obj)} non serializzabile JSON (fallback fallito: {e_fallback}).")
        return f"Error: Non-serializable type {type(obj)}"


# --- Funzioni per Salvataggio Risultati Analisi (JSON/TXT) ---
# (save_results_to_json, save_results_to_txt rimangono invariate)
def save_results_to_json(results: Dict, filename: str) -> bool:
    """Salva un dizionario di risultati in un file JSON, usando il converter robusto."""
    logger.info(f"Tentativo salvataggio risultati JSON in: {filename}")
    try:
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory creata: {dir_path}")
        serializable_results = convert_to_json_serializable(results)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Risultati JSON salvati con successo in {filename}")
        return True
    except IOError as e:
        logger.error(f"Errore I/O salvataggio JSON {filename}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Errore imprevisto salvataggio JSON {filename}: {e}", exc_info=True)
        return False

def save_results_to_txt(results: Dict, filename: str) -> bool:
    """
    Salva un dizionario di risultati in un file di testo (.txt),
    formattato come stringa JSON per preservare struttura e leggibilità LLM.
    """
    logger.info(f"Tentativo salvataggio risultati TXT in: {filename}")
    try:
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory creata: {dir_path}")
        serializable_results = convert_to_json_serializable(results)
        json_string = json.dumps(serializable_results, indent=4, ensure_ascii=False)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json_string)
        logger.info(f"Risultati TXT salvati con successo in {filename}")
        return True
    except IOError as e:
        logger.error(f"Errore I/O salvataggio TXT {filename}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Errore imprevisto salvataggio TXT {filename}: {e}", exc_info=True)
        return False


# --- Funzioni per Salvataggio/Caricamento Statistiche Storiche (Cicli, Intra-Candela) ---
def save_historical_stats_to_json(stats_data: Dict, filename: str) -> bool:
    """
    Salva le statistiche storiche (es. cicli, intra-candela) in un file JSON.
    Utilizza la stessa logica di serializzazione robusta.
    """
    # Log più generico
    logger.info(f"Tentativo salvataggio statistiche storiche JSON in: {filename}")
    try:
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory per statistiche storiche creata: {dir_path}")
        serializable_stats = convert_to_json_serializable(stats_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=4, ensure_ascii=False)
        logger.info(f"Statistiche storiche JSON salvate con successo in {filename}")
        return True
    except IOError as e:
        logger.error(f"Errore I/O salvataggio statistiche storiche JSON {filename}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Errore imprevisto salvataggio statistiche storiche JSON {filename}: {e}", exc_info=True)
        return False

def load_historical_stats_from_json(filename: str) -> Optional[Dict]:
    """
    Carica le statistiche storiche (es. cicli, intra-candela) da un file JSON.
    """
    # Log più generico
    logger.debug(f"Tentativo caricamento statistiche storiche JSON da: {filename}")
    if not os.path.exists(filename):
        logger.warning(f"File statistiche storiche non trovato: {filename}")
        return None
    if not os.path.isfile(filename):
        logger.error(f"Il percorso specificato per le statistiche storiche non è un file: {filename}")
        return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        if not isinstance(stats_data, dict):
             logger.error(f"Il contenuto del file JSON {filename} non è un dizionario valido.")
             return None
        logger.debug(f"Statistiche storiche caricate con successo da {filename}")
        return stats_data
    except json.JSONDecodeError as e:
        logger.error(f"Errore parsing JSON durante caricamento statistiche storiche da {filename}: {e}", exc_info=True)
        return None
    except IOError as e:
        logger.error(f"Errore I/O caricamento statistiche storiche JSON {filename}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Errore imprevisto caricamento statistiche storiche JSON {filename}: {e}", exc_info=True)
        return None

# --- Funzioni per Storico Deribit (load_deribit_history, save_deribit_history rimangono invariate) ---
def load_deribit_history(filename: str) -> List[Dict[str, Any]]:
    """
    Carica la storia dei dati aggregati Deribit da un file JSON.

    Args:
        filename (str): Percorso del file JSON storico.

    Returns:
        List[Dict[str, Any]]: Lista di dizionari storici,
                              o lista vuota se file non esiste o errore.
    """
    logger.debug(f"Tentativo caricamento storia Deribit da: {filename}")
    history = []
    if not os.path.exists(filename) or not os.path.isfile(filename):
        logger.info(f"File storico Deribit non trovato o non è un file: {filename}. Restituita storia vuota.")
        return history

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        # Verifica che sia una lista di dizionari
        if isinstance(loaded_data, list) and all(isinstance(item, dict) for item in loaded_data):
            history = loaded_data
            logger.info(f"Caricati {len(history)} record storici Deribit da {filename}")
        else:
            logger.error(f"Contenuto del file storico Deribit {filename} non è una lista di dizionari valida.")
    except json.JSONDecodeError as e:
        logger.error(f"Errore parsing JSON storico Deribit {filename}: {e}", exc_info=True)
    except IOError as e:
        logger.error(f"Errore I/O caricamento storico Deribit {filename}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Errore imprevisto caricamento storico Deribit {filename}: {e}", exc_info=True)

    return history # Ritorna la lista (vuota in caso di errore)

def save_deribit_history(
    history_data: List[Dict[str, Any]],
    new_entry: Dict[str, Any],
    filename: str,
    max_entries: int = 180 # Mantieni circa 6 mesi di dati giornalieri
) -> bool:
    """
    Aggiunge una nuova entry alla storia Deribit, rimuove le più vecchie
    se si supera max_entries, e salva il file JSON.

    Args:
        history_data (List[Dict[str, Any]]): La storia attuale caricata.
        new_entry (Dict[str, Any]): Il nuovo dizionario da aggiungere.
                                     Dovrebbe contenere un campo 'timestamp'.
        filename (str): Percorso del file JSON storico.
        max_entries (int): Numero massimo di record da mantenere.

    Returns:
        bool: True se il salvataggio ha avuto successo, False altrimenti.
    """
    logger.debug(f"Tentativo salvataggio storia Deribit in: {filename} (Max Entries: {max_entries})")

    if not isinstance(history_data, list):
        logger.error("Formato history_data non valido per salvataggio storico Deribit (non è una lista).")
        return False
    if not isinstance(new_entry, dict) or 'timestamp' not in new_entry:
        logger.error("Nuova entry per storico Deribit non valida (non dict o manca 'timestamp').")
        return False

    # Aggiungi la nuova entry
    history_data.append(new_entry)

    # Ordina per timestamp (dal più vecchio al più recente) - Assicurati che il timestamp sia confrontabile
    try:
        # Usa il timestamp UNIX (in secondi) per l'ordinamento robusto
        history_data.sort(key=lambda x: x.get('timestamp', 0) / 1000 if isinstance(x.get('timestamp'), (int, float)) else 0)
    except TypeError as e:
        logger.error(f"Errore di tipo durante l'ordinamento dello storico Deribit per timestamp: {e}. Il salvataggio potrebbe non troncare correttamente.")
        # Non bloccare il salvataggio, ma logga l'errore

    # Mantieni solo le ultime max_entries
    if len(history_data) > max_entries:
        logger.info(f"Storico Deribit supera {max_entries} entries ({len(history_data)}). Tronco le più vecchie.")
        history_data = history_data[-max_entries:]

    # Ora salva la lista aggiornata (usa la stessa logica di save_results_to_json)
    try:
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory per storico Deribit creata: {dir_path}")

        # Assicurati che i dati siano serializzabili (anche se dovrebbero già esserlo)
        serializable_history = convert_to_json_serializable(history_data)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False) # Indent 2 per file più compatti

        logger.info(f"Storia Deribit ({len(history_data)} entries) salvata con successo in {filename}")
        return True
    except IOError as e:
        logger.error(f"Errore I/O salvataggio storico Deribit {filename}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Errore imprevisto salvataggio storico Deribit {filename}: {e}", exc_info=True)
        return False

# --- Altri Helper ---
def get_date_range(start_date: Any, end_date: Any = None, fmt: str = '%Y-%m-%d') -> str:
    """Formatta un intervallo di date in stringa."""
    try:
        # Usa safe_strftime che ora gestisce diversi tipi di input
        start_str = safe_strftime(start_date, fmt, fallback="N/A")
        # Usa il timestamp corrente se end_date è None
        end_ts = end_date if end_date is not None else datetime.now(timezone.utc)
        end_str = safe_strftime(end_ts, fmt, fallback="N/A")
        return f"{start_str} to {end_str}"
    except Exception as e:
        logger.error(f"Errore formattazione date range: {e}")
        return f"{start_date} to {end_date}" # Fallback grezzo

logger.debug("Modulo statistical_analyzer_helpers.py caricato.")
# --- END OF FILE statistical_analyzer_helpers.py ---