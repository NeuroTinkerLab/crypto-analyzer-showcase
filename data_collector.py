# --- START OF FILE data_collector.py ---
# data_collector.py
"""
Modulo responsabile del recupero e della gestione (caching)
dei dati storici OHLCV da exchange tramite ccxt.
Include logica robusta per la gestione della cache, staleness check,
retry su errori API e preparazione dati base.
"""
import os
import time
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union, List
from datetime import datetime, timedelta, timezone
import pyarrow as pa # type: ignore
import pyarrow.parquet as pq # type: ignore
# --- MODIFICA IMPORT FREQUENCIES ---
# Usa import diretto da pandas.tseries
from pandas.tseries import frequencies
# --- FINE MODIFICA ---
import re # Aggiunto per fallback timeframe
import math # Aggiunto per fallback timeframe

def safe_strftime(dt, fmt, fallback='N/A'):
    try:
        return dt.strftime(fmt)
    except Exception:
        return fallback

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

# Importa da config in modo sicuro
try:
    from config import (
        API_TIMEOUT, CACHE_DIR, CACHE_EXPIRY, DATA_LOOKBACK_PERIODS as BACKTEST_PERIODS, # Usa nuovo nome
        API_KEY, API_SECRET, FALLBACK_START_YEAR
    )
    if 'FALLBACK_START_YEAR' not in locals(): FALLBACK_START_YEAR = 2017 # Fallback anno default
    if 'BACKTEST_PERIODS' not in locals() or not isinstance(BACKTEST_PERIODS, dict): # Fallback lookback se manca
        logging.warning("DATA_LOOKBACK_PERIODS non trovato in config. Uso fallback.")
        BACKTEST_PERIODS = {tf: 365 for tf in ['1h', '4h', '1d']}
        BACKTEST_PERIODS.update({'1w': 1825, '1M': 2555, '5m': 14, '15m': 60}) # Lookback aumentati

except ImportError:
    logging.warning("config.py non trovato o errore import. Uso valori di default per DataCollector.")
    API_TIMEOUT = 30
    CACHE_DIR = "cache_analysis" # Usa nome cache aggiornato
    CACHE_EXPIRY = 3600 * 4 # Usa scadenza aggiornata
    # Fallback periodi essenziali
    BACKTEST_PERIODS = {tf: 365 for tf in ['1h', '4h', '1d']}
    BACKTEST_PERIODS.update({'1w': 1825, '1M': 2555, '5m': 14, '15m': 60}) # Lookback aumentati
    API_KEY = None
    API_SECRET = None
    FALLBACK_START_YEAR = 2018 # Usa anno aggiornato


import ccxt  # type: ignore

logger = logging.getLogger(__name__)

# --- Singleton Leggero per l'Istanza Exchange (INVARIATO)---
_exchange_instance: Optional[ccxt.Exchange] = None
_exchange_initialization_failed: bool = False
def _initialize_global_exchange() -> Optional[ccxt.Exchange]:
    global _exchange_instance, _exchange_initialization_failed
    if _exchange_instance is None and not _exchange_initialization_failed:
        logger.info("Inizializzazione istanza globale ccxt.Exchange (Binance)...")
        try:
            exchange_options = {'timeout': API_TIMEOUT * 1000, 'options': {'adjustForTimeDifference': True,}, 'enableRateLimit': True,}
            api_key_to_use = API_KEY if isinstance(API_KEY, str) and API_KEY and "YOUR_API_KEY" not in API_KEY else None
            api_secret_to_use = API_SECRET if isinstance(API_SECRET, str) and API_SECRET and "YOUR_SECRET_KEY" not in API_SECRET else None
            if api_key_to_use and api_secret_to_use: exchange_options['apiKey'] = api_key_to_use; exchange_options['secret'] = api_secret_to_use; logger.info("Istanza globale ccxt: Utilizzo API Key fornita.")
            else: logger.warning("Istanza globale ccxt: API Key/Secret non trovate o placeholder. Utilizzo API pubblica.")
            exchange_id = 'binance'; exchange_class = getattr(ccxt, exchange_id); exchange = exchange_class(exchange_options)
            exchange.load_markets(reload=False); logger.info(f"Istanza globale ccxt.{exchange.id} inizializzata e mercati caricati.")
            _exchange_instance = exchange
        except (ccxt.AuthenticationError) as e: logger.error(f"Errore Autenticazione Globale ccxt: {e}."); _exchange_initialization_failed = True; _exchange_instance = None
        except (ccxt.ExchangeError, ccxt.NetworkError, ccxt.RequestTimeout) as e: logger.error(f"Errore Inizializzazione/Caricamento Mercati Globale ccxt: {e}", exc_info=False); _exchange_initialization_failed = True; _exchange_instance = None
        except Exception as e: logger.error(f"Errore Sconosciuto Inizializzazione Globale ccxt: {e}", exc_info=True); _exchange_initialization_failed = True; _exchange_instance = None
    elif _exchange_initialization_failed: logger.debug("Inizializzazione globale ccxt fallita precedentemente. Salto.")
    return _exchange_instance
def get_exchange_instance() -> Optional[ccxt.Exchange]: return _initialize_global_exchange()


class DataCollector:
    """Recupera e gestisce i dati storici OHLCV con caching e verifica aggiornamento."""
    FETCH_LIMIT = 1000; FALLBACK_START_DATE = datetime(FALLBACK_START_YEAR, 1, 1, tzinfo=timezone.utc); MAX_FETCH_RETRIES = 3

    def __init__(self):
        """Inizializza il DataCollector."""
        self.exchange = get_exchange_instance()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Crea la directory cache se non esiste."""
        try: os.makedirs(CACHE_DIR, exist_ok=True)
        except OSError as e: logger.error(f"Impossibile creare cache dir {CACHE_DIR}: {e}", exc_info=True)

    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        """Genera il percorso del file cache Parquet."""
        safe_symbol = symbol.replace('/', '_').replace('\\', '_').replace(':', '_'); filename = f"{safe_symbol}_{timeframe}.parquet"; return os.path.join(CACHE_DIR, filename)

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Controlla se la cache esiste, non è scaduta e non è corrotta."""
        if not os.path.exists(cache_path):
            return False
        try:
            file_mod_time = os.path.getmtime(cache_path)
            if (time.time() - file_mod_time) >= CACHE_EXPIRY:
                logger.debug(f"Cache scaduta: {cache_path}")
                return False
            pq.read_metadata(cache_path)
            return True
        except OSError as e:
            logger.warning(f"Errore OS accesso cache {cache_path}: {e}")
            return False
        except (pa.ArrowIOError, pa.ArrowInvalid, Exception) as e:
            logger.warning(f"Cache file {cache_path} corrotto: {str(e)}. Rimuovo.")
            try:
                os.remove(cache_path)
                logger.info(f"File cache corrotto rimosso: {cache_path}")
            except OSError as rm_e:
                logger.error(f"Impossibile rimuovere cache corrotta {cache_path}: {rm_e}")
            return False

    def _timeframe_to_milliseconds(self, timeframe: str) -> Optional[int]:
        """Converte una stringa timeframe (es. '1h', '1d') in millisecondi."""
        try:
            if timeframe == '1w': return 7 * 24 * 60 * 60 * 1000
            if timeframe == '1M': return 30 * 24 * 60 * 60 * 1000 # Stima
            # --- CORREZIONE TIMEDELTA ---
            # Mappa le abbreviazioni comuni a quelle che pd.Timedelta capisce meglio
            # --- CORREZIONE H -> h ---
            tf_map = {'m': 'min', 'h': 'h', 'd': 'D', 'w': 'W', 'M': 'D'} # Mappa M a D (stima 30 giorni)
            match = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
            if match:
                val = match.group(1)
                unit = match.group(2).lower()
                pd_unit = tf_map.get(unit)
                if pd_unit:
                     # Gestione speciale per M (stima 30 giorni)
                     if pd_unit == 'D' and unit == 'm': # Se era '1M'
                          delta = pd.Timedelta(days=30 * int(val))
                     else:
                          # --- CORREZIONE H -> h ---
                          delta = pd.Timedelta(f"{val}{pd_unit}") # Usa pd_unit che ora è 'h' minuscolo se era 'h'
                else: # Prova direttamente se non mappato
                     delta = pd.Timedelta(timeframe)
            else: # Prova direttamente se non c'è numero (es. 'W')
                delta = pd.Timedelta(timeframe)
            # --- FINE CORREZIONE ---
            return int(delta.total_seconds() * 1000)
        except ValueError: logger.error(f"Formato timeframe non riconosciuto da pd.Timedelta: '{timeframe}'"); return None
        except Exception as e: logger.error(f"Errore _timeframe_to_milliseconds per '{timeframe}': {e}"); return None

    def _calculate_start_time_ms(self, timeframe: str, days_lookback: Optional[float] = None) -> int:
        """Calcola il timestamp di inizio (ms UTC) basato sul lookback o config."""
        if days_lookback is not None and days_lookback > 0: logger.debug(f"Usando lookback fornito: {days_lookback} giorni")
        else: days_lookback = BACKTEST_PERIODS.get(timeframe);
        if days_lookback is None or days_lookback <= 0: logger.warning(f"Lookback non valido per {timeframe}. Default 365."); days_lookback = 365
        else: logger.debug(f"Usando lookback config per {timeframe}: {days_lookback} giorni")
        max_days = 7 * 365; days_lookback = min(days_lookback, max_days)
        try: start_delta = timedelta(days=days_lookback); start_time = datetime.now(timezone.utc) - start_delta; start_ms = max(int(start_time.timestamp() * 1000), int(self.FALLBACK_START_DATE.timestamp() * 1000)); return start_ms
        except Exception as e: logger.error(f"Errore calcolo start time: {e}. Uso fallback."); return int(self.FALLBACK_START_DATE.timestamp() * 1000)

    # --- Metodo _get_expected_last_candle_close_time_ms CORRETTO ---
    def _get_expected_last_candle_close_time_ms(self, current_time_utc: datetime, timeframe: str) -> Optional[int]:
        """Calcola timestamp (ms UTC) chiusura ultima candela completa."""
        pd_freq_str : Optional[str] = None # Inizializza a None
        offset: Optional[pd.DateOffset] = None # Inizializza a None
        try:
            # --- CORREZIONE H -> h ---
            tf_map = {'m': 'min', 'h': 'h', 'd': 'D', 'w': 'W', 'M': 'ME'} # Usa 'h' minuscolo
            tf_unit = ""
            tf_val_str = ""

            # Tenta di estrarre valore e unità
            match_num_unit = re.match(r'(\d+)([a-zA-Z]+)', timeframe)
            match_unit_only = None # Inizializza a None qui

            if match_num_unit:
                tf_val_str = match_num_unit.group(1)
                tf_unit = match_num_unit.group(2).lower()
            else:
                # Prova con solo unità SOLO SE il primo match fallisce
                match_unit_only = re.match(r'([a-zA-Z]+)', timeframe)
                if match_unit_only: # Controlla se questo match ha avuto successo
                    tf_unit = match_unit_only.group(1).lower()
                    tf_val_str = "1"
                else:
                    logger.error(f"Formato timeframe non riconoscibile: '{timeframe}'")
                    return None

            # Tenta di ottenere l'offset
            pd_freq_unit = tf_map.get(tf_unit)
            if pd_freq_unit:
                pd_freq_str = f"{tf_val_str}{pd_freq_unit}" # pd_freq_unit sarà 'h' se timeframe era '1h'
                try:
                    # --- CORREZIONE H -> h ---
                    # Frequncy string is now correct ('1h', '4h', etc.)
                    offset = frequencies.to_offset(pd_freq_str)
                except ValueError as e_map:
                    logger.warning(f"Errore to_offset con freq mappata '{pd_freq_str}': {e_map}. Tento offset diretto.")
                    try: offset = frequencies.to_offset(timeframe)
                    except ValueError as e_direct: logger.error(f"Errore to_offset anche con freq diretta '{timeframe}': {e_direct}"); return None
            else:
                 logger.warning(f"Unità timeframe '{tf_unit}' non mappata, tento to_offset diretto con '{timeframe}'")
                 try: offset = frequencies.to_offset(timeframe)
                 except ValueError as e_unmapped: logger.error(f"Errore to_offset con freq non mappata '{timeframe}': {e_unmapped}"); return None

            if offset is None:
                 logger.error(f"Impossibile determinare offset pandas per '{timeframe}' (Str provata: '{pd_freq_str if pd_freq_str else timeframe}').")
                 return None

            # Determina l'inizio della candela CORRENTE (quella che include current_time_utc)
            # rollback trova l'inizio del periodo che CONTIENE il timestamp
            current_candle_start_time = offset.rollback(current_time_utc)

            # L'inizio della candela corrente corrisponde alla chiusura della candela *precedente completa*.
            # Esempio: se sono le 10:35 e il TF è 1h, current_candle_start_time è 10:00.
            # La candela completa precedente è quella che si è chiusa alle 10:00.
            expected_last_close_time = current_candle_start_time

            # Restituisci il timestamp in millisecondi
            return int(expected_last_close_time.timestamp() * 1000)

        except Exception as e:
            logger.error(f"Errore imprevisto _get_expected_last_candle_close_time_ms '{timeframe}': {e}", exc_info=True)
            return None
    # --- FINE CORREZIONE ---

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str,
        force_refresh: bool = False,
        start_time_ms: Optional[int] = None,
        days_lookback: Optional[float] = None,
        limit_override: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Recupera dati storici OHLCV, usando cache e verificando staleness."""
        # (Logica controllo cache invariata rispetto alla versione precedente corretta)
        cache_path = self._get_cache_path(symbol, timeframe); cache_is_stale = False; df_cached = None
        if not force_refresh and limit_override is None and self._is_cache_valid(cache_path):
            try:
                logger.debug(f"Cache valida {symbol} {timeframe}. Verifico agg..."); df_cached = pq.read_table(cache_path).to_pandas()
                if df_cached.empty: logger.warning(f"Cache {symbol} {timeframe} valida ma vuota."); cache_is_stale = True
                else:
                    if not isinstance(df_cached.index, pd.DatetimeIndex): df_cached.index = pd.to_datetime(df_cached.index, utc=True)
                    elif df_cached.index.tz is None or str(df_cached.index.tz).upper() != 'UTC': df_cached.index = df_cached.index.tz_localize('UTC') if df_cached.index.tz is None else df_cached.index.tz_convert('UTC')
                    cached_last_ts_ms = int(df_cached.index[-1].timestamp() * 1000)
                    expected_last_close_ms = self._get_expected_last_candle_close_time_ms(datetime.now(timezone.utc), timeframe)
                    if expected_last_close_ms is not None:
                        # Verifica se la cache è STALE (l'ultima candela in cache è ANTECEDENTE all'ultima candela COMPLETA attesa)
                        if cached_last_ts_ms < expected_last_close_ms:
                            logger.info(f"Cache {symbol} {timeframe} STALE (Ultima cache: {pd.Timestamp(cached_last_ts_ms, unit='ms', tz='UTC')}, Attesa: {pd.Timestamp(expected_last_close_ms, unit='ms', tz='UTC')}). Forza Refresh.")
                            cache_is_stale = True
                        else:
                            logger.info(f"Cache VALIDA e AGGIORNATA {symbol} {timeframe}.")
                            # Ulteriore controllo: la cache copre lo start_time richiesto?
                            if start_time_ms is not None:
                                cache_start_ms = int(df_cached.index.min().timestamp() * 1000)
                                if cache_start_ms > start_time_ms:
                                    logger.info(f"Cache non copre start_time richiesto ({pd.Timestamp(start_time_ms, unit='ms', tz='UTC')}). Riscarico.")
                                    cache_is_stale = True
                                else: return df_cached # Cache valida, aggiornata e copre il periodo
                            else: return df_cached # Cache valida e aggiornata, nessun start_time specifico richiesto
                    else:
                         logger.warning(f"Cache {symbol} {timeframe}: Impossibile calc timestamp atteso. Uso cache.")
                         # Controlla comunque start_time
                         if start_time_ms is not None:
                             cache_start_ms = int(df_cached.index.min().timestamp() * 1000)
                             if cache_start_ms > start_time_ms:
                                logger.info(f"Cache (staleness N/A) non copre start_time richiesto ({pd.Timestamp(start_time_ms, unit='ms', tz='UTC')}). Riscarico.")
                                cache_is_stale = True
                             else: return df_cached
                         else: return df_cached
            except Exception as e: logger.warning(f"Errore lettura/verifica cache {cache_path}: {e}. Riscarico."); cache_is_stale = True
        elif limit_override is not None and not force_refresh and not cache_is_stale and self._is_cache_valid(cache_path):
             logger.info(f"Cache valida {symbol} {timeframe}, limit={limit_override}. Carico e taglio.")
             try:
                 df_cached_limit = pq.read_table(cache_path).to_pandas()
                 if not df_cached_limit.empty and isinstance(df_cached_limit.index, pd.DatetimeIndex):
                     if df_cached_limit.index.tz is None or str(df_cached_limit.index.tz).upper() != 'UTC': df_cached_limit.index = df_cached_limit.index.tz_localize('UTC') if df_cached_limit.index.tz is None else df_cached_limit.index.tz_convert('UTC')
                     if len(df_cached_limit) >= limit_override: return df_cached_limit.iloc[-limit_override:]
                     else: logger.info(f"Cache < {limit_override} righe. Riscarico."); cache_is_stale = True
                 else: logger.warning(f"Cache {symbol} {timeframe} vuota/invalida per limit. Riscarico."); cache_is_stale = True
             except Exception as e: logger.warning(f"Errore lettura/taglio cache {cache_path} per limit: {e}. Riscarico."); cache_is_stale = True

        should_fetch = force_refresh or cache_is_stale or not self._is_cache_valid(cache_path)
        if not should_fetch:
             if df_cached is not None: logger.info(f"Restituzione dati cache non aggiornata {symbol} {timeframe}."); return df_cached
             else: logger.error(f"Nessuna cache valida e nessun fetch per {symbol} {timeframe}."); return None

        if should_fetch:
            # (Logica Fetch API, Retry, Elaborazione DF, Rimozione Candela Incompleta, Salvataggio Cache
            #  INVARIATA rispetto alla versione precedente corretta - include fix UnboundLocalError)
            if self.exchange is None: logger.error(f"Fetch {symbol}/{timeframe}: Exchange non disponibile."); return None
            try:
                if not self.exchange.markets: self.exchange.load_markets()
                if symbol not in self.exchange.markets: logger.error(f"Fetch: Simbolo '{symbol}' non supportato."); return None
                if timeframe not in self.exchange.timeframes: logger.error(f"Fetch: Timeframe '{timeframe}' non supportato."); return None
            except Exception as e: logger.error(f"Fetch {symbol}/{timeframe}: Errore verifica simbolo/TF: {e}."); return None
            fetch_reason = f"(Force={force_refresh}, Stale={cache_is_stale}, CacheValid={self._is_cache_valid(cache_path)})"; logger.info(f"Avvio download/refresh {symbol} {timeframe} {fetch_reason}.")
            effective_start_ms: Optional[int] = None
            if limit_override is not None: logger.info(f"Download ULTIME {limit_override} candele...")
            elif start_time_ms is not None: effective_start_ms = max(start_time_ms, int(self.FALLBACK_START_DATE.timestamp() * 1000)); logger.info(f"Download da timestamp: {datetime.fromtimestamp(effective_start_ms/1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S %Z}...")
            else: effective_start_ms = self._calculate_start_time_ms(timeframe, days_lookback); logger.info(f"Download da lookback auto: {datetime.fromtimestamp(effective_start_ms/1000, tz=timezone.utc):%Y-%m-%d %H:%M:%S %Z}...")
            all_klines: List[list] = []; current_since = effective_start_ms; tf_ms = self._timeframe_to_milliseconds(timeframe);
            if tf_ms is None: return None;
            # --- Definizione fetch_limit SPOSTATA QUI ---
            fetch_limit = limit_override if limit_override is not None else self.FETCH_LIMIT
            # --- FINE SPOSTAMENTO ---
            while True:
                retries = 0; klines = None;
                params = {'limit': fetch_limit} # Usa variabile definita sopra
                # Modifica: Passa 'since' solo se non è specificato limit_override
                if limit_override is None and current_since is not None: params['since'] = current_since
                elif limit_override is not None: params.pop('since', None) # Assicurati che 'since' non venga passato con 'limit'

                while retries <= self.MAX_FETCH_RETRIES:
                    try: logger.debug(f"API Req {symbol} {timeframe}: P={params} (T {retries+1})"); klines = self.exchange.fetch_ohlcv(symbol, timeframe, **params); break
                    except ccxt.RateLimitExceeded as e: logger.warning(f"Rate limit fetch {symbol} {timeframe}: {e}. Attesa..."); time.sleep((self.exchange.rateLimit / 1000 * 1.2) if hasattr(self.exchange, 'rateLimit') else 5)
                    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                        retries += 1
                        logger.warning(f"Net/Timeout fetch {symbol} {timeframe} (T {retries}): {e}")
                        if retries > self.MAX_FETCH_RETRIES: logger.error(f"Max tentativi Net/Timeout fetch {symbol} {timeframe}."); time.sleep(retries * 2); return None # Ritorna None qui
                        time.sleep(retries * 2) # Aggiunto sleep tra retry
                    except ccxt.ExchangeError as e: logger.error(f"Errore Exchange fetch {symbol} {timeframe}: {e}."); return None
                    except Exception as e: logger.error(f"Errore Sconosciuto fetch {symbol} {timeframe}: {e}", exc_info=True); return None
                if klines is None: logger.error(f"Fallito fetch klines {symbol} {timeframe}."); return None
                all_klines.extend(klines); logger.debug(f"Ricevuti {len(klines)} klines. Tot: {len(all_klines)}.")
                # Interrompi se abbiamo un limit, se non riceviamo più dati, o se i dati sono meno del limite fetch (siamo alla fine)
                if limit_override is not None or not klines or len(klines) < fetch_limit: break
                # Gestione loop 'since': trova l'ultimo timestamp e aggiungi il delta del timeframe
                last_ts = klines[-1][0]; next_since = last_ts + tf_ms;
                if next_since <= current_since: logger.warning(f"API non avanza TS per {symbol} {timeframe}. Ultimo TS: {last_ts}, Prossimo Since calcolato: {next_since}, Since Corrente: {current_since}. Interruzione."); break
                current_since = next_since
            if not all_klines: logger.warning(f"Nessun dato storico recuperato {symbol} {timeframe}."); return None
            df = pd.DataFrame(all_klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume']); initial_len = len(df)
            df = df.drop_duplicates(subset=['open_time'], keep='first')
            if len(df) < initial_len: logger.debug(f"Rimosse {initial_len - len(df)} duplicati (open_time).")
            numeric_columns = ["open", "high", "low", "close", "volume"]; df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True); df = df.set_index("timestamp"); df = df[numeric_columns]; df = df.sort_index(); df = df.dropna(subset=numeric_columns, how='any')
            if df.empty: logger.warning(f"DataFrame vuoto dopo processing {symbol} {timeframe}."); return None

            # Taglia il DataFrame se è stato specificato un limit_override e abbiamo scaricato più dati
            if limit_override is not None and len(df) > limit_override:
                logger.debug(f"Taglio DataFrame a {limit_override} candele finali.")
                df = df.iloc[-limit_override:]

            logger.debug(f"({symbol}|{timeframe}) Controllo candela incompleta post-fetch...")
            if not df.empty:
                last_candle_ts_obj = df.index[-1]; expected_close_ms = self._get_expected_last_candle_close_time_ms(datetime.now(timezone.utc), timeframe)
                if expected_close_ms is not None:
                     # Verifica se l'ultima candela è *dopo o uguale* alla chiusura attesa della candela completa precedente
                     # (es. se expected è 10:00, la candela delle 10:00 è quella incompleta)
                     expected_dt = pd.Timestamp(expected_close_ms, unit='ms', tz='UTC')
                     if last_candle_ts_obj >= expected_dt:
                         logger.info(f"({symbol}|{timeframe}) Rimozione candela incompleta post-fetch: TS={last_candle_ts_obj} >= ExpClose={expected_dt}")
                         df = df.iloc[:-1] # Rimuovi l'ultima riga
                         if df.empty: logger.warning(f"({symbol}|{timeframe}) Dati vuoti dopo rimozione candela incompleta."); return None
                     else: logger.debug(f"({symbol}|{timeframe}) Ultima candela {last_candle_ts_obj} è completa (precede exp close {expected_dt}).")
                else: logger.warning(f"({symbol}|{timeframe}) Impossibile det. chiusura attesa, candela incompleta non rimossa.")

            logger.info(f"Recuperati e preparati {len(df)} periodi unici per {symbol} {timeframe}.")
            # Salva in cache solo se NON è stato usato limit_override (perché la cache deve essere completa)
            if limit_override is None:
                try: self._ensure_cache_dir(); table = pa.Table.from_pandas(df, preserve_index=True); pq.write_table(table, cache_path); logger.info(f"Dati SALVATI/AGGIORNATI cache: {cache_path}")
                except Exception as e: logger.error(f"Impossibile SALVARE cache {cache_path}: {e}", exc_info=True)
            else: logger.debug(f"Limit override usato, non aggiorno cache completa {cache_path}.")
            return df
        else:
            logger.error(f"Stato imprevisto in fetch_historical_data per {symbol} {timeframe}."); return None

# --- Funzione wrapper get_data_with_cache (INVARIATA rispetto alla versione precedente corretta) ---
def get_data_with_cache( pair: str, timeframe: str, days: Optional[float] = None, force_fetch: bool = False, start_dt: Optional[datetime] = None, limit: Optional[int] = None ) -> Optional[pd.DataFrame]:
    # (Logica invariata)
    try:
        collector = DataCollector(); start_time_ms: Optional[int] = None; days_lookback_override: Optional[float] = days; limit_override: Optional[int] = limit
        if limit_override is not None:
            if limit_override <= 0: logger.error("'limit' > 0."); return None; logger.debug(f"get_data: limit={limit_override}."); start_time_ms = None; days_lookback_override = None
        elif start_dt is not None:
            if start_dt.tzinfo is None: start_dt = start_dt.replace(tzinfo=timezone.utc)
            elif start_dt.tzinfo != timezone.utc: start_dt = start_dt.astimezone(timezone.utc)
            start_time_ms = int(start_dt.timestamp() * 1000); logger.debug(f"get_data: start_dt={start_dt:%Y-%m-%d %H:%M:%S %Z}."); days_lookback_override = None
        elif days_lookback_override is not None:
            if days_lookback_override <= 0: logger.error("'days' > 0."); return None; logger.debug(f"get_data: days={days_lookback_override}."); start_time_ms = None
        else: logger.debug("get_data: Uso lookback automatico."); start_time_ms = None; days_lookback_override = None
        symbol_ccxt = pair
        if '/' not in pair and collector.exchange is not None and hasattr(collector.exchange, 'markets') and collector.exchange.markets:
            potential_symbol_usdt = f"{pair.upper()}/USDT"
            if potential_symbol_usdt in collector.exchange.markets: symbol_ccxt = potential_symbol_usdt
            else:
                 common_quotes = ['USDT', 'BUSD', 'FDUSD', 'USDC', 'TUSD', 'BTC', 'ETH', 'EUR', 'GBP']
                 found_match = False
                 for quote in common_quotes:
                     potential_symbol = f"{pair.upper()}/{quote}"
                     if potential_symbol in collector.exchange.markets:
                         symbol_ccxt = potential_symbol
                         found_match = True
                         break
                 if not found_match: logger.warning(f"Impossibile normalizzare '{pair}'.")
            if symbol_ccxt != pair: logger.debug(f"Simbolo normalizzato: '{pair}' -> '{symbol_ccxt}'")
        elif '/' not in pair: logger.warning(f"Exchange/mercati non caricati, formato simbolo '{pair}' potrebbe fallire."); symbol_ccxt = f"{pair.upper()}/USDT"
        data = collector.fetch_historical_data(symbol=symbol_ccxt, timeframe=timeframe, force_refresh=force_fetch, start_time_ms=start_time_ms, days_lookback=days_lookback_override, limit_override=limit_override)
        return data
    except Exception as e: logger.error(f"Errore imprevisto get_data_with_cache: {e}", exc_info=True); return None

# --- Funzione get_last_n_candles_multiple_tf (INVARIATA rispetto alla versione precedente corretta) ---
def get_last_n_candles_multiple_tf( symbol: str, timeframes: List[str] = ['1d', '1w', '1M'], n_candles: int = 3, force_fetch: bool = False ) -> Dict[str, List[Dict[str, Any]]]:
    # (Logica invariata)
    results: Dict[str, List[Dict[str, Any]]] = {tf: [] for tf in timeframes}; logger.info(f"Recupero DWM {n_candles} candele per {symbol} - TF: {timeframes}"); limit_fetch = n_candles + 5
    for tf in timeframes:
        try:
            logger.debug(f"Recupero DWM {tf} per {symbol} (limit={limit_fetch}). Force={force_fetch}")
            # Usa limit per fetch solo N candele
            df_tf = get_data_with_cache(pair=symbol, timeframe=tf, limit=limit_fetch, force_fetch=force_fetch)
            if df_tf is None or df_tf.empty: raise ValueError(f"Nessun dato DWM per {tf}.")
            # Prendi le ultime 'n_candles' tra quelle recuperate
            candles_to_extract = min(n_candles, len(df_tf));
            if candles_to_extract < n_candles: logger.warning(f"Recuperate solo {candles_to_extract}/{n_candles} candele DWM per {tf}.")
            df_last_n = df_tf.iloc[-candles_to_extract:]; ohlc_list = []
            for timestamp, row in df_last_n.iterrows():
                ts_str = safe_strftime(timestamp, '%Y-%m-%d %H:%M:%S', fallback='N/A')
                ohlc_list.append({'timestamp_utc': ts_str, 'open': _safe_float(row.get('open')), 'high': _safe_float(row.get('high')), 'low': _safe_float(row.get('low')), 'close': _safe_float(row.get('close')), 'volume': _safe_float(row.get('volume'))})
            results[tf] = ohlc_list; logger.debug(f"Estratte {len(ohlc_list)} candele DWM per {tf}.")
        except Exception as e: logger.error(f"Errore recupero/elaborazione DWM {tf} per {symbol}: {e}", exc_info=False); results[tf] = [{'error': str(e)}]
    return results

# --- END OF FILE data_collector.py ---