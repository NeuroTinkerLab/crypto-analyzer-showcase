# --- START OF FILE deribit_collector.py ---
import requests
import logging
import time
from typing import Optional, Dict, Any, List, Union
import json
import re # Aggiunto per parsing nomi strumenti
from datetime import datetime, timezone # Aggiunto per timestamp scadenza

# ... (Import helper e costanti come prima) ...
try:
    from statistical_analyzer_helpers import _safe_float, safe_strftime
except ImportError:
    logging.warning("Import statistical_analyzer_helpers fallito in deribit_collector. Uso fallback.")
    def _safe_float(value, default=None):
        try:
            if value is None: return default
            return float(value)
        except (ValueError, TypeError):
            return default
    def safe_strftime(dt, fmt, fallback='N/A'): # Fallback anche per strftime
        try:
            # Gestisce sia datetime che timestamp in ms
            if isinstance(dt, (int, float)):
                dt_obj = datetime.fromtimestamp(dt / 1000, tz=timezone.utc)
            elif isinstance(dt, datetime):
                 dt_obj = dt
            else: return fallback
            return dt_obj.strftime(fmt)
        except Exception:
            return fallback


logger = logging.getLogger(__name__)
DERIBIT_API_URL = "https://www.deribit.com/api/v2"
API_TIMEOUT_SECONDS = 15 # Potrebbe essere necessario aumentarlo per opzioni

# --- _make_deribit_public_request (invariato) ---
def _make_deribit_public_request(endpoint_path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Union[Dict[str, Any], List[Any]]]:
    url = f"{DERIBIT_API_URL}{endpoint_path}"
    retries = 2 # Magari aumentare a 3 per le opzioni?
    last_exception_obj = None
    for attempt in range(retries):
        try:
            prepared_request = requests.Request('GET', url, params=params).prepare()
            final_url = prepared_request.url
            logger.debug(f"Deribit API Request: GET {final_url} (Attempt {attempt + 1}/{retries})")
            # Timeout aumentato leggermente per chiamate potenzialmente più pesanti
            response = requests.get(url, params=params, timeout=API_TIMEOUT_SECONDS + 5)
            if not response.ok:
                 last_exception_obj = response
                 try:
                     error_data = response.json()
                     if isinstance(error_data, dict) and 'error' in error_data:
                          error_details = error_data['error']
                          api_error_msg = f"API Error {error_details.get('code')}: {error_details.get('message')} {error_details.get('data') or ''}".strip()
                          logger.error(f"Errore API Deribit ({endpoint_path}): {api_error_msg}")
                          return error_data
                     else: response.raise_for_status()
                 except (requests.exceptions.JSONDecodeError, AttributeError, KeyError):
                     logger.error(f"Errore HTTP {response.status_code} da Deribit ({endpoint_path}). Corpo: {response.text[:200]}")
                     response.raise_for_status()
                 break # Esce dal loop di retry se c'è stato un errore HTTP gestito
            data = response.json()
            # Verifica la struttura JSON-RPC standard di Deribit
            if isinstance(data, dict) and 'result' in data:
                logger.debug(f"Deribit API Success: GET {endpoint_path}")
                return data['result'] # Ritorna solo il campo 'result'
            elif isinstance(data, dict) and 'error' in data:
                 error_details = data['error']
                 api_error_msg = f"API Error {error_details.get('code')}: {error_details.get('message')} {error_details.get('data') or ''}".strip()
                 logger.error(f"Errore API Deribit (JSON-RPC in 2xx OK) ({endpoint_path}): {api_error_msg}")
                 return data # Ritorna l'intero dizionario errore
            else:
                 # Risposta inattesa ma OK, ritorna come errore strutturato
                 logger.warning(f"Risposta API Deribit 2xx OK ma struttura JSON-RPC inattesa ({endpoint_path}): {str(data)[:200]}...")
                 return {"error": {"message": f"Unexpected success JSON-RPC structure: {str(data)[:100]}"}} # Struttura come errore API

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout API Deribit ({endpoint_path}) (Attempt {attempt + 1}/{retries})")
            last_exception_obj = requests.exceptions.Timeout("Request Timeout")
            time.sleep(1.5 * (attempt + 1)) # Aumenta leggermente sleep
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore Richiesta API Deribit ({endpoint_path}) (Attempt {attempt + 1}/{retries}): {e}")
            last_exception_obj = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            logger.error(f"Errore Generico Richiesta Deribit ({endpoint_path}) (Attempt {attempt + 1}/{retries}): {e}", exc_info=True)
            last_exception_obj = e
            break # Esce dal loop per errori generici

    # Gestione errore dopo i tentativi
    error_message = f"{type(last_exception_obj).__name__}: {str(last_exception_obj)}" if last_exception_obj else "Unknown Error after retries"
    # Prova a estrarre un errore API strutturato se l'ultimo errore era una response
    if isinstance(last_exception_obj, requests.Response):
         try:
             error_data = last_exception_obj.json()
             if isinstance(error_data, dict) and 'error' in error_data:
                 return error_data # Ritorna l'errore API strutturato
         except: pass # Ignora errori nel parsing dell'errore
    # Ritorna un errore generico strutturato
    return {"error": {"message": error_message}}

# --- find_instrument_details (invariato) ---
def find_instrument_details(currency: str, kind: str = "future", expired: bool = False) -> List[Dict[str, Any]]:
    logger.info(f"Ricerca strumenti Deribit per: currency={currency}, kind={kind}, expired={expired}")
    endpoint_path = "/public/get_instruments"
    params = {"currency": currency, "kind": kind, "expired": str(expired).lower()}
    result = _make_deribit_public_request(endpoint_path=endpoint_path, params=params)

    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error"); api_error_msg_str = f"{error_msg}"
        # Assicurati che l'errore sia una stringa per l'output
        if isinstance(error_msg, dict):
            api_error_msg_str = f"API Error {error_msg.get('code')}: {error_msg.get('message')} {error_msg.get('data') or ''}".strip()
        elif not isinstance(error_msg, str):
             api_error_msg_str = str(error_msg)
        logger.error(f"Errore API durante la ricerca strumenti Deribit per {currency}: {api_error_msg_str}")
        return [{"error": api_error_msg_str}] # Lista con dict errore

    elif result is None:
        logger.error(f"Chiamata API ricerca strumenti Deribit {currency} ha restituito None.")
        return [{"error": "API call returned None"}] # Lista con dict errore

    elif not isinstance(result, list):
        logger.error(f"Risposta ricerca strumenti Deribit {currency} non è una lista: {type(result)}")
        return [{"error": f"Unexpected API response type: {type(result)}"}] # Lista con dict errore

    # Se result è una lista (caso successo), la ritorna direttamente
    logger.debug(f"Trovati {len(result)} strumenti Deribit per {currency}, kind={kind}")
    return result # Ritorna la lista di strumenti


# --- get_deribit_ticker_data (invariato) ---
def get_deribit_ticker_data(instrument_name: str) -> Optional[Dict[str, Any]]:
    logger.info(f"Recupero Book Summary (ticker substitute) Deribit per: {instrument_name}")
    endpoint_path = "/public/get_book_summary_by_instrument"
    params = {"instrument_name": instrument_name}
    result = _make_deribit_public_request(endpoint_path=endpoint_path, params=params)

    # Gestione errore robusta
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error"); api_error_msg_str = f"{error_msg}"
        if isinstance(error_msg, dict):
            api_error_msg_str = f"API Error {error_msg.get('code')}: {error_msg.get('message')} {error_msg.get('data') or ''}".strip()
        elif not isinstance(error_msg, str):
            api_error_msg_str = str(error_msg)
        logger.error(f"Errore API Deribit (get_book_summary per {instrument_name}): {api_error_msg_str}")
        return {"error": api_error_msg_str} # Ritorna dict errore

    elif result is None:
        return {"error": "API call returned None unexpectedly"} # Ritorna dict errore

    elif not isinstance(result, list):
         logger.warning(f"Tipo risultato Book Summary Deribit inatteso (non lista): {type(result)}")
         return {"error": f"Unexpected book summary data type: {type(result)}"} # Ritorna dict errore

    elif not result:
         logger.warning(f"Book Summary Deribit vuoto per {instrument_name}.")
         return {"error": f"Empty book summary list for {instrument_name}"} # Ritorna dict errore

    # Estrai il primo (e unico) elemento della lista
    summary = result[0]
    if not isinstance(summary, dict):
         logger.warning(f"Elemento Book Summary Deribit non è un dizionario: {type(summary)}")
         return {"error": f"Unexpected item type in book summary list: {type(summary)}"} # Ritorna dict errore

    # Estrazione dati (come prima)
    ticker_data = {
        "instrument": instrument_name,
        "open_interest_usd": _safe_float(summary.get("open_interest")), # Già in USD per i book summary
        "estimated_delivery_price": _safe_float(summary.get("estimated_delivery_price")), # aka index_price
        "mark_price": _safe_float(summary.get("mark_price")),
        "volume_24h": _safe_float(summary.get("volume")), # Volume in numero contratti base (e.g. BTC)
        "volume_24h_usd": _safe_float(summary.get("volume_usd")), # Volume in USD
        "mark_iv_pct": _safe_float(summary.get("mark_iv")), # In percentuale (es. 65.3)
        "last_price": _safe_float(summary.get("last")),
        "bid_price": _safe_float(summary.get("bid_price")),
        "ask_price": _safe_float(summary.get("ask_price")),
        "underlying_index": summary.get("underlying_index"), # Es. 'BTC-USD'
        "underlying_price": _safe_float(summary.get("underlying_price")), # Prezzo spot/index sottostante
        "interest_rate": _safe_float(summary.get("interest_rate")),
        "error": None
    }
    # Calcolo basis (come prima)
    if ticker_data["mark_price"] is not None and ticker_data["estimated_delivery_price"] is not None:
        ticker_data["basis_usd"] = ticker_data["mark_price"] - ticker_data["estimated_delivery_price"]
        if ticker_data["estimated_delivery_price"] != 0:
             ticker_data["basis_pct"] = (ticker_data["basis_usd"] / ticker_data["estimated_delivery_price"]) * 100
        else: ticker_data["basis_pct"] = None
    else: ticker_data["basis_usd"] = None; ticker_data["basis_pct"] = None

    logger.info(f"Book Summary (ticker substitute) recuperato per {instrument_name}")
    return ticker_data

# --- NUOVA FUNZIONE: get_deribit_options_summary ---
def get_deribit_options_summary(currency: str) -> List[Dict[str, Any]]:
    """
    Recupera i book summary per TUTTE le opzioni attive per una data valuta.

    Args:
        currency (str): La valuta base (es. 'BTC', 'ETH').

    Returns:
        List[Dict[str, Any]]: Una lista di dizionari, ognuno rappresentante
                              il summary di un'opzione. Include un dizionario
                              con 'error' se la chiamata API fallisce.
    """
    logger.info(f"Recupero Options Book Summary per currency={currency}")
    endpoint_path = "/public/get_book_summary_by_currency"
    params = {"currency": currency, "kind": "option"}
    result = _make_deribit_public_request(endpoint_path=endpoint_path, params=params)

    # Gestione robusta dell'errore o risposta inattesa
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error"); api_error_msg_str = f"{error_msg}"
        if isinstance(error_msg, dict):
            api_error_msg_str = f"API Error {error_msg.get('code')}: {error_msg.get('message')} {error_msg.get('data') or ''}".strip()
        elif not isinstance(error_msg, str):
            api_error_msg_str = str(error_msg)
        logger.error(f"Errore API Deribit recupero options summary per {currency}: {api_error_msg_str}")
        # Ritorna una lista contenente un singolo dizionario di errore
        return [{"error": api_error_msg_str}]

    elif result is None:
        logger.error(f"Chiamata API options summary per {currency} ha restituito None.")
        return [{"error": "API call returned None unexpectedly"}]

    elif not isinstance(result, list):
        logger.error(f"Risposta options summary Deribit per {currency} non è una lista: {type(result)}")
        return [{"error": f"Unexpected API response type: {type(result)}"}]

    # Se la chiamata ha successo, ritorna la lista di summary
    logger.info(f"Recuperati {len(result)} options summary per {currency}.")
    # Aggiungiamo un parsing preliminare per uniformità e facilità d'uso
    parsed_summaries = []
    for summary in result:
        if not isinstance(summary, dict):
            logger.warning(f"Trovato elemento non dizionario nella lista summary opzioni: {type(summary)}")
            continue
        # Aggiunge un nuovo dizionario vuoto alla lista
        parsed_summaries.append({})
        # Popola l'ultimo dizionario aggiunto
        parsed_summaries[-1]["instrument_name"] = summary.get("instrument_name")
        parsed_summaries[-1]["underlying_price"] = _safe_float(summary.get("underlying_price"))
        parsed_summaries[-1]["volume"] = _safe_float(summary.get("volume")) # in numero di contratti
        parsed_summaries[-1]["volume_usd"] = _safe_float(summary.get("volume_usd"))
        parsed_summaries[-1]["open_interest"] = _safe_float(summary.get("open_interest")) # in numero di contratti
        parsed_summaries[-1]["mark_iv"] = _safe_float(summary.get("mark_iv")) # Mark IV in % (e.g., 55.3)
        parsed_summaries[-1]["bid_iv"] = _safe_float(summary.get("bid_iv"))
        parsed_summaries[-1]["ask_iv"] = _safe_float(summary.get("ask_iv"))

        # Aggiungiamo strike ed expiry parsati se possibile
        instrument_name = summary.get("instrument_name")
        strike = None
        expiry_ts = None
        option_type = None
        if isinstance(instrument_name, str):
             parts = instrument_name.split('-')
             if len(parts) == 4:
                 try:
                     expiry_str = parts[1] # e.g., 26JUL24
                     try:
                          expiry_dt = datetime.strptime(expiry_str, "%d%b%y")
                          expiry_ts = int(expiry_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
                     except ValueError:
                          logger.debug(f"Formato scadenza non DDMMMYY per {instrument_name}, parsing fallito.")
                     strike = _safe_float(parts[2])
                     option_type = parts[3] # 'C' o 'P'
                 except Exception as parse_err:
                      logger.warning(f"Errore parsing nome strumento opzione '{instrument_name}': {parse_err}")

        parsed_summaries[-1]["strike_price"] = strike
        parsed_summaries[-1]["expiration_timestamp"] = expiry_ts
        parsed_summaries[-1]["option_type"] = option_type # 'C', 'P', or None
    # --- CORREZIONE: Fine del loop for ---

    return parsed_summaries
# --- FINE NUOVA FUNZIONE ---


# --- get_deribit_funding_rate_history (invariato) ---
def get_deribit_funding_rate_history(instrument_name: str, count: int = 3) -> Optional[Dict[str, Any]]:
    logger.info(f"Recupero storico funding Deribit per: {instrument_name} (count={count})")
    end_timestamp = int(time.time() * 1000); start_timestamp = end_timestamp - (24 * 60 * 60 * 1000) # Ultimo giorno circa
    endpoint_path = "/public/get_funding_rate_history"
    params = {"instrument_name": instrument_name, "start_timestamp": start_timestamp, "end_timestamp": end_timestamp}
    result = _make_deribit_public_request(endpoint_path=endpoint_path, params=params)

    # Gestione errore robusta
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error"); api_error_msg_str = f"{error_msg}"
        if isinstance(error_msg, dict): api_error_msg_str = f"API Error {error_msg.get('code')}: {error_msg.get('message')} {error_msg.get('data') or ''}".strip()
        elif not isinstance(error_msg, str): api_error_msg_str = str(error_msg)
        logger.error(f"Errore API recupero storico funding {instrument_name}: {api_error_msg_str}")
        return {"history": [], "error": api_error_msg_str} # Ritorna dict con errore

    elif result is None:
        logger.error(f"Chiamata API storico funding {instrument_name} ha restituito None.")
        return {"history": [], "error": "API call returned None unexpectedly"} # Ritorna dict con errore

    elif not isinstance(result, list):
         logger.warning(f"Risposta storico funding Deribit inattesa (non lista): {result}")
         return {"history": [], "error": f"Unexpected funding history data type: {type(result)}"} # Ritorna dict con errore

    # Elaborazione successo
    history_list = [];
    try:
        # Filtra e ordina
        sorted_history = sorted([x for x in result if isinstance(x, dict)], key=lambda x: x.get('timestamp', 0), reverse=True)
        limited_history = sorted_history[:count]
        for item in limited_history:
            # Privilegia interest_1h se presente, altrimenti interest_8h
            rate_key = 'interest_1h' if 'interest_1h' in item else 'interest_8h'
            rate = _safe_float(item.get(rate_key))
            history_list.append({
                "timestamp_ms": item.get('timestamp'),
                "rate_pct_per_period": rate, # Tasso percentuale per il periodo (1h o 8h)
                "index_price": _safe_float(item.get('index_price'))
            })
        return {"history": history_list, "error": None} # Successo

    except Exception as e:
        logger.error(f"Errore processamento storico funding Deribit {instrument_name}: {e}", exc_info=True)
        return {"history": [], "error": f"Processing error: {e}"} # Ritorna dict con errore


# --- get_deribit_last_trades (invariato) ---
def get_deribit_last_trades(instrument_name: str, count: int = 100) -> Optional[Dict[str, Any]]:
    logger.info(f"Recupero ultimi {count} trade Deribit per: {instrument_name}")
    endpoint_path = "/public/get_last_trades_by_instrument"
    params = {"instrument_name": instrument_name, "count": min(count, 1000)} # Max 1000 per API
    result = _make_deribit_public_request(endpoint_path=endpoint_path, params=params)

    # Gestione errore robusta
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error"); api_error_msg_str = f"{error_msg}"
        if isinstance(error_msg, dict): api_error_msg_str = f"API Error {error_msg.get('code')}: {error_msg.get('message')} {error_msg.get('data') or ''}".strip()
        elif not isinstance(error_msg, str): api_error_msg_str = str(error_msg)
        logger.error(f"Errore API recupero ultimi trade {instrument_name}: {api_error_msg_str}")
        return {"trades": [], "error": api_error_msg_str} # Ritorna dict con errore

    elif result is None:
        logger.error(f"Chiamata API ultimi trade {instrument_name} ha restituito None.")
        return {"trades": [], "error": "API call returned None unexpectedly"} # Ritorna dict con errore

    elif not isinstance(result, dict) or 'trades' not in result or not isinstance(result['trades'], list):
        logger.warning(f"Risposta ultimi trade Deribit inattesa: {result}")
        return {"trades": [], "error": f"Unexpected trades data structure: {type(result)}"} # Ritorna dict con errore

    # Elaborazione successo
    trade_list = [];
    try:
        for trade in result['trades']:
            if not isinstance(trade, dict): continue
            trade_list.append({
                "timestamp_ms": trade.get("timestamp"),
                "trade_id": trade.get("trade_id"),
                "price": _safe_float(trade.get("price")),
                "amount": _safe_float(trade.get("amount")), # Numero contratti
                "side": trade.get("direction"), # 'buy' or 'sell'
                "liquidation": trade.get("liquidation") # 'none', 'liquidation', 'maker', 'taker' ?
            })
        return {"trades": trade_list, "error": None} # Successo

    except Exception as e:
        logger.error(f"Errore processamento ultimi trade Deribit {instrument_name}: {e}", exc_info=True)
        return {"trades": [], "error": f"Processing error: {e}"} # Ritorna dict con errore


# --- get_deribit_liquidations (invariato) ---
def get_deribit_liquidations(instrument_name: str, lookback_trades: int = 200) -> Optional[Dict[str, Any]]:
     logger.info(f"Controllo liquidazioni Deribit per: {instrument_name} (ultimi {lookback_trades} trade)")
     trade_data = get_deribit_last_trades(instrument_name, count=lookback_trades)

     # Gestione errore da get_deribit_last_trades
     if trade_data is None or trade_data.get("error") is not None:
         err_msg = trade_data.get("error", "Failed to fetch trades for liquidation check") if trade_data else "Failed to fetch trades (None response)"
         logger.error(f"Impossibile ottenere trade per controllo liquidazioni {instrument_name}: {err_msg}")
         return {"liquidations": [], "error": err_msg} # Ritorna dict con errore

     liquidations = [];
     try:
         if "trades" in trade_data and isinstance(trade_data["trades"], list):
             for trade in trade_data["trades"]:
                 # Verifica se 'liquidation' esiste, non è None e non è 'none'
                 liquidation_status = trade.get("liquidation")
                 if liquidation_status and liquidation_status != 'none':
                     liquidations.append(trade) # Aggiunge l'intero dizionario del trade liquidato
         elif "trades" in trade_data:
             logger.warning(f"Campo 'trades' inatteso per liquidazioni {instrument_name}: tipo {type(trade_data['trades'])}")
             # Non è un errore fatale, ma loggalo

     except Exception as e:
         logger.error(f"Errore filtraggio liquidazioni {instrument_name}: {e}", exc_info=True)
         return {"liquidations": [], "error": f"Liquidation filtering error: {e}"} # Ritorna dict con errore

     logger.debug(f"Trovate {len(liquidations)} liquidazioni negli ultimi {lookback_trades} trade per {instrument_name}")
     return {"liquidations": liquidations, "error": None} # Successo


# --- get_deribit_trade_volumes (invariato) ---
def get_deribit_trade_volumes(currency: str = "BTC") -> Optional[Dict[str, Any]]:
    """
    Recupera i volumi di trade 24h per futures e opzioni per la valuta specificata,
    interpretando correttamente la risposta API basata sulla valuta.
    """
    logger.info(f"Recupero volumi 24h Deribit per: {currency}")
    endpoint_path = "/public/get_trade_volumes"
    result = _make_deribit_public_request(endpoint_path=endpoint_path, params=None)

    volumes = {"futures_volume_usd": None, "options_volume_usd": None, "error": None}
    currency_upper = currency.upper()

    # Gestione Errore API o Risposta Non Valida
    if isinstance(result, dict) and "error" in result:
        error_msg = result.get("error"); api_error_msg_str = f"{error_msg}"
        if isinstance(error_msg, dict): api_error_msg_str = f"API Error {error_msg.get('code')}: {error_msg.get('message')} {error_msg.get('data') or ''}".strip()
        elif not isinstance(error_msg, str): api_error_msg_str = str(error_msg)
        volumes["error"] = api_error_msg_str
        logger.error(f"Errore API recupero volumi Deribit: {api_error_msg_str}")
        return volumes
    elif result is None:
        volumes["error"] = "API call returned None unexpectedly"
        logger.error(f"Chiamata API volumi Deribit ha restituito None.")
        return volumes
    elif not isinstance(result, list):
        volumes["error"] = f"Unexpected API response structure: {type(result)}"
        logger.warning(f"Risposta volumi trade Deribit inattesa (non lista): {result}")
        return volumes
    elif not result:
         volumes["error"] = f"Empty volume list received from API"
         logger.warning(f"Ricevuta lista volumi vuota da API Deribit.")
         return volumes

    # Elaborazione Risposta API (Lista di dizionari per diverse valute)
    try:
        target_currency_data = None
        for item in result:
            if isinstance(item, dict) and item.get("currency") == currency_upper:
                target_currency_data = item
                logger.debug(f"Trovato item per la valuta {currency_upper}: {target_currency_data}")
                break

        if target_currency_data is None:
            logger.warning(f"Nessun dato volume trovato per la valuta specifica '{currency_upper}' nella risposta API.")
            return volumes

        # Estrai i volumi (Futures volume è già in USD, opzioni sono in valuta base)
        futures_vol_raw = target_currency_data.get("futures_volume") # Questo è in USD
        calls_vol_raw = target_currency_data.get("calls_volume") # Questo è in valuta base (es. BTC)
        puts_vol_raw = target_currency_data.get("puts_volume") # Questo è in valuta base (es. BTC)

        # Ottieni il prezzo corrente dell'indice per la conversione
        index_price_for_conversion = None
        perp_instrument_name = f"{currency_upper}-PERPETUAL"
        ticker_for_index = get_deribit_ticker_data(perp_instrument_name)
        if ticker_for_index and not ticker_for_index.get("error"):
            index_price_for_conversion = ticker_for_index.get("estimated_delivery_price") or ticker_for_index.get("underlying_price")
            if index_price_for_conversion:
                logger.debug(f"({currency_upper}) Prezzo indice per conversione volume opzioni: {index_price_for_conversion}")
            else:
                 logger.warning(f"({currency_upper}) Impossibile ottenere prezzo indice da ticker {perp_instrument_name} per conversione volume opzioni.")
        else:
             logger.warning(f"({currency_upper}) Impossibile ottenere ticker {perp_instrument_name} per prezzo indice conversione volume opzioni.")

        # Converti futures (già USD)
        futures_vol_usd = _safe_float(futures_vol_raw)

        # Converti opzioni se possibile
        calls_vol_usd = None
        puts_vol_usd = None
        if index_price_for_conversion is not None and index_price_for_conversion > 0:
             calls_vol_base = _safe_float(calls_vol_raw, default=0.0)
             puts_vol_base = _safe_float(puts_vol_raw, default=0.0)
             if calls_vol_base is not None: calls_vol_usd = calls_vol_base * index_price_for_conversion
             if puts_vol_base is not None: puts_vol_usd = puts_vol_base * index_price_for_conversion
        else:
            logger.warning(f"({currency_upper}) Conversione volume opzioni a USD fallita per mancanza prezzo indice.")

        # Assegna volume futures
        volumes["futures_volume_usd"] = futures_vol_usd

        # Calcola volume totale opzioni USD
        if calls_vol_usd is not None and puts_vol_usd is not None:
             total_options_vol = calls_vol_usd + puts_vol_usd
             volumes["options_volume_usd"] = total_options_vol
             logger.debug(f"({currency_upper}) Volume Opzioni Calcolato (USD): {total_options_vol} (Calls: {calls_vol_usd}, Puts: {puts_vol_usd})")
        else:
             logger.warning(f"({currency_upper}) Impossibile calcolare volume totale opzioni USD (calls={calls_vol_usd}, puts={puts_vol_usd}).")
             volumes["options_volume_usd"] = None

        volumes.pop("error", None)
        logger.info(f"Volumi Deribit ({currency_upper}) estratti: Futures={volumes['futures_volume_usd']}, Options={volumes['options_volume_usd']}")
        return volumes

    except Exception as e:
        logger.error(f"Errore processamento volumi trade Deribit {currency}: {e}", exc_info=True)
        volumes["error"] = f"Processing error: {e}"
        volumes["futures_volume_usd"] = None
        volumes["options_volume_usd"] = None
        return volumes

# --- END OF FILE deribit_collector.py ---