# --- START OF FILE binance_data_fetcher.py ---
"""
Modulo dedicato al recupero di dati specifici dal mercato
Futures e Spot di Binance tramite API REST pubblica.
Include calcoli derivati come OI Change e CVD periodico.
"""
import os
import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
from dotenv import load_dotenv
from decimal import Decimal, getcontext, ROUND_HALF_UP
import logging
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd # Necessario per Decimal e gestione errori
import numpy as np  # Necessario per np.isinf

# Setup logging
logger = logging.getLogger(__name__)

# Importa configurazioni e helper necessari
try:
    import config # Assume config.py sia accessibile
    from statistical_analyzer_helpers import _safe_float # Usa helper esistente
    CONFIG_LOADED = True
except ImportError:
    logger.error("ERRORE CRITICO: Impossibile importare config o statistical_analyzer_helpers in binance_data_fetcher.")
    # Definisci fallback minimali se l'import fallisce
    class ConfigFallback:
        API_KEY = os.environ.get("BINANCE_API_KEY")
        API_SECRET = os.environ.get("BINANCE_API_SECRET")
        BINANCE_FUTURES_BASE_URL = 'https://fapi.binance.com'
        BINANCE_SPOT_BASE_URL = 'https://api.binance.com'
    config = ConfigFallback()
    CONFIG_LOADED = False
    # Fallback safe_float
    def _safe_float(value, default=None):
        try: return float(value) if pd.notna(value) and not np.isinf(value) else default
        except: return default

# --- Configurazione Specifica Binance ---
# Carica da .env per sicurezza (sovrascrive config se presenti)
load_dotenv()
API_KEY = os.environ.get('BINANCE_API_KEY', getattr(config, 'API_KEY', None)) # Priorità a .env, poi config
SECRET_KEY = os.environ.get('BINANCE_SECRET_KEY', getattr(config, 'API_SECRET', None))

FUTURES_BASE_URL = getattr(config, 'BINANCE_FUTURES_BASE_URL', 'https://fapi.binance.com')
SPOT_BASE_URL = getattr(config, 'BINANCE_SPOT_BASE_URL', 'https://api.binance.com')
OPTIONS_BASE_URL = getattr(config, 'BINANCE_OPTIONS_BASE_URL', 'https://eapi.binance.com')

# Imposta precisione per Decimal
getcontext().prec = 28

# --- Funzioni Helper (Come nello script precedente) ---
def generate_signature(query_string):
    """Genera la firma HMAC SHA256 richiesta da Binance."""
    if not SECRET_KEY: raise ValueError("SECRET_KEY Binance non trovata")
    return hmac.new(SECRET_KEY.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

def make_request(base_url, endpoint, params=None, requires_signature=False):
    """Esegue una richiesta API GET a Binance (specificando base_url)."""
    method = 'GET'
    if requires_signature and (not API_KEY or not SECRET_KEY):
        logger.error(f"ERRORE: Chiavi API Binance mancanti per {base_url}{endpoint}")
        return None, "Chiavi API mancanti"

    headers = {}
    if API_KEY: headers['X-MBX-APIKEY'] = API_KEY

    query_params = params.copy() if params else {}
    if requires_signature:
        query_params['timestamp'] = int(time.time() * 1000)
        query_string = urlencode(query_params)
        try: query_params['signature'] = generate_signature(query_string)
        except ValueError as e: return None, str(e)
    else:
        query_string = urlencode(query_params)

    url = f"{base_url}{endpoint}"
    logger.debug(f"Binance API Request: {url} Params: {query_params}")

    try:
        response = requests.get(url, headers=headers, params=query_params, timeout=15)
        status_code = response.status_code

        if status_code == 200:
            try: return response.json(), None
            except requests.exceptions.JSONDecodeError: return None, "Risposta non JSON"
        else:
            error_msg = f"Errore {status_code}"
            error_code = 'N/A'
            try:
                err_data = response.json()
                error_msg = err_data.get('msg', response.text)
                error_code = err_data.get('code', 'N/A')
            except requests.exceptions.JSONDecodeError: error_msg = response.text if response.text else error_msg
            logger.warning(f"Richiesta FALLITA a {url}. Codice: {error_code}, Messaggio: {error_msg[:200]}") # Limita lunghezza msg
            return None, f"Errore {status_code} - Codice: {error_code}"

    except requests.exceptions.Timeout: return None, "Timeout"
    except requests.exceptions.RequestException as e: return None, f"Errore Connessione: {e}"
    except Exception as e: return None, f"Errore Script: {e}"

# --- Funzioni di Recupero Dati Specifiche (Adattate) ---

def get_premium_index(symbol):
    data, error = make_request(FUTURES_BASE_URL, '/fapi/v1/premiumIndex', params={'symbol': symbol})
    if error or not isinstance(data, dict): return None, error
    return {'mark_price': data.get('markPrice'), 'index_price': data.get('indexPrice'), 'last_funding_rate': data.get('lastFundingRate'), 'next_funding_time_ms': data.get('nextFundingTime')}, None

def get_open_interest(symbol):
    data, error = make_request(FUTURES_BASE_URL, '/fapi/v1/openInterest', params={'symbol': symbol})
    if error or not isinstance(data, dict): return None, error
    return data.get('openInterest'), None

def get_open_interest_history(symbol, period, limit):
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    data, error = make_request(FUTURES_BASE_URL, '/futures/data/openInterestHist', params=params)
    if error or not isinstance(data, list): return None, error
    return data, None

def get_funding_rate_history(symbol, limit=1):
    data, error = make_request(FUTURES_BASE_URL, '/fapi/v1/fundingRate', params={'symbol': symbol, 'limit': limit})
    if error or not isinstance(data, list): return None, error
    return data[0].get('fundingRate') if data else None, None

def get_futures_ticker_24hr(symbol):
    data, error = make_request(FUTURES_BASE_URL, '/fapi/v1/ticker/24hr', params={'symbol': symbol})
    if error or not isinstance(data, dict): return None, error
    return {'price_change_percent': data.get('priceChangePercent'), 'last_price': data.get('lastPrice'), 'volume_24h_quote': data.get('quoteVolume')}, None

def get_klines(symbol, interval, limit):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    data, error = make_request(FUTURES_BASE_URL, '/fapi/v1/klines', params=params)
    if error or not isinstance(data, list): return None, error
    klines_formatted = []
    for k in data:
        if len(k) >= 11:
            klines_formatted.append({
                'open_time_ms': k[0], 'open': k[1], 'high': k[2], 'low': k[3], 'close': k[4],
                'volume_base': k[5], 'close_time_ms': k[6], 'volume_quote': k[7], 'trades': k[8],
                'taker_buy_volume_base': k[9], 'taker_buy_volume_quote': k[10] })
    return klines_formatted, None

def get_global_long_short_ratio(symbol, period):
    params = {'symbol': symbol, 'period': period, 'limit': 1}
    data, error = make_request(FUTURES_BASE_URL, '/futures/data/globalLongShortAccountRatio', params=params)
    if error or not isinstance(data, list) or not data: return None, error
    return data[0].get('longShortRatio'), None

def get_top_long_short_pos_ratio(symbol, period):
    params = {'symbol': symbol, 'period': period, 'limit': 1}
    data, error = make_request(FUTURES_BASE_URL, '/futures/data/topLongShortPositionRatio', params=params)
    if error or not isinstance(data, list) or not data: return None, error
    return data[0].get('longShortRatio'), None

def get_basis(pair, contract_type, period):
    params = {'pair': pair, 'contractType': contract_type, 'period': period, 'limit': 1}
    data, error = make_request(FUTURES_BASE_URL, '/futures/data/basis', params=params)
    if error or not isinstance(data, list) or not data: return None, error
    basis_data = data[0]
    return {'basis': basis_data.get('basis'), 'basis_rate': basis_data.get('basisRate')}, None

def get_spot_ticker_24hr(symbol):
    data, error = make_request(SPOT_BASE_URL, '/api/v3/ticker/24hr', params={'symbol': symbol})
    if error or not isinstance(data, dict): return None, error
    return {'volume_24h_quote': data.get('quoteVolume'), 'last_price': data.get('lastPrice'), 'price_change_percent': data.get('priceChangePercent')}, None

def get_aggregated_trades(symbol, limit=1):
     params = {'symbol': symbol, 'limit': limit}
     data, error = make_request(FUTURES_BASE_URL, '/fapi/v1/aggTrades', params=params)
     if error or not isinstance(data, list) or not data: return None, error
     trade = data[-1]
     return {'price': trade.get('p'), 'quantity_base': trade.get('q'), 'time_ms': trade.get('T'), 'is_buyer_maker': trade.get('m')}, None

# --- Funzioni per Calcoli Derivati (Invariate) ---
def calculate_oi_change(current_oi_usd, oi_history):
    """Calcola la variazione % dell'OI rispetto a ~24h fa."""
    if current_oi_usd is None or not oi_history: return None
    try:
        # L'API restituisce dal più recente al più vecchio.
        # Il primo elemento [0] corrisponde al dato di 'limit' periodi fa.
        if len(oi_history) < 2: return None # Necessita almeno 2 punti per confronto
        oldest_oi_data = oi_history[0] # Primo = più vecchio tra gli ultimi 'limit' richiesti
        previous_oi_usd_str = oldest_oi_data.get('sumOpenInterestValue')
        if previous_oi_usd_str is None: return None
        previous_oi_usd = Decimal(previous_oi_usd_str)
        if previous_oi_usd == 0: return None # Evita divisione per zero
        change_pct = ((current_oi_usd - previous_oi_usd) / previous_oi_usd) * Decimal(100)
        return change_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except (IndexError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Errore calcolo OI Change: {e}")
        return None

def calculate_cvd_period(klines_data):
    """Calcola il CVD sommando i delta delle klines fornite."""
    if not klines_data: return None
    total_cvd = Decimal(0)
    try:
        for kline in klines_data:
            vol_quote_str = kline.get('volume_quote')
            taker_buy_vol_quote_str = kline.get('taker_buy_volume_quote')
            if vol_quote_str is None or taker_buy_vol_quote_str is None:
                logger.debug("Dati volume mancanti in kline per CVD, salto.")
                continue # Salta questa kline se mancano dati volume
            vol_quote = Decimal(vol_quote_str)
            taker_buy_vol_quote = Decimal(taker_buy_vol_quote_str)
            taker_sell_vol_quote = vol_quote - taker_buy_vol_quote
            delta = taker_buy_vol_quote - taker_sell_vol_quote
            total_cvd += delta
        return total_cvd.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except (TypeError, ValueError) as e:
        logger.warning(f"Errore calcolo CVD: {e}")
        return None

# --- Funzione Principale di Fetching (MODIFICATA per rimuovere taker_buy_sell_ratio) ---
def fetch_all_binance_data(
    futures_symbol: str = 'BTCUSDT',
    spot_symbol: str = 'BTCUSDT',
    kline_interval: str = '1h',
    kline_limit: int = 24,
    ratio_period: str = '1h',
    basis_pair: str = 'BTCUSDT',
    basis_contract: str = 'PERPETUAL',
    basis_period: str = '1h',
    oi_hist_period: str = '1h',
    oi_hist_limit: int = 25
) -> Dict[str, Any]:
    """
    Orchestra il recupero di tutti i dati Binance necessari e calcola i derivati.
    """
    logger.info(f"Avvio recupero dati Binance combinati per {futures_symbol}...")
    fetch_start_time = time.time()
    final_data = {} # Inizializza dizionario vuoto
    errors = []

    # --- Recupero Dati Grezzi ---
    premium_data, err = get_premium_index(futures_symbol)
    if err: errors.append(f"Premium Index: {err}")
    mark_price_str = premium_data.get('mark_price') if premium_data else None
    mark_price = Decimal(mark_price_str) if mark_price_str else None

    oi_base_str, err = get_open_interest(futures_symbol)
    if err: errors.append(f"Open Interest: {err}")

    oi_history, err = get_open_interest_history(futures_symbol, oi_hist_period, oi_hist_limit)
    if err: errors.append(f"OI History: {err}")

    funding_rate_hist, err = get_funding_rate_history(futures_symbol, limit=1)
    if err: errors.append(f"Funding Rate History: {err}")

    futures_ticker_data, err = get_futures_ticker_24hr(futures_symbol)
    if err: errors.append(f"Futures Ticker 24hr: {err}")

    klines_data, err = get_klines(futures_symbol, kline_interval, kline_limit)
    if err: errors.append(f"Klines: {err}")

    global_lsr, err = get_global_long_short_ratio(futures_symbol, ratio_period)
    if err: errors.append(f"Global LSR: {err}")
    top_pos_lsr, err = get_top_long_short_pos_ratio(futures_symbol, ratio_period)
    if err: errors.append(f"Top Pos LSR: {err}")

    last_agg_trade, err = get_aggregated_trades(futures_symbol, limit=1)
    if err: errors.append(f"Agg Trades: {err}")

    basis_data, err = get_basis(basis_pair, basis_contract, basis_period)
    if err: errors.append(f"Basis: {err}")

    spot_ticker_data, err = get_spot_ticker_24hr(spot_symbol)
    if err: errors.append(f"Spot Ticker 24hr: {err}")

    # --- Calcoli Derivati ---
    current_oi_usd = None
    oi_base = None
    if oi_base_str and mark_price:
        try:
            oi_base = Decimal(oi_base_str)
            current_oi_usd = (oi_base * mark_price).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        except Exception as e: errors.append(f"Calcolo OI USD: {e}")

    oi_change_pct = calculate_oi_change(current_oi_usd, oi_history)
    cvd_period = calculate_cvd_period(klines_data)

    # --- Assembla l'Output Finale (SENZA taker_buy_sell_ratio) ---
    final_data['fetch_timestamp_utc'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    final_data['symbol_futures'] = futures_symbol
    final_data['symbol_spot'] = spot_symbol
    final_data['market_snapshot'] = {
        'futures_last_price': _safe_float(futures_ticker_data.get('last_price')) if futures_ticker_data else None,
        'futures_mark_price': str(mark_price) if mark_price else None,
        'futures_index_price': _safe_float(premium_data.get('index_price')) if premium_data else None,
        'spot_last_price': _safe_float(spot_ticker_data.get('last_price')) if spot_ticker_data else None,
    }
    final_data['volume_activity'] = {
        'futures_volume_24h_quote': _safe_float(futures_ticker_data.get('volume_24h_quote')) if futures_ticker_data else None,
        'spot_volume_24h_quote': _safe_float(spot_ticker_data.get('volume_24h_quote')) if spot_ticker_data else None,
        'futures_last_agg_trade': last_agg_trade,
    }
    final_data['open_interest'] = {
        'value_base': str(oi_base) if oi_base else None,
        'value_usd': str(current_oi_usd) if current_oi_usd else None,
    }
    final_data['funding_basis'] = {
        'funding_rate_last': _safe_float(funding_rate_hist),
        'funding_rate_next': _safe_float(premium_data.get('last_funding_rate')) if premium_data else None,
        'next_funding_time_ms': premium_data.get('next_funding_time_ms') if premium_data else None,
        'basis': _safe_float(basis_data.get('basis')) if basis_data else None,
        'basis_rate': _safe_float(basis_data.get('basis_rate')) if basis_data else None,
    }
    # --- Sezione Modificata ---
    final_data['sentiment_ratios'] = {
        'period': ratio_period,
        'global_lsr_account': _safe_float(global_lsr),
        'top_trader_lsr_position': _safe_float(top_pos_lsr)
        # Chiave 'taker_buy_sell_ratio' RIMOSSA
    }
    # --- Fine Sezione Modificata ---
    final_data['derived_metrics'] = {
        'oi_change_pct_approx_24h': str(oi_change_pct) if oi_change_pct is not None else None,
        'cvd_period': str(cvd_period) if cvd_period is not None else None,
        'cvd_period_details': f"{kline_limit} x {kline_interval} klines"
    }
    # Non includiamo klines grezze qui per brevità, ma potrebbero essere utili
    # final_data['raw_klines_period'] = klines_data

    # Aggiungi errori riscontrati
    if errors:
        final_data['fetch_errors'] = errors

    fetch_end_time = time.time()
    logger.info(f"Recupero dati Binance completato in {fetch_end_time - fetch_start_time:.2f} sec. Errori: {len(errors)}")
    return final_data

# --- Esempio di utilizzo (può essere commentato o rimosso) ---
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     binance_data = fetch_all_binance_data()
#     print("\n--- Dati Binance Recuperati ---")
#     import json
#     print(json.dumps(binance_data, indent=2, default=str))

# --- END OF FILE binance_data_fetcher.py ---