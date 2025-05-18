# --- START OF FILE config.py ---

# config.py
"""
File di configurazione centrale per parametri, costanti e liste predefinite.
Versione rifattorizzata per analisi singola asset/timeframe orientata a LLM.
Include parametri aggiuntivi per analisi tecnica dettagliata.
"""
import os
import logging # Aggiunto per warning fallback

# --- Configurazione API Exchange (Default: Binance) ---
# !!! ATTENZIONE: Inserire qui le chiavi API è FORTEMENTE SCONSIGLIATO per sicurezza. !!!
# !!! Usare preferibilmente variabili d'ambiente o un sistema di gestione segreti.  !!!
# !!! Se inserite qui, NON committare questo file in repository pubblici/condivisi.   !!!
# --- Sostituisci con le tue chiavi o lasciale vuote/None per API pubblica ---
API_KEY = "****************************************" # os.environ.get("BINANCE_API_KEY", None)
API_SECRET = "*************************************" # os.environ.get("BINANCE_API_SECRET", None)
# ------------------------------------------------------------------------------------
API_TIMEOUT = 30 # Secondi per timeout richieste API

# --- NUOVO: Configurazione Base URL Binance ---
# Usati da binance_data_fetcher.py
BINANCE_FUTURES_BASE_URL = 'https://fapi.binance.com'
BINANCE_SPOT_BASE_URL = 'https://api.binance.com'
BINANCE_OPTIONS_BASE_URL = 'https://eapi.binance.com' # Anche se non usato attivamente ora
# ----------------------------------------------

# --- Configurazione Cache Dati ---
CACHE_DIR = "cache_analysis" # Nome directory cache
CACHE_EXPIRY = 3600 * 4 # Scadenza cache in secondi (es. 4 ore)

# --- Configurazione Dati Storici ---
# Anno di inizio da cui scaricare i dati se la cache è vuota o il lookback richiesto va più indietro
FALLBACK_START_YEAR = 2018 # Anno fallback per inizio storico

# Periodi di lookback (in giorni) suggeriti per ogni timeframe per analisi completa
# Usati da data_collector per determinare da quando scaricare i dati.
# Usati ANCHE per calcolare le statistiche storiche aggregate (cicli, intra-candela).
DATA_LOOKBACK_PERIODS = {
    "1m": 7,      # Lookback più brevi per TF minori
    "5m": 14,
    "15m": 30,
    "30m": 60,
    "1h": 365,     # Almeno 1 anno per indicatori lenti su 1h
    "2h": 400,
    "4h": 500,     # Circa 1.5 anni per 4h
    "6h": 600,
    "8h": 730,     # 2 anni per sicurezza su 8h/12h/1d
    "12h": 730,
    "1d": 1095,    # 3 anni per daily (include SMA 200, stagionalità base)
    "3d": 1460,    # 4 anni per 3d
    "1w": 1825,    # 5 anni per weekly <-- Lookback già presente e adeguato
    "1M": 2555     # ~7 anni per monthly
}

# Numero minimo di punti dati richiesti DOPO la pulizia per eseguire l'analisi
MIN_DATA_POINTS_FOR_ANALYSIS = 250 # Necessario per SMA 200 e altre analisi lunghe

# --- Timeframes Definiti ---
# Usati per validazione, menu, logica interna
TIMEFRAMES = {
    "1m": "1 minuto", "5m": "5 minuti", "15m": "15 minuti", "30m": "30 minuti",
    "1h": "1 ora", "2h": "2 ore", "4h": "4 ore", "6h": "6 ore", "8h": "8 ore", "12h": "12 ore",
    "1d": "1 giorno", "3d": "3 giorni", "1w": "1 settimana", "1M": "1 mese" # '1w' già presente
}
# Lista ordinata (utile per logica MTF se necessaria)
ORDERED_TIMEFRAMES = [
    "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M" # '1w' già presente
]
# Validazione coerenza
try:
    assert all(tf in TIMEFRAMES for tf in ORDERED_TIMEFRAMES), "ORDERED_TIMEFRAMES non corrisponde a TIMEFRAMES"
    assert len(ORDERED_TIMEFRAMES) == len(TIMEFRAMES), "ORDERED_TIMEFRAMES ha lunghezza diversa da TIMEFRAMES"
except AssertionError as e:
     logging.warning(f"Errore validazione timeframe in config.py: {e}")


# --- Parametri Menu Interattivo (main.py) ---
MENU_ASSET_LIST = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "WOO/USDT", "SOL/USDT"]
# --- MODIFICA QUI ---
MENU_TIMEFRAME_LIST = ["5m", "15m", "1h", "4h", "1d", "1w"] # Aggiunto '1w' (1 settimana)
# --- FINE MODIFICA ---

# --- Parametri Analisi Tecnica Base (Default usati da TechnicalAnalyzer) ---
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BBANDS_PERIOD = 20
BBANDS_STDDEV = 2.0
ATR_PERIOD = 14
ADX_PERIOD = 14
SMA_PERIODS = [20, 50, 200] # Periodi SMA standard da calcolare
EMA_PERIODS = [12, 26, 50] # Periodi EMA standard da calcolare

# --- Parametri Aggiuntivi Analisi Tecnica (per technical_analyzer.py) ---
TA_VOLUME_SMA_SHORT = 10        # Periodo SMA breve per confronto volume spike
TA_VOLUME_ZSCORE_PERIOD = 20    # Periodo per calcolo Z-score volume
TA_UPDOWN_VOL_PERIOD = 10       # Periodo per calcolo rapporto Volume Up/Down
TA_VWAP_ROLLING_PERIOD = 20     # Periodo per VWAP Rolling
TA_VOLATILITY_PERCENTILE_PERIOD = 100 # Periodo per calcolo percentili ATR/BBW
TA_ATR_TARGET_MULTIPLIERS = [1.0, 1.5, 2.0, 3.0] # Moltiplicatori ATR per target potenziali
# -------------------------------------------------------------------------------

# --- Parametri Analisi Livelli e Prossimità ---
FIB_LOOKBACK_PERIOD = 90 # Periodo (numero candele) per calcolare High/Low per Fibonacci
FIB_NEARNESS_THRESHOLD = 0.015 # 1.5% per considerare prezzo vicino a livello Fibo
VP_NEARNESS_THRESHOLD = 0.010 # 1.0% per vicinanza a VAH/VAL/POC (Volume Profile)
SR_NEARNESS_THRESHOLD = 0.010 # 1.0% per vicinanza a Supporto/Resistenza (basati su pivot)
MA_NEARNESS_THRESHOLD = 0.005 # 0.5% per vicinanza a Medie Mobili
# NUOVO: Soglia distanza ATR per prossimità (usata in TradingAdvisor._populate_level_proximity)
LEVEL_NEARNESS_THRESHOLD_ATR = 0.25 # Considera "vicino" se a <= 0.25 ATR di distanza

# --- Parametri Analisi Multi-Timeframe (MTF) ---
# Timeframe superiori da analizzare sempre per contesto
MTF_TARGET_TIMEFRAMES = ['1h', '4h', '1d', '1w'] # '1w' già presente
# Lookback per dati MTF (potrebbe essere derivato da DATA_LOOKBACK_PERIODS, ma mantenuto per ora)
# Assicura che sia sufficiente per gli indicatori sui TF più alti (es. 1w)
MTF_DAYS_LOOKBACK = 1095 # 3 anni (sufficienti per SMA 200 su daily/weekly)


# --- Parametri Aggiornamento Batch Statistiche Storiche ---
# Timeframe inclusi quando si esegue l'aggiornamento batch
# (per cicli e statistiche intra-candela)
# --- MODIFICA OPZIONALE (ma consigliata): Aggiungere '1w' anche qui se si vuole aggiornare in batch ---
HISTORICAL_STATS_BATCH_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]
# --- FINE MODIFICA OPZIONALE ---

# --- Parametri Statistiche Intra-Candela ---
# Percentili da calcolare e salvare per le metriche intra-candela storiche
INTRACANDLE_PERCENTILES = [10, 25, 50, 75, 90, 95, 99]
# ----------------------------------------------------

# --- Configurazione Output e Logging ---
LOG_FILE = "analysis_tool.log" # Nome file di log
LOG_LEVEL = "INFO" # Livello di log di default per file e console
REPORT_DIR = "analysis_reports" # Directory per salvare i report JSON/TXT
# Directory per statistiche storiche (Cicli, Intra-Candela, Deribit)
HISTORICAL_STATS_DIR = "historical_stats"

# --- Costanti Storico Deribit ---
# Numero massimo di record giornalieri da conservare nello storico Deribit
DERIBIT_HISTORY_MAX_ENTRIES = 180
# Periodo (in giorni/record) su cui calcolare le medie storiche per Deribit
DERIBIT_HISTORY_AVG_PERIOD = 30

# --- END OF FILE config.py ---