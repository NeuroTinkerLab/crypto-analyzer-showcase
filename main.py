# --- START OF FILE main.py ---
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import time
import inquirer  # type: ignore
from typing import Optional, List, Dict, Any, Tuple
import warnings

# --- Aggiungi la directory dello script a sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --------------------------------------------------------------

# --- Import Moduli Locali ---
try:
    import config
    from data_collector import get_data_with_cache, get_exchange_instance
    from statistical_analyzer_helpers import (
        save_results_to_json, save_results_to_txt, _safe_get,
        load_historical_stats_from_json, # Usato per display info
        safe_strftime,
        convert_to_json_serializable # Importato per logging
    )
    from trading_advisor import TradingAdvisor
    # Importa StatisticalAnalyzerAdvanced per l'aggiornamento delle statistiche
    from statistical_analyzer_advanced import StatisticalAnalyzerAdvanced
    # --- NUOVO IMPORT ---
    from binance_data_fetcher import fetch_all_binance_data

except ImportError as e:
    print(f"\nERRORE CRITICO: Impossibile importare moduli locali necessari ({e}).")
    print("Verifica che tutti i file .py del progetto siano nella stessa directory o nel PYTHONPATH.")
    print("Assicurati che non ci siano errori di sintassi nei file importati.")
    print("Controlla specificamente se 'binance_data_fetcher.py' esiste e non ha errori.")
    print("sys.path attuale:")
    for p in sys.path:
        print(f"  - {p}")
    # Definisci funzione fallback se import fallisce
    def fetch_all_binance_data(*args, **kwargs) -> Dict[str, Any]:
        logging.error("Chiamata a funzione fallback fetch_all_binance_data!")
        return {"error": "binance_data_fetcher module not loaded or failed to import"}
    # Definisci fallback per convert_to_json_serializable se l'helper non carica
    def convert_to_json_serializable(obj: Any) -> Any: return str(obj) # Fallback molto basico
    # sys.exit(1) # Potresti voler uscire qui, ma per ora definiamo il fallback
except Exception as general_import_err:
    print(f"\nERRORE CRITICO: Errore imprevisto durante import moduli locali: {general_import_err}")
    sys.exit(1)


# --- CONFIGURAZIONE LOGGING ---
def setup_logging(log_level_console_str: str):
    # (Funzione setup_logging invariata)
    log_level_console = getattr(logging, log_level_console_str.upper(), logging.INFO)
    log_level_file = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level_console, log_level_file))
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_filename = config.LOG_FILE
    log_dir = os.path.dirname(log_filename)
    try:
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir); print(f"Directory log creata: {log_dir}")
        file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
        file_handler.setLevel(log_level_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging su file abilitato: {log_filename} (Livello: {logging.getLevelName(log_level_file)})")
    except Exception as log_file_err:
        print(f"Attenzione: Log su file '{log_filename}' non configurato: {log_file_err}", file=sys.stderr)
        logging.warning(f"Logging su file non configurato: {log_file_err}")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_console)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Logging su console abilitato (Livello: {logging.getLevelName(log_level_console)})")


# Logger globale per questo modulo
logger = logging.getLogger(__name__)

# --- FUNZIONE PARSE ARGUMENTS (Invariata) ---
def parse_arguments():
    parser = argparse.ArgumentParser(description='Crypto Market Analysis Tool for LLM')
    parser.add_argument('--fetch', action='store_true', help='Forza download dati (ignora cache) - Usato anche con --update-all.')
    parser.add_argument('--loglevel', type=str, default=config.LOG_LEVEL, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Livello logging console.')
    single_op_group = parser.add_argument_group('Single Analysis/Update Options')
    single_op_group.add_argument('--pair', type=str, default=None, help="Specifica coppia (es. BTC/USDT) per analisi singola o update singolo.")
    single_op_group.add_argument('--timeframe', type=str, default=None, help="Specifica timeframe (es. 1h) per analisi singola o update singolo.")
    single_op_group.add_argument('--update-historical-stats', action='store_true', help='Forza il ricalcolo e salvataggio delle statistiche storiche (cicli e intra-candela) per la coppia/timeframe specificati (richiede --pair e --timeframe).')
    batch_op_group = parser.add_argument_group('Batch Update Options')
    batch_op_group.add_argument('--update-all-historical-stats', action='store_true', help='Aggiorna le statistiche storiche (cicli e intra-candela) per TUTTE le coppie e timeframe definiti in config.py (ignora --pair, --timeframe, --update-historical-stats).')
    return parser.parse_args()


# --- FUNZIONE HELPER PER INFO AGGIORNAMENTO (Invariata) ---
def display_historical_stats_update_info(pair: str, timeframe: str):
    if not pair or not timeframe: return
    safe_symbol = pair.replace('/', '_').replace('\\', '_').replace(':', '_')
    base_filename = f"{safe_symbol}_{timeframe}.json"
    cycle_stats_filename = os.path.join(config.HISTORICAL_STATS_DIR, f"cycle_stats_{base_filename}")
    candle_stats_filename = os.path.join(config.HISTORICAL_STATS_DIR, f"candle_stats_{base_filename}")
    update_info_lines = []
    try:
        stats_data_cycles = load_historical_stats_from_json(cycle_stats_filename)
        if stats_data_cycles:
            calc_time = stats_data_cycles.get('calculation_timestamp', 'N/D')
            data_points = stats_data_cycles.get('data_points_used', 'N/D')
            up_count = stats_data_cycles.get('up_cycles_count', 'N/D')
            down_count = stats_data_cycles.get('down_cycles_count', 'N/D')
            update_info_lines.append(f"  - Cicli:   {calc_time} (Dati:{data_points}, Cicli U/D:{up_count}/{down_count})")
        else: update_info_lines.append("  - Cicli:   Non ancora calcolate.")
    except Exception as e: update_info_lines.append("  - Cicli:   Errore recupero info.")
    try:
        stats_data_candle = load_historical_stats_from_json(candle_stats_filename)
        if stats_data_candle:
            calc_time = stats_data_candle.get('calculation_timestamp', 'N/D')
            data_points = stats_data_candle.get('data_points_used', 'N/D')
            median_range = _safe_get(stats_data_candle, ['stats_by_metric', 'ic_range_pct', 'median'])
            range_str = f"{median_range:.2f}%" if median_range is not None else "N/D"
            update_info_lines.append(f"  - Candele: {calc_time} (Dati:{data_points}, Mediana Range:{range_str})")
        else: update_info_lines.append("  - Candele: Non ancora calcolate.")
    except Exception as e: update_info_lines.append("  - Candele: Errore recupero info.")
    print(f"\n--- Info Statistiche Storiche ({pair} - {timeframe}) ---")
    for line in update_info_lines: print(line)
    print("-" * (34 + len(pair) + len(timeframe)))


# --- FUNZIONE MENU INTERATTIVO (Invariata) ---
def get_analysis_parameters_interactive() -> Optional[Tuple[str, str, str]]:
    print("-" * 30 + "\n Selezione Asset e Timeframe \n" + "-" * 30)
    try:
        pair_choices = config.MENU_ASSET_LIST; tf_choices = config.MENU_TIMEFRAME_LIST
        pair_question = [inquirer.List('pair', message="Seleziona coppia", choices=pair_choices, default=pair_choices[0])]
        pair_answer = inquirer.prompt(pair_question, raise_keyboard_interrupt=True)
        if pair_answer is None: return None
        selected_pair = pair_answer['pair']
        tf_question = [inquirer.List('timeframe', message="Seleziona timeframe", choices=tf_choices, default='1h')]
        tf_answer = inquirer.prompt(tf_question, raise_keyboard_interrupt=True)
        if tf_answer is None: return None
        selected_tf = tf_answer['timeframe']
        display_historical_stats_update_info(selected_pair, selected_tf)
        confirm_question = [
            inquirer.List('action', message="Azione successiva?", choices=[('Esegui Analisi Completa', 'analyze'), ('Aggiorna Statistiche Storiche (Cicli & Candele)', 'update_stats'), ('Seleziona altra Coppia/Timeframe', 'change'), ('Esci', 'exit')], default='analyze') ]
        confirm_answer = inquirer.prompt(confirm_question, raise_keyboard_interrupt=True)
        if confirm_answer is None or confirm_answer['action'] == 'exit': return None
        elif confirm_answer['action'] in ['analyze', 'update_stats']:
            action = confirm_answer['action']; logger.info(f"Parametri selezionati da menu: Pair={selected_pair}, Timeframe={selected_tf}, Azione={action}"); return selected_pair, selected_tf, action
        elif confirm_answer['action'] == 'change': return None
        else: logger.error(f"Azione menu non riconosciuta: {confirm_answer.get('action')}"); return None
    except KeyboardInterrupt: print("\nOperazione interrotta dall'utente."); return None
    except Exception as e: logger.error(f"Errore durante il menu interattivo: {e}", exc_info=True); return None


# --- load_and_prepare_data (Invariata) ---
def load_and_prepare_data(symbol: str, timeframe: str, force_fetch: bool, days_override: Optional[float] = None) -> Optional[pd.DataFrame]:
    # (Funzione Invariata - Mantiene la logica di caricamento OHLCV)
    logger.info(f"Richiesta dati OHLCV per {symbol} ({timeframe}). Force Fetch: {force_fetch}, Days Override: {days_override}")
    try:
        # Verifica se il timeframe richiesto è valido per l'exchange (es. Binance)
        # Questo controllo è più per robustezza, ccxt dovrebbe gestirlo internamente
        exchange = get_exchange_instance()
        if exchange and timeframe not in exchange.timeframes:
            logger.error(f"({symbol}|{timeframe}) Timeframe non supportato dall'exchange {exchange.id}.")
            return None

        data = get_data_with_cache(pair=symbol, timeframe=timeframe, force_fetch=force_fetch, days=days_override)

        if data is None or data.empty:
            logger.warning(f"({symbol}|{timeframe}) Nessun dato OHLCV ottenuto da get_data_with_cache.")
            return None
        logger.debug(f"({symbol}|{timeframe}) Dati OHLCV grezzi ricevuti: {len(data)} righe.")
        if not isinstance(data.index, pd.DatetimeIndex): data.index = pd.to_datetime(data.index, utc=True)
        if data.index.tz is None or str(data.index.tz).upper() != 'UTC': data.index = data.index.tz_localize('UTC') if data.index.tz is None else data.index.tz_convert('UTC')
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols: logger.error(f"({symbol}|{timeframe}) Colonne OHLCV mancanti: {missing_cols}"); return None
        for col in required_cols: data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)
        if 'simple_return' not in data.columns: data['simple_return'] = (data['close'].replace(0, np.nan).pct_change()).replace([np.inf, -np.inf], np.nan)
        if 'log_return' not in data.columns: data['log_return'] = (np.log(data['close'].replace(0, np.nan)).diff()).replace([np.inf, -np.inf], np.nan)
        if 'simple_return_scaled' not in data.columns and 'simple_return' in data.columns: data['simple_return_scaled'] = data['simple_return'].fillna(0) * 100000
        if 'returns_squared' not in data.columns and 'simple_return' in data.columns: data['returns_squared'] = data['simple_return'].fillna(0) ** 2
        initial_rows = len(data)
        cols_to_check_nan = [c for c in ['open', 'high', 'low', 'close', 'volume', 'simple_return', 'simple_return_scaled', 'returns_squared'] if c in data.columns]
        data = data.dropna(subset=cols_to_check_nan, how='any')
        if (initial_rows - len(data)) > 0: logger.debug(f"({symbol}|{timeframe}) Rimosse {initial_rows - len(data)} righe con NaN.")
        if data.empty: logger.warning(f"({symbol}|{timeframe}) Dati OHLCV vuoti dopo pulizia."); return None
        min_required_len = config.MIN_DATA_POINTS_FOR_ANALYSIS
        if len(data) < min_required_len:
             logger.warning(f"({symbol}|{timeframe}) Dati finali ({len(data)} righe) < {min_required_len} richiesti per analisi completa. L'analisi potrebbe essere limitata o fallire.")
        else:
             logger.info(f"({symbol}|{timeframe}) Dati OHLCV pronti per analisi: {len(data)} righe.")
        return data
    except Exception as e: logger.error(f"({symbol}|{timeframe}) Errore critico in load/prepare OHLCV: {e}", exc_info=True); return None


# --- run_single_analysis_and_save (MODIFICATO per logging più chiaro) ---
def run_single_analysis_and_save(symbol: str, timeframe: str, analysis_results: Dict[str, Any]) -> bool:
    """Salva i risultati dell'analisi (se validi) in file JSON e TXT."""
    # --- MODIFICA: Log iniziale più chiaro ---
    logger.info(f"Verifica risultati analisi per {symbol} ({timeframe}) prima del salvataggio...")
    if analysis_results is None:
        logger.error(f"Salvataggio saltato ({symbol}|{timeframe}): Risultati analisi sono None.")
        return False
    if not isinstance(analysis_results, dict):
        logger.error(f"Salvataggio saltato ({symbol}|{timeframe}): Risultati analisi non sono un dizionario (tipo: {type(analysis_results)}).")
        return False
    if analysis_results.get('error'):
        logger.error(f"Salvataggio saltato ({symbol}|{timeframe}): Dizionario risultati contiene chiave 'error': {analysis_results['error']}")
        return False
    # --- FINE MODIFICA ---

    logger.info(f"({symbol}|{timeframe}) Tentativo salvataggio report JSON/TXT...") # Log spostato qui
    try:
        report_dir = config.REPORT_DIR
        # Usa ensure_directory_exists per gestire la creazione e gli errori
        from utils import ensure_directory_exists # Importa qui per evitare dipendenza circolare potenziale
        ensure_directory_exists(report_dir) # Solleva eccezione se fallisce

        timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        safe_pair = symbol.replace('/', '_').replace('\\', '_')
        output_filename_base = f"analysis_{safe_pair}_{timeframe}_{timestamp_str}"
        output_path_base = os.path.join(report_dir, output_filename_base)

        # Assicura che market_info esista prima di modificarlo
        if 'market_info' not in analysis_results or not isinstance(analysis_results['market_info'], dict):
             analysis_results['market_info'] = {}
        analysis_results['market_info']['symbol'] = symbol
        analysis_results['market_info']['timeframe'] = timeframe
        analysis_results['market_info']['analysis_save_time'] = timestamp_str

        json_filepath = f"{output_path_base}.json"
        json_success = save_results_to_json(analysis_results, json_filepath)

        txt_filepath = f"{output_path_base}.txt"
        txt_success = save_results_to_txt(analysis_results, txt_filepath)

        if not json_success or not txt_success:
            # Log migliorato se una delle scritture fallisce
            logger.error(f"({symbol}|{timeframe}) Fallimento salvataggio (JSON: {json_success}, TXT: {txt_success}). Controlla log precedenti per errori I/O.")
            return False

        logger.info(f"Report salvati con successo: {output_filename_base}.[json/txt]")
        return True
    except NotADirectoryError as e: # Cattura errore da ensure_directory_exists
         logger.error(f"({symbol}|{timeframe}) Salvataggio fallito - Percorso report non è una directory: {e}", exc_info=True)
         return False
    except Exception as save_err:
        # Log migliorato in caso di errore generico nel salvataggio
        logger.error(f"({symbol}|{timeframe}) Errore imprevisto durante salvataggio report: {save_err}", exc_info=True)
        return False


# --- update_and_save_historical_stats (Invariata) ---
def update_and_save_historical_stats(symbol: str, timeframe: str, force_fetch: bool) -> bool:
    # (Funzione Invariata - Usa StatisticalAnalyzerAdvanced per update)
    logger.info(f"*** AVVIO AGGIORNAMENTO STATISTICHE STORICHE per {symbol} ({timeframe}) ***")
    update_start_time = time.time()
    days_lookback_hist = config.DATA_LOOKBACK_PERIODS.get(timeframe, 1095)
    logger.info(f"Caricamento storico completo (~{days_lookback_hist} giorni) per analisi statistiche...")
    historical_data = load_and_prepare_data(symbol, timeframe, force_fetch, days_override=days_lookback_hist)
    if historical_data is None or historical_data.empty:
        logger.error(f"({symbol}|{timeframe}) Impossibile caricare storico per update stats.")
        return False
    logger.info(f"({symbol}|{timeframe}) Dati storici caricati: {len(historical_data)} righe, dal {safe_strftime(historical_data.index.min())} al {safe_strftime(historical_data.index.max())}")
    try:
        hist_analyzer = StatisticalAnalyzerAdvanced(historical_data.copy())
        hist_analyzer.set_symbol_timeframe(symbol, timeframe)
        success_cycles, success_intracandle = hist_analyzer.update_historical_stats()
        update_end_time = time.time(); total_duration = update_end_time - update_start_time
        logger.info(f"*** AGGIORNAMENTO STATS {symbol} ({timeframe}) - ESITO: Cicli={success_cycles}, Candele={success_intracandle} (Tot: {total_duration:.2f} sec) ***")
        return success_cycles and success_intracandle
    except Exception as e: logger.error(f"({symbol}|{timeframe}) Errore durante update stats: {e}", exc_info=True); return False


# --- run_analysis_flow (MODIFICATO per logging più chiaro) ---
def run_analysis_flow(args, analysis_params_tuple: Optional[Tuple[str, str, str]]):
    start_flow_time = time.time()
    if analysis_params_tuple is None:
        logger.error("Parametri analisi non validi.")
        return
    symbol, timeframe, action = analysis_params_tuple
    force_load_refresh = args.fetch

    if action == 'update_stats':
        logger.info(f"Modalità Update Stats Storiche: {symbol} ({timeframe}).")
        update_success = update_and_save_historical_stats(symbol, timeframe, force_load_refresh)
        if update_success: logger.info(f"Update stats storiche completato con successo per {symbol} ({timeframe}).")
        else: logger.error(f"Update stats storiche fallito o parzialmente fallito per {symbol} ({timeframe}).")
    elif action == 'analyze':
        logger.info(f"Avvio Flusso Analisi Standard per {symbol} ({timeframe})...")
        if force_load_refresh: logger.info("Forza refresh dati OHLCV abilitato.")

        # 1. Carica dati OHLCV
        data_ohlcv = load_and_prepare_data(symbol, timeframe, force_load_refresh)

        if data_ohlcv is not None and not data_ohlcv.empty:
            # 2. Esegui Analisi Principale (TradingAdvisor)
            logger.info(f"Esecuzione analisi TradingAdvisor per {symbol} ({timeframe})...")
            advisor = TradingAdvisor(data=data_ohlcv.copy(), symbol=symbol, timeframe=timeframe)
            analysis_results = advisor.analyze() # Questo ora fa stat, TA, MTF, DWM, Deribit, Fibo, Pattern, Targets, etc.

            # --- MODIFICA: Log Dettagliato dello Stato di analysis_results ---
            if analysis_results is None:
                logger.error(f"({symbol}|{timeframe}) TradingAdvisor.analyze() ha restituito None.")
                analysis_failed = True
            elif not isinstance(analysis_results, dict):
                logger.error(f"({symbol}|{timeframe}) TradingAdvisor.analyze() non ha restituito un dizionario (tipo: {type(analysis_results)}).")
                analysis_failed = True
            else:
                analysis_failed = analysis_results.get('error') is not None
                if analysis_failed:
                    logger.error(f"({symbol}|{timeframe}) TradingAdvisor.analyze() ha restituito un errore: {analysis_results.get('error')}")
                else:
                    # Logga un riassunto se l'analisi sembra ok finora
                    keys_present = list(analysis_results.keys())
                    logger.debug(f"({symbol}|{timeframe}) Risultati analisi preliminari OK. Chiavi presenti: {keys_present}")
            # --- FINE MODIFICA ---

            if analysis_failed:
                logger.error(f"Analisi principale fallita per {symbol} ({timeframe}). Non si procede all'integrazione dati Binance e al salvataggio.")
            else:
                # 3. --- INTEGRAZIONE DATI BINANCE ---
                logger.info(f"Recupero dati aggiuntivi Binance API per {symbol}...")
                try:
                    binance_symbol = symbol.replace('/', '').upper()
                    binance_specific_data = fetch_all_binance_data(
                        futures_symbol=binance_symbol,
                        spot_symbol=binance_symbol,
                        kline_interval=timeframe,
                    )
                    analysis_results['binance_market_data'] = binance_specific_data
                    logger.info("Dati Binance aggiunti ai risultati.")
                    if binance_specific_data.get('fetch_errors'):
                         logger.warning(f"Errori durante recupero dati Binance: {binance_specific_data['fetch_errors']}")
                except Exception as binance_fetch_err:
                     logger.error(f"Errore imprevisto chiamata fetch_all_binance_data: {binance_fetch_err}", exc_info=True)
                     analysis_results['binance_market_data'] = {'error': f'Fetch Binance API failed: {binance_fetch_err}'}
                # --- FINE INTEGRAZIONE ---

                # --- MODIFICA: Log prima del salvataggio ---
                logger.debug(f"({symbol}|{timeframe}) Stato finale analysis_results prima del salvataggio: "
                             f"Errore presente? {'Si' if analysis_results.get('error') else 'No'}")
                # Per debug estremo, si potrebbe loggare l'intero dizionario (ma è molto verboso)
                # logger.debug(f"Contenuto analysis_results: {json.dumps(convert_to_json_serializable(analysis_results), indent=2)}")
                # --- FINE MODIFICA ---

                # 4. --- SALVATAGGIO FINALE ---
                save_success = run_single_analysis_and_save(symbol, timeframe, analysis_results)
                if save_success:
                    logger.info(f"Analisi {symbol} ({timeframe}) completata e report salvati.")
                else:
                    logger.error(f"Salvataggio report {symbol} ({timeframe}) fallito o saltato. Controllare log precedenti.")
        else:
            logger.error(f"Impossibile eseguire analisi {symbol} ({timeframe}) - Dati OHLCV mancanti/invalidi dopo il caricamento.")
    else:
        logger.error(f"Azione non riconosciuta: {action}")

    end_flow_time = time.time()
    logger.info(f"Flusso per {symbol}({timeframe})[{action}] completato in {end_flow_time - start_flow_time:.2f} sec.")

# --- FUNZIONE MAIN (Invariata) ---
def main():
    args = parse_arguments()
    setup_logging(args.loglevel)
    if args.update_all_historical_stats and (args.pair or args.timeframe or args.update_historical_stats):
        logger.critical("ERRORE: --update-all-historical-stats non può essere usato con --pair, --timeframe, o --update-historical-stats."); sys.exit(1)
    if args.update_historical_stats and (not args.pair or not args.timeframe):
        logger.critical("ERRORE: --update-historical-stats richiede --pair e --timeframe."); sys.exit(1)
    try:
        dirs_to_ensure = [config.CACHE_DIR, config.REPORT_DIR, config.HISTORICAL_STATS_DIR];
        # Usa la funzione di utils per creare directory
        from utils import ensure_directory_exists
        for dir_path in dirs_to_ensure:
            if dir_path: ensure_directory_exists(dir_path); logger.info(f"Verificata/Creata directory: {dir_path}")
    except Exception as e:
        logger.critical(f"Impossibile creare directory necessarie ({dirs_to_ensure}): {e}. Uscita.", exc_info=True)
        sys.exit(1)
    logger.info("Inizializzazione Exchange..."); exchange_instance = get_exchange_instance()
    if exchange_instance is None: logger.critical("Fallimento inizializzazione exchange."); sys.exit(1)
    else: logger.info(f"Istanza Exchange ({exchange_instance.id}) inizializzata.")

    if args.update_all_historical_stats:
        logger.info("========= AVVIO AGGIORNAMENTO BATCH STATISTICHE STORICHE =========")
        pairs_to_update = config.MENU_ASSET_LIST; timeframes_to_update = config.HISTORICAL_STATS_BATCH_TIMEFRAMES; total_updates = len(pairs_to_update) * len(timeframes_to_update); success_count = 0; fail_count = 0; batch_start_time = time.time()
        logger.info(f"Coppie: {pairs_to_update}"); logger.info(f"Timeframes per coppia: {timeframes_to_update}"); logger.info(f"Totale previsto: {total_updates}")
        for i, pair in enumerate(pairs_to_update):
            for j, tf in enumerate(timeframes_to_update):
                current_job_num = i * len(timeframes_to_update) + j + 1; logger.info(f"\n--- Aggiornamento {current_job_num}/{total_updates}: {pair} ({tf}) ---")
                try: success = update_and_save_historical_stats(pair, tf, args.fetch);
                except Exception as batch_err: logger.error(f"Errore non gestito update {pair} ({tf}): {batch_err}", exc_info=True); success = False
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                time.sleep(1) # Pausa tra un task e l'altro

        batch_end_time = time.time(); logger.info("========= AGGIORNAMENTO BATCH COMPLETATO ========="); logger.info(f"Tempo totale: {(batch_end_time - batch_start_time):.2f} sec."); logger.info(f"Riusciti: {success_count} / {total_updates}"); logger.info(f"Falliti: {fail_count} / {total_updates}")
    else:
        is_cli_execution = bool(args.pair and args.timeframe); action_cli = 'update_stats' if args.update_historical_stats else 'analyze'
        while True:
            analysis_params_tuple: Optional[Tuple[str, str, str]] = None
            if is_cli_execution:
                 if args.pair and args.timeframe:
                     pair_normalized = args.pair.upper().replace('-', '/'); tf_valid = True; pair_valid = True
                     # Verifica validità timeframe per l'azione specifica
                     if action_cli == 'update_stats' and args.timeframe not in config.TIMEFRAMES: # Controllo più generico
                         logger.error(f"Errore CLI: Timeframe '{args.timeframe}' non definito in config.TIMEFRAMES."); tf_valid = False
                     elif action_cli == 'analyze':
                         if pair_normalized not in config.MENU_ASSET_LIST: logger.warning(f"Warning CLI: Coppia '{args.pair}' non nella lista del menu standard."); # Non bloccare, ma avvisa
                         if args.timeframe not in config.MENU_TIMEFRAME_LIST: logger.warning(f"Warning CLI: Timeframe '{args.timeframe}' non nella lista del menu standard."); # Non bloccare, ma avvisa
                         if args.timeframe not in config.TIMEFRAMES: # Controllo se esiste comunque
                             logger.error(f"Errore CLI: Timeframe '{args.timeframe}' non definito in config.TIMEFRAMES."); tf_valid = False
                     # Verifica se il simbolo esiste sull'exchange (richiede istanza)
         
                     if pair_valid and exchange_instance and hasattr(exchange_instance, 'markets') and exchange_instance.markets:
                          if pair_normalized not in exchange_instance.markets:
                              logger.error(f"Errore CLI: Simbolo '{pair_normalized}' non trovato su exchange {exchange_instance.id}.")
                              pair_valid = False
                     elif pair_valid and not exchange_instance:
                          logger.warning("Impossibile verificare validità simbolo (exchange non inizializzato).")

                     if pair_valid and tf_valid: analysis_params_tuple = (pair_normalized, args.timeframe, action_cli)
                     else: logger.error("Parametri CLI non validi.")
            else: analysis_params_tuple = get_analysis_parameters_interactive()

            if analysis_params_tuple is None:
                 if is_cli_execution: logger.info("Esecuzione singola CLI terminata o parametri non validi."); break
                 else: logger.info("Uscita o nuova selezione richiesta..."); time.sleep(1); print("\n\n"); continue # Aggiungi spazio per chiarezza menu

            run_analysis_flow(args, analysis_params_tuple)

            if is_cli_execution: logger.info("Esecuzione singola CLI completata."); break
            else:
                 print("\n" + "="*20 + " Operazione Completata " + "="*20);
                 try:
                     post_op_question = [ inquirer.List('post_action', message="Cosa vuoi fare ora?", choices=[('Nuova Selezione', 'new'), ('Esci', 'exit')], default='new') ]
                     post_answer = inquirer.prompt(post_op_question, raise_keyboard_interrupt=True)
                     if post_answer is None or post_answer['post_action'] == 'exit': logger.info("Uscita richiesta."); break
                     else: logger.info("\n" + "*"*15 + " NUOVA SELEZIONE " + "*"*15 + "\n"); time.sleep(0.5); print("\n")
                 except KeyboardInterrupt: print("\nOperazione interrotta."); break
                 except Exception as post_err: logger.error(f"Errore prompt post-op: {post_err}. Uscita."); break
        logger.info("Fine esecuzione script.")
    return 0

# --- Blocco Esecuzione Principale ---
if __name__ == "__main__":
    # Aggiungi filtro warning specifici se necessario
    # warnings.filterwarnings("ignore", message="Mean of empty slice")
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    sys.exit(main())
# --- END OF FILE main.py ---