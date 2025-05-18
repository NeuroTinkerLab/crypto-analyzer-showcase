# --- START OF FILE utils.py ---
# utils.py
"""
Funzioni di utilità generiche per il progetto.
Versione rifattorizzata: mantenute solo funzioni non duplicate altrove.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone # Aggiunto timezone
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> None:
    """
    Assicura che una directory esista, creandola se necessario.

    Args:
        directory (str): Percorso della directory.
    """
    if not directory:
        logger.warning("Tentativo di creare una directory con percorso vuoto.")
        return
    try:
        # Check esistenza e creazione
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True) # exist_ok=True evita errori se esiste già
            logger.info(f"Directory creata: {directory}")
        # Check se è effettivamente una directory (potrebbe essere un file)
        elif not os.path.isdir(directory):
             logger.error(f"Il percorso specificato '{directory}' esiste ma non è una directory.")
             raise NotADirectoryError(f"Path '{directory}' is not a directory.")

    except OSError as e:
        # Gestisce errori specifici di creazione (es. permessi)
        logger.error(f"Errore OSError nella creazione della directory {directory}: {e}", exc_info=True)
        raise # Rilancia l'eccezione per segnalare il fallimento
    except Exception as e:
        logger.error(f"Errore imprevisto in ensure_directory_exists per {directory}: {e}", exc_info=True)
        raise # Rilancia l'eccezione

def timestamp_to_str(timestamp_ms: Optional[Union[int, float]]) -> str:
    """
    Converte un timestamp (in millisecondi, es. da API) in una stringa data/ora UTC.

    Args:
        timestamp_ms (Optional[Union[int, float]]): Timestamp in millisecondi.

    Returns:
        str: Data e ora formattate (YYYY-MM-DD HH:MM:S) in UTC, o "N/A" in caso di errore/input non valido.
    """
    fallback_str = "N/A"
    if timestamp_ms is None or not isinstance(timestamp_ms, (int, float)) or pd.isna(timestamp_ms):
        # logger.debug(f"Timestamp non valido fornito a timestamp_to_str: {timestamp_ms}")
        return fallback_str
    try:
        # Converti millisecondi in secondi
        timestamp_sec = timestamp_ms / 1000.0
        # Crea oggetto datetime UTC
        # Usare fromtimestamp con tz=timezone.utc è più robusto di utcfromtimestamp
        dt_utc = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc)
        # Formatta la stringa
        return dt_utc.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, OverflowError, OSError) as e:
         # Errori comuni nella conversione di timestamp
         logger.warning(f"Errore nella conversione del timestamp {timestamp_ms}: {e}")
         return fallback_str
    except Exception as e:
        # Errori imprevisti
        logger.error(f"Errore imprevisto in timestamp_to_str per {timestamp_ms}: {e}", exc_info=True)
        return fallback_str

# --- Funzioni Rimosse ---
# - format_percentage (duplicata in statistical_analyzer_helpers)
# - create_histogram (non necessaria per output LLM)
# - calculate_rolling_metric (logica ora in StatisticalAnalyzerAdvanced)
# - calculate_drawdowns (duplicata in statistical_analyzer_helpers)

logger.debug("Modulo utils.py caricato (versione rifattorizzata).")
# --- END OF FILE utils.py ---