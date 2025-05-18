import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
from typing import Optional, Dict, Any
# import joblib # Opzionale: per salvare/caricare il modello

logger = logging.getLogger(__name__)

class MLPredictor:
    """
    Classe per addestrare un modello di machine learning (Random Forest)
    e prevedere la direzione del prezzo successivo (rialzo/ribasso).
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Inizializza il predittore ML.

        Args:
            n_estimators (int): Numero di alberi nel RandomForest.
            random_state (int): Seed per la riproducibilità.
        """
        # Non memorizza i dati qui, li riceve per training/prediction
        # self.data = data
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False
        self.feature_columns = None # Memorizza le colonne usate per l'addestramento

    # --- Calcolo Indicatori (Interno - Potenziale Duplicazione) ---
    # Nota: Questi calcoli duplicano logica da TechnicalAnalyzer.
    # Considerare l'uso di una libreria (pandas_ta) o di TechnicalAnalyzer.

    def _calculate_rsi(self, series: pd.Series, period=14) -> pd.Series:
        """Calcola RSI."""
        if not isinstance(series, pd.Series) or series.empty: return pd.Series(dtype=float)
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).fillna(0.0)
        loss = -delta.where(delta < 0, 0.0).fillna(0.0)

        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0) # Riempi NaN iniziali con 50 (neutro)

    def _calculate_macd_hist(self, series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
        """Calcola l'istogramma MACD."""
        if not isinstance(series, pd.Series) or series.empty: return pd.Series(dtype=float)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return histogram.fillna(0.0) # Riempi NaN iniziali con 0

    def _calculate_bollinger_percent_b(self, series: pd.Series, window=20, num_std=2) -> pd.Series:
        """Calcola %B delle Bande di Bollinger."""
        if not isinstance(series, pd.Series) or series.empty: return pd.Series(dtype=float)
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std().replace(0, 1e-10) # Evita divisione per zero

        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        percent_b = (series - lower_band) / (upper_band - lower_band).replace(0, 1e-10)
        return percent_b.fillna(0.5) # Riempi NaN iniziali con 0.5 (centro)

    # --- Preparazione Feature ---

    def prepare_features(self, data: pd.DataFrame, window=10) -> tuple[pd.DataFrame, pd.Series, list]:
        """
        Prepara le feature e il target per l'addestramento del modello.

        Args:
            data (pd.DataFrame): DataFrame con dati OHLCV e DatetimeIndex.
            window (int): Finestra temporale per le feature di rendimento/volume passate.

        Returns:
            tuple[pd.DataFrame, pd.Series, list]:
                - DataFrame X: Feature pronte per il training.
                - Series y: Variabile target (0 o 1).
                - list: Nomi delle colonne delle feature utilizzate.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
             raise ValueError("Dati di input non validi per prepare_features.")
        if not all(c in data.columns for c in ['open', 'high', 'low', 'close', 'volume']):
             raise ValueError("Colonne OHLCV mancanti nei dati di input.")

        df = data.copy()

        # 1. Feature basate su rendimenti e volume passati
        for i in range(1, window + 1):
            df[f'return_{i}'] = df['close'].pct_change(i)
            df[f'volume_change_{i}'] = df['volume'].pct_change(i)

        # 2. Feature basate su indicatori tecnici
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['macd_hist'] = self._calculate_macd_hist(df['close'])
        df['bb_percent_b'] = self._calculate_bollinger_percent_b(df['close'])
        # Aggiungi altri indicatori rilevanti se necessario (es. ATR, ADX, Stochastics...)
        # df['atr_14'] = ...
        # df['adx_14'] = ...

        # 3. Crea la variabile target (1 se il prezzo *successivo* sale, 0 altrimenti)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

        # 4. Pulizia: Rimuovi NaN introdotti dai calcoli (pct_change, rolling, shift)
        # Questo rimuove anche l'ultima riga che ha target NaN a causa dello shift(-1)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # 5. Seleziona le colonne delle feature
        # Escludi colonne OHLCV, timestamp (se esiste come colonna), e target
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target', 'timestamp', df.index.name]]
        # Verifica se rimangono feature
        if not feature_cols:
            raise ValueError("Nessuna colonna feature valida trovata dopo la preparazione.")

        X = df[feature_cols]
        y = df['target']

        return X, y, feature_cols

    def prepare_features_for_prediction(self, data: pd.DataFrame, window=10) -> pd.DataFrame:
        """
        Prepara le feature da nuovi dati per la previsione.
        Importante: 'data' deve contenere storia sufficiente per calcolare tutte le feature.

        Args:
            data (pd.DataFrame): DataFrame con dati OHLCV recenti, inclusa storia sufficiente.
            window (int): Finestra temporale usata per creare le feature (deve corrispondere al training).

        Returns:
            pd.DataFrame: DataFrame contenente le feature per l'ultima riga (o più righe).
                          Restituisce DataFrame vuoto se non ci sono dati validi dopo la preparazione.
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
             logger.warning("Dati input vuoti per prepare_features_for_prediction.")
             return pd.DataFrame()
        if not all(c in data.columns for c in ['open', 'high', 'low', 'close', 'volume']):
             logger.warning("Colonne OHLCV mancanti nei dati input.")
             return pd.DataFrame()

        # Verifica se ci sono abbastanza dati per il calcolo delle feature
        min_required_data = window + 30 # Stima approssimativa (window per pct_change + periodi indicatori)
        if len(data) < min_required_data:
             logger.warning(f"Dati insufficienti ({len(data)}) per preparare feature per predizione. Minimo richiesto: ~{min_required_data}.")
             return pd.DataFrame()

        df = data.copy()

        # Calcola le stesse feature del training (SENZA creare 'target')
        for i in range(1, window + 1):
            df[f'return_{i}'] = df['close'].pct_change(i)
            df[f'volume_change_{i}'] = df['volume'].pct_change(i)
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['macd_hist'] = self._calculate_macd_hist(df['close'])
        df['bb_percent_b'] = self._calculate_bollinger_percent_b(df['close'])
        # Aggiungi altri indicatori calcolati durante il training

        # Rimuovi eventuali NaN (principalmente all'inizio a causa dei calcoli)
        df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

        if df_clean.empty:
             logger.warning("Nessun dato valido rimasto dopo il calcolo delle feature per la predizione.")
             return pd.DataFrame()

        # Seleziona SOLO le colonne delle feature usate durante l'addestramento
        if self.feature_columns is None:
            raise RuntimeError("Il modello non è stato addestrato o le feature columns non sono state salvate.")

        # Verifica che tutte le feature necessarie siano presenti dopo i calcoli
        missing_features = [col for col in self.feature_columns if col not in df_clean.columns]
        if missing_features:
             logger.error(f"Feature mancanti dopo la preparazione per predizione: {missing_features}")
             return pd.DataFrame()

        # Restituisci il DataFrame con solo le feature richieste, potenzialmente solo l'ultima riga valida
        return df_clean[self.feature_columns]


    # --- Training e Prediction ---

    def train_model(self, data: pd.DataFrame, window=10, test_size=0.2):
        """
        Addestra il modello RandomForest sui dati forniti.

        Args:
            data (pd.DataFrame): DataFrame completo con dati storici OHLCV.
            window (int): Finestra per la creazione delle feature.
            test_size (float): Percentuale di dati da usare per il test set.

        Returns:
            float: Accuratezza del modello sul test set.
                   Restituisce None se l'addestramento fallisce.
        """
        logger.info(f"Inizio addestramento modello ML con finestra={window}, test_size={test_size}...")
        try:
            X, y, feature_cols = self.prepare_features(data, window=window)

            if X.empty or y.empty:
                 logger.error("Nessuna feature o target disponibile dopo la preparazione dati per il training.")
                 self.is_trained = False
                 return None

            self.feature_columns = feature_cols # Salva i nomi delle feature usate
            logger.info(f"Numero di feature utilizzate: {len(self.feature_columns)}")
            logger.debug(f"Feature: {self.feature_columns}")

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.model.random_state, shuffle=False # Non mischiare serie temporali!
            )
            logger.info(f"Dimensioni Train set: {X_train.shape}, Test set: {X_test.shape}")

            if len(X_train) == 0 or len(X_test) == 0:
                 logger.error("Train o Test set vuoti dopo lo split. Impossibile addestrare.")
                 self.is_trained = False
                 return None

            # Addestra il modello
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info("Modello addestrato con successo.")

            # Valuta sul test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Accuratezza modello su Test set: {accuracy:.4f}")

            # Opzionale: Salvare il modello addestrato
            # model_filename = 'ml_predictor_model.joblib'
            # feature_filename = 'ml_predictor_features.json'
            # joblib.dump(self.model, model_filename)
            # with open(feature_filename, 'w') as f:
            #     json.dump(self.feature_columns, f)
            # logger.info(f"Modello salvato in {model_filename}, Features in {feature_filename}")

            return accuracy

        except ValueError as ve:
             logger.error(f"Errore di valore durante la preparazione feature o training: {ve}", exc_info=True)
             self.is_trained = False
             return None
        except Exception as e:
            logger.error(f"Errore imprevisto durante l'addestramento del modello: {e}", exc_info=True)
            self.is_trained = False
            return None

    def predict(self, current_data: pd.DataFrame, window=10) -> Optional[Dict[str, Any]]:
        """
        Effettua previsioni sulla direzione del prezzo per l'ultimo punto disponibile
        nei dati forniti.

        Args:
            current_data (pd.DataFrame): DataFrame contenente i dati più recenti,
                                         INCLUSA la storia necessaria per calcolare le feature
                                         (almeno `window +` periodi per indicatori).
            window (int): Finestra usata per creare le feature (deve corrispondere al training).

        Returns:
            Optional[Dict[str, Any]]: Dizionario con 'prediction' ('rialzo' o 'ribasso')
                                     e 'probability' (float %).
                                     Restituisce None se la previsione fallisce.
        """
        if not self.is_trained:
            logger.error("Il modello deve essere addestrato prima di effettuare previsioni. Chiamare train_model().")
            return None
        if self.feature_columns is None:
             logger.error("Nomi delle feature non disponibili. Addestrare nuovamente il modello.")
             return None

        logger.debug("Preparazione feature per la previsione...")
        try:
            # Prepara le feature usando gli ultimi dati + storia necessaria
            X_predict = self.prepare_features_for_prediction(current_data, window=window)

            if X_predict.empty:
                logger.warning("Nessuna feature valida generata per la previsione.")
                return None

            # Usa solo l'ultima riga disponibile dopo la preparazione per la previsione
            X_last = X_predict.iloc[-1:]
            logger.debug(f"Feature per l'ultima previsione:\n{X_last}")

            # Effettua la previsione
            prediction_int = self.model.predict(X_last)[0]
            probabilities = self.model.predict_proba(X_last)[0]
            prediction_label = 'rialzo' if prediction_int == 1 else 'ribasso'
            probability_pct = probabilities[prediction_int] * 100

            logger.info(f"Previsione ML: {prediction_label} (Probabilità: {probability_pct:.2f}%)")

            return {
                'prediction': prediction_label,
                'probability': probability_pct
            }

        except RuntimeError as re:
             logger.error(f"Errore di runtime durante la predizione: {re}")
             return None
        except Exception as e:
            logger.error(f"Errore imprevisto durante la previsione: {e}", exc_info=True)
            return None

    # Opzionale: Metodo per caricare un modello salvato
    # def load_model(self, model_path='ml_predictor_model.joblib', features_path='ml_predictor_features.json'):
    #     try:
    #         self.model = joblib.load(model_path)
    #         with open(features_path, 'r') as f:
    #             self.feature_columns = json.load(f)
    #         self.is_trained = True
    #         logger.info(f"Modello caricato da {model_path}, Features da {features_path}")
    #         return True
    #     except FileNotFoundError:
    #         logger.error("File modello o feature non trovati.")
    #         return False
    #     except Exception as e:
    #         logger.error(f"Errore caricamento modello: {e}")
    #         return False