# src/model_building/classification_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import logging
from preprocessing import preprocess_opinion_dataframe  # Asegúrate de que la ruta sea correcta
from sentiment_analysis import analyze_opinion_sentiment  # Asegúrate de que la ruta sea correcta

logging.basicConfiglevel=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'

def load_and_prepare_data(file_path):
    """Carga los datos, los preprocesa y realiza el análisis de sentimiento."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Datos cargados desde: {file_path}")

        # Preprocesar los datos
        df_pros_processed, df_cons_processed = preprocess_opinion_dataframe(df.copy())

        # Realizar análisis de sentimiento
        df_pros_sentiment, df_cons_sentiment = analyze_opinion_sentiment(df_pros_processed, df_cons_processed)

        # Fusionar los resultados (esto dependerá de cómo quieras estructurar tus datos para el modelo)
        # Un ejemplo simple sería concatenar los textos preprocesados y usar una etiqueta para pro/con
        df_pros_sentiment['label'] = 'positive'
        df_cons_sentiment['label'] = 'negative'
        combined_df = pd.concat([df_pros_sentiment[['lemmatized_text', 'sentiment', 'positive', 'negative', 'neutral', 'label']].rename(columns={'lemmatized_text': 'text'}),
                                  df_cons_sentiment[['lemmatized_text', 'sentiment', 'positive', 'negative', 'neutral', 'label']].rename(columns={'lemmatized_text': 'text'})],
                                 ignore_index=True)
        combined_df.dropna(subset=['text'], inplace=True) # Eliminar filas con texto vacío después del preprocesamiento

        logging.info(f"Datos preprocesados y con sentimiento combinados. Total de registros: {len(combined_df)}")
        return combined_df

    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error al cargar, preprocesar o analizar el sentimiento de los datos: {e}")
        return None

def train_model(df):
    """Entrena un modelo de clasificación."""
    if df is None or df.empty:
        logging.warning("No hay datos para entrenar el modelo.")
        return None, None

    X = df['text']
    y = df['label']

    # Vectorización TF-IDF
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Entrenar un modelo de regresión logística (puedes cambiarlo por el modelo que prefieras)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info(f"Modelo entrenado con una precisión de: {accuracy:.4f}")
    logging.info(f"Reporte de clasificación:\n{report}")

    return model, vectorizer

if _name_ == "_main_":
    # Define la ruta del archivo CSV generado por web_scraper.py
    output_filepath = "data/glassdoor_all_spanish_reviews.csv"

    # Cargar, preprocesar y obtener sentimiento de los datos
    processed_df = load_and_prepare_data(output_filepath)

    # Entrenar el modelo si los datos se procesaron correctamente
    if processed_df is not None:
        model, vectorizer = train_model(processed_df)

        # Puedes guardar el modelo y el vectorizador aquí para su uso posterior
        # import joblib
        # joblib.dump(model, 'sentiment_classification_model.pkl')
        # joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        # logging.info("Modelo y vectorizador guardados.")