# src/model_building/sentiment_analysis.py
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar analizador de sentimiento VADER
vader_analyzer = None

def get_vader_analyzer():
    """Carga y devuelve el analizador de sentimiento VADER."""
    global vader_analyzer
    if vader_analyzer is None:
        try:
            vader_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logging.error(f"Error al cargar el analizador VADER: {e}")
            return None
    return vader_analyzer

def analyze_sentiment_vader(text, lang='en'):
    """Analiza el sentimiento del texto utilizando VADER."""
    analyzer = get_vader_analyzer()
    if analyzer and isinstance(text, str):
        try:
            scores = analyzer.polarity_scores(text)
            # VADER devuelve 'compound', 'pos', 'neg', 'neu'
            # Adaptamos para devolver una etiqueta de sentimiento similar a pysentimiento
            if scores['compound'] >= 0.05:
                sentiment = 'POS'
            elif scores['compound'] <= -0.05:
                sentiment = 'NEG'
            else:
                sentiment = 'NEU'
            return sentiment, scores['pos'], scores['neg'], scores['neu']
        except Exception as e:
            logging.error(f"Error al analizar el sentimiento del texto '{text}' con VADER en idioma '{lang}': {e}")
    return None, None, None, None

def analyze_opinion_sentiment(df_pros_processed, df_cons_processed):
    """Realiza análisis de sentimiento en los DataFrames de ventajas y desventajas."""
    df_pros_sentiment = pd.DataFrame()
    df_cons_sentiment = pd.DataFrame()

    if not df_pros_processed.empty:
        df_pros_sentiment['ventaja'] = df_pros_processed['ventaja']
        df_pros_sentiment['lemmatized_text'] = df_pros_processed['lemmatized_text']
        sentiment_results = df_pros_processed.apply(
            lambda row: analyze_sentiment_vader(row['lemmatized_text'], lang=row.get('lang', 'es')), axis=1
        )
        df_pros_sentiment[['sentiment', 'positive', 'negative', 'neutral']] = pd.DataFrame(sentiment_results.tolist(), index=df_pros_sentiment.index)

    if not df_cons_processed.empty:
        df_cons_sentiment['desventaja'] = df_cons_processed['desventaja']
        df_cons_sentiment['lemmatized_text'] = df_cons_processed['lemmatized_text']
        sentiment_results = df_cons_processed.apply(
            lambda row: analyze_sentiment_vader(row['lemmatized_text'], lang=row.get('lang', 'es')), axis=1
        )
        df_cons_sentiment[['sentiment', 'positive', 'negative', 'neutral']] = pd.DataFrame(sentiment_results.tolist(), index=df_cons_sentiment.index)

    return df_pros_sentiment, df_cons_sentiment

if __name__ == "__main__":
    import pandas as pd
    import logging
    from preprocessing import preprocess_opinion_dataframe  # Importa la función de preprocesamiento

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define la ruta del archivo CSV generado por web_scraper.py
    output_filepath = "data/glassdoor_all_spanish_reviews.csv"

    try:
        # Lee el DataFrame desde el archivo CSV
        all_reviews_df = pd.read_csv(output_filepath)
        logging.info(f"Se cargaron {len(all_reviews_df)} registros desde {output_filepath}")

        # Preprocesa el DataFrame
        if not all_reviews_df.empty:
            df_pros_processed, df_cons_processed = preprocess_opinion_dataframe(all_reviews_df.copy())

            # Realiza el análisis de sentimiento con VADER
            df_pros_sentiment, df_cons_sentiment = analyze_opinion_sentiment(df_pros_processed, df_cons_processed)

            print("\nSentimiento en Ventajas (con VADER):")
            print(df_pros_sentiment.head())

            print("\nSentimiento en Desventajas (con VADER):")
            print(df_cons_sentiment.head())
        else:
            logging.warning("El DataFrame de reseñas está vacío. No se puede realizar el análisis de sentimiento.")

    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {output_filepath}. Asegúrate de ejecutar primero web_scraper.py y preprocessing.py")
    except Exception as e:
        logging.error(f"Ocurrió un error al leer, preprocesar o analizar el sentimiento del archivo CSV: {e}")
