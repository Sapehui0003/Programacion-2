# src/model_building/preprocessing.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.util import ngrams
from collections import Counter
import logging
# from src.data_extraction.web_scraper import scrape_glassdoor_paged_spanish, save_dataframe  # Asegúrate de que la ruta sea correcta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words_es = set(stopwords.words('spanish'))
stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def detect_language(text):
    """Intenta detectar el idioma predominante del texto usando rangos de caracteres Unicode."""
    if not isinstance(text, str):
        return None
    spanish_count = 0
    english_count = 0
    total_count = 0
    for char in text:
        if 0xC0 <= ord(char) <= 0xFF:  # Rango para caracteres latinos extendidos (acentos españoles, etc.)
            spanish_count += 1
        elif 'a' <= char <= 'z':
            english_count += 1
        total_count += 1

    if total_count == 0:
        return None
    elif spanish_count / total_count > 0.5:
        return 'es'
    elif english_count / total_count > 0.5:
        return 'en'
    return None

def create_language_dataframe(df, text_column):
    """Crea DataFrames separados para español e inglés usando una detección de idioma simple."""
    df_es = df[df[text_column].apply(detect_language) == 'es']
    df_en = df[df[text_column].apply(detect_language) == 'en']
    logging.info(f"DataFrame en español creado con {len(df_es)} registros.")
    logging.info(f"DataFrame en inglés creado con {len(df_en)} registros.")
    return df_es, df_en

def clean_text(text):
    """Limpia el texto eliminando caracteres especiales y convirtiendo a minúsculas."""
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', '', text)
        return text.lower().strip()
    return ''

def remove_stopwords(text, lang='en'):
    """Elimina las stopwords del texto."""
    if isinstance(text, str):
        words = text.split()
        if lang == 'es':
            return ' '.join([word for word in words if word not in stop_words_es])
        else:
            return ' '.join([word for word in words if word not in stop_words_en])
    return ''

def lemmatize_text(text, lang='en'):
    """Lematiza el texto."""
    if isinstance(text, str):
        words = text.split()
        if lang == 'es':
            return ' '.join(words)  # Sin lematización compleja en español por ahora
        else:
            return ' '.join([lemmatizer.lemmatize(word) for word in words])
    return ''

def get_ngrams(texts, n=2):
    """Calcula la distribución de n-gramas."""
    all_ngrams = []
    for text in texts:
        if isinstance(text, str):
            n_grams = ngrams(text.split(), n)
            all_ngrams.extend([' '.join(ngram) for ngram in n_grams])
    return Counter(all_ngrams).most_common(10) # Obtener los 10 n-gramas más comunes

def preprocess_dataframe(df, text_column, lang='en'):
    """Aplica el preprocesamiento al DataFrame."""
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['stopwords_removed'] = df['cleaned_text'].apply(lambda x: remove_stopwords(x, lang))
    df['lemmatized_text'] = df['stopwords_removed'].apply(lambda x: lemmatize_text(x, lang))
    logging.info(f"Preprocesamiento completado para el idioma: {lang}")
    return df

def preprocess_opinion_dataframe(df):
    """Preprocesa el DataFrame con columnas 'ventaja' y 'desventaja'."""
    df_pros_es, df_pros_en = create_language_dataframe(df.copy(), 'ventaja')
    df_cons_es, df_cons_en = create_language_dataframe(df.copy(), 'desventaja')

    df_pros_processed = pd.DataFrame()
    df_cons_processed = pd.DataFrame()

    if not df_pros_es.empty:
        df_pros_processed = pd.concat([df_pros_processed, preprocess_dataframe(df_pros_es.copy(), 'ventaja', 'es')])
    if not df_pros_en.empty:
        df_pros_processed = pd.concat([df_pros_processed, preprocess_dataframe(df_pros_en.copy(), 'ventaja', 'en')])
    if not df_cons_es.empty:
        df_cons_processed = pd.concat([df_cons_processed, preprocess_dataframe(df_cons_es.copy(), 'desventaja', 'es')])
    if not df_cons_en.empty:
        df_cons_processed = pd.concat([df_cons_processed, preprocess_dataframe(df_cons_en.copy(), 'desventaja', 'en')])

    return df_pros_processed, df_cons_processed

if __name__ == "__main__":
    # Importa pandas aquí para leer el CSV
    import pandas as pd
    # Define la ruta del archivo CSV generado por web_scraper.py
    output_filepath = "data/glassdoor_all_spanish_reviews.csv"

    try:
        # Lee el DataFrame desde el archivo CSV
        all_reviews_df = pd.read_csv(output_filepath)
        logging.info(f"Se cargaron {len(all_reviews_df)} registros desde {output_filepath}")

        # Preprocesa el DataFrame
        if not all_reviews_df.empty:
            df_pros_processed, df_cons_processed = preprocess_opinion_dataframe(all_reviews_df.copy())

            print("\nVentajas preprocesadas:")
            print(df_pros_processed[['ventaja', 'lemmatized_text']].head())
            ngrams_pros_es = get_ngrams(df_pros_processed[df_pros_processed['ventaja'].apply(detect_language) == 'es']['lemmatized_text'], n=2)
            print("\nN-gramas (ventajas - español):", ngrams_pros_es)
            ngrams_pros_en = get_ngrams(df_pros_processed[df_pros_processed['ventaja'].apply(detect_language) == 'en']['lemmatized_text'], n=2)
            print("\nN-gramas (ventajas - inglés):", ngrams_pros_en)

            print("\nDesventajas preprocesadas:")
            print(df_cons_processed[['desventaja', 'lemmatized_text']].head())
            ngrams_cons_es = get_ngrams(df_cons_processed[df_cons_processed['desventaja'].apply(detect_language) == 'es']['lemmatized_text'], n=2)
            print("\nN-gramas (desventajas - español):", ngrams_cons_es)
            ngrams_cons_en = get_ngrams(df_cons_processed[df_cons_processed['desventaja'].apply(detect_language) == 'en']['lemmatized_text'], n=2)
            print("\nN-gramas (desventajas - inglés):", ngrams_cons_en)
        else:
            logging.warning("El DataFrame de reseñas está vacío. No se puede realizar el preprocesamiento.")

    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {output_filepath}. Asegúrate de ejecutar primero web_scraper.py")
    except Exception as e:
        logging.error(f"Ocurrió un error al leer o procesar el archivo CSV: {e}")