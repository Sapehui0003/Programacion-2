# src/model_building/mlops_pipeline.py

import argparse
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
import numpy as np
from preprocessing import preprocess_opinion_dataframe
from sentiment_analysis import analyze_opinion_sentiment
import joblib
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
mlflow.set_tracking_uri("http://localhost:5000")

def load_and_prepare_data(data_path):
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Datos cargados desde: {data_path}")
        df_pros_processed, df_cons_processed = preprocess_opinion_dataframe(df.copy())
        df_pros_sentiment, df_cons_sentiment = analyze_opinion_sentiment(df_pros_processed, df_cons_processed)
        df_pros_sentiment['label'] = 'positive'
        df_cons_sentiment['label'] = 'negative'
        combined_df = pd.concat([df_pros_sentiment[['lemmatized_text', 'sentiment', 'positive', 'negative', 'neutral', 'label']].rename(columns={'lemmatized_text': 'text'}),
                                  df_cons_sentiment[['lemmatized_text', 'sentiment', 'positive', 'negative', 'neutral', 'label']].rename(columns={'lemmatized_text': 'text'})],
                                 ignore_index=True)
        combined_df.dropna(subset=['text'], inplace=True)
        logging.info(f"Datos preprocesados y con sentimiento combinados. Total de registros: {len(combined_df)}")
        return combined_df
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {data_path}")
        return None
    except Exception as e:
        logging.error(f"Error al cargar, preprocesar o analizar el sentimiento de los datos: {e}")
        return None

def split_data(df):
    if df is None or df.empty:
        logging.warning("No hay datos para dividir.")
        return None, None, None, None
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    logging.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_vectorized, y_train)
    logging.info("Modelo de regresión logística entrenado.")
    vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, vectorizer_path)
    logging.info(f"Vectorizador guardado localmente (después de entrenamiento) en: {vectorizer_path}")
    return model, vectorizer, vectorizer_path

def evaluate_and_log_model(model, vectorizer, X_test, y_test, vectorizer_path):
    logging.info("Entrando a la función evaluate_and_log_model")
    with mlflow.start_run():
        logging.info("Dentro del bloque mlflow.start_run()")
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Contenido del diccionario 'report':")
        print(report)

        logging.info(f"Precisión del modelo en el conjunto de prueba: {accuracy:.4f}")
        logging.info(f"Reporte de clasificación en el conjunto de prueba:\n{classification_report(y_test, y_pred)}")

        # Log parámetros
        mlflow.log_param("model_name", "LogisticRegression")
        mlflow.log_param("tfidf_vectorizer", "TfidfVectorizer")
        mlflow.log_param("split_ratio", 0.2)
        mlflow.log_param("random_state", 42)

        # Log métricas
        mlflow.log_metric("accuracy", accuracy)

        # Prueba con tag para negative precision
        mlflow.set_tag("negative_precision_tag", report['negative']['precision'])
        # Elimina o comenta la línea original de logueo de esta métrica
        # mlflow.log_metric("negative_precision", report['negative']['precision'])
        mlflow.log_metric("negative_recall", report['negative']['recall'])
        mlflow.log_metric("negative_f1_score", report['negative']['f1-score'])

        mlflow.log_metric("positive_precision", report['positive']['precision'])
        mlflow.log_metric("positive_recall", report['positive']['recall'])
        mlflow.log_metric("positive_f1_score", report['positive']['f1-score'])

        mlflow.log_metric("macro_average_precision", report['macro avg']['precision'])
        mlflow.log_metric("macro_average_recall", report['macro avg']['recall'])
        mlflow.log_metric("macro_average_f1_score", report['macro avg']['f1-score'])
        mlflow.log_metric("weighted_average_precision", report['weighted avg']['precision'])
        mlflow.log_metric("weighted_average_recall", report['weighted avg']['recall'])
        mlflow.log_metric("weighted_average_f1_score", report['weighted avg']['f1-score'])

        # Inferir firma y loguear el modelo
        signature = infer_signature(X_test, model.predict(vectorizer.transform(X_test)))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        if os.path.exists(vectorizer_path):
            mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")
            logging.info(f"Vectorizador logueado desde: {vectorizer_path}")
        else:
            logging.error(f"El archivo del vectorizador no se encuentra en: {vectorizer_path}")

        # Guardar y loguear la matriz de confusión
        cm = ConfusionMatrixDisplay.from_estimator(model, vectorizer.transform(X_test), y_test, cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
        plt.close()

        logging.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de MLOps para clasificación de opiniones")
    parser.add_argument("--data_path", type=str, required=True, help="Ruta del archivo CSV con los datos")
    args = parser.parse_args()

    logging.info("Inicio del script mlops_pipeline.py")
    processed_df = load_and_prepare_data(args.data_path)
    logging.info(f"Resultado de load_and_prepare_data: {processed_df is not None}")

    if processed_df is not None:
        X_train, X_test, y_train, y_test = split_data(processed_df)
        logging.info(f"Resultado de split_data: X_train is not None: {X_train is not None}, X_test is not None: {X_test is not None}, y_train is not None: {y_train is not None}, y_test is not None: {y_test is not None}")

        if X_train is not None:
            model, vectorizer, vectorizer_path = train_model(X_train, y_train)
            logging.info(f"Resultado de train_model: model is not None: {model is not None}, vectorizer is not None: {vectorizer is not None}, vectorizer_path: {vectorizer_path}")

            if model is not None:
                evaluate_and_log_model(model, vectorizer, X_test, y_test, vectorizer_path)
                logging.info("Finalizada la función evaluate_and_log_model")
            else:
                logging.error("El modelo es None, no se puede evaluar.")
        else:
            logging.error("El DataFrame procesado es None, no se puede continuar.")

    logging.info("Fin del script mlops_pipeline.py")