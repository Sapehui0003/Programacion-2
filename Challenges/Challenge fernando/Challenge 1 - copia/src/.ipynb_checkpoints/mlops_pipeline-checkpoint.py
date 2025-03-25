# mlops_pipeline.py

import argparse
from preprocessing import load_data, preprocess_data, split_data
from training import train_model
from evaluation import evaluate_and_log_model

def main(data_path):
    df = load_data(data_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

    # Entrenar modelo
    model = train_model(X_train, y_train, n_estimators=100)

    # Evaluar y registrar en MLflow
    evaluate_and_log_model(model, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de MLOps para detección de cáncer")
    parser.add_argument("--data_path", type=str, required=True, help="Ruta del archivo CSV con los datos")

    args = parser.parse_args()
    main(args.data_path)
