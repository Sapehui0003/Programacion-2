# preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    """Carga el dataset desde un archivo CSV."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Realiza limpieza y preprocesamiento del dataset."""
    # Eliminar columnas innecesarias
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

    # Codificar la variable objetivo (M = 1, B = 0)
    df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])

    # Separar características y variable objetivo
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    # Normalizar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
