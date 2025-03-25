# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def preprocess_module(df):
    """
    Performs data exploration and preprocessing on the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A tuple containing the scaled features (X_scaled) and the target variable (y) if present,
               otherwise just the scaled features (X_scaled, None).
    """
    print("--- Data Exploration ---")
    print("Data Info:")
    df.info()
    print("\nData Describe:")
    print(df.describe())
    print("\nData Value Counts for Each Column:")
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())
    print("\nNull Value Counts:")
    print(df.isnull().sum())
    # Add checks for other characters if needed (e.g., non-numeric in numeric columns)
    print("\n--- End of Data Exploration ---")

    print("\n--- Data Preprocessing ---")
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')
    if 'diagnosis' in df.columns:
        print("Encoding 'diagnosis' column.")
        df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]
    else:
        print("Warning: 'diagnosis' column not found for label encoding.")
        X = df
        y = None

    print("Normalizing features.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("--- End of Data Preprocessing ---")

    return X_scaled, y

if __name__ == '__main__':
    from module_load_data import load_breast_cancer_data
    import os

    # Explicitly define the path to your dataset
    file_path_example = r"C:\Users\SABRINA PEREZ\anaconda3\Porgramacion-2\Challenges\Challenge 1\data\breast-cancer-wisconsin.data.csv"

    df = load_breast_cancer_data(file_path_example)
    if df is not None:
        X_scaled, y = preprocess_module(df)
        print("\nProcessed Data (first 5 rows of scaled features):")
        print(X_scaled[:5])
        if y is not None:
            print("\nTarget variable (first 5 values):")
            print(y[:5])
        else:
            print("\nNo target variable found after preprocessing.")
    else:
        print(f"Could not load data for preprocessing example from: {file_path_example}")