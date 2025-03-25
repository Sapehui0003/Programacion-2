# data/load_data.py
import pandas as pd
import os

def load_breast_cancer_data(file_path):
    """Loads the breast cancer Wisconsin dataset from a CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage with the specific path
    file_path_example = r"C:\Users\SABRINA PEREZ\anaconda3\Porgramacion-2\data"
    df = load_breast_cancer_data(file_path_example)
    if df is not None:
        print("Data loaded successfully (from example with specific path).")
        print(df.head())