# mlops_pipeline.py
import os
import mlflow as mlf
import mlflow.sklearn
from mlflow.models import infer_signature

# Import your modules
from module_load_data import load_breast_cancer_data
from module_preprocessing import preprocess_module
from module_training import train_model_module
from module_evaluation import evaluate_model_module

# Set MLflow tracking URI
mlf.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlf.set_experiment("Breast Cancer Prediction Pipeline")

def main():
    # --- Load Data ---
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'breast-cancer-wisconsin.data.csv')
    df = load_breast_cancer_data(file_path)
    if df is None:
        return

    # --- Preprocess Data ---
    X, y = preprocess_module(df)
    if y is None:
        print("Target variable not found. Exiting.")
        return

    # --- Model Training ---
    model_name = "LogisticRegression"  # You can change this to train different models
    with mlf.start_run(run_name=f"{model_name} Training"):
        # Log parameters
        mlf.log_param("model_name", model_name)
        test_size = 0.2
        random_state = 42
        mlf.log_param("test_size", test_size)
        mlf.log_param("random_state", random_state)

        # Train the model
        model, X_test, y_test = train_model_module(
            X, y, model_name=model_name, test_size=test_size, random_state=random_state
        )

        # Infer model signature
        signature = infer_signature(X_train=X, model_input=X)
        mlf.sklearn.log_model(model, f"{model_name}_model", signature=signature)
        print(f"MLflow: Logged model signature.")

    # --- Model Evaluation ---
    with mlf.start_run(run_name=f"{model_name} Evaluation"):
        # Load the trained model (optional, if you want a separate evaluation run)
        # logged_model = f"runs:/{mlf.active_run().info.run_id}/{model_name}_model"
        # loaded_model = mlflow.sklearn.load_model(logged_model)

        # Evaluate the model
        evaluate_model_module(model, X_test, y_test, model_name=model_name)

        # Get the run ID for tracking URL
        run_id = mlf.active_run().info.run_id
        tracking_url = mlf.get_tracking_uri()
        print(f"\nMLflow Tracking URL: {tracking_url}")
        print(f"MLflow Run ID for Evaluation: {run_id}")

if __name__ == "__main__":
    main()