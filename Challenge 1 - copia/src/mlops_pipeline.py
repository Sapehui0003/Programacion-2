# mlops_pipeline.py
import os
import mlflow as mlf
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime

# Import your modules (assuming they are in the same directory or in 'src' if you followed the previous structure)
from module_load_data import load_breast_cancer_data
from module_preprocessing import preprocess_module
from module_training import train_model_module
from module_evaluation import evaluate_model_module

# Set MLflow tracking URI (replace with your MLflow server address if needed)
mlf.set_tracking_uri("http://localhost:5000")

# Set the experiment name
mlf.set_experiment("Breast Cancer Prediction Pipeline")

def main():
    # --- Main Pipeline Run ---
    with mlf.start_run(run_name="Full_Pipeline_Run"):
        mlf.log_param("pipeline_start_time", datetime.now().isoformat())

        try:
            # --- Load Data ---
            with mlf.start_run(run_name="Data_Loading", nested=True):
                file_path = r"C:\Users\SABRINA PEREZ\anaconda3\Porgramacion-2\Challenges\Challenge 1\data\breast-cancer-wisconsin.data.csv"
                mlf.log_param("data_path", file_path)
                df = load_breast_cancer_data(file_path)
                if df is None:
                    mlf.log_param("data_load_status", "Failed")
                    return
                mlf.log_param("data_load_status", "Success")
                mlf.log_param("data_row_count", df.shape[0])
                mlf.log_param("data_column_count", df.shape[1])
                print("Data loading completed.")

            try:
                # --- Preprocess Data ---
                with mlf.start_run(run_name="Data_Preprocessing", nested=True):
                    X, y = preprocess_module(df.copy())
                    if y is None:
                        mlf.log_param("preprocessing_status", "Failed - Target variable not found")
                        print("Target variable not found. Exiting.")
                        return
                    mlf.log_param("preprocessing_status", "Success")
                    mlf.log_param("feature_count", X.shape[1] if hasattr(X, 'shape') else None)
                    mlf.log_param("sample_count", X.shape[0] if hasattr(X, 'shape') else None)
                    print("Data preprocessing completed.")

                    try:
                        # --- Model Training ---
                        with mlf.start_run(run_name="LogisticRegression_Training", nested=True):
                            model_name = "logistic_regression"
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
                            signature = infer_signature(X, params=model.get_params())
                            mlf.sklearn.log_model(model, f"{model_name}_model", signature=signature)
                            print(f"MLflow: Logged {model_name} model with signature.")
                            mlf.log_param("model_logged", True)

                            try:
                                # --- Model Evaluation ---
                                with mlf.start_run(run_name="LogisticRegression_Evaluation", nested=True):
                                    evaluated_model_name = model_name
                                    mlf.log_param("evaluated_model_name", evaluated_model_name)
                                    evaluation_metrics = evaluate_model_module(model, X_test, y_test, model_name=evaluated_model_name)
                                    mlf.log_param("evaluation_completed", True)
                                    if evaluation_metrics:
                                        for metric, value in evaluation_metrics.items():
                                            mlf.log_metric(metric, value)
                                    print("Model evaluation completed.")

                            except Exception as e:
                                mlf.log_param("evaluation_error", str(e))
                                print(f"Error during evaluation: {e}")
                                raise

                    except Exception as e:
                        mlf.log_param("training_error", str(e))
                        print(f"Error during training: {e}")
                        raise

            except Exception as e:
                mlf.log_param("preprocessing_error", str(e))
                print(f"Error during preprocessing: {e}")
                raise

        except Exception as e:
            mlf.log_param("data_loading_error", str(e))
            print(f"Error during data loading: {e}")
            raise

        mlf.log_param("pipeline_end_time", datetime.now().isoformat())
        print("\nMLOps Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()