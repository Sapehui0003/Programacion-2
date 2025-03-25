# src/evaluation.py
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow as mlf

def evaluate_model_module(model, X_test, y_test, model_name="trained_model"):
    """
    Evaluates the trained model and prints metrics and plots, logging them with MLflow.

    Args:
        model: Trained machine learning model (must have predict and predict_proba methods).
        X_test (pd.DataFrame or np.ndarray): Test features.
        y_test (pd.Series or np.ndarray): True labels for the test set.
        model_name (str): Name of the model for plot titles and MLflow run.
    """
    print(f"--- Model Evaluation: {model_name} ---")
    with mlf.start_run(run_name=f"{model_name}Evaluation"):
        try:
            y_pred = model.predict(X_test)

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            mlf.log_metric("precision", precision)
            mlf.log_metric("recall", recall)
            mlf.log_metric("f1_score", f1)

            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)

            # Plot Confusion Matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {model_name}')
            cm_filename = f"confusion_matrix_{model_name}.png"
            plt.savefig(cm_filename)
            plt.close()
            mlf.log_artifact(cm_filename, "confusion_matrix")
            print(f"\nConfusion matrix saved as {cm_filename}")

            # Calculate ROC Curve (only if the model has predict_proba)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
                plt.xlabel('False Positive Rate (FPR)')
                plt.ylabel('True Positive Rate (TPR)')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc='lower right')
                plt.grid()
                roc_filename = f"roc_curve_{model_name}.png"
                plt.savefig(roc_filename)
                plt.close()
                mlf.log_artifact(roc_filename, "roc_curve")
                mlf.log_metric("roc_auc", roc_auc)
                print(f"ROC curve saved as {roc_filename}")
                print(f"ROC AUC: {roc_auc:.2f}")
            else:
                print(f"\n{model_name} does not have predict_proba method, ROC curve cannot be calculated.")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            mlf.log_param("error", f"Error during evaluation: {e}")
            raise

    print(f"--- End of Model Evaluation: {model_name} ---")

if __name__ == '__main__':
    from module_load_data import load_breast_cancer_data
    from module_preprocessing import preprocess_module
    from module_training import train_model_module # Import the training module
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    # Example usage
    file_path_example = r"C:\Users\SABRINA PEREZ\anaconda3\Porgramacion-2\Challenges\Challenge 1\data\breast-cancer-wisconsin.data.csv"
    df = load_breast_cancer_data(file_path_example)

    if df is not None:
        X, y = preprocess_module(df)
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Train a simple model for demonstration
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the trained model
            evaluate_model_module(model, X_test, y_test, model_name="LogisticRegression_Example")

        else:
            print("\nNo target variable available, cannot evaluate model.")
    else:
        print(f"Could not load data for evaluation example from: {file_path_example}")
    