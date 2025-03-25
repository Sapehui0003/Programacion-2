# src/module_evaluation.py

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
import numpy as np

def evaluate_and_log_model(model, X_test, y_test):
    """Evalúa el modelo y registra las métricas y artefactos en MLflow."""
    print("--- Evaluating Model ---")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = 0.0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    # Log ROC AUC if the model has predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # Plot and save ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        roc_curve_path = "roc_curve.png"
        plt.savefig(roc_curve_path)
        mlflow.log_artifact(roc_curve_path)
        plt.close()

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['B', 'M'])) # Assuming 'B' and 'M' are your labels
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # Infer and log model signature
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)
    print("Model and evaluation metrics logged to MLflow.")