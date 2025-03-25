import os  # Agregar esta línea
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Crear la carpeta "plots" si no existe
os.makedirs("plots", exist_ok=True)

def evaluate_and_log_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo: {accuracy:.2f}")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Cáncer", "Cáncer"], yticklabels=["No Cáncer", "Cáncer"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.savefig("plots/confusion_matrix.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("plots/roc_curve.png")
    plt.close()

    # Registrar modelo en MLflow
    input_example = pd.DataFrame(X_test[:1]).astype("float64")  # Convertir a float64
    with mlflow.start_run():
        mlflow.log_param("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model", input_example=input_example)
        mlflow.log_artifact("plots/confusion_matrix.png")
        mlflow.log_artifact("plots/roc_curve.png")

