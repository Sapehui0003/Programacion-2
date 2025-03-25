# Breast Cancer Prediction MLOps Pipeline

## 1. Documentation

This document provides a technical overview of the Breast Cancer Prediction MLOps pipeline.

### ● Dataset Extraction

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset. This dataset can be obtained from the UCI Machine Learning Repository or directly from the provided `data` directory within this repository.

To load the dataset, the `load_data` function in `src/module_preprocessing.py` is used. This function takes the file path of the CSV file as input and returns a pandas DataFrame.

### ● Model Construction

The pipeline currently trains a Random Forest Classifier using the `train_model` function in `src/module_training.py`.

**Model:** Random Forest Classifier
**Parameters (currently hardcoded):**
  - `n_estimators`: 100
  - `random_state`: 42 (for reproducibility)

The `train_model` function takes the training features (`X_train`) and target variable (`y_train`) as input, initializes a `RandomForestClassifier` with the specified parameters, trains the model using the training data, and returns the trained model.

### ● MLOps

This project utilizes MLflow to track experiments, logging parameters, metrics, artifacts, and saving the trained model.

- **Tracking URI:** The MLflow tracking server is configured to run locally at `http://localhost:5000`. This is set in `mlops_pipeline.py`.
- **Experiment:** All runs are logged under the "Default" experiment, which is also set in `mlops_pipeline.py`.
- **Metric Logging:** The `evaluate_and_log_model` function in `src/module_evaluation.py` calculates and logs the following metrics on the test set:
    - `test_accuracy`
    - `test_precision`
    - `test_recall`
    - `test_f1`
    - `test_roc_auc` (if the model supports probability predictions)
- **Artifact Logging:** The `evaluate_and_log_model` function also logs the following artifacts:
    - `roc_curve.png`: The Receiver Operating Characteristic (ROC) curve plot.
    - `confusion_matrix.png`: The confusion matrix plot.
- **Model Logging:** The trained `RandomForestClassifier` model is saved as an MLflow Sklearn model with an inferred signature in `src/module_evaluation.py`.



### ● Execution Guide

This section provides instructions for technical users on how to set up and run the MLOps pipeline.

#### 1. Code Retrieval from GitHub

Clone the repository using the following command:
git clone<repository_url>
cd <repository_name>

#### 2.Dependency installation

pip install -r requirements.txt

Ensure that the requirements.txt file includes the necessary libraries such as pandas, scikit-learn, mlflow, and matplotlib. A sample requirements.txt would look like this:

pandas
scikit-learn
mlflow
matplotlib
numpy

#### 3.Pipeline Execution
python src/mlops_pipeline.py --data_path data/breast-cancer-wisconsin.data.csv

Ensure that the data directory and the breast-cancer-wisconsin.data.csv file are located in the correct relative path from where you are running the command. The --data_path argument should point to the location where you have stored the breast-cancer-wisconsin.data.csv file within your local file system. For example, if you have placed the data directory at the root of your mlops project, the command shown above is correct.

#### 4. Monitoring MLFlow
mlflow ui
Then, open your web browser and navigate to http://127.0.0.1:5000 to view the registered experiments, metrics, models, and visualizations.

#### 4. Uploading results to GitHub
To automatically upload your results to GitHub, you can use the provided upload_results.py script. Ensure this script is in the root of your project directory.

You only need to execute:
python upload_results.py

This script will automatically add all changes, commit them with the message 'Resultados del challenge', and push them to the origin main branch of your GitHub repository.

### ● Architecture Diagram
graph LR
    A[Data Source: breast-cancer-wisconsin.data.csv] --> B(Load Data: module_preprocessing.py);
    B --> C(Preprocess Data: module_preprocessing.py);
    C --> D(Split Data: module_preprocessing.py);
    D -- Training Data (X_train, y_train) --> E(Train Model: module_training.py);
    D -- Testing Data (X_test, y_test) --> F(Evaluate Model: module_evaluation.py);
    E --> G{MLflow Tracking};
    F --> G;
    G -- Parameters, Metrics, Artifacts, Model --> H[MLflow UI (http://localhost:5000)];