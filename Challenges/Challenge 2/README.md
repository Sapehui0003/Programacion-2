# Opinion Sentiment Analysis MLOps Pipeline

## 1. Documentation

This document provides a technical overview of the Opinion Sentiment Analysis MLOps pipeline.

### ● Dataset Extraction

The dataset used in this project consists of product reviews in Spanish and English. The data is expected to be in a CSV file, where each row represents a review and contains text and potentially other metadata. The pipeline is designed to process this data for sentiment analysis.

To load the dataset, the `load_and_prepare_data` function in `src/model_building/mlops_pipeline.py` is used. This function takes the file path of the CSV file as input and returns a pandas DataFrame that has been preprocessed and had sentiment analysis applied.
It has been downloaded from Glssdor website and the reviews are from Google Evaluations https://www.glassdoor.com.mx/Evaluaciones/Google-Evaluaciones-E9079_P3.htm?filter.iso3Language=spa 

### ● Data Preprocessing and Sentiment Analysis

The pipeline includes steps for preprocessing the text data and analyzing its sentiment. These steps are performed by functions in `preprocessing.py` and `sentiment_analysis.py`.

- **Preprocessing:** The `preprocess_opinion_dataframe` function handles tasks such as tokenization, lemmatization, and removal of stop words, tailored for both Spanish and English.
- **Sentiment Analysis:** The `analyze_opinion_sentiment` function uses a sentiment lexicon or model to determine the sentiment of each review, categorizing it as positive, negative, or neutral, and provides scores for each category.

The `load_and_prepare_data` function in the MLOps pipeline orchestrates these preprocessing and sentiment analysis steps before preparing the data for model training.

### ● Model Construction

The pipeline currently trains a Logistic Regression model using the `train_model` function in `src/model_building/mlops_pipeline.py`.

**Model:** Logistic Regression
**Parameters (currently hardcoded):**
  - `solver`: 'liblinear'

The `train_model` function takes the training features (vectorized text) and the target variable (`y_train`) as input, initializes a `LogisticRegression` model with the specified parameters, trains the model using the training data, and returns the trained model and the fitted `TfidfVectorizer`.

### ● Feature Engineering

The pipeline uses `TfidfVectorizer` from scikit-learn for feature engineering. This vectorizer converts the text data into a numerical representation (TF-IDF features) that can be used by the Logistic Regression model. The vectorizer is fitted on the training data and then used to transform both the training and testing data.

### ● MLOps

This project utilizes MLflow to track experiments, logging parameters, metrics, artifacts, and saving the trained model and the vectorizer.

- **Tracking URI:** The MLflow tracking server is configured to run locally at `http://localhost:5000`. This is set in `src/model_building/mlops_pipeline.py`.
- **Experiment:** All runs are logged under the default MLflow experiment.
- **Parameter Logging:** The `evaluate_and_log_model` function in `src/model_building/mlops_pipeline.py` logs the following parameters:
    - `model_name`: The name of the model used (LogisticRegression).
    - `tfidf_vectorizer`: Indicates the use of TfidfVectorizer.
    - `split_ratio`: The ratio used for splitting the data (0.2).
    - `random_state`: The random state used for reproducibility (42).
- **Metric Logging:** The `evaluate_and_log_model` function calculates and logs the following metrics on the test set:
    - `accuracy`
    - `negative_precision`
    - `negative_recall`
    - `negative_f1_score`
    - `positive_precision`
    - `positive_recall`
    - `positive_f1_score`
    - `macro_average_precision`
    - `macro_average_recall`
    - `macro_average_f1_score`
    - `weighted_average_precision`
    - `weighted_average_recall`
    - `weighted_average_f1_score`
- **Artifact Logging:** The `evaluate_and_log_model` function also logs the following artifacts:
    - `tfidf_vectorizer.pkl`: The serialized `TfidfVectorizer` object.
    - `confusion_matrix.png`: The confusion matrix plot.
- **Model Logging:** The trained `LogisticRegression` model is saved as an MLflow Sklearn model with an inferred signature in `src/model_building/mlops_pipeline.py`.

### ● Execution Guide

This section provides instructions for technical users on how to set up and run the MLOps pipeline.

#### 1. Code Retrieval

Clone the repository using the appropriate command (e.g., from GitHub or your version control system).
```bash

 "git clone <repository_url>"
 "cd <repository_name>".


### 2. Dependency installation

"pip install -r requirements.txt".

Make  the requirements.txt file includes the necessary libraries such as pandas, scikit-learn, mlflow, and matplotlib. A sample requirements.txt would look like this:

pandas
scikit-learn
mlflow
matplotlib
numpy

#### 3.Pipeline Execution

python src/model_building/mlops_pipeline.py --data_path data/glassdoor_all_spanish_reviews.csv



Ensure that the data directory and the /glassdoor_all_spanish_reviews.csv file are located in the correct relative path from where you are running the command. The --data_path argument should point to the location where you have stored the breast-cancer-wisconsin.data.csv file within your local file system. For example, if you have placed the data directory at the root of your mlops project, the command shown above is correct.



#### 4. Monitoring MLFlow

mlflow ui

Then, open your web browser and navigate to http://127.0.0.1:5000 to view the registered experiments, metrics, models, and visualizations.


