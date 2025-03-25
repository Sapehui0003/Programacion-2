# model_training.py
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from datetime import datetime
import mlflow as mlf
import mlflow.sklearn

def train_model_module(X, y, model_name="random_forest", test_size=0.2, random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3, n_neighbors=5, solver='lbfgs', max_iter=100, kernel='rbf', C=1.0, cv_folds=5):
    """
    Trains a specified machine learning model with cross-validation and returns the trained model and test sets,
    logging metrics and the model with MLflow.

    Args:
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target variable.
        model_name (str): The name of the model to train ('random_forest', 'gradient_boosting', 'logistic_regression', 'knn', 'decision_tree', 'svm').
        test_size (float): Proportion of the data to use for the test set.
        random_state (int): Seed for random number generation.
        n_estimators (int): Number of trees in the Random Forest or Gradient Boosting.
        learning_rate (float): Learning rate for Gradient Boosting.
        max_depth (int): Maximum depth of the Decision Tree or Gradient Boosting.
        n_neighbors (int): Number of neighbors for KNN.
        solver (str): Solver to use for Logistic Regression.
        max_iter (int): Maximum number of iterations for Logistic Regression or SVM.
        kernel (str): Kernel type for SVM.
        C (float): Regularization parameter for Logistic Regression and SVM.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        tuple: (trained model, X_test, y_test)
    """
    print(f"--- Model Training: {model_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    with mlf.start_run(run_name=f"{model_name.capitalize()}Training"):
        # Log parameters
        mlf.log_param("model_name", model_name)
        mlf.log_param("test_size", test_size)
        mlf.log_param("random_state", random_state)
        mlf.log_param("cv_folds", cv_folds)

        if model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            mlf.log_param("n_estimators", n_estimators)
            print(f"Training RandomForest with {n_estimators} estimators.")
        elif model_name == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
            mlf.log_param("n_estimators", n_estimators)
            mlf.log_param("learning_rate", learning_rate)
            mlf.log_param("max_depth", max_depth)
            print(f"Training GradientBoosting with {n_estimators} estimators, learning rate={learning_rate}, max depth={max_depth}.")
        elif model_name == "logistic_regression":
            model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state, C=C)
            mlf.log_param("solver", solver)
            mlf.log_param("max_iter", max_iter)
            mlf.log_param("C", C)
            print(f"Training Logistic Regression with solver='{solver}', max iterations={max_iter}, C={C}.")
        elif model_name == "knn":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            mlf.log_param("n_neighbors", n_neighbors)
            print(f"Training KNN with {n_neighbors} neighbors.")
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            mlf.log_param("max_depth", max_depth)
            print(f"Training Decision Tree with max depth={max_depth}.")
        elif model_name == "svm":
            model = SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=random_state, probability=True) # probability=True for ROC curve
            mlf.log_param("kernel", kernel)
            mlf.log_param("C", C)
            mlf.log_param("max_iter", max_iter)
            print(f"Training SVM with kernel='{kernel}', C={C}, max iterations={max_iter}.")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        model.fit(X_train, y_train)
        print("Model trained.")

        print(f"Performing {cv_folds}-fold stratified cross-validation.")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
        mlf.log_metric(f"{model_name}_cv_mean_accuracy", cv_scores.mean())
        mlf.sklearn.log_model(model, f"{model_name}_model")
        print(f"MLflow: Logged model as {model_name}_model")
        print(f"--- End of Model Training: {model_name} ---")

    return model, X_test, y_test

if __name__ == '__main__':
    from module_load_data import load_breast_cancer_data
    from module_preprocessing import preprocess_module
    import os

    