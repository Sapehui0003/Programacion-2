# training.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow
import mlflow.sklearn
import datetime

def train_model(X_train, y_train, model_name="random_forest", n_estimators=100, learning_rate=0.1, max_depth=3,
                solver='liblinear', max_iter=100, C=1.0, n_neighbors=5, kernel='rbf', random_state=42, cv_folds=5):
    """Entrena un modelo de clasificación y lo evalúa con validación cruzada."""
    print(f"--- Training {model_name} Model ---")
    mlflow.set_tag("model_name", model_name)
    start_time = datetime.datetime.now()
    mlflow.log_param("start_time", str(start_time))

    try:
        if model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            mlflow.log_param("n_estimators", n_estimators)
            print(f"Training RandomForest with {n_estimators} estimators.")
        elif model_name == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            print(f"Training GradientBoosting with {n_estimators} estimators, learning rate={learning_rate}, max depth={max_depth}.")
        elif model_name == "logistic_regression":
            model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state, C=C)
            mlflow.log_param("solver", solver)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("C", C)
            print(f"Training Logistic Regression with solver='{solver}', max iterations={max_iter}, C={C}.")
        elif model_name == "knn":
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            mlflow.log_param("n_neighbors", n_neighbors)
            print(f"Training KNN with {n_neighbors} neighbors.")
        elif model_name == "decision_tree":
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
            mlflow.log_param("max_depth", max_depth)
            print(f"Training Decision Tree with max depth={max_depth}.")
        elif model_name == "svm":
            model = SVC(kernel=kernel, C=C, max_iter=max_iter, random_state=random_state, probability=True) # probability=True for ROC curve
            mlflow.log_param("kernel", kernel)
            mlflow.log_param("C", C)
            mlflow.log_param("max_iter", max_iter)
            print(f"Training SVM with kernel='{kernel}', C={C}, max iterations={max_iter}.")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        print("--- Training Model ---")
        model.fit(X_train, y_train)
        print("Model trained.")
        mlflow.log_param("training_end_time", str(datetime.datetime.now()))

        print("--- Cross-validation ---")
        print(f"Performing {cv_folds}-fold stratified cross-validation.")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Cross-validation accuracy scores: {cv_scores}")
        print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
        mlflow.log_metric(f"{model_name}_cv_mean_accuracy", cv_scores.mean())
        mlflow.log_param("cross_validation_folds", cv_folds)

        mlflow.sklearn.log_model(model, f"{model_name}_model")
        print(f"MLflow: Logged model as {model_name}_model")
        end_time = datetime.datetime.now()
        mlflow.log_param("end_time", str(end_time))
        print(f"Training duration: {end_time - start_time}")
        print("-" * 30)
        return model

    except Exception as e:
        print(f"Error training {model_name}: {e}")
        raise