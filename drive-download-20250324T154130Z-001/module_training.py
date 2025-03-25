# model_training.py
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train_model(X, y, test_size=0.2, random_state=42, n_estimators=100, cv_folds=5):
    """
    Trains a model with cross-validation and returns the trained model and test sets.

    Args:
        X (pd.DataFrame or np.ndarray): Features.
        y (pd.Series or np.ndarray): Target variable.
        test_size (float): Proportion of the data to use for the test set.
        random_state (int): Seed for random number generation.
        n_estimators (int): Number of trees in the Random Forest.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        tuple: (trained model, X_test, y_test)
    """
    print("--- Model Training ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    print(f"Training model with {n_estimators} estimators.")
    model.fit(X_train, y_train)
    print("Model trained.")

    print(f"Performing {cv_folds}-fold stratified cross-validation.")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.2f}")
    print("--- End of Model Training ---")

    return model, X_test, y_test

if __name__ == '__main__':
    from data.load_data import load_breast_cancer_data
    from preprocessing import preprocess_data

    file_path_example = r"data/breast-cancer-wisconsin.data.csv"
    df = load_breast_cancer_data(file_path_example)
    if df is not None:
        X_scaled, y = preprocess_data(df)
        if y is not None:
            model, X_test, y_test = train_model(X_scaled, y)
            print("\nModel and test data obtained.")
        else:
            print("\nNo target variable available, cannot train model.")