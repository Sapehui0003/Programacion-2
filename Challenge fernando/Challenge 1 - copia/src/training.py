# training.py
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, n_estimators=100):
    """Entrena un modelo RandomForest y lo devuelve."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model
