import json
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

# Chemins des fichiers
MODEL_PATH = "Streamlit/docker_test/best_lightgbm_model.pkl"
SCALER_PATH = "Streamlit/docker_test/scaler.pkl"
IMPUTER_PATH = "Streamlit/docker_test/imputer.pkl"
TEST_DATA_PATH = "Tests/test_data.json"

def load_model(path):
    """Charge un modèle depuis un fichier."""
    with open(path, "rb") as file:
        return pickle.load(file)

def load_test_data(path):
    """Charge les données de test et les sorties attendues depuis un fichier JSON."""
    with open(path, "r") as file:
        data = json.load(file)
    inputs = data["inputs"]
    expected_outputs = data["expected_outputs"]
    return inputs, expected_outputs

def test_model_predictions():
    """Teste si les prédictions du modèle sont correctes."""
    # Charger le modèle, le scaler et l'imputer
    model = load_model(MODEL_PATH)
    scaler = load_model(SCALER_PATH)
    imputer = load_model(IMPUTER_PATH)

    # Charger les données de test
    inputs, expected_outputs = load_test_data(TEST_DATA_PATH)

    # Obtenir les colonnes attendues à partir du modèle ou du fichier de données
    expected_columns = list(inputs[0].keys())  # Assume que toutes les lignes ont les mêmes colonnes

    # Préparer les données d'entrée
    X_test = []
    for item in inputs:
        row = []
        for col in expected_columns:
            value = item.get(col, np.nan)  # Utiliser NaN si la colonne est manquante
            if isinstance(value, (int, float)):
                row.append(float(value))
            elif value == "True":
                row.append(1.0)
            elif value == "False":
                row.append(0.0)
            else:
                row.append(np.nan)  # Pour toute autre valeur non prise en charge
        X_test.append(row)

    # Diagnostiquer les lignes
    for i, row in enumerate(X_test):
        print(f"Row {i}: {row}, Length: {len(row)}")

    # Convertir en tableau NumPy
    X_test = np.array(X_test)

    # Appliquer l'imputation et la normalisation
    X_imputed = imputer.transform(X_test)
    X_normalized = scaler.transform(X_imputed)

    # Prédire avec le modèle
    predictions_proba = model.predict_proba(X_normalized)[:, 1]
    predictions = [1 if proba >= 0.5 else 0 for proba in predictions_proba]

    # Vérifier les prédictions avec les sorties attendues
    assert accuracy_score(expected_outputs, predictions) == 1.0, \
        f"Test échoué. Prédictions: {predictions}, Attendu: {expected_outputs}"
    print("Tous les tests sont passés avec succès ! ✅")

if __name__ == "__main__":
    test_model_predictions()
