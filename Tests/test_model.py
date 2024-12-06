import json
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Charger le modèle, le scaler et l'imputer
with open("Streamlit/docker_test/best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("Streamlit/docker_test/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("Streamlit/docker_test/imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)

# Charger les données de test
with open("Tests/test_data.json", "r") as json_file:
    test_data = json.load(json_file)

inputs = test_data["inputs"]
expected_outputs = test_data["expected_outputs"]

def test_model_predictions():
    """
    Fonction pour tester les prédictions du modèle.
    Vérifie si les prédictions correspondent aux résultats attendus.
    """
    X_test = []
    for i, item in enumerate(inputs):
        # Convertir les données en format numérique
        row = []
        for value in item.values():
            try:
                row.append(float(value) if value not in [None, "", "False", "True"] else (1.0 if value == "True" else 0.0))
            except ValueError:
                row.append(np.nan)
        X_test.append(row)

    # Conversion en numpy array
    X_test = np.array(X_test)

    # Étapes de prétraitement
    X_imputed = imputer.transform(X_test)  # Imputation des données manquantes
    X_scaled = scaler.transform(X_imputed)  # Normalisation des données

    # Prédictions
    predictions_proba = model.predict_proba(X_scaled)[:, 1]
    predictions = [1 if proba >= 0.5 else 0 for proba in predictions_proba]

    # Afficher les résultats
    print("Résultats des prédictions du modèle :", predictions)
    print("Résultat attendu :", expected_outputs)

    # Sauvegarder les résultats dans un fichier log
    with open("test_results.log", "w") as log_file:
        log_file.write(f"Résultat attendu : {expected_outputs}\n")
        log_file.write(f"Résultat obtenu : {predictions}\n")

    # Vérification des prédictions
    if predictions != expected_outputs:
        print(f"ÉCHEC DU TEST : Les résultats diffèrent. Attendu {expected_outputs}, mais obtenu {predictions}")
        return False
    else:
        print("TEST RÉUSSI : Les résultats sont conformes.")
        return True


# Exécution des tests
if __name__ == "__main__":
    success = test_model_predictions()
    import sys
    sys.exit(0 if success else 1)
