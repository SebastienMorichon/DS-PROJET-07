import json
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Charger les fichiers nécessaires (imputer, scaler, modèle)
with open("imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Charger les colonnes nécessaires en ignorant la première colonne
data = pd.read_csv("Base De Donnée Prétraitée.csv")
required_columns = list(data.iloc[:, 1:].columns)  # Ignorer la première colonne sans nom


# Charger les données de test depuis le fichier JSON
with open("Tests/test_data.json", "r") as test_file:
    test_data = json.load(test_file)["inputs"]

def validate_features(row, required_columns):
    """
    S'assure que la ligne possède les mêmes colonnes que celles attendues par le modèle.
    Ajoute des valeurs manquantes pour les colonnes absentes et ignore celles en trop.
    """
    validated_row = {feature: np.nan for feature in required_columns}
    validated_row.update({key: value for key, value in row.items() if key in required_columns})
    return [validated_row[feature] for feature in required_columns]

def test_model_predictions():
    X_test = []
    for i, item in enumerate(test_data):
        try:
            # Validation des colonnes et transformation des valeurs
            row = validate_features(item, required_columns)
            row = [float(value) if value not in [None, "", "False", "True"] else np.nan for value in row]
            X_test.append(row)
        except Exception as e:
            print(f"Erreur lors du traitement de la ligne {i}: {e}")
            continue

    # Convertir en tableau numpy
    X_test = np.array(X_test)
    print(f"Shape de X_test après validation : {X_test.shape}")

    # Appliquer l'imputer et le scaler
    try:
        X_imputed = imputer.transform(X_test)
        X_scaled = scaler.transform(X_imputed)
    except Exception as e:
        print(f"Erreur pendant le prétraitement : {e}")
        return False

    # Faire les prédictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Comparer les prédictions aux cibles (si disponibles)
    target_key = "TARGET"
    targets = [item.get(target_key) for item in test_data if target_key in item]
    if targets:
        for i, (pred, prob, target) in enumerate(zip(predictions, probabilities, targets)):
            print(f"Ligne {i}: Prédiction={pred}, Probabilité={prob:.2f}, Cible={target}")
            if int(pred) != int(target):
                print(f"Incohérence dans la prédiction à la ligne {i}")
    else:
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Ligne {i}: Prédiction={pred}, Probabilité={prob:.2f}")

    return True

# Exécuter le test
if __name__ == "__main__":
    success = test_model_predictions()
    print(f"Test {'réussi' if success else 'échoué'}")
