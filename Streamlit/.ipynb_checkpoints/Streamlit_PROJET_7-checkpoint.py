import streamlit as st
import pickle
import pandas as pd
import shap
import streamlit.components.v1 as components
from lightgbm import LGBMClassifier

# Charger le modèle et les transformations
with open("best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)

# Configuration de l'interface Streamlit
st.title("Évaluation de la Demande de Prêt")
st.write(
    "Téléchargez un fichier CSV contenant les informations des clients ou entrez manuellement les données d'un client pour évaluer la probabilité d'acceptation de la demande de prêt."
)

# Ajouter un bouton pour charger un fichier CSV
uploaded_file = st.file_uploader("Téléchargez un fichier CSV", type=["csv"])

# Fonction pour effectuer des prédictions
def make_prediction(input_data):
    # Appliquer l'imputer et le scaler aux données
    data_imputed = imputer.transform(input_data)
    normalized_data = scaler.transform(data_imputed)

    # Faire les prédictions
    prediction_proba = model.predict_proba(normalized_data)[:, 1]
    predictions = ["Accordé" if proba < 0.5 else "Refusé" for proba in prediction_proba]

    # Retourner les résultats sous forme de DataFrame
    results = pd.DataFrame({
        "Probabilité de défaut": prediction_proba,
        "Décision": predictions
    })
    return results, normalized_data

# Si un fichier CSV est téléchargé
if uploaded_file is not None:
    try:
        # Charger le fichier CSV
        input_data = pd.read_csv(uploaded_file)

        # Vérifier que toutes les colonnes nécessaires sont présentes
        missing_features = [col for col in model.feature_name_ if col not in input_data.columns]
        if missing_features:
            st.error(f"Les colonnes suivantes sont manquantes dans le fichier CSV : {missing_features}")
        else:
            # Effectuer les prédictions
            results, normalized_data = make_prediction(input_data)

            # Afficher les résultats
            st.write("Résultats des prédictions :")
            st.write(results)

            # Afficher l'importance des caractéristiques (SHAP)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(normalized_data)

            st.write("Importance globale des caractéristiques :")
            shap.summary_plot(shap_values, normalized_data, feature_names=model.feature_name_, plot_type="bar")
            components.html(shap.force_plot(explainer.expected_value[1], shap_values[1], normalized_data), height=400)

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")

