import streamlit as st
import pickle
import pandas as pd
import shap
import streamlit.components.v1 as components
from lightgbm import LGBMClassifier

# Charger le modèle
with open("../Exports Modèles/lightgbm_model_5_features.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("../Exports Modèles/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("../Exports Modèles/imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)

# Liste des 5 features importantes
important_features = ["EXT_SOURCE_3", "EXT_SOURCE_2", "CREDIT_TERM", "EXT_SOURCE_1", "AMT_GOODS_PRICE"]

# Configuration de l'interface Streamlit
st.title("Évaluation de la Demande de Prêt")
st.write("Entrez les informations du client pour évaluer la probabilité d'acceptation de la demande de prêt.")

# Interface utilisateur pour saisir les valeurs des 5 features principales
user_input = {}
for feature in important_features:
    user_input[feature] = st.number_input(f"{feature}")

# Créer un DataFrame avec les valeurs saisies par l'utilisateur
data = pd.DataFrame([user_input])

# Faire la prédiction et afficher les valeurs SHAP
if st.button("Évaluer la demande de prêt"):
    # Normaliser les données avant de faire la prédiction
    normalized_data = scaler.transform(data)
    normalized_data = imputer.transform(data)

    prediction_proba = model.predict_proba(data)[:, 1][0]
    prediction = "Accordé" if prediction_proba < 0.5 else "Refusé"
    
    # Affichage des résultats
    st.write(f"Probabilité de défaut : {prediction_proba:.2f}")
    st.write(f"Décision : **{prediction}**")

    # Explication locale avec SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    # Vérifier si shap_values est une liste ou un tableau
    shap_value_to_use = shap_values[0] if isinstance(shap_values, list) else shap_values

    # Utiliser directement explainer.expected_value sans index [0] si c'est un scalaire
    shap.initjs()
    shap_plot = shap.force_plot(explainer.expected_value, shap_value_to_use, data)

    # Afficher le graphique SHAP en tant qu'HTML interactif dans Streamlit
    # Convertir le graphique en HTML
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
    components.html(shap_html, height=400)
