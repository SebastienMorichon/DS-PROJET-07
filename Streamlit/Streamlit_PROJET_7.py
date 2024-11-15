import streamlit as st
import pickle
import pandas as pd
import shap
import streamlit.components.v1 as components
from lightgbm import LGBMClassifier

# Charger le modèle et les transformations
with open("lightgbm_model_5_features.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("imputer.pkl", "rb") as imputer_file:
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
    # Appliquer l'imputer et le scaler aux données
    data_imputed = imputer.transform(data)
    normalized_data = scaler.transform(data_imputed)

    # Faire la prédiction
    prediction_proba = model.predict_proba(normalized_data)[:, 1][0]
    prediction = "Accordé" if prediction_proba < 0.5 else "Refusé"
    
    # Affichage des résultats
    st.write(f"Probabilité de défaut : {prediction_proba:.2f}")
    st.write(f"Décision : **{prediction}**")

    # Explication locale avec SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(normalized_data)

    # Afficher un graphique SHAP alternatif (summary plot)
    st.write("Importance des caractéristiques pour cette décision:")
    shap.summary_plot(shap_values, normalized_data, feature_names=important_features, plot_type="bar")
