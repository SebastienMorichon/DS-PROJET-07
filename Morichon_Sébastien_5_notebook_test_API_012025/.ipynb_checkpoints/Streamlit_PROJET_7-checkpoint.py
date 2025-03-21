import streamlit as st
import pandas as pd
import pickle
import shap
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Charger le modèle, l'imputer et le scaler
with open("../Model/best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("../Model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("../Model/imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)

# Charger les colonnes nécessaires
data = pd.read_csv("../Bases de données/Base De Donnée Prétraitée.csv")

required_columns = list(data.drop(columns=['TARGET']).columns)

# Configuration de l'application Streamlit
st.title("Prédiction d'Acceptation de Prêt")
st.write("Téléversez un fichier Excel contenant les données des clients pour évaluer leurs probabilités d'acceptation.")

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléversez un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Charger le fichier Excel
        client_data = pd.read_excel(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(client_data.head())

        # Vérifier les colonnes manquantes
        missing_columns = [col for col in required_columns if col not in client_data.columns]
        if missing_columns:
            st.warning(f"Les colonnes suivantes sont manquantes dans le fichier Excel : {', '.join(missing_columns)}")
            for col in missing_columns:
                client_data[col] = 0  # Ajouter les colonnes manquantes avec des valeurs par défaut
            st.info("Les colonnes manquantes ont été ajoutées avec des valeurs par défaut (0).")

        # Prétraitement des données
        st.info("Prétraitement des données...")
        data_to_predict = client_data[required_columns]  # Garder uniquement les colonnes nécessaires
        data_imputed = imputer.transform(data_to_predict)
        normalized_data = scaler.transform(data_imputed)

        # Vérification des dimensions des données normalisées
        if normalized_data.shape[0] == 0:
            st.error("Le fichier téléversé ne contient aucune ligne valide après prétraitement. Veuillez vérifier le fichier.")
            st.stop()

        # Prédictions
        st.info("Calcul des prédictions...")
        predictions_proba = model.predict_proba(normalized_data)[:, 1]
        predictions = ["Accordé" if proba < 0.5 else "Refusé" for proba in predictions_proba]

        # Ajouter les résultats au DataFrame
        client_data["Probabilité de défaut"] = predictions_proba
        client_data["Décision"] = predictions

        # Affichage des résultats
        st.success("Résultats des prédictions :")
        st.dataframe(client_data[["SK_ID_CURR", "Probabilité de défaut", "Décision"]])

        # Explications globales avec SHAP
        st.write("**Importance globale des caractéristiques :**")
        explainer = shap.Explainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(normalized_data)

        # Vérification des SHAP values
        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1]  # Classe 1 : défaut de paiement
        else:
            shap_values_to_use = shap_values

        # Calcul des moyennes absolues pour les 10 variables les plus importantes
        mean_abs_shap_values = abs(shap_values_to_use).mean(axis=0)
        top_indices = mean_abs_shap_values.argsort()[-10:][::-1]  # Indices des 10 plus importantes
        top_features = [required_columns[i] for i in top_indices]  # Noms des 10 caractéristiques

        # Filtrage des valeurs SHAP et des données pour les graphiques
        shap_values_top = shap_values_to_use[:, top_indices]
        normalized_data_top = normalized_data[:, top_indices]

        # Visualisation globale avec Matplotlib
        fig_summary = plt.figure()
        shap.summary_plot(shap_values_top, normalized_data_top, feature_names=top_features, show=False)
        st.pyplot(fig_summary)

        # Sélection du client
        st.write("**Sélectionnez un client pour voir les explications locales :**")
        client_index = st.selectbox(
            "Choisissez l'index du client",
            options=range(len(client_data)),
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )

        # Affichage de la décision pour le client sélectionné
        st.write(f"**Décision pour le client {client_data['SK_ID_CURR'].iloc[client_index]} :**")
        st.write(f"Décision : {client_data['Décision'].iloc[client_index]}")
        st.write(f"Probabilité de défaut : {client_data['Probabilité de défaut'].iloc[client_index]:.2f}")

        # Préparation des données pour le client spécifique
        shap_values_client = shap_values_top[client_index].reshape(1, -1)  # SHAP values pour un seul client
        data_client = normalized_data_top[client_index].reshape(1, -1)  # Données normalisées pour ce client

        # Visualisation locale comme un summary plot
        st.write(f"**Explications locales pour le client {client_data['SK_ID_CURR'].iloc[client_index]} :**")
        fig_local = plt.figure()
        shap.summary_plot(shap_values_client, data_client, feature_names=top_features, show=False)
        st.pyplot(fig_local)

except Exception as e:
    st.error(f"Une erreur s'est produite lors du traitement du fichier : {e}")
