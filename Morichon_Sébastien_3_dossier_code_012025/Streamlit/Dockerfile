# Étape 1 : Utiliser une image de base Python légère
FROM python:3.9-slim

# Étape 2 : Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Étape 3 : Copier les fichiers nécessaires dans le conteneur
COPY requirements.txt ./
COPY Streamlit_PROJET_7.py ./
COPY best_lightgbm_model.pkl ./
COPY scaler.pkl ./
COPY imputer.pkl ./
COPY Base_De_Donnée_Prétraitée.csv ./

# Étape 4 : Installer les outils de base nécessaires pour la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Étape 5 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6 : Exposer le port utilisé par Streamlit
EXPOSE 8501

# Étape 7 : Lancer l'application Streamlit
CMD ["streamlit", "run", "Streamlit_PROJET_7.py"]
