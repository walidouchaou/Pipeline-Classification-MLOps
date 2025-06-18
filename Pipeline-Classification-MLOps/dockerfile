# --- Étape 1: Base ---
# Utilisation d'une image de base Python légère pour optimiser la taille finale 
FROM python:3.9-slim-buster

# Définir le répertoire de travail dans le conteneur
WORKDIR /API

# --- Étape 2: Installation des Dépendances ---
# Copier uniquement le fichier requirements.txt pour utiliser le cache de Docker.
# Cette étape ne sera ré-exécutée que si le fichier requirements.txt change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape 3: Copie des Fichiers de l'Application ---
# Copier les modèles et le code de l'application
# S'assurer que les modèles pré-entraînés sont correctement inclus 
COPY ./models ./models
COPY ./API ./API

# --- Étape 4: Exécution ---
# Exposer le port sur lequel l'API s'exécutera
EXPOSE 8000

# Commande pour lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]