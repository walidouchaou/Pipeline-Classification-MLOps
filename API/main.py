import os
import io
import pathlib
import tensorflow as tf
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image

# --- Configuration et Initialisation ---

# Construction de chemins absolus pour plus de robustesse
BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_IMAGE_PATH = BASE_DIR.parent / "models" / "image_classifier_model.keras"
MODEL_TEXT_PATH = BASE_DIR.parent / "models" / "huggingFace" / "text-classifier-ag-news-distilbert"

# Définition des classes pour la lisibilité
IMAGE_CLASS_NAMES = [
    'Bicycle', 'Bird', 'Bread', 'Car', 'Cat', 'Dog', 'Elephant', 'Fish',
    'Flower', 'Frog', 'Hamburger', 'Horse', 'Lasagna', 'Lion', 'Men',
    'Monkey', 'Moose', 'Motorcycle', 'Pencil', 'Pickup', 'Pizza', 'Rabbit',
    'Rat', 'Rhinoceros', 'Robot', 'Shark', 'Spaghetti', 'Squirrel', 'Tiger',
    'Turtle', 'Women', 'Zebra'
]
TEXT_CLASS_NAMES_MAP = {"LABEL_0": "World", "LABEL_1": "Sports", "LABEL_2": "Business", "LABEL_3": "Sci/Tech"}

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API de Classification d'Image et de Texte",
    description="Une API pour classifier des images et des textes en utilisant des modèles de Deep Learning.",
    version="1.0.0"
)

# Dictionnaire pour contenir les modèles chargés
models = {}

# --- Chargement des Modèles au Démarrage ---

@app.on_event("startup")
async def load_models():
    """Charge les modèles au démarrage de l'application pour éviter la latence."""
    print("--- Chargement des modèles en cours... ---")
    
    # Modèle de classification d'images
    print(f"Chargement du modèle d'image depuis : {MODEL_IMAGE_PATH}")
    models["image_classifier"] = tf.keras.models.load_model(MODEL_IMAGE_PATH)
    print("✅ Modèle d'image chargé.")

    # Modèle de classification de texte
    print(f"Chargement du modèle de texte depuis : {MODEL_TEXT_PATH}")
    models["text_classifier"] = pipeline("text-classification", model=MODEL_TEXT_PATH)
    print("✅ Modèle de texte chargé.")
    
    print("--- Tous les modèles sont prêts. ---")


# --- Modèles Pydantic pour la Validation des Données ---

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    predicted_class: str
    confidence: float

# --- Endpoints de l'API ---

@app.get("/", summary="Endpoint racine pour vérifier l'état de l'API")
def root():
    return {"message": "Bienvenue sur l'API de classification. L'API est opérationnelle."}

@app.post("/predict/text", response_model=PredictionOut, summary="Classifier un texte")
async def predict_text(payload: TextIn):
    """
    Accepte une chaîne de caractères et retourne la catégorie de texte prédite. 
    """
    if "text_classifier" not in models:
        raise HTTPException(status_code=503, detail="Modèle de texte non disponible.")
    
    try:
        prediction = models["text_classifier"](payload.text)[0]
        predicted_class_name = TEXT_CLASS_NAMES_MAP.get(prediction['label'], prediction['label'])
        
        return PredictionOut(
            predicted_class=predicted_class_name,
            confidence=prediction['score']
        )
    except Exception as e:
        # Gestion d'erreur robuste 
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction du texte : {e}")

@app.post("/predict/image", response_model=PredictionOut, summary="Classifier une image")
async def predict_image(file: UploadFile = File(..., description="Fichier image à classifier")):
    """
    Accepte une image en entrée et retourne la classe prédite. 
    """
    if "image_classifier" not in models:
        raise HTTPException(status_code=503, detail="Modèle d'image non disponible.")

    # Vérification du type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image.")

    try:
        # Lire le contenu de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prétraitement de l'image
        image_resized = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        image_array = tf.expand_dims(image_array, 0)

        # Prédiction
        predictions = models["image_classifier"].predict(image_array)
        scores = tf.nn.softmax(predictions[0])
        
        predicted_class_name = IMAGE_CLASS_NAMES[tf.argmax(scores)]
        confidence = float(tf.reduce_max(scores))

        # Retour JSON clair 
        return PredictionOut(
            predicted_class=predicted_class_name,
            confidence=confidence
        )
    except Exception as e:
        # Gestion d'erreur robuste 
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction de l'image : {e}")