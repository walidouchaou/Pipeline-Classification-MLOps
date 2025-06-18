# Pipeline Classification MLOps

Projet de classification d'images utilisant des techniques de Deep Learning et MLOps pour classifier 32 catégories d'images différentes.

## 🎯 Objectif

Développer un pipeline complet de classification d'images incluant :
- Exploration et prétraitement des données
- Entraînement de modèles CNN avec transfer learning
- Évaluation des performances
- Déploiement via API REST

## 📁 Structure du Projet

```
Pipeline-Classification-MLOps/
├── data/                          # Données (gitignored)
├── models/                       # Modèles entraînés (gitignored)
├── notebooks/                    # Notebooks Jupyter d'exploration
├── src/                          # Code source
│   ├── data_utils.py            # Utilitaires de gestion des données
│   ├── model_architecture.py    # Architectures de modèles
│   └── train.py                 # Script d'entraînement
├── requirements.txt             # Dépendances Python
├── run_training.py             # Script principal d'entraînement
└── README.md                   # Ce fichier
```

## 🚀 Installation

### 1. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'Entraînement

```bash
python run_training.py
```

## 📊 Dataset

**Image Classification - 32 Classes - Variety**
- Source : Kaggle
- Format : Images PNG 512x512 pixels
- Classes : 32 catégories variées

## 🏗️ Architectures Supportées

1. **ResNet50** (Recommandé) - Transfer learning avec ImageNet
2. **EfficientNetB0** - Architecture efficace et moderne
3. **VGG16** - Architecture classique et robuste
4. **Custom CNN** - CNN personnalisé

---

**Développé avec ❤️ pour l'apprentissage du MLOps**

