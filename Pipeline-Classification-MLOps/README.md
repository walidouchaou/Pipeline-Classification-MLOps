# 🚀 Pipeline Classification MLOps - Classification Multi-Classes

Un pipeline complet de **Machine Learning** utilisant **Deep Learning** et **Transfer Learning** pour deux tâches de classification :
- **🖼️ Classification d'Images** : 32 catégories d'objets avec EfficientNetV2B0
- **📝 Classification de Texte** : 4 catégories d'actualités avec DistilBERT

## 🎯 Objectif

Développer des systèmes robustes de classification incluant :
- ✅ **Images** : Classification multi-classe (32 catégories)
- ✅ **Texte** : Classification d'actualités (4 catégories)  
- ✅ **Transfer Learning** : EfficientNetV2B0 + DistilBERT
- ✅ **Évaluation complète** des performances
- ✅ **Pipeline reproductible** et documenté

## 📈 Performances des Modèles

### 🖼️ **Classification d'Images**
- **🎯 Accuracy**: 100% sur le jeu de test
- **📊 Dataset**: 12,155 images, 32 classes
- **🏗️ Architecture**: EfficientNetV2B0 + Transfer Learning
- **⚡ Temps d'entraînement**: ~13 epochs (early stopping)

### 📝 **Classification de Texte**
- **🎯 Accuracy**: ~92% sur le jeu de test
- **📊 Dataset**: AG News (127,600 articles, 4 classes)
- **🏗️ Architecture**: DistilBERT fine-tuné
- **⚡ Temps d'entraînement**: ~10 epochs

## 📁 Structure du Projet

```
Pipeline-Classification-MLOps/
├── data/                          # Données (auto-téléchargées)
├── models/                        # Modèles entraînés sauvegardés
├── notebooks/                     # Notebooks Jupyter
│   ├── classificationMulti_Class.ipynb      # Classification d'images
│   └── Classification_texte_huginface.ipynb # Classification de texte
├── docs/                          # Documentation
├── reports/                       # Rapports et résultats
├── references/                    # Références et ressources
├── requirements.txt               # Dépendances Python
├── pyproject.toml                 # Configuration du projet
├── Makefile                       # Commandes automatisées
└── README.md                      # Ce fichier
```

## 🚀 Démarrage Rapide

### 1. Cloner le Projet

```bash
git clone https://github.com/votre-username/Pipeline-Classification-MLOps.git
cd Pipeline-Classification-MLOps
```

### 2. Créer un Environnement Virtuel

```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (macOS/Linux)
source venv/bin/activate
```

### 3. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'Entraînement

#### Option A: Via Jupyter Notebook (Recommandé)

```bash
# Lancer Jupyter
jupyter notebook

# Choisir le notebook selon votre tâche :
# - Classification d'images : notebooks/classificationMulti_Class.ipynb
# - Classification de texte : notebooks/Classification_texte_huginface.ipynb
```

#### Option B: Via Kaggle/Google Colab

1. Télécharger un des notebooks :
   - `classificationMulti_Class.ipynb` (images)
   - `Classification_texte_huginface.ipynb` (texte)
2. L'importer dans Kaggle ou Google Colab
3. Exécuter toutes les cellules

## 📊 Datasets

### 🖼️ **Images : Classification Dataset - 32 Classes**
- **Source**: [Image Classification Dataset - 32 Classes](https://www.kaggle.com/datasets/anthonytherrien/image-classification-dataset-32-classes)
- **Taille**: 12,155 images
- **Classes**: 32 catégories variées
- **Format**: Images de taille variable (redimensionnées à 224×224)

### 📝 **Texte : AG News Dataset**
- **Source**: [AG News Dataset](https://huggingface.co/datasets/ag_news) (Hugging Face)
- **Taille**: 127,600 articles de presse
- **Classes**: 4 catégories d'actualités
- **Format**: Texte brut (tokenisé automatiquement)

### 📋 Classes Disponibles

#### 🖼️ **Images (32 classes)** :
```
['bicycle', 'bird', 'bread', 'car', 'cat', 'dog', 'elephant', 'fish', 
 'flower', 'frog', 'hamburger', 'horse', 'lasagna', 'lion', 'men', 
 'monkey', 'moose', 'motorcycle', 'pencil', 'pickup', 'pizza', 'rabbit', 
 'rat', 'rhinoceros', 'robot', 'shark', 'spaghetti', 'squirrel', 'tiger', 
 'turtle', 'women', 'zebra']
```

#### 📝 **Texte (4 classes)** :
```
['World', 'Sports', 'Business', 'Sci/Tech']
```

## 🏗️ Architectures des Modèles

### 🖼️ **Images : EfficientNetV2B0 + Transfer Learning**

```python
Model: "functional"
├── Data Augmentation (RandomFlip + RandomRotation)
├── EfficientNetV2B0 (pré-entraîné ImageNet, frozen)
├── GlobalAveragePooling2D
├── Dropout (0.2)
└── Dense (32 classes, softmax)
```

**Configuration d'Entraînement :**
- **Optimiseur**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Batch Size**: 32 | **Input Size**: 224×224×3

### 📝 **Texte : DistilBERT Fine-tuné**

```python
Model: "DistilBertForSequenceClassification"
├── DistilBERT (6 layers, 768 hidden, 12 heads)
├── Pre-classifier (Dense + Dropout)
└── Classifier (Dense, 4 classes)
```

**Configuration d'Entraînement :**
- **Optimiseur**: AdamW (lr=2e-5)
- **Loss**: Cross Entropy Loss
- **Batch Size**: 16 | **Max Length**: 512 tokens

## 📋 Prérequis

### 🔧 Système
- Python 3.8+
- 8GB+ RAM recommandés
- GPU optionnel (améliore les performances)

### 📦 Dépendances Principales
```
# Deep Learning
tensorflow>=2.13.0
torch>=2.0.0

# Hugging Face Ecosystem
transformers>=4.30.0
datasets>=2.12.0
tokenizers>=0.13.0
huggingface-hub>=0.15.0

# Data Science
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Environment
jupyter>=1.0.0
kagglehub>=0.2.0
```

## 🎮 Utilisation

### 📓 Notebooks Disponibles

#### 🖼️ **Classification d'Images** : `classificationMulti_Class.ipynb`
1. **Installation automatique des dépendances**
2. **Téléchargement automatique du dataset** (via kagglehub)
3. **Préparation des données** (split 80/10/10)
4. **Entraînement EfficientNetV2B0** avec early stopping
5. **Évaluation complète** (accuracy, confusion matrix, classification report)
6. **Sauvegarde du modèle** (.keras)
7. **Tests de prédiction** sur images réelles

#### 📝 **Classification de Texte** : `Classification_texte_huginface.ipynb`
1. **Installation des dépendances Hugging Face**
2. **Chargement du dataset AG News** (automatique)
3. **Tokenisation avec DistilBERT**
4. **Fine-tuning du modèle** pré-entraîné
5. **Évaluation des performances** (classification report)
6. **Sauvegarde du modèle** (.bin + tokenizer)
7. **Tests de prédiction** sur textes variés

### 💾 Modèles Sauvegardés

Après l'entraînement, les modèles sont automatiquement sauvegardés :

#### 🖼️ **Images** :
```
models/image_classifier_model.keras
```

#### 📝 **Texte** :
```
models/huggingFace/text-classifier-ag-news-distilbert/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
└── tokenizer.json
```

### 🔮 Prédiction sur Nouvelles Images

Le notebook inclut une section complète de tests avec :
- Tests sur images de base
- Tests de robustesse (classes similaires)
- Tests hors distribution

## 📊 Résultats

### 🎯 Métriques de Performance
- **Test Accuracy**: 100%
- **Validation Accuracy**: 100% 
- **Training Time**: ~13 epochs (early stopping)

### 📈 Points Forts
- ✅ Excellent transfer learning
- ✅ Convergence rapide
- ✅ Généralisation parfaite sur le test set
- ✅ Pipeline entièrement automatisé

## 🛠️ Développement

### 🔄 Relancer l'Entraînement

Pour relancer un entraînement avec des paramètres différents :

1. Modifier les hyperparamètres dans le notebook
2. Relancer les cellules d'entraînement
3. Le nouveau modèle remplacera l'ancien

### 🧪 Personnalisation

```python
# Modifier ces paramètres dans le notebook :
IMG_SIZE = 224          # Taille des images
BATCH_SIZE = 32         # Taille des batches
LEARNING_RATE = 0.001   # Taux d'apprentissage
EPOCHS = 50             # Nombre d'epochs max
```

## 🤝 Contribution

1. Fork du projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit des changes (`git commit -am 'Add new feature'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Créer une Pull Request


