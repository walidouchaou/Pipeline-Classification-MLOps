#!/usr/bin/env python3
"""
Script simple pour lancer l'entraînement du modèle de classification d'images.
Usage: python run_training.py
"""

import sys
from pathlib import Path

# Ajouter le dossier src au path Python
sys.path.append(str(Path(__file__).parent / "src"))

from src.train import main

if __name__ == "__main__":
    print("🚀 Démarrage de l'entraînement du modèle de classification d'images...")
    print("=" * 60)
    
    try:
        main()
        print("=" * 60)
        print("✅ Entraînement terminé avec succès!")
    except Exception as e:
        print("=" * 60)
        print(f"❌ Erreur lors de l'entraînement: {e}")
        sys.exit(1) 