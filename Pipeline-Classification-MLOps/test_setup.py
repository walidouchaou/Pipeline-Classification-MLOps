#!/usr/bin/env python3
"""
Script de test pour vérifier que l'installation et la configuration sont correctes.
"""

import sys
from pathlib import Path

def test_imports():
    """Test des imports principaux."""
    print("🔍 Test des imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} importé avec succès")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib importé avec succès")
        
        import seaborn as sns
        print("✅ Seaborn importé avec succès")
        
        import numpy as np
        print("✅ NumPy importé avec succès")
        
        import pandas as pd
        print("✅ Pandas importé avec succès")
        
        import sklearn
        print("✅ Scikit-learn importé avec succès")
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    
    return True

def test_custom_modules():
    """Test des modules personnalisés."""
    print("\n🔍 Test des modules personnalisés...")
    
    # Ajouter le dossier src au path
    sys.path.append(str(Path(__file__).parent / "src"))
    
    try:
        from data_utils import DataManager
        print("✅ DataManager importé avec succès")
        
        from model_architecture import ImageClassifierBuilder
        print("✅ ImageClassifierBuilder importé avec succès")
        
        # Test d'instanciation
        data_manager = DataManager()
        print("✅ DataManager instancié avec succès")
        
        builder = ImageClassifierBuilder(num_classes=32)
        print("✅ ImageClassifierBuilder instancié avec succès")
        
    except ImportError as e:
        print(f"❌ Erreur d'import des modules personnalisés: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de l'instanciation: {e}")
        return False
    
    return True

def test_gpu():
    """Test de la disponibilité du GPU."""
    print("\n🔍 Test de la disponibilité du GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) détecté(s): {[gpu.name for gpu in gpus]}")
            
            # Test de mémoire GPU
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✅ Croissance mémoire activée pour {gpu.name}")
                except:
                    print(f"⚠️ Impossible d'activer la croissance mémoire pour {gpu.name}")
        else:
            print("⚠️ Aucun GPU détecté. L'entraînement se fera sur CPU.")
            
    except Exception as e:
        print(f"❌ Erreur lors du test GPU: {e}")
        return False
    
    return True

def test_directories():
    """Test de la structure des dossiers."""
    print("\n🔍 Test de la structure des dossiers...")
    
    required_dirs = ['src', 'data', 'notebooks', 'models']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Dossier '{dir_name}' trouvé")
        else:
            print(f"⚠️ Dossier '{dir_name}' manquant - création...")
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Dossier '{dir_name}' créé")
    
    return True

def main():
    """Fonction principale de test."""
    print("🧪 Test de Configuration du Projet MLOps")
    print("=" * 50)
    
    tests = [
        ("Imports principaux", test_imports),
        ("Modules personnalisés", test_custom_modules),
        ("GPU", test_gpu),
        ("Structure des dossiers", test_directories)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur inattendue dans {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Résumé des Tests:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 Tous les tests sont passés! Votre environnement est prêt.")
        print("Vous pouvez maintenant lancer: python run_training.py")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        print("Assurez-vous d'avoir installé toutes les dépendances: pip install -r requirements.txt")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 