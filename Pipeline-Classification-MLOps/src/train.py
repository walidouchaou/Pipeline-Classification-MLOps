"""
Main training script for image classification model.
"""

import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from data_utils import DataManager
from model_architecture import ImageClassifierBuilder, create_data_augmentation_layer, create_preprocessing_layer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles the complete training pipeline for image classification."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_manager = DataManager()
        self.model = None
        self.history = None
        
        # Create output directories
        self.output_dir = Path("models") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard logging
        self.log_dir = Path("logs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self):
        """Download and prepare the dataset."""
        logger.info("Preparing dataset...")
        
        # Download dataset if not already present
        if not self.data_manager.verify_dataset_structure():
            logger.info("Dataset not found. Attempting to download...")
            if not self.data_manager.download_dataset():
                raise RuntimeError("Failed to download dataset. Please download manually.")
        
        # Load datasets
        self.train_ds, self.val_ds = self.data_manager.load_datasets(
            image_size=self.config['image_size'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split']
        )
        
        # Get dataset info
        self.dataset_info = self.data_manager.get_dataset_info()
        logger.info(f"Dataset info: {self.dataset_info}")
        
        # Apply preprocessing and augmentation
        self.prepare_datasets()
    
    def prepare_datasets(self):
        """Apply preprocessing and data augmentation."""
        # Create preprocessing and augmentation layers
        preprocessing = create_preprocessing_layer()
        augmentation = create_data_augmentation_layer()
        
        # Apply to training dataset (with augmentation)
        self.train_ds = self.train_ds.map(
            lambda x, y: (preprocessing(augmentation(x)), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply to validation dataset (preprocessing only)
        self.val_ds = self.val_ds.map(
            lambda x, y: (preprocessing(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Optimize datasets for performance
        self.train_ds = self.train_ds.cache().prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(tf.data.AUTOTUNE)
        
        logger.info("Data preprocessing and augmentation applied")
    
    def build_model(self):
        """Build and compile the model."""
        logger.info(f"Building {self.config['architecture']} model...")
        
        builder = ImageClassifierBuilder(
            num_classes=self.dataset_info['num_classes'],
            input_shape=(*self.config['image_size'], 3)
        )
        
        # Build model based on architecture choice
        if self.config['architecture'] == 'resnet50':
            self.model = builder.build_resnet50(trainable_base=self.config['trainable_base'])
        elif self.config['architecture'] == 'efficientnet_b0':
            self.model = builder.build_efficientnet_b0(trainable_base=self.config['trainable_base'])
        elif self.config['architecture'] == 'vgg16':
            self.model = builder.build_vgg16(trainable_base=self.config['trainable_base'])
        elif self.config['architecture'] == 'custom_cnn':
            self.model = builder.build_custom_cnn()
        else:
            raise ValueError(f"Unsupported architecture: {self.config['architecture']}")
        
        # Compile model
        self.model = builder.compile_model(
            self.model,
            learning_rate=self.config['learning_rate'],
            optimizer=self.config['optimizer']
        )
        
        # Print model summary
        self.model.summary()
        logger.info("Model built and compiled successfully")
    
    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        
        callbacks = [
            ModelCheckpoint(
                filepath=str(self.output_dir / "best_model.keras"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        self.history = self.model.fit(
            self.train_ds,
            epochs=self.config['epochs'],
            validation_data=self.val_ds,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        # Save final model
        self.model.save(str(self.output_dir / "final_model.keras"))
        logger.info(f"Model saved to {self.output_dir}")

def main():
    """Main training function."""
    config = {
        'architecture': 'resnet50',
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'image_size': (224, 224),
        'validation_split': 0.2,
        'trainable_base': False,
        'optimizer': 'adam',
        'early_stopping_patience': 10
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Execute training pipeline
    trainer.prepare_data()
    trainer.build_model()
    trainer.train()
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main() 