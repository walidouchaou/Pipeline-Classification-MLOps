"""
Data utilities for downloading and preprocessing the image classification dataset.
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import Tuple, Optional

import kaggle
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data downloading, extraction, and loading for the image classification project."""
    
    def __init__(self, data_dir: str = "data", dataset_name: str = "anthonytherrien/image-classification-dataset-32-classes"):
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
    
    def download_dataset(self) -> bool:
        """
        Download the dataset from Kaggle.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading dataset: {self.dataset_name}")
            kaggle.api.dataset_download_files(
                self.dataset_name, 
                path=str(self.raw_data_dir), 
                unzip=True
            )
            logger.info("Dataset downloaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            logger.info("Please ensure you have:")
            logger.info("1. Installed kaggle: pip install kaggle")
            logger.info("2. Set up your Kaggle API credentials")
            logger.info("3. Accepted the dataset terms on Kaggle website")
            return False
    
    def verify_dataset_structure(self) -> bool:
        """
        Verify that the dataset has been downloaded and extracted properly.
        
        Returns:
            bool: True if dataset structure is valid
        """
        # Look for the main dataset directory
        dataset_dirs = list(self.raw_data_dir.glob("*"))
        
        if not dataset_dirs:
            logger.error("No dataset directories found in raw data folder")
            return False
        
        # Check if we have class directories
        main_dir = dataset_dirs[0] if dataset_dirs[0].is_dir() else self.raw_data_dir
        class_dirs = [d for d in main_dir.iterdir() if d.is_dir()]
        
        if len(class_dirs) < 30:  # Expecting around 32 classes
            logger.warning(f"Found only {len(class_dirs)} class directories. Expected around 32.")
        
        logger.info(f"Dataset structure verified. Found {len(class_dirs)} classes.")
        return True
    
    def load_datasets(self, 
                     image_size: Tuple[int, int] = (224, 224),
                     batch_size: int = 32,
                     validation_split: float = 0.2,
                     seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Load training and validation datasets.
        
        Args:
            image_size: Target size for images (height, width)
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, validation_dataset)
        """
        # Find the actual data directory
        dataset_dirs = list(self.raw_data_dir.glob("*"))
        data_path = dataset_dirs[0] if dataset_dirs and dataset_dirs[0].is_dir() else self.raw_data_dir
        
        logger.info(f"Loading datasets from: {data_path}")
        
        # Create training dataset
        train_ds = image_dataset_from_directory(
            data_path,
            validation_split=validation_split,
            subset="training",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Create validation dataset
        val_ds = image_dataset_from_directory(
            data_path,
            validation_split=validation_split,
            subset="validation",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size
        )
        
        logger.info(f"Training dataset: {len(train_ds)} batches")
        logger.info(f"Validation dataset: {len(val_ds)} batches")
        logger.info(f"Class names: {train_ds.class_names}")
        
        return train_ds, val_ds
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            dict: Dataset information
        """
        dataset_dirs = list(self.raw_data_dir.glob("*"))
        if not dataset_dirs:
            return {"error": "Dataset not found"}
        
        data_path = dataset_dirs[0] if dataset_dirs[0].is_dir() else self.raw_data_dir
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        info = {
            "num_classes": len(class_dirs),
            "class_names": [d.name for d in class_dirs],
            "total_images": sum(len(list(d.glob("*"))) for d in class_dirs)
        }
        
        return info 