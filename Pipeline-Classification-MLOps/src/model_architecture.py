"""
Model architecture definitions for image classification.
Includes various CNN architectures with transfer learning capabilities.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import logging

logger = logging.getLogger(__name__)

class ImageClassifierBuilder:
    """Builder class for creating different CNN architectures."""
    
    def __init__(self, num_classes: int, input_shape: tuple = (224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        
    def build_resnet50(self, trainable_base: bool = False) -> tf.keras.Model:
        """
        Build ResNet50-based classifier with transfer learning.
        
        Args:
            trainable_base: Whether to make the base model trainable
            
        Returns:
            Compiled Keras model
        """
        # Load pre-trained ResNet50
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = trainable_base
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        logger.info(f"ResNet50 model built with {self.num_classes} classes")
        return model
    
    def build_efficientnet_b0(self, trainable_base: bool = False) -> tf.keras.Model:
        """
        Build EfficientNetB0-based classifier.
        
        Args:
            trainable_base: Whether to make the base model trainable
            
        Returns:
            Compiled Keras model
        """
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = trainable_base
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        logger.info(f"EfficientNetB0 model built with {self.num_classes} classes")
        return model
    
    def build_vgg16(self, trainable_base: bool = False) -> tf.keras.Model:
        """
        Build VGG16-based classifier.
        
        Args:
            trainable_base: Whether to make the base model trainable
            
        Returns:
            Compiled Keras model
        """
        base_model = applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        base_model.trainable = trainable_base
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        logger.info(f"VGG16 model built with {self.num_classes} classes")
        return model
    
    def build_custom_cnn(self) -> tf.keras.Model:
        """
        Build a custom CNN from scratch.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        logger.info(f"Custom CNN model built with {self.num_classes} classes")
        return model
    
    def compile_model(self, 
                     model: tf.keras.Model, 
                     learning_rate: float = 0.001,
                     optimizer: str = 'adam') -> tf.keras.Model:
        """
        Compile the model with appropriate optimizer, loss, and metrics.
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            
        Returns:
            Compiled model
        """
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        model.compile(
            optimizer=opt,
            loss=CategoricalCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            ]
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
        return model

def create_data_augmentation_layer():
    """
    Create a data augmentation layer for training.
    
    Returns:
        Sequential layer with data augmentation
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1)
    ])

def create_preprocessing_layer():
    """
    Create a preprocessing layer for normalization.
    
    Returns:
        Rescaling layer
    """
    return layers.Rescaling(1./255) 