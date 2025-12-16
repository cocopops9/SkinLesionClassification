"""
Enhanced classification module for melanoma detection.
Integrates with explainability engine and validation system.
"""

import os
import time
import gdown
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from typing import Tuple, Dict, Optional, List
import cv2


# Disable GPU for CPU-only environments
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MelanomaClassifier:
    """
    Enhanced melanoma classification with model ensemble support.
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'EfficientNetB3': {
            'url': 'https://drive.google.com/uc?id=1lebK-70tcon9hUWjfTm-sqsjqH2Ny83f',
            'filename': 'EfficientNetB3.h5',
            'img_size': 224,
            'accuracy': 0.85
        },
        'InceptionV3': {
            'url': 'https://drive.google.com/uc?id=1wmNirs6NwvLEamvGSwLzRlJHLwHivPit',
            'filename': 'Inceptionv3.h5',
            'img_size': 299,
            'accuracy': 0.84
        },
        'InceptionResNetV2': {
            'url': 'https://drive.google.com/uc?id=14xPWqyeiz4S2XPiizEeDTTEQn5sSYBbE',
            'filename': 'InceptionResNetv2.h5',
            'img_size': 299,
            'accuracy': 0.79
        }
    }
    
    # Class labels
    CLASS_LABELS = {
        0: {'name': 'AKIEC', 'full': "Actinic keratosis / Bowen's disease", 'type': 'Pre-Malignant/Malignant'},
        1: {'name': 'BCC', 'full': 'Basal cell carcinoma', 'type': 'Malignant'},
        2: {'name': 'BKL', 'full': 'Benign keratosis', 'type': 'Benign'},
        3: {'name': 'DF', 'full': 'Dermatofibroma', 'type': 'Benign'},
        4: {'name': 'MEL', 'full': 'Melanoma', 'type': 'Malignant'},
        5: {'name': 'NV', 'full': 'Melanocytic nevus', 'type': 'Benign'},
        6: {'name': 'VASC', 'full': 'Vascular lesion', 'type': 'Indecidable'}
    }
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize classifier with models directory.
        
        Args:
            models_dir: Directory to store model weights
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
    
    def download_weights(self, model_name: str) -> bool:
        """
        Download model weights if not present.
        
        Args:
            model_name: Name of the model
        
        Returns:
            True if successful
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.MODEL_CONFIGS[model_name]
        output_path = self.models_dir / config['filename']
        
        if not output_path.exists():
            try:
                print(f"Downloading weights for {model_name}...")
                gdown.download(config['url'], str(output_path), quiet=False)
                return True
            except Exception as e:
                print(f"Error downloading {model_name}: {e}")
                return False
        return True
    
    def build_model(self, model_name: str) -> Model:
        """
        Build model architecture with pre-trained weights.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Compiled Keras model
        """
        config = self.MODEL_CONFIGS[model_name]
        img_size = config['img_size']
        weights_path = self.models_dir / config['filename']
        
        # Import appropriate model class
        if model_name == 'EfficientNetB3':
            from tensorflow.keras.applications.efficientnet import EfficientNetB3 as BaseModel
        elif model_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel
        elif model_name == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as BaseModel
        
        # Build architecture
        base_model = BaseModel(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights=None
        )
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        predictions = Dense(7, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Load weights
        model.load_weights(str(weights_path))
        
        return model
    
    def get_model(self, model_name: str) -> Model:
        """
        Get model, loading if necessary with caching.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Loaded model
        """
        if model_name not in self.loaded_models:
            if not self.download_weights(model_name):
                raise RuntimeError(f"Failed to download weights for {model_name}")
            self.loaded_models[model_name] = self.build_model(model_name)
        
        return self.loaded_models[model_name]
    
    def preprocess_image(self, img_path: str, model_name: str) -> np.ndarray:
        """
        Preprocess image for specific model.
        
        Args:
            img_path: Path to image file
            model_name: Name of the model
        
        Returns:
            Preprocessed image array
        """
        config = self.MODEL_CONFIGS[model_name]
        img_size = config['img_size']
        
        # Load and resize image
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply model-specific preprocessing
        if model_name == 'EfficientNetB3':
            from tensorflow.keras.applications.efficientnet import preprocess_input
        elif model_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        elif model_name == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        
        img_array = preprocess_input(img_array)
        img_array = img_array / 255.0
        
        return img_array
    
    def predict_single(self, img_path: str, model_name: str) -> Tuple[int, np.ndarray]:
        """
        Predict using single model.
        
        Args:
            img_path: Path to image file
            model_name: Name of the model
        
        Returns:
            Tuple of (predicted class index, confidence scores)
        """
        model = self.get_model(model_name)
        img_array = self.preprocess_image(img_path, model_name)
        
        predictions = model.predict(img_array, verbose=0)
        pred_class = np.argmax(predictions[0])
        
        return pred_class, predictions[0]
    
    def predict_ensemble(self, img_path: str, model_names: List[str]) -> Dict[str, any]:
        """
        Predict using model ensemble with weighted voting.
        
        Args:
            img_path: Path to image file
            model_names: List of model names to use
        
        Returns:
            Dictionary with ensemble results
        """
        if len(model_names) < 2:
            raise ValueError("Ensemble requires at least 2 models")
        
        ensemble_scores = np.zeros((1, 7))
        total_weight = 0
        individual_predictions = {}
        
        for model_name in model_names:
            try:
                pred_class, scores = self.predict_single(img_path, model_name)
                
                # Weight by model accuracy
                weight = self.MODEL_CONFIGS[model_name]['accuracy']
                ensemble_scores += scores * weight
                total_weight += weight
                
                individual_predictions[model_name] = {
                    'class': pred_class,
                    'scores': scores.tolist(),
                    'confidence': float(scores[pred_class])
                }
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        if total_weight == 0:
            raise RuntimeError("No models succeeded in prediction")
        
        # Normalize ensemble scores
        ensemble_scores = ensemble_scores[0] / total_weight
        final_pred = np.argmax(ensemble_scores)
        
        return {
            'predicted_class': int(final_pred),
            'ensemble_scores': ensemble_scores.tolist(),
            'confidence': float(ensemble_scores[final_pred]),
            'individual_predictions': individual_predictions,
            'diagnosis': self.CLASS_LABELS[final_pred]['full'],
            'lesion_type': self.CLASS_LABELS[final_pred]['type']
        }
    
    def get_prediction_with_metadata(self, img_path: str, 
                                    model_names: List[str]) -> Dict[str, any]:
        """
        Get prediction with full metadata for database storage.
        
        Args:
            img_path: Path to image file
            model_names: List of model names or single model name
        
        Returns:
            Complete prediction metadata
        """
        start_time = time.time()
        
        if isinstance(model_names, str):
            model_names = [model_names]
        
        if len(model_names) == 1:
            pred_class, scores = self.predict_single(img_path, model_names[0])
            result = {
                'predicted_class': int(pred_class),
                'confidence_scores': {
                    self.CLASS_LABELS[i]['name']: float(scores[i]) 
                    for i in range(7)
                },
                'confidence': float(scores[pred_class]),
                'diagnosis': self.CLASS_LABELS[pred_class]['full'],
                'lesion_type': self.CLASS_LABELS[pred_class]['type'],
                'model_used': model_names[0],
                'ensemble_used': False
            }
        else:
            ensemble_result = self.predict_ensemble(img_path, model_names)
            result = {
                'predicted_class': ensemble_result['predicted_class'],
                'confidence_scores': {
                    self.CLASS_LABELS[i]['name']: ensemble_result['ensemble_scores'][i] 
                    for i in range(7)
                },
                'confidence': ensemble_result['confidence'],
                'diagnosis': ensemble_result['diagnosis'],
                'lesion_type': ensemble_result['lesion_type'],
                'model_used': ', '.join(model_names),
                'ensemble_used': True,
                'individual_predictions': ensemble_result['individual_predictions']
            }
        
        result['processing_time'] = time.time() - start_time
        
        return result
    
    def get_model_for_explainability(self, model_name: str) -> Tuple[Model, int]:
        """
        Get model and configuration for explainability analysis.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Tuple of (model, image_size)
        """
        model = self.get_model(model_name)
        img_size = self.MODEL_CONFIGS[model_name]['img_size']
        
        return model, img_size
    
    def clear_cache(self):
        """Clear loaded models from memory."""
        self.loaded_models.clear()
        tf.keras.backend.clear_session()
