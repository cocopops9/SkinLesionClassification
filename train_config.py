"""
Training configuration for skin lesion classification models.
"""

import os
from pathlib import Path


class TrainConfig:
    """Configuration parameters for model training."""

    # Dataset paths
    DATA_DIR = 'data'  # Directory containing HAM10000 dataset
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VAL_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')

    # Model configurations
    AVAILABLE_MODELS = {
        'EfficientNetB3': {
            'img_size': 224,
            'base_lr': 1e-4,
            'fine_tune_lr': 1e-5
        },
        'InceptionV3': {
            'img_size': 299,
            'base_lr': 1e-4,
            'fine_tune_lr': 1e-5
        },
        'InceptionResNetV2': {
            'img_size': 299,
            'base_lr': 1e-4,
            'fine_tune_lr': 1e-5
        }
    }

    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS_PHASE1 = 15  # Train only classification head
    EPOCHS_PHASE2 = 30  # Fine-tune entire model
    NUM_CLASSES = 7

    # Class labels
    CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']

    # Data augmentation parameters
    AUGMENTATION = {
        'rotation_range': 40,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'nearest'
    }

    # Model saving
    MODELS_SAVE_DIR = 'trained_models'
    LOGS_DIR = 'training_logs'
    CHECKPOINT_DIR = 'checkpoints'

    # Training behavior
    EARLY_STOPPING_PATIENCE = 10
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5

    # Class weights (for imbalanced dataset)
    # HAM10000 is imbalanced - adjust these based on your dataset
    USE_CLASS_WEIGHTS = True

    # Validation split (if not using separate directories)
    VALIDATION_SPLIT = 0.2

    # Random seed for reproducibility
    RANDOM_SEED = 42

    @staticmethod
    def ensure_dirs():
        """Create necessary directories if they don't exist."""
        Path(TrainConfig.MODELS_SAVE_DIR).mkdir(exist_ok=True)
        Path(TrainConfig.LOGS_DIR).mkdir(exist_ok=True)
        Path(TrainConfig.CHECKPOINT_DIR).mkdir(exist_ok=True)
