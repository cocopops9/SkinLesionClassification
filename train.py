"""
Training script for skin lesion classification models.
Implements transfer learning with two-phase training:
1. Train only classification head with frozen base
2. Fine-tune entire network with lower learning rate

Usage:
    python train.py --model EfficientNetB3 --epochs1 15 --epochs2 30
    python train.py --model InceptionV3 --freeze-layers 100
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)

from train_config import TrainConfig

# Set random seeds for reproducibility
np.random.seed(TrainConfig.RANDOM_SEED)
tf.random.set_seed(TrainConfig.RANDOM_SEED)


class ModelTrainer:
    """Handles model training with transfer learning."""

    def __init__(self, model_name: str):
        """
        Initialize trainer.

        Args:
            model_name: Name of the model to train
        """
        if model_name not in TrainConfig.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name
        self.config = TrainConfig.AVAILABLE_MODELS[model_name]
        self.img_size = self.config['img_size']
        self.model = None
        self.history_phase1 = None
        self.history_phase2 = None

        TrainConfig.ensure_dirs()

    def build_model(self, freeze_base: bool = True) -> Model:
        """
        Build model architecture with transfer learning.

        Args:
            freeze_base: Whether to freeze base model layers

        Returns:
            Compiled Keras model
        """
        print(f"\n{'='*60}")
        print(f"Building {self.model_name} architecture...")
        print(f"{'='*60}")

        # Import appropriate base model
        if self.model_name == 'EfficientNetB3':
            from tensorflow.keras.applications.efficientnet import EfficientNetB3 as BaseModel
        elif self.model_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3 as BaseModel
        elif self.model_name == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as BaseModel

        # Load base model with ImageNet weights
        base_model = BaseModel(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'  # Start with ImageNet pre-trained weights
        )

        # Freeze base model layers if requested
        base_model.trainable = not freeze_base

        if freeze_base:
            print(f"✓ Base model frozen ({len(base_model.layers)} layers)")
        else:
            print(f"✓ Base model unfrozen ({len(base_model.layers)} layers)")

        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = Dropout(0.2, name='dropout_0.2')(x)
        x = Dense(128, activation='relu', name='dense_128')(x)
        x = Dropout(0.1, name='dropout_0.1')(x)
        predictions = Dense(TrainConfig.NUM_CLASSES, activation='softmax', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions, name=f'{self.model_name}_skinlesion')

        print(f"✓ Added custom classification head")
        print(f"✓ Total parameters: {model.count_params():,}")

        return model

    def get_data_generators(self):
        """
        Create data generators for training and validation.

        Returns:
            Tuple of (train_generator, validation_generator)
        """
        print(f"\n{'='*60}")
        print("Setting up data generators...")
        print(f"{'='*60}")

        # Model-specific preprocessing
        if self.model_name == 'EfficientNetB3':
            from tensorflow.keras.applications.efficientnet import preprocess_input
        elif self.model_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        elif self.model_name == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: preprocess_input(x) / 255.0,
            rotation_range=TrainConfig.AUGMENTATION['rotation_range'],
            width_shift_range=TrainConfig.AUGMENTATION['width_shift_range'],
            height_shift_range=TrainConfig.AUGMENTATION['height_shift_range'],
            shear_range=TrainConfig.AUGMENTATION['shear_range'],
            zoom_range=TrainConfig.AUGMENTATION['zoom_range'],
            horizontal_flip=TrainConfig.AUGMENTATION['horizontal_flip'],
            vertical_flip=TrainConfig.AUGMENTATION['vertical_flip'],
            fill_mode=TrainConfig.AUGMENTATION['fill_mode'],
            validation_split=TrainConfig.VALIDATION_SPLIT if not os.path.exists(TrainConfig.VAL_DIR) else 0
        )

        # Validation data generator without augmentation
        val_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: preprocess_input(x) / 255.0
        )

        # Check if using separate train/val directories or single directory with split
        if os.path.exists(TrainConfig.VAL_DIR):
            # Separate directories
            train_generator = train_datagen.flow_from_directory(
                TrainConfig.TRAIN_DIR,
                target_size=(self.img_size, self.img_size),
                batch_size=TrainConfig.BATCH_SIZE,
                class_mode='categorical',
                shuffle=True,
                seed=TrainConfig.RANDOM_SEED
            )

            val_generator = val_datagen.flow_from_directory(
                TrainConfig.VAL_DIR,
                target_size=(self.img_size, self.img_size),
                batch_size=TrainConfig.BATCH_SIZE,
                class_mode='categorical',
                shuffle=False
            )
        else:
            # Single directory with split
            train_generator = train_datagen.flow_from_directory(
                TrainConfig.TRAIN_DIR,
                target_size=(self.img_size, self.img_size),
                batch_size=TrainConfig.BATCH_SIZE,
                class_mode='categorical',
                subset='training',
                shuffle=True,
                seed=TrainConfig.RANDOM_SEED
            )

            val_generator = val_datagen.flow_from_directory(
                TrainConfig.TRAIN_DIR,
                target_size=(self.img_size, self.img_size),
                batch_size=TrainConfig.BATCH_SIZE,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )

        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        print(f"✓ Classes: {list(train_generator.class_indices.keys())}")

        return train_generator, val_generator

    def compute_class_weights(self, train_generator):
        """
        Compute class weights for imbalanced dataset.

        Args:
            train_generator: Training data generator

        Returns:
            Dictionary of class weights
        """
        if not TrainConfig.USE_CLASS_WEIGHTS:
            return None

        print("\nComputing class weights for imbalanced dataset...")
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weight_dict = dict(enumerate(class_weights))

        for cls_idx, weight in class_weight_dict.items():
            cls_name = list(train_generator.class_indices.keys())[cls_idx]
            print(f"  {cls_name}: {weight:.2f}")

        return class_weight_dict

    def get_callbacks(self, phase: str):
        """
        Get training callbacks.

        Args:
            phase: 'phase1' or 'phase2'

        Returns:
            List of Keras callbacks
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=os.path.join(
                    TrainConfig.CHECKPOINT_DIR,
                    f'{self.model_name}_{phase}_{timestamp}.h5'
                ),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),

            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=TrainConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=TrainConfig.REDUCE_LR_FACTOR,
                patience=TrainConfig.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),

            # Log metrics to CSV
            CSVLogger(
                filename=os.path.join(
                    TrainConfig.LOGS_DIR,
                    f'{self.model_name}_{phase}_{timestamp}.csv'
                )
            ),

            # TensorBoard logging
            TensorBoard(
                log_dir=os.path.join(
                    TrainConfig.LOGS_DIR,
                    'tensorboard',
                    f'{self.model_name}_{phase}_{timestamp}'
                ),
                histogram_freq=1
            )
        ]

        return callbacks

    def train_phase1(self, epochs: int):
        """
        Phase 1: Train only classification head with frozen base.

        Args:
            epochs: Number of epochs for phase 1
        """
        print(f"\n{'='*60}")
        print("PHASE 1: Training classification head (frozen base)")
        print(f"{'='*60}")

        # Build model with frozen base
        self.model = self.build_model(freeze_base=True)

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['base_lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # Get data generators
        train_gen, val_gen = self.get_data_generators()
        class_weights = self.compute_class_weights(train_gen)

        # Train
        print(f"\nStarting Phase 1 training for {epochs} epochs...")
        self.history_phase1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=self.get_callbacks('phase1'),
            verbose=1
        )

        print("\n✓ Phase 1 training complete")

    def train_phase2(self, epochs: int, unfreeze_layers: int = None):
        """
        Phase 2: Fine-tune entire model with lower learning rate.

        Args:
            epochs: Number of epochs for phase 2
            unfreeze_layers: Number of layers to unfreeze (None = all)
        """
        print(f"\n{'='*60}")
        print("PHASE 2: Fine-tuning entire model")
        print(f"{'='*60}")

        if self.model is None:
            raise RuntimeError("Must run train_phase1 first")

        # Unfreeze base model layers
        base_model = self.model.layers[0] if hasattr(self.model.layers[0], 'layers') else None

        if base_model:
            if unfreeze_layers is None:
                # Unfreeze all layers
                base_model.trainable = True
                print(f"✓ Unfroze all {len(base_model.layers)} base layers")
            else:
                # Unfreeze only last N layers
                base_model.trainable = True
                for layer in base_model.layers[:-unfreeze_layers]:
                    layer.trainable = False
                print(f"✓ Unfroze last {unfreeze_layers} base layers")
        else:
            # No separate base model, unfreeze all
            self.model.trainable = True
            print(f"✓ Unfroze all model layers")

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['fine_tune_lr']),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        # Get data generators
        train_gen, val_gen = self.get_data_generators()
        class_weights = self.compute_class_weights(train_gen)

        # Train
        print(f"\nStarting Phase 2 training for {epochs} epochs...")
        self.history_phase2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=self.get_callbacks('phase2'),
            verbose=1
        )

        print("\n✓ Phase 2 training complete")

    def save_final_model(self):
        """Save the final trained model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(
            TrainConfig.MODELS_SAVE_DIR,
            f'{self.model_name}_final_{timestamp}.h5'
        )

        self.model.save(save_path)
        print(f"\n✓ Final model saved to: {save_path}")

        return save_path

    def evaluate(self):
        """Evaluate model on test set if available."""
        if not os.path.exists(TrainConfig.TEST_DIR):
            print("\n⚠ No test directory found, skipping evaluation")
            return

        print(f"\n{'='*60}")
        print("Evaluating on test set...")
        print(f"{'='*60}")

        # Model-specific preprocessing
        if self.model_name == 'EfficientNetB3':
            from tensorflow.keras.applications.efficientnet import preprocess_input
        elif self.model_name == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        elif self.model_name == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

        test_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: preprocess_input(x) / 255.0
        )

        test_generator = test_datagen.flow_from_directory(
            TrainConfig.TEST_DIR,
            target_size=(self.img_size, self.img_size),
            batch_size=TrainConfig.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        results = self.model.evaluate(test_generator, verbose=1)

        print(f"\nTest Results:")
        print(f"  Loss: {results[0]:.4f}")
        print(f"  Accuracy: {results[1]:.4f}")
        if len(results) > 2:
            print(f"  AUC: {results[2]:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(TrainConfig.AVAILABLE_MODELS.keys()),
                        help='Model architecture to train')
    parser.add_argument('--epochs1', type=int, default=TrainConfig.EPOCHS_PHASE1,
                        help='Number of epochs for phase 1 (frozen base)')
    parser.add_argument('--epochs2', type=int, default=TrainConfig.EPOCHS_PHASE2,
                        help='Number of epochs for phase 2 (fine-tuning)')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip phase 1 (load pre-trained model)')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip phase 2 (only train classification head)')
    parser.add_argument('--unfreeze-layers', type=int, default=None,
                        help='Number of base layers to unfreeze in phase 2 (default: all)')
    parser.add_argument('--batch-size', type=int, default=TrainConfig.BATCH_SIZE,
                        help='Batch size for training')

    args = parser.parse_args()

    # Update config with command line args
    TrainConfig.BATCH_SIZE = args.batch_size

    # Initialize trainer
    trainer = ModelTrainer(args.model)

    # Training pipeline
    try:
        if not args.skip_phase1:
            trainer.train_phase1(epochs=args.epochs1)

        if not args.skip_phase2:
            trainer.train_phase2(epochs=args.epochs2, unfreeze_layers=args.unfreeze_layers)

        # Save final model
        model_path = trainer.save_final_model()

        # Evaluate on test set
        trainer.evaluate()

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Model saved to: {model_path}")
        print(f"Logs saved to: {TrainConfig.LOGS_DIR}")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        if trainer.model is not None:
            save = input("Save current model? (y/n): ")
            if save.lower() == 'y':
                trainer.save_final_model()

    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
