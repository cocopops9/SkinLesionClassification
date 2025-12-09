# Model Training Guide

This guide explains how to train and fine-tune your own skin lesion classification models on the HAM10000 dataset.

## Overview

The training pipeline implements **two-phase transfer learning**:

1. **Phase 1**: Train only the classification head (frozen base model)
2. **Phase 2**: Fine-tune the entire network with a lower learning rate

This approach is based on best practices for medical imaging with limited datasets.

---

## Prerequisites

### 1. Install Dependencies

All dependencies are already in `requirements.txt`. If you need to reinstall:

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the HAM10000 dataset and organize it in the following structure:

```
data/
├── train/
│   ├── AKIEC/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── BCC/
│   ├── BKL/
│   ├── DF/
│   ├── MEL/
│   ├── NV/
│   └── VASC/
├── val/  (optional - will auto-split if not present)
│   ├── AKIEC/
│   ├── BCC/
│   └── ...
└── test/  (optional)
    ├── AKIEC/
    ├── BCC/
    └── ...
```

**Class names:**
- `AKIEC`: Actinic keratosis / Bowen's disease
- `BCC`: Basal cell carcinoma
- `BKL`: Benign keratosis
- `DF`: Dermatofibroma
- `MEL`: Melanoma
- `NV`: Melanocytic nevus
- `VASC`: Vascular lesion

**Dataset sources:**
- [HAM10000 on Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)
- [ISIC Archive](https://www.isic-archive.com/)

---

## Training Configuration

Edit `train_config.py` to customize training parameters:

```python
# Key parameters:
BATCH_SIZE = 32              # Batch size for training
EPOCHS_PHASE1 = 15           # Epochs for phase 1
EPOCHS_PHASE2 = 30           # Epochs for phase 2
VALIDATION_SPLIT = 0.2       # If no separate val directory

# Data augmentation:
AUGMENTATION = {
    'rotation_range': 40,
    'horizontal_flip': True,
    'vertical_flip': True,
    # ... etc
}
```

---

## Training Commands

### Basic Training (All Phases)

Train a model with default settings:

```bash
python train.py --model EfficientNetB3
```

This will:
1. Train classification head for 15 epochs (frozen base)
2. Fine-tune entire model for 30 epochs
3. Save best models to `checkpoints/`
4. Save final model to `trained_models/`
5. Log metrics to `training_logs/`

### Available Models

- `EfficientNetB3` (recommended, 224x224, best accuracy)
- `InceptionV3` (299x299)
- `InceptionResNetV2` (299x299)

### Custom Training Parameters

```bash
# Custom epoch counts
python train.py --model InceptionV3 --epochs1 20 --epochs2 40

# Custom batch size
python train.py --model EfficientNetB3 --batch-size 16

# Unfreeze only last 50 layers in phase 2
python train.py --model EfficientNetB3 --unfreeze-layers 50

# Skip phase 1 (if you already have a trained classification head)
python train.py --model InceptionV3 --skip-phase1 --epochs2 30

# Skip phase 2 (train only classification head)
python train.py --model EfficientNetB3 --skip-phase2 --epochs1 20
```

---

## Training Process

### Phase 1: Classification Head Training

**Duration:** ~15 epochs (depends on dataset size)

**What happens:**
- Base model (EfficientNetB3/InceptionV3/etc.) is frozen
- Only the custom dense layers are trained
- Uses learning rate: `1e-4`
- Data augmentation is applied

**Why this phase:**
- Allows custom layers to learn from scratch
- Prevents destroying pre-trained ImageNet features
- Faster initial convergence

### Phase 2: Fine-Tuning

**Duration:** ~30 epochs

**What happens:**
- All layers are unfrozen (or last N layers)
- Entire model is fine-tuned
- Uses lower learning rate: `1e-5`
- Data augmentation continues

**Why this phase:**
- Adapts pre-trained features to skin lesions
- Achieves better accuracy than frozen base
- Lower LR prevents overfitting

---

## Monitoring Training

### TensorBoard

Launch TensorBoard to monitor training in real-time:

```bash
tensorboard --logdir training_logs/tensorboard
```

Then open: `http://localhost:6006`

**Metrics tracked:**
- Training/validation loss
- Training/validation accuracy
- Learning rate changes
- AUC scores

### CSV Logs

Training metrics are saved to CSV files in `training_logs/`:

```
training_logs/
├── EfficientNetB3_phase1_20231209_143022.csv
├── EfficientNetB3_phase2_20231209_150145.csv
└── tensorboard/
```

---

## Output Files

After training completes, you'll have:

### 1. Checkpoints (Best Models)

Saved in `checkpoints/` during training:
- `{model}_phase1_{timestamp}.h5` - Best phase 1 model
- `{model}_phase2_{timestamp}.h5` - Best phase 2 model

These are saved based on **validation accuracy** (best model only).

### 2. Final Model

Saved in `trained_models/`:
- `{model}_final_{timestamp}.h5` - Final trained model

This is the complete model after both phases.

### 3. Training Logs

- CSV logs: `training_logs/{model}_{phase}_{timestamp}.csv`
- TensorBoard logs: `training_logs/tensorboard/`

---

## Using Your Trained Model

### Option 1: Replace Pre-trained Weights

1. Rename your trained model:
   ```bash
   cp trained_models/EfficientNetB3_final_20231209.h5 models/EfficientNetB3.h5
   ```

2. The app will automatically use your custom weights!

### Option 2: Update classification.py

Edit `classification.py` to use local models instead of Google Drive:

```python
# Before:
'url': 'https://drive.google.com/...'

# After:
'url': None  # Will skip download
'filename': 'EfficientNetB3.h5'  # Must exist in models/ directory
```

---

## Handling Class Imbalance

HAM10000 is **highly imbalanced** (NV class dominates).

The training script automatically computes **class weights**:

```python
# In train_config.py
USE_CLASS_WEIGHTS = True  # Enabled by default
```

This balances training so minority classes (like MEL) get more importance.

**Manual weights** (if needed):
```python
class_weights = {
    0: 1.5,  # AKIEC
    1: 3.0,  # BCC
    2: 1.0,  # BKL
    3: 5.0,  # DF (rarest)
    4: 2.5,  # MEL
    5: 0.5,  # NV (most common)
    6: 2.0   # VASC
}
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1:** Reduce batch size
```bash
python train.py --model EfficientNetB3 --batch-size 16
```

**Solution 2:** Use CPU only
```python
# Add to top of train.py
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**Solution 3:** Use smaller model
```bash
python train.py --model InceptionV3  # Smaller than EfficientNetB3
```

### Low Validation Accuracy

**Possible causes:**
1. **Insufficient data** - Need at least 500-1000 images per class
2. **Too much regularization** - Reduce dropout in `train_config.py`
3. **Learning rate too high/low** - Adjust in `train_config.py`
4. **Class imbalance** - Ensure `USE_CLASS_WEIGHTS = True`

**Solutions:**
- Increase data augmentation
- Train for more epochs
- Try different models
- Check data quality (corrupted images, wrong labels)

### Training Taking Too Long

**Speed up training:**
1. Use GPU (automatically detected if available)
2. Increase batch size: `--batch-size 64`
3. Reduce image size (edit model config)
4. Skip phase 2: `--skip-phase2`

---

## Advanced Options

### Custom Architecture

Edit `train.py` to modify the classification head:

```python
# Current architecture:
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)
predictions = Dense(7, activation='softmax')(x)

# Example: Simpler head
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(7, activation='softmax')(x)
```

### Different Base Models

To add new base models (e.g., ResNet, DenseNet):

1. Add to `train_config.py`:
```python
AVAILABLE_MODELS = {
    'ResNet50': {
        'img_size': 224,
        'base_lr': 1e-4,
        'fine_tune_lr': 1e-5
    }
}
```

2. Add imports to `train.py`:
```python
elif model_name == 'ResNet50':
    from tensorflow.keras.applications.resnet import ResNet50 as BaseModel
```

---

## Performance Benchmarks

**Expected results on HAM10000:**

| Model | Phase 1 Acc | Phase 2 Acc | Training Time |
|-------|-------------|-------------|---------------|
| EfficientNetB3 | 75-80% | 82-85% | ~2-3 hours (GPU) |
| InceptionV3 | 73-78% | 80-84% | ~2-3 hours (GPU) |
| InceptionResNetV2 | 72-77% | 78-82% | ~3-4 hours (GPU) |

*Results may vary based on dataset split and augmentation*

---

## Best Practices

1. **Always use validation set** - Monitor for overfitting
2. **Enable early stopping** - Prevents wasted training time
3. **Save checkpoints frequently** - Don't lose progress
4. **Use data augmentation** - Critical for small medical datasets
5. **Monitor class weights** - Ensure balanced learning
6. **Start with small epochs** - Test pipeline before long training
7. **Use TensorBoard** - Visualize training progress
8. **Test on held-out data** - Evaluate generalization

---

## Example Training Session

```bash
# 1. Prepare data
ls data/train/
# Output: AKIEC/ BCC/ BKL/ DF/ MEL/ NV/ VASC/

# 2. Quick test run (2 epochs)
python train.py --model EfficientNetB3 --epochs1 2 --epochs2 2

# 3. Full training
python train.py --model EfficientNetB3 --epochs1 15 --epochs2 30

# 4. Monitor with TensorBoard (in separate terminal)
tensorboard --logdir training_logs/tensorboard

# 5. After training completes, copy to models/
cp trained_models/EfficientNetB3_final_20231209_154523.h5 models/EfficientNetB3.h5

# 6. Test in web app
streamlit run app.py
```

---

## Citation

If you use this training code, please cite:

```bibtex
@article{tschandl2018ham10000,
  title={The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author={Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal={Scientific data},
  volume={5},
  pages={180161},
  year={2018}
}
```

---

## Support

For issues or questions:
1. Check this README first
2. Review `train_config.py` settings
3. Check TensorBoard logs
4. Create an issue in the repository

---

**Happy Training! 🚀**
