# Enhanced Melanoma Detection System

## Overview

Advanced melanoma detection system with user authentication, explainable AI, and comprehensive image validation. This system provides enterprise-grade features for medical image analysis with full transparency and user management.

**New:** This system now includes full training capabilities - train your own models on custom datasets! See [TRAINING_README.md](TRAINING_README.md) for details.

## Key Enhancements

### 1. Authentication System
- **Secure user registration and login**
- **bcrypt password hashing** (salt rounds: 12)
- **Session management** with Streamlit session state
- **User-specific data isolation**

### 2. User-Specific Image Storage
- **SQLAlchemy ORM** with SQLite backend
- **UUID-based image identification**
- **Relational data model** linking users to images
- **Persistent storage** of analysis history

### 3. Explainability Layer
- **Grad-CAM visualization** for CNN decision transparency
- **Occlusion sensitivity analysis**
- **Comprehensive textual explanations** based on:
  - Dermoscopic features
  - Clinical correlations
  - ABCDE criteria
  - Color distribution analysis
  - Asymmetry scoring

### 4. Image Validation
- **MobileNetV2-based classification** to detect non-skin images
- **Statistical validation** including:
  - Color space analysis (HSV, LAB)
  - Edge density computation
  - Skin tone detection
  - Text/document detection
- **Validation confidence scoring**

## Technical Architecture

```
SkinLesionClassification/
├── app.py                    # Main Streamlit application
├── database.py              # SQLAlchemy models and database management
├── classification.py        # Enhanced CNN classifier with ensemble support
├── explainability.py       # Grad-CAM and explanation generation
├── image_validator.py      # Non-skin image detection
├── train.py                # Model training script (NEW)
├── train_config.py         # Training configuration (NEW)
├── update_drive_links.py   # Helper to update Drive URLs (NEW)
├── requirements.txt        # Python dependencies
├── README.md               # Main documentation
├── TRAINING_README.md      # Training guide (NEW)
├── melanoma_app.db        # SQLite database (auto-created)
├── models/                # Downloaded/trained model weights
├── trained_models/        # Newly trained models (NEW)
├── training_logs/         # Training metrics and logs (NEW)
├── checkpoints/           # Training checkpoints (NEW)
├── data/                  # Training dataset (user-provided)
├── user_uploads/          # User-specific image storage
└── heatmaps/             # Grad-CAM heatmap storage
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)
- 8GB+ RAM recommended

### Setup Instructions

1. **Clone or extract the project:**
```bash
cd melanoma_enhanced
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download model weights (automatic on first use):**
The system will automatically download required model weights (~500MB total) on first prediction.

5. **Run the application:**
```bash
streamlit run app.py
```

6. **Access the application:**
Open browser to `http://localhost:8501`

## Usage Guide

### 1. Registration
- Create account with username, email, and password
- Password must be minimum 8 characters
- Email validation enforced

### 2. Login
- Authenticate with username and password
- Session persists until logout

### 3. Image Analysis
- Upload dermoscopic or clinical skin images
- Select single model or ensemble
- Enable/disable explainability features
- View results with confidence scores

### 4. Features

#### Model Selection
- **Single Model**: Choose one CNN architecture
- **Ensemble**: Combine multiple models with weighted voting

#### Available Models
| Model | Accuracy | Input Size | Parameters |
|-------|----------|------------|------------|
| EfficientNetB3 | 85% | 224x224 | 12M |
| InceptionV3 | 84% | 299x299 | 24M |
| InceptionResNetV2 | 79% | 299x299 | 56M |

#### Analysis Options
- **Explainability Report**: Detailed AI reasoning
- **Grad-CAM Heatmap**: Visual attention mapping
- **Save to History**: Persistent storage

### 5. History
- View all previous analyses
- Filter by date, diagnosis, validity
- Export data as CSV

## Training Your Own Models

This system now includes comprehensive training capabilities to fine-tune models on custom datasets.

### Quick Start

```bash
# Train EfficientNetB3 on your dataset
python train.py --model EfficientNetB3

# Train with custom parameters
python train.py --model InceptionV3 --epochs1 20 --epochs2 40 --batch-size 16
```

### Training Features

- **Two-phase transfer learning**: Train classification head first, then fine-tune entire model
- **Automated data augmentation**: Rotation, flipping, zooming for robustness
- **Class weight balancing**: Handles imbalanced datasets automatically
- **Multiple callbacks**: Early stopping, learning rate reduction, model checkpointing
- **TensorBoard integration**: Real-time training visualization
- **Flexible configuration**: Customize via `train_config.py`

### Training Pipeline

1. **Prepare HAM10000 dataset** in `data/train/` directory
2. **Configure training** in `train_config.py`
3. **Run training**: `python train.py --model EfficientNetB3`
4. **Monitor progress**: `tensorboard --logdir training_logs/tensorboard`
5. **Use trained model**: Copy from `trained_models/` to `models/`

**For complete training documentation, see [TRAINING_README.md](TRAINING_README.md)**

### Updating Model Weights

To use models from your Google Drive folder:

```bash
# Run the helper script
python update_drive_links.py

# Or manually edit classification.py MODEL_CONFIGS
```

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at DATETIME,
    last_login DATETIME,
    is_active BOOLEAN
);
```

### Images Table
```sql
CREATE TABLE images (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) FOREIGN KEY,
    filename VARCHAR(255),
    file_path VARCHAR(500),
    upload_time DATETIME,
    diagnosis VARCHAR(100),
    lesion_type VARCHAR(50),
    confidence_scores TEXT (JSON),
    model_used VARCHAR(100),
    explanation_text TEXT,
    heatmap_path VARCHAR(500),
    is_valid_skin_image BOOLEAN,
    validation_message VARCHAR(500),
    processing_time FLOAT
);
```

## API Documentation

### DatabaseManager
```python
# Create user
user = DatabaseManager.create_user(username, password, email)

# Authenticate
user = DatabaseManager.authenticate_user(username, password)

# Save image record
record = DatabaseManager.save_image_record(user_id, image_data)

# Get user images
images = DatabaseManager.get_user_images(user_id, limit=100)
```

### MelanomaClassifier
```python
# Initialize
classifier = MelanomaClassifier()

# Single prediction
pred_class, scores = classifier.predict_single(img_path, model_name)

# Ensemble prediction
results = classifier.predict_ensemble(img_path, model_names)

# With metadata
metadata = classifier.get_prediction_with_metadata(img_path, model_names)
```

### ExplainabilityEngine
```python
# Grad-CAM
heatmap = ExplainabilityEngine.compute_gradcam(model, img_array, pred_idx)

# Generate explanation
explanation = ExplainabilityEngine.generate_explanation(
    pred_idx, scores, color_metrics, asymmetry
)

# Overlay heatmap
overlaid = ExplainabilityEngine.overlay_heatmap(img, heatmap)
```

### ImageValidator
```python
# Initialize
validator = ImageValidator()

# Validate image
results = validator.validate_image(img)

# Generate report
report = validator.generate_validation_report(results)
```

## Security Considerations

1. **Password Security**
   - bcrypt with automatic salt generation
   - Minimum 8 character enforcement
   - No password recovery (demo limitation)

2. **Data Isolation**
   - User-specific directories
   - Foreign key constraints
   - Session-based access control

3. **Input Validation**
   - Image format verification
   - File size limits
   - Content validation

## Performance Optimization

1. **Model Caching**
   - Models loaded once and cached
   - TensorFlow session management
   - Memory-efficient loading

2. **Database Optimization**
   - Indexed columns for queries
   - Connection pooling
   - Batch operations where applicable

3. **Image Processing**
   - Efficient numpy operations
   - OpenCV for fast image processing
   - Lazy loading of large files

## Limitations

1. **Medical Disclaimer**
   - Intelligent decision support system for dermatologists
   - Only dermatologists are responsible for diagnosis
   - AI predictions serve as supplementary clinical information

2. **Technical Limitations**
   - SQLite database (not for production scale)
   - Local file storage (not distributed)
   - No email verification
   - No password recovery

3. **Model Limitations**
   - Training on HAM10000 dataset
   - Limited to 7 diagnosis categories
   - Requires good quality images
   - Best with dermoscopic images

## Future Enhancements

1. **Production Features**
   - PostgreSQL/MySQL support
   - Cloud storage (S3/GCS)
   - Email verification
   - Password recovery
   - 2FA authentication

2. **Advanced AI**
   - LIME explanations
   - SHAP values
   - Uncertainty quantification
   - Active learning

3. **Clinical Integration**
   - DICOM support
   - HL7/FHIR integration
   - Audit logging
   - HIPAA compliance

## Troubleshooting

### Common Issues

1. **Model download fails**
   - Check internet connection
   - Verify Google Drive accessibility
   - Manual download from provided URLs

2. **Database errors**
   - Delete `melanoma_app.db` and restart
   - Check write permissions

3. **Memory errors**
   - Reduce batch size
   - Use single model instead of ensemble
   - Clear model cache

4. **Invalid image errors**
   - Ensure image is skin lesion
   - Check image format (JPG/PNG)
   - Verify image quality

## Testing

### Unit Tests (to implement)
```python
# Test user creation
test_user_registration()
test_password_hashing()
test_duplicate_user()

# Test classification
test_single_prediction()
test_ensemble_prediction()

# Test validation
test_skin_detection()
test_non_skin_rejection()

# Test explainability
test_gradcam_generation()
test_explanation_text()
```

## Contributing

Contributions welcome for:
- Additional model architectures
- Improved explainability techniques
- Enhanced validation methods
- Bug fixes and optimizations

## License

This project is for educational and research purposes. Model weights are subject to their original licenses.

## Citations

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

## Contact

For technical questions or issues, please create an issue in the repository.

---

**⚠️ MEDICAL DISCLAIMER**: This is an intelligent decision support system designed to assist dermatologists in their clinical practice. Only qualified dermatologists are responsible for the final diagnosis and treatment decisions. The AI-generated predictions serve as supplementary information and do not replace professional medical judgment. The authors assume no liability for decisions made based on this system's output.
