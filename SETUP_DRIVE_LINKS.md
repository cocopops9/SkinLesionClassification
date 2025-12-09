# Setup Guide: Update Google Drive Links

Your models are now hosted in: https://drive.google.com/drive/folders/1uSlAS4t8_1AkUZmX0q_61jf5_5-TpYtZ

## Quick Update Method

### Option 1: Automated Script (Recommended)

```bash
python update_drive_links.py
```

Follow the interactive prompts to paste file IDs or full Drive URLs.

### Option 2: Manual Update

1. Open `classification.py`
2. Find the `MODEL_CONFIGS` dictionary (around line 29)
3. Update each model's `'url'` field with the new Google Drive file ID

## Getting Google Drive File IDs

For each `.h5` file in your Drive folder:

1. **Navigate to your folder**: https://drive.google.com/drive/folders/1uSlAS4t8_1AkUZmX0q_61jf5_5-TpYtZ

2. **For each model file** (EfficientNetB3.h5, Inceptionv3.h5, InceptionResNetv2.h5):
   - Right-click the file
   - Select "Get link" or "Share"
   - Copy the link

3. **Extract the FILE_ID** from the URL:
   ```
   URL format: https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing

   Example: https://drive.google.com/file/d/1abc123XYZ/view?usp=sharing
   FILE_ID: 1abc123XYZ
   ```

## Manual Update Example

Edit `classification.py`:

```python
MODEL_CONFIGS = {
    'EfficientNetB3': {
        'url': 'https://drive.google.com/uc?id=YOUR_EFFICIENTNET_FILE_ID',
        'filename': 'EfficientNetB3.h5',
        'img_size': 224,
        'accuracy': 0.85
    },
    'InceptionV3': {
        'url': 'https://drive.google.com/uc?id=YOUR_INCEPTION_FILE_ID',
        'filename': 'Inceptionv3.h5',
        'img_size': 299,
        'accuracy': 0.84
    },
    'InceptionResNetV2': {
        'url': 'https://drive.google.com/uc?id=YOUR_INCEPTIONRESNET_FILE_ID',
        'filename': 'InceptionResNetv2.h5',
        'img_size': 299,
        'accuracy': 0.79
    }
}
```

## Files to Update

You need file IDs for these 3 models:

- [ ] `EfficientNetB3.h5`
- [ ] `Inceptionv3.h5` (note: lowercase 'v')
- [ ] `InceptionResNetv2.h5`

## Verification

After updating, test the download:

```bash
# Start the app
streamlit run app.py

# Try to analyze an image
# The system will download models from your new Drive links
```

## Notes

- File names must match exactly (case-sensitive)
- The `uc?id=` format is required for gdown to work
- If download fails, check:
  - File sharing permissions (should be "Anyone with link can view")
  - File ID is correct
  - File name matches configuration

## Alternative: Use Local Models

If you prefer to skip Drive downloads and use local models:

1. Place your `.h5` files directly in the `models/` directory
2. Edit `classification.py` and set `'url': None` for each model
3. The system will use local files instead of downloading

Example:
```python
'EfficientNetB3': {
    'url': None,  # Skip download, use local file
    'filename': 'EfficientNetB3.h5',
    'img_size': 224,
    'accuracy': 0.85
}
```
