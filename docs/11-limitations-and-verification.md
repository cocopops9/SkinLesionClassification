# 11 · Limitations and verification

[← 10 Application, storage and security](10-application-storage-security.md) · [Docs index](README.md)

This chapter collects what was checked while writing the documentation, how it was checked, and the limitations that follow. Confidence labels: **verified** (reproduced by running code), **source** (read directly from the code, no ambiguity), **conditional** (true under a stated assumption).

## Findings

| # | Topic | Finding | Status |
|---|---|---|---|
| 1 | Accuracy figures | 0.85, 0.84 and 0.79 are constants in `MODEL_CONFIGS`; no evaluation script, test split, confusion matrix or sensitivity per class is included to reproduce them. | source |
| 2 | Weights provenance | The `.h5` files come from Google Drive; the training data, split and code version used are not recorded. | source |
| 3 | Preprocessing | Inputs are divided by 255 once more than the backbones expect (ranges in [chapter 04](04-preprocessing-and-architecture.md)); identical in training and inference. | source |
| 4 | EfficientNetB3 graph | With `weights='imagenet'` Keras adds `Rescaling(1/√σ)`; with `weights=None` it does not. Training and inference graphs therefore scale inputs differently (factor ≈ 0.48). | conditional: weights trained by `train.py` under current Keras |
| 5 | CLIP gate | Softmax without CLIP's logit scale; probabilities cluster near 0.5, decision unaffected. | source + math |
| 6 | Grad-CAM layer | `top_conv`, `conv2d_93` (192 channels of `mixed10`) and `conv_7b_ac`; first selected model only. | verified (TF 2.20) |
| 7 | Grad-CAM normalisation | Divisor is the pre-ReLU maximum without epsilon. | source |
| 8 | ABCDE | Bright-region foreground; on all 10 `media/` images the contour touches the border; D assumes 10 px/mm. | verified |
| 9 | Training split | Without `data/val/` the validation generator has 0 images. | verified (TF 2.20) |
| 10 | `--unfreeze-layers` | Ignored; the whole model is always unfrozen in phase 2. | verified (TF 2.20) |
| 11 | Shear augmentation | `shear_range=0.2` means ±0.2 degrees. | source (Keras) |
| 12 | Non-RGB uploads | RGBA: the alpha channel enters the colour variance; single-channel greyscale: the explanation step raises an OpenCV error. | verified |
| 13 | Upload copies | The upload is written to disk again at every Streamlit rerun. | source |
| 14 | Processing time | Measures the classification call only. | source |
| 15 | Occlusion sensitivity | Implemented, never called by the UI; last grid row and column are never visited. | source |

## How the checks were done

* **Model shapes, parameter counts, Grad-CAM layers, preprocessing ranges**: the architectures of `classification.py` were rebuilt with TensorFlow 2.20 / Keras 3.15 (`weights=None`) and inspected.
* **Training split and unfreezing**: `ImageDataGenerator` was run on a small dummy directory with the same arguments as `train.py`; `model.layers[0]` was inspected.
* **ABCDE behaviour**: `ExplainabilityEngine` was imported and run on every image in `media/`.
* **JavaScript port**: [`lab/verify/`](lab/verify/) decodes images in Python, computes reference values with OpenCV and the project's functions, and compares them with `lab/heuristics.js` in Node. The OpenCV greyscale formula was matched exhaustively over all $2^{24}$ RGB triples.
* **Callback markers**: the training figure replays `EarlyStopping` and `ReduceLROnPlateau` with the improvement rules read from the Keras 3 source.

## General limitations of the approach

* **Not a diagnostic device.** The system has no regulatory clearance and no prospective clinical validation. It is intended to support, not replace, a dermatologist.
* **Dataset shift.** HAM10000 is dermatoscopic and skewed towards lighter skin types. Clinical (non-dermoscopic) phone photos, other devices and darker skin types are out of distribution, and performance there is unknown.
* **Seven classes only.** Lesions outside these categories (for example squamous cell carcinoma other than Bowen's disease, or inflammatory conditions) will still be forced into one of the seven.
* **No calibration, no uncertainty.** Confidence values are raw softmax outputs.
* **Explanations are approximations.** Grad-CAM is coarse and layer-dependent; the ABCDE numbers are image-level heuristics as described in [chapter 09](09-abcde-heuristics.md).

## References

* Tschandl P., Rosendahl C., Kittler H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161.
* Tan M., Le Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
* Szegedy C., Vanhoucke V., Ioffe S., Shlens J., Wojna Z. (2016). Rethinking the Inception architecture for computer vision. *CVPR*.
* Szegedy C., Ioffe S., Vanhoucke V., Alemi A. (2017). Inception-v4, Inception-ResNet and the impact of residual connections on learning. *AAAI*.
* Radford A. et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
* Selvaraju R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
* Zeiler M. D., Fergus R. (2014). Visualizing and understanding convolutional networks. *ECCV*.
* Otsu N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics* 9(1), 62–66.
* Suzuki S., Abe K. (1985). Topological structural analysis of digitized binary images by border following. *Computer Vision, Graphics, and Image Processing* 30(1), 32–46.
* Provos N., Mazières D. (1999). A future-adaptable password scheme. *USENIX Annual Technical Conference*.

[← 10 Application, storage and security](10-application-storage-security.md) · [Docs index](README.md)
