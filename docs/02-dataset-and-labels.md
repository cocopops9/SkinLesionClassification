# 02 · Labels and data

[← 01 System overview](01-system-overview.md) · [Docs index](README.md) · Next: [03 Input validation →](03-input-validation.md)

## The seven classes

Every model ends in `Dense(7, activation='softmax')`. The index of each output is fixed by `MelanomaClassifier.CLASS_LABELS`:

| Index | Code | Full name | `lesion_type` string | Red alert in UI | Counted as "Malignant finding" |
|---:|---|---|---|:---:|:---:|
| 0 | AKIEC | Actinic keratosis / Bowen's disease | `Pre-Malignant/Malignant` | yes | yes |
| 1 | BCC | Basal cell carcinoma | `Malignant` | yes | yes |
| 2 | BKL | Benign keratosis | `Benign` | no | no |
| 3 | DF | Dermatofibroma | `Benign` | no | no |
| 4 | MEL | Melanoma | `Malignant` | yes | yes |
| 5 | NV | Melanocytic nevus | `Benign` | no | no |
| 6 | VASC | Vascular lesion | `Malignant/Benign` | no | yes |

Two rules use these labels and they differ on purpose or by accident, so it is worth knowing both:

* the **alert banner** in `analysis_page` and the "ALERT" line in `generate_explanation` fire for indices `{0, 1, 4}`;
* the **"Malignant Findings"** counter on the History page and the red/green dot next to "Lesion Type" use the substring test `"Malignant" in lesion_type`, which is also true for VASC.

### Index consistency between training and inference

`train.py` reads images with `flow_from_directory`, which assigns class indices in **alphanumeric order of the sub-folder names**. With the folder names documented in `TRAINING_README.md` (`AKIEC, BCC, BKL, DF, MEL, NV, VASC`) this order equals the order of `CLASS_LABELS`, so a model trained by `train.py` and served by `classification.py` agrees on what index 4 means. Renaming a folder (for example `mel` instead of `MEL`) would silently break this mapping.

## HAM10000

The intended dataset is HAM10000 (Tschandl, Rosendahl and Kittler, 2018): 10,015 dermatoscopic images from several populations and acquisition devices, with diagnoses confirmed by histopathology, follow-up, expert consensus or confocal microscopy.

| Class | Images | Share |
|---|---:|---:|
| NV | 6705 | 66.9 % |
| MEL | 1113 | 11.1 % |
| BKL | 1099 | 11.0 % |
| BCC | 514 | 5.1 % |
| AKIEC | 327 | 3.3 % |
| VASC | 142 | 1.4 % |
| DF | 115 | 1.1 % |

A classifier that always answers NV is about 67 % accurate, so **accuracy alone is a weak metric** on this data; per-class recall (sensitivity), especially for MEL, is the clinically relevant quantity.

## Balanced class weights

With `USE_CLASS_WEIGHTS = True`, `train.py` calls `sklearn.utils.class_weight.compute_class_weight('balanced', ...)`:

$$
w_c = \frac{N}{K \, N_c}
$$

where $N$ is the number of training images, $K = 7$ and $N_c$ the images of class $c$. Keras multiplies each sample's loss by $w_{y_i}$, so every class contributes the same total weight $N/K$ to the loss. On the full HAM10000 counts this gives DF ≈ 12.4, VASC ≈ 10.1, AKIEC ≈ 4.4, BCC ≈ 2.8, BKL ≈ 1.30, MEL ≈ 1.29 and NV ≈ 0.21; the script computes the values from the actual training split.

Class weighting changes the optimisation target, not the data: rare classes still provide few distinct examples, which is why augmentation ([chapter 05](05-training.md)) is used alongside it.

## A note on data leakage

HAM10000 contains several images of the same lesion. A random image-level split can place images of one lesion in both training and validation, inflating validation scores. The repository does not include a split script; when preparing `data/train`, `data/val` and `data/test`, split by `lesion_id` (provided in the HAM10000 metadata) rather than by image.

[← 01 System overview](01-system-overview.md) · [Docs index](README.md) · Next: [03 Input validation →](03-input-validation.md)
