# 06 · Inference and ensemble

[← 05 Training](05-training.md) · [Docs index](README.md) · Next: [07 Grad-CAM →](07-grad-cam.md)

## Single model

`predict_single(img_path, model_name)` preprocesses the image for that backbone, runs `model.predict` and returns `argmax` and the 7 probabilities. `get_prediction_with_metadata` then attaches the labels from `CLASS_LABELS`:

```python
{
  'predicted_class': 4,
  'confidence_scores': {'AKIEC': ..., 'BCC': ..., ..., 'VASC': ...},
  'confidence': scores[4],
  'diagnosis': 'Melanoma',
  'lesion_type': 'Malignant',
  'model_used': 'EfficientNetB3',
  'ensemble_used': False,
  'processing_time': ...
}
```

## Ensemble: accuracy-weighted soft voting

With two or three models selected, `predict_ensemble` averages the probability vectors with weights proportional to the accuracies declared in `MODEL_CONFIGS` ($a$ = 0.85, 0.84, 0.79):

$$
\mathbf{p}_\text{ens} = \frac{\sum_m a_m \, \mathbf{p}_m}{\sum_m a_m}, \qquad \hat{y} = \arg\max_i \, p_{\text{ens},i}.
$$

<p align="center"><img src="assets/ensemble.svg" alt="Animated accuracy-weighted ensemble" width="100%"></p>

Properties worth knowing:

* **Why it can help.** If models make partly independent errors, averaging their probabilities reduces variance; soft voting also uses each model's confidence, unlike majority voting.
* **The weights are nearly uniform.** Normalised, they are 0.343, 0.339 and 0.319 for all three models, or 0.503 and 0.497 for the default pair, so the ensemble is very close to a plain mean. The accuracies are constants in the code; the repository contains no evaluation that reproduces them, so they should not be read as validated performance.
* **Robustness.** If one model raises an exception, it is skipped and the remaining ones are averaged; if all fail, a `RuntimeError` is raised.
* **Fewer than two models.** In "Ensemble" mode the UI warns when fewer than two models are selected; with one model the single-model path is used.

## What the result panel shows

| Element | Rule |
|---|---|
| Red alert banner | predicted index in {0, 1, 4} (AKIEC, BCC, MEL) |
| Diagnosis | `CLASS_LABELS[i]['full']` |
| Lesion Type with red or green dot | red when the string contains "Malignant" |
| Confidence | $100 \cdot p_{\hat y}$ |
| Bar chart | all 7 classes sorted by probability; the MEL bar is red, the others blue |
| Explanation "Differential Consideration" | second-highest class, shown only if its probability exceeds 0.10 |
| Explanation "certainty" | "high certainty" if confidence > 70 %, otherwise "moderate certainty" |

## Interpreting the probabilities

Softmax outputs of deep networks are often **over-confident** and are not calibrated probabilities of disease unless calibration has been measured and corrected (for example with temperature scaling on a held-out set). No calibration step exists in this project. The confidence value is best read as the model's relative preference among the seven classes for this image.

Build your own ensemble outputs and watch every rule above react in the [interactive lab](https://cocopops9.github.io/SkinLesionClassification/#inference).

[← 05 Training](05-training.md) · [Docs index](README.md) · Next: [07 Grad-CAM →](07-grad-cam.md)
