# ABCDE Criteria Analysis - Implementation Guide

## Overview

The system now computes and reports complete **ABCDE criteria** for every analyzed lesion image. This provides standardized, quantitative metrics that align with clinical dermatology practice.

---

## What Are ABCDE Criteria?

The ABCDE criteria are the gold standard for melanoma detection:

```
A = Asymmetry          → Is one half different from the other?
B = Border             → Are borders irregular, notched, or blurred?
C = Color              → Multiple colors present?
D = Diameter           → Larger than 6mm (pencil eraser size)?
E = Evolution          → Has the lesion changed over time?
```

---

## Implementation Details

### **A - Asymmetry Score**

**Computation:**
```python
asymmetry_score = detect_asymmetry(img)
# Range: 0.0 (symmetric) to 1.0 (very asymmetric)
```

**Method:**
1. Find lesion contour using Otsu thresholding
2. Compute center of mass
3. Split image vertically through center
4. Flip right half horizontally
5. Compute mean pixel difference between halves
6. Normalize to 0-1 range

**Interpretation:**
- `< 0.3` → **Low asymmetry** (typical for benign moles)
- `≥ 0.3` → **High asymmetry** (concerning for melanoma)

**Example Output:**
```
• A - Asymmetry: 0.42 (High)
  → Significant left-right asymmetry detected
  → ABCDE criterion A - POSITIVE
```

---

### **B - Border Irregularity Score**

**Computation:**
```python
border_score = detect_border_irregularity(img)
# Range: 0.0 (circular) to 1.0 (very irregular)
```

**Method:**
1. Find lesion contour
2. Compute perimeter and area
3. Calculate circularity: `4π × area / perimeter²`
   - Perfect circle = 1.0
   - Irregular shape < 1.0
4. Convert to irregularity: `1.0 - circularity`

**Interpretation:**
- `< 0.4` → **Regular borders** (smooth, well-defined)
- `≥ 0.4` → **Irregular borders** (notched, jagged, blurred)

**Example Output:**
```
• B - Border: 0.58 (Irregular)
  → Borders are notched and irregular
  → ABCDE criterion B - POSITIVE
```

---

### **C - Color Variation**

**Computation:**
```python
color_variance = analyze_color_distribution(img)['color_variance']
color_score = min(color_variance / 2000.0, 1.0)
```

**Method:**
1. Compute variance across all RGB channels
2. Normalize to 0-1 scale
3. Classify as uniform or multiple colors

**Interpretation:**
- `< 1000` → **Uniform color** (single shade)
- `≥ 1000` → **Multiple colors** (2+ distinct colors)

**Additional Metrics:**
- Blue-white veil detection (melanoma-specific)
- Darkness ratio
- HSV color space analysis

**Example Output:**
```
• C - Color: Variance 1523 (Multiple colors)
  → Multiple distinct colors detected
  → ABCDE criterion C - POSITIVE

Additional Quantitative Details:
• Blue-white areas: 15.3%
  → Blue-white veil present (melanoma-specific sign)
```

---

### **D - Diameter**

**Computation:**
```python
diameter_mm = compute_diameter(img, pixels_per_mm=10.0)
# Estimated diameter in millimeters
```

**Method:**
1. Find lesion contour
2. Get bounding rectangle
3. Take maximum dimension (width or height)
4. Convert pixels to millimeters using calibration factor
   - Default: 10 pixels/mm (approximate for dermoscopic images)

**Interpretation:**
- `≤ 6mm` → **Small lesion** (low concern)
- `> 6mm` → **Large lesion** (increased concern)

**Note:** Diameter estimation assumes standard dermoscopic magnification. Results are approximate without proper calibration.

**Example Output:**
```
• D - Diameter: 8.2mm (>6mm)
  → Lesion exceeds 6mm threshold
  → ABCDE criterion D - POSITIVE
```

---

### **E - Evolution**

**Status:** Not computable from single image

**Explanation:**
Evolution requires temporal comparison (has the lesion changed?). Single-timepoint analysis cannot assess this criterion.

**Example Output:**
```
• E - Evolution: Not assessed (single timepoint)
  → Requires comparison with previous images
  → Dermatologists should inquire about changes
```

---

## Complete ABCDE Report Example

### Malignant Lesion (Melanoma)

```markdown
**ABCDE Criteria Analysis:**
• A - Asymmetry: 0.42 (High)
• B - Border: 0.58 (Irregular)
• C - Color: Variance 1523 (Multiple colors)
• D - Diameter: 8.2mm (>6mm)
• E - Evolution: Not assessed (single timepoint)

ABCDE Score: 4/5 criteria positive
→ High concern for melanoma
```

### Benign Lesion (Nevus)

```markdown
**ABCDE Criteria Analysis:**
• A - Asymmetry: 0.12 (Low)
• B - Border: 0.28 (Regular)
• C - Color: Variance 487 (Uniform)
• D - Diameter: 4.5mm (≤6mm)
• E - Evolution: Not assessed (single timepoint)

ABCDE Score: 0/5 criteria positive
→ Consistent with benign nevus
```

---

## Malignant Lesion Alerts

### UI Alert

When a potentially malignant lesion is detected (AKIEC, BCC, or Melanoma), a **prominent red alert** appears:

```
⚠️ ALERT: Potentially Malignant or Pre-Malignant Lesion Detected

This analysis suggests a lesion that may require immediate medical attention.
Please consult a qualified dermatologist as soon as possible for proper evaluation.

Remember: This is a decision support tool, not a diagnostic device.
Only a dermatologist can provide a definitive diagnosis.
```

### Explanation Alert

The explanation text also includes:

```markdown
⚠️ ALERT: Potentially Malignant Lesion Detected

**Confidence Level:** 85.3%

[... rest of explanation ...]
```

---

## Clinical Recommendations Updated

### Old Recommendations (Removed)

```
✗ Immediate dermatological consultation recommended
✗ Biopsy may be warranted for definitive diagnosis
```

### New Recommendations (Current)

```
✓ This tool must NOT be used for self-diagnosis
✓ Always consult a qualified dermatologist for proper evaluation
✓ Only dermatologists are responsible for diagnosis and treatment decisions
✓ This is a decision support tool, not a diagnostic device
```

**Rationale:** Aligns with the system's purpose as a **decision support tool for dermatologists**, not a direct-to-consumer diagnostic device.

---

## Integration Points

### 1. Explainability Engine

```python
# Compute all ABCDE criteria
abcde = ExplainabilityEngine.compute_abcde_criteria(img, color_metrics)

# Returns dictionary:
{
    'asymmetry': {'score': 0.42, 'status': 'High', 'criterion': 'A'},
    'border': {'score': 0.58, 'status': 'Irregular', 'criterion': 'B'},
    'color': {'score': 0.76, 'variance': 1523, 'status': 'Multiple colors', 'criterion': 'C'},
    'diameter': {'value_mm': 8.2, 'status': '>6mm', 'criterion': 'D'},
    'evolution': {'status': 'Not assessed (single timepoint)', 'criterion': 'E'}
}
```

### 2. Explanation Generation

```python
explanation = ExplainabilityEngine.generate_explanation(
    pred_index=4,           # Melanoma
    confidence_scores=scores_array,
    color_metrics=color_metrics,
    img=img_array          # Image for ABCDE computation
)
```

### 3. App Display

```python
# app.py automatically displays:
# 1. Malignant alert (if applicable)
# 2. Full ABCDE analysis in explanation
# 3. Updated clinical recommendations
```

---

## Benefits

### For Dermatologists

1. **Standardized Metrics**: ABCDE scores provide objective measurements
2. **Clinical Alignment**: Matches existing clinical practice
3. **Quantitative Support**: Numbers support visual assessment
4. **Transparency**: Clear what the AI analyzed

### For System Quality

1. **Reproducible**: Same image → same ABCDE scores
2. **Explainable**: Each metric has clear meaning
3. **Verifiable**: Dermatologists can validate scores
4. **Complete**: All standard criteria addressed

---

## Limitations

### Diameter Calibration

- Default assumes 10 pixels/mm
- Actual calibration varies by camera/magnification
- Results are **approximate estimates**
- Should not be used for precise measurement

**Recommendation:** Use diameter as relative indicator (large vs small), not absolute measurement.

### Evolution Assessment

- Cannot be computed from single image
- Requires temporal comparison
- Important criterion for melanoma detection
- **Dermatologists must assess clinically**

### Border Detection

- Depends on image quality
- May struggle with poorly defined borders
- Hair or artifacts can affect results

---

## Future Enhancements

1. **Calibration Tool**: Allow users to set pixels/mm ratio
2. **Temporal Comparison**: Store and compare previous images for E criterion
3. **Confidence Intervals**: Provide uncertainty estimates for each metric
4. **Additional Metrics**: Add other dermoscopic features (network, globules, etc.)

---

## Summary

The system now provides **comprehensive ABCDE analysis** for every lesion:

✅ **A** - Asymmetry computed and reported
✅ **B** - Border irregularity computed and reported
✅ **C** - Color variation computed and reported
✅ **D** - Diameter estimated and reported
✅ **E** - Evolution status noted (not computable)

Combined with:
- ⚠️ Malignant lesion alerts
- 📋 Updated clinical recommendations
- 🧠 Complete explainability reports

This makes the system a more valuable **decision support tool for dermatologists**.
