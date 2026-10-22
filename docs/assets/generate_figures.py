"""
Generate the numeric, animated SVG figures used by the README and the docs.

Hand-drawn schematic figures (hero, pipeline, validation, architecture,
augmentation) are edited directly. The figures produced here contain numbers,
so they are computed instead of drawn:

    training.svg   callback decisions replayed with the Keras rules used by train.py
    ensemble.svg   weighted average with the accuracies declared in MODEL_CONFIGS
    gradcam.svg    Grad-CAM arithmetic on a toy 7x7 example + the real overlay_heatmap
    occlusion.svg  the real compute_occlusion_sensitivity driven by a toy scorer
    abcde.svg      the real ABCDE functions applied to a synthetic lesion

Run from the repository root with the project requirements installed:

    python docs/assets/generate_figures.py
"""

import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from explainability import ExplainabilityEngine  # noqa: E402

# Accuracies declared in classification.MelanomaClassifier.MODEL_CONFIGS.
# They are duplicated here so that this script does not need gdown or the weights.
MODEL_ACCURACY = {'EfficientNetB3': 0.85, 'InceptionV3': 0.84, 'InceptionResNetV2': 0.79}
CLASS_NAMES = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
ALERT_CLASSES = {0, 1, 4}

FONT = '-apple-system, BlinkMacSystemFont, &quot;Segoe UI&quot;, Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

BASE_STYLE = f"""
    .t {{ font-family: {FONT}; fill: #e6edf3; }}
    .m {{ font-family: {MONO}; }}
    .mut {{ fill: #8b949e; }}
"""


# ---------------------------------------------------------------- helpers

def svg_document(width, height, title, desc, style, body):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{title}</title>
  <desc id="desc">{desc}</desc>
  <defs>
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M24 0H0V24" fill="none" stroke="#30363d" stroke-width="0.6" opacity="0.45"/>
    </pattern>
  </defs>
  <style>{BASE_STYLE}{style}
  </style>
  <rect width="{width}" height="{height}" rx="14" fill="#0d1117"/>
  <rect width="{width}" height="{height}" rx="14" fill="url(#grid)"/>
{body}
</svg>
"""


def blink(duration, appear, disappear=0.95):
    """SMIL opacity animation: hidden until `appear`, visible until `disappear` (fractions of the loop)."""
    a = min(max(appear, 0.0), 0.9)
    b = min(a + 0.02, disappear - 0.01)
    return (f'<animate attributeName="opacity" dur="{duration}s" repeatCount="indefinite" '
            f'keyTimes="0;{a:.4f};{b:.4f};{disappear};1" values="0;0;1;1;0"/>')


def data_uri(rgb, fmt='JPEG', quality=88):
    buf = io.BytesIO()
    image = Image.fromarray(rgb)
    if fmt == 'JPEG':
        image.save(buf, format='JPEG', quality=quality)
    else:
        image.save(buf, format='PNG', optimize=True)
    mime = 'image/jpeg' if fmt == 'JPEG' else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def jpeg_roundtrip(rgb, quality=88):
    """Return the pixels a viewer actually sees, so that metrics match the embedded image."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))


def colormap_rgb(name, value):
    r, g, b, _ = colormaps[name](float(np.clip(value, 0, 1)))
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'


def synthetic_lesion(size=240, seed=7):
    """Dark, slightly irregular pigmented lesion on lighter skin with sensor-like noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy, cx = size * 0.5, size * 0.47
    theta = np.arctan2(yy - cy, xx - cx)
    radius = np.hypot(yy - cy, xx - cx)
    boundary = size * 0.27 * (1 + 0.16 * np.sin(3 * theta + 0.6) + 0.07 * np.sin(7 * theta))
    inside = np.clip((boundary - radius) / 6.0, 0, 1)[..., None]

    skin = np.array([224, 182, 156], dtype=np.float64) - 18 * (radius / size)[..., None]
    lesion = np.array([96, 58, 42], dtype=np.float64) - 40 * np.clip(1 - radius / boundary, 0, 1)[..., None]
    img = skin * (1 - inside) + lesion * inside
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- training

def simulate_callbacks(val_losses, lr0, patience_lr=5, factor=0.5, min_lr=1e-7, min_delta_lr=1e-4, patience_es=10):
    """Replay ReduceLROnPlateau and EarlyStopping exactly as Keras 3 evaluates them on val_loss."""
    lr = lr0
    best_lr, wait_lr = None, 0
    best_es, wait_es, best_epoch = None, 0, 0
    events, lrs = [], []
    for epoch, loss in enumerate(val_losses):
        lrs.append(lr)
        wait_es += 1
        if best_es is None or loss < best_es:
            best_es, best_epoch, wait_es = loss, epoch, 0
        elif wait_es >= patience_es and epoch > 0:
            events.append(('stop', epoch))
            return events, lrs, best_epoch, epoch

        if best_lr is None or loss + min_delta_lr < best_lr:
            best_lr, wait_lr = loss, 0
        else:
            wait_lr += 1
            if wait_lr >= patience_lr and lr > min_lr:
                lr = max(lr * factor, min_lr)
                wait_lr = 0
                events.append(('reduce', epoch))
    return events, lrs, best_epoch, len(val_losses) - 1


def training_figure():
    phase1 = [1.62, 1.41, 1.27, 1.18, 1.12, 1.085, 1.07, 1.074, 1.071, 1.0702, 1.076, 1.072, 1.031, 1.012, 1.005]
    phase2 = [0.97, 0.91, 0.872, 0.851, 0.838, 0.8379, 0.842, 0.839, 0.8385, 0.841, 0.812, 0.801,
              0.805, 0.803, 0.809, 0.802, 0.806, 0.804, 0.807, 0.805, 0.808, 0.806, 0.81, 0.809,
              0.811, 0.812, 0.81, 0.813, 0.812, 0.814]
    ev1, lr1, best1, last1 = simulate_callbacks(phase1, 1e-4)
    ev2, lr2, best2, last2 = simulate_callbacks(phase2, 1e-5)

    losses = phase1[:last1 + 1] + phase2[:last2 + 1]
    lrs = lr1 + lr2[:last2 + 1]
    n1 = last1 + 1
    total_epochs = 45

    x0, x1 = 90, 1150
    def ex(epoch_index):
        return x0 + (x1 - x0) * epoch_index / (total_epochs - 1)
    y_top, y_bot = 150, 330
    lo, hi = 0.75, 1.7
    def ey(loss):
        return y_bot - (y_bot - y_top) * (loss - lo) / (hi - lo)
    lr_top, lr_bot = 372, 420
    log_hi, log_lo = np.log10(1e-4), np.log10(5e-6)
    def ly(lr):
        return lr_bot - (lr_bot - lr_top) * (np.log10(lr) - log_lo) / (log_hi - log_lo)

    duration = 12
    draw_end = 0.8
    def t_at(epoch_index):
        return draw_end * epoch_index / (total_epochs - 1)

    points = ' '.join(f'{ex(i):.1f},{ey(v):.1f}' for i, v in enumerate(losses))
    lr_path = []
    for i, lr in enumerate(lrs):
        x_start = ex(i - 0.5) if i else ex(0)
        lr_path.append(f'{"M" if i == 0 else "L"}{x_start:.1f},{ly(lr):.1f} H{ex(i + 0.5 if i < len(lrs) - 1 else i):.1f}')
    lr_d = ' '.join(lr_path)

    stop_epoch = n1 + last2
    body = []
    body.append('  <text x="30" y="40" class="t" font-size="20" font-weight="700">train.py: two-phase transfer learning</text>')
    body.append('  <text x="30" y="62" class="t mut" font-size="13">The val_loss values are illustrative. Every marker is the decision Keras 3 takes on them with the callbacks configured in train.py.</text>')

    # phase bands
    body.append(f'  <rect x="{ex(-0.5):.1f}" y="84" width="{ex(14.5) - ex(-0.5):.1f}" height="50" rx="8" fill="#13233a" stroke="#1f6feb"/>')
    body.append(f'  <text x="{ex(-0.3):.1f}" y="104" class="t" font-size="13" font-weight="700">Phase 1 · head only, up to 15 epochs</text>')
    body.append(f'  <text x="{ex(-0.3):.1f}" y="124" class="t m mut" font-size="11">backbone frozen · Adam 1e-4</text>')
    body.append(f'  <rect x="{ex(14.5):.1f}" y="84" width="{ex(44.5) - ex(14.5):.1f}" height="50" rx="8" fill="#1f1530" stroke="#8957e5"/>')
    body.append(f'  <text x="{ex(14.7):.1f}" y="104" class="t" font-size="13" font-weight="700">Phase 2 · fine-tune all layers, up to 30 epochs</text>')
    body.append(f'  <text x="{ex(14.7):.1f}" y="124" class="t m mut" font-size="11">recompiled · Adam 1e-5 · fresh callbacks</text>')

    # axes
    for v in (0.8, 1.0, 1.2, 1.4, 1.6):
        body.append(f'  <line x1="{x0}" y1="{ey(v):.1f}" x2="{x1}" y2="{ey(v):.1f}" stroke="#21262d"/>')
        body.append(f'  <text x="{x0 - 8}" y="{ey(v) + 4:.1f}" class="t m mut" font-size="10" text-anchor="end">{v:.1f}</text>')
    body.append(f'  <text x="{x0 - 50}" y="{(y_top + y_bot) / 2:.0f}" class="t mut" font-size="11" transform="rotate(-90 {x0 - 50} {(y_top + y_bot) / 2:.0f})" text-anchor="middle">val_loss</text>')
    for e in range(0, total_epochs, 5):
        body.append(f'  <text x="{ex(e):.1f}" y="{y_bot + 16}" class="t m mut" font-size="10" text-anchor="middle">{e + 1}</text>')
    body.append(f'  <text x="{x1}" y="{y_bot + 30}" class="t mut" font-size="10.5" text-anchor="end">global epoch</text>')

    # curve revealed by an expanding clip
    body.append('  <clipPath id="reveal"><rect x="0" y="0" width="0" height="460">'
                f'<animate attributeName="width" dur="{duration}s" repeatCount="indefinite" '
                f'keyTimes="0;{draw_end};0.95;1" values="{x0};{x1 + 10};{x1 + 10};{x0}"/></rect></clipPath>')
    body.append(f'  <polyline clip-path="url(#reveal)" points="{points}" fill="none" stroke="#39d0d8" stroke-width="2.5" stroke-linejoin="round"/>')

    # learning-rate strip
    body.append(f'  <text x="{x0 - 8}" y="{ly(1e-4) + 4:.1f}" class="t m mut" font-size="10" text-anchor="end">1e-4</text>')
    body.append(f'  <text x="{x0 - 8}" y="{ly(1e-5) + 4:.1f}" class="t m mut" font-size="10" text-anchor="end">1e-5</text>')
    body.append(f'  <text x="{x0 - 50}" y="{(lr_top + lr_bot) / 2:.0f}" class="t mut" font-size="11" transform="rotate(-90 {x0 - 50} {(lr_top + lr_bot) / 2:.0f})" text-anchor="middle">lr</text>')
    body.append(f'  <path clip-path="url(#reveal)" d="{lr_d}" fill="none" stroke="#d29922" stroke-width="2"/>')

    # events
    marks = []
    for kind, epoch in ev1:
        marks.append((kind, epoch))
    for kind, epoch in ev2:
        marks.append((kind, n1 + epoch))
    for k, (kind, g) in enumerate(marks):
        x = ex(g)
        if kind == 'reduce':
            label = 'ReduceLROnPlateau: lr × 0.5'
            colour = '#d29922'
        else:
            label = 'EarlyStopping'
            colour = '#f85149'
        label_y = y_top + 2 + 16 * (k % 3)
        body.append(f'  <g opacity="0">{blink(duration, t_at(g))}'
                    f'<line x1="{x:.1f}" y1="{y_top - 6}" x2="{x:.1f}" y2="{lr_bot}" stroke="{colour}" stroke-dasharray="4 4"/>'
                    f'<text x="{x - 6:.1f}" y="{label_y}" class="t" font-size="11.5" style="fill:{colour}" text-anchor="end">{label}</text></g>')
    for phase_best, offset, text_dy in ((best1, 0, -12), (best2, n1, 22)):
        g = phase_best + offset
        x, y = ex(g), ey(losses[g])
        body.append(f'  <g opacity="0">{blink(duration, t_at(stop_epoch if offset else n1 - 1))}'
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="none" stroke="#3fb950" stroke-width="2"/>'
                    f'<text x="{x:.1f}" y="{y + text_dy:.1f}" class="t" font-size="11" style="fill:#3fb950" text-anchor="middle">best val_loss: weights restored</text></g>')

    body.append(f'  <text x="30" y="452" class="t mut" font-size="12">ModelCheckpoint keeps the best val_accuracy per phase in checkpoints/. Phase 2 stopped at global epoch {stop_epoch + 1}; the remaining epochs are never run.</text>')

    svg = svg_document(1200, 470, 'Two-phase training with Keras callbacks',
                       'Animated validation-loss curve over two training phases with markers for learning-rate reductions, early stopping and restored best weights.',
                       '', '\n'.join(body))
    (OUT_DIR / 'training.svg').write_text(svg)
    return {'events_phase1': ev1, 'events_phase2': ev2, 'best1': best1, 'best2': best2, 'stop_global_epoch': stop_epoch + 1}


# ---------------------------------------------------------------- ensemble

def ensemble_figure():
    per_model = {
        'EfficientNetB3': np.array([0.02, 0.03, 0.10, 0.01, 0.58, 0.25, 0.01]),
        'InceptionV3': np.array([0.03, 0.05, 0.15, 0.01, 0.41, 0.34, 0.01]),
        'InceptionResNetV2': np.array([0.04, 0.06, 0.22, 0.02, 0.30, 0.35, 0.01]),
    }
    accuracies = np.array([MODEL_ACCURACY[m] for m in per_model])
    weights = accuracies / accuracies.sum()
    fused = sum(MODEL_ACCURACY[m] * p for m, p in per_model.items()) / accuracies.sum()

    duration = 10
    body = []
    body.append('  <text x="30" y="40" class="t" font-size="20" font-weight="700">MelanomaClassifier.predict_ensemble: accuracy-weighted soft voting</text>')
    body.append(f'  <text x="30" y="62" class="t mut" font-size="13">p = Σ a·p_m / Σ a with a = (0.85, 0.84, 0.79) from MODEL_CONFIGS, so w = ({weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}). Per-model outputs are illustrative.</text>')

    panels = list(per_model.items()) + [('Ensemble', fused)]
    panel_w, gap, top = 270, 20, 86
    for k, (name, probs) in enumerate(panels):
        px = 30 + k * (panel_w + gap)
        is_fused = name == 'Ensemble'
        stroke = '#3fb950' if is_fused else '#30363d'
        body.append(f'  <rect x="{px}" y="{top}" width="{panel_w}" height="300" rx="10" fill="#161b22" stroke="{stroke}"/>')
        title = name if is_fused else f'{name}'
        body.append(f'  <text x="{px + 16}" y="{top + 26}" class="t" font-size="14" font-weight="700">{title}</text>')
        subtitle = 'argmax = diagnosis' if is_fused else f'a = {MODEL_ACCURACY[name]:.2f} · w = {weights[k]:.3f}'
        body.append(f'  <text x="{px + 16}" y="{top + 46}" class="t m mut" font-size="11">{subtitle}</text>')
        start = 0.08 + 0.17 * k
        winner = int(np.argmax(probs))
        for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
            y = top + 70 + i * 31
            width = 150 * p / 0.6
            colour = '#f85149' if i in ALERT_CLASSES else '#58a6ff'
            body.append(f'  <text x="{px + 16}" y="{y + 12}" class="t m" font-size="11">{cls}</text>')
            body.append(f'  <g opacity="0">{blink(duration, start + 0.01 * i)}'
                        f'<rect x="{px + 66}" y="{y}" width="{width:.1f}" height="16" rx="3" fill="{colour}"/>'
                        f'<text x="{px + 72 + width:.1f}" y="{y + 12}" class="t m" font-size="11">{p:.3f}</text></g>')
            if i == winner:
                body.append(f'  <g opacity="0">{blink(duration, start + 0.09)}'
                            f'<rect x="{px + 8}" y="{y - 5}" width="{panel_w - 16}" height="26" rx="6" fill="none" stroke="#e6edf3" stroke-width="1.2"/></g>')
        if k < 3:
            body.append(f'  <text x="{px + panel_w + gap / 2:.0f}" y="{top + 160}" class="t mut" font-size="18" text-anchor="middle">{"+" if k < 2 else "="}</text>')

    verdict = CLASS_NAMES[int(np.argmax(fused))]
    body.append(f'  <text x="30" y="420" class="t" font-size="13">Individually, InceptionResNetV2 prefers NV; the weighted mean selects {verdict} ({fused.max():.3f}) and the second class {CLASS_NAMES[int(np.argsort(fused)[-2])]} ({np.sort(fused)[-2]:.3f}) is reported as "Differential Consideration" because it exceeds 0.10.</text>')
    body.append('  <text x="30" y="442" class="t mut" font-size="12">Because the three weights differ by less than 0.025, the ensemble is close to a plain arithmetic mean. The UI default ensemble is EfficientNetB3 + InceptionV3 (w = 0.503, 0.497).</text>')

    svg = svg_document(1200, 460, 'Accuracy-weighted ensemble',
                       'Three model probability vectors are combined into the ensemble vector using weights proportional to the declared accuracies.',
                       '', '\n'.join(body))
    (OUT_DIR / 'ensemble.svg').write_text(svg)
    return {'weights': weights.round(4).tolist(), 'fused': fused.round(4).tolist()}


# ---------------------------------------------------------------- Grad-CAM

def gradcam_figure():
    size = 7
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    centre = np.exp(-((yy - 3.2) ** 2 + (xx - 3.0) ** 2) / 3.0)
    ring = np.exp(-((np.hypot(yy - 3.2, xx - 3.0) - 2.2) ** 2) / 0.6)
    background = np.clip(np.hypot(yy - 3.2, xx - 3.0) / 4.0, 0, 1)
    corner = np.zeros((size, size))
    corner[0:2, 5:7] = 1.0
    activations = np.stack([centre, ring, background, corner], axis=-1)

    rng = np.random.default_rng(3)
    mean_gradient = np.array([0.9, 0.5, -0.6, 0.05])
    gradients = mean_gradient + rng.normal(0, 0.15, (size, size, 4))
    alphas = gradients.mean(axis=(0, 1))

    weighted = activations @ alphas
    heatmap = np.maximum(weighted, 0) / np.max(weighted)

    image = jpeg_roundtrip(synthetic_lesion(224))
    overlay = ExplainabilityEngine.overlay_heatmap(image.copy(), heatmap.astype(np.float32))

    duration = 12
    stage_times = [0.04, 0.16, 0.28, 0.42, 0.56, 0.7]

    def grid(x, y, values, cell, cmap, vmin, vmax):
        parts = []
        for r in range(size):
            for c in range(size):
                v = (values[r, c] - vmin) / (vmax - vmin + 1e-12)
                parts.append(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell - 1}" height="{cell - 1}" fill="{colormap_rgb(cmap, v)}"/>')
        return ''.join(parts)

    body = []
    body.append('  <text x="30" y="40" class="t" font-size="20" font-weight="700">ExplainabilityEngine.compute_gradcam, step by step</text>')
    body.append('  <text x="30" y="62" class="t mut" font-size="13">Toy numbers on a 7×7 grid (the EfficientNetB3 top_conv resolution) with 4 of the 1536 channels; the overlay is produced by the real overlay_heatmap.</text>')

    cell = 15
    labels = ['A¹ lesion centre', 'A² lesion border', 'A³ background skin', 'A⁴ corner artefact']
    for k in range(4):
        x, y = 40, 92 + k * 118
        body.append(f'  <g opacity="0">{blink(duration, stage_times[0] + 0.02 * k)}'
                    f'<text x="{x}" y="{y - 4}" class="t" font-size="11">{labels[k]}</text>'
                    f'{grid(x, y, activations[..., k], cell, "viridis", 0, 1)}</g>')
        gx = 180
        body.append(f'  <g opacity="0">{blink(duration, stage_times[1] + 0.02 * k)}'
                    f'<text x="{gx}" y="{y - 4}" class="t" font-size="11">∂y/∂A{"¹²³⁴"[k]}</text>'
                    f'{grid(gx, y, gradients[..., k], cell, "coolwarm", -1.2, 1.2)}</g>')
        colour = '#f85149' if alphas[k] > 0 else '#58a6ff'
        body.append(f'  <g opacity="0">{blink(duration, stage_times[2] + 0.02 * k)}'
                    f'<text x="320" y="{y + 50}" class="t m" font-size="13" style="fill:{colour}">α{"¹²³⁴"[k]} = {alphas[k]:+.3f}</text></g>')

    body.append('  <text x="40" y="574" class="t m mut" font-size="11">tape.gradient(loss, conv_outputs)</text>')
    body.append('  <text x="300" y="574" class="t m mut" font-size="11">reduce_mean(grads, axis=(0,1,2))</text>')

    sx = 470
    vmax = np.abs(weighted).max()
    body.append(f'  <g opacity="0">{blink(duration, stage_times[3])}'
                f'<text x="{sx}" y="146" class="t" font-size="12">Σₖ αₖ Aᵏ (signed)</text>'
                f'{grid(sx, 152, weighted, 22, "coolwarm", -vmax, vmax)}'
                f'<text x="{sx}" y="324" class="t m mut" font-size="11">conv_outputs @ pooled_grads</text></g>')
    hx = 660
    body.append(f'  <g opacity="0">{blink(duration, stage_times[4])}'
                f'<text x="{hx}" y="146" class="t" font-size="12">ReLU, then ÷ max</text>'
                f'{grid(hx, 152, heatmap, 22, "jet", 0, 1)}'
                f'<text x="{hx}" y="324" class="t m mut" font-size="11">maximum(h, 0) / reduce_max(h)</text></g>')

    ox = 860
    body.append(f'  <g opacity="0">{blink(duration, stage_times[5])}'
                f'<text x="{ox}" y="100" class="t" font-size="12">resize + COLORMAP_JET, 0.4 · heat + 0.6 · image</text>'
                f'<image x="{ox}" y="110" width="150" height="150" href="{data_uri(image)}"/>'
                f'<image x="{ox + 160}" y="110" width="150" height="150" href="{data_uri(overlay)}"/>'
                f'<text x="{ox}" y="278" class="t m mut" font-size="11">original</text>'
                f'<text x="{ox + 160}" y="278" class="t m mut" font-size="11">overlay_heatmap(img, h)</text></g>')

    body.append('  <g class="t" font-size="12.5">'
                '<text x="470" y="390">• A positive αₖ means "more of channel k raises the class score"; A³ gets a negative α, so skin is suppressed.</text>'
                '<text x="470" y="414">• ReLU keeps only evidence for the class; dividing by the maximum maps the result to [0, 1].</text>'
                '<text x="470" y="438">• The app computes the map on the first selected model only, for the class chosen</text>'
                '<text x="482" y="456">by the (possibly ensembled) prediction, and stores it as heatmaps/&lt;id&gt;_heatmap.npy.</text>'
                '<text x="470" y="480" class="mut">• The divisor is the maximum of the map before ReLU and has no epsilon: if it is 0 the map becomes NaN,</text>'
                '<text x="482" y="498" class="mut">if it is negative the map is all zeros.</text></g>')

    svg = svg_document(1200, 596, 'Grad-CAM step by step',
                       'Feature maps and their gradients are pooled into channel weights, combined, rectified, normalised and overlaid on the image with the JET colormap.',
                       '', '\n'.join(body))
    (OUT_DIR / 'gradcam.svg').write_text(svg)
    return {'alphas': alphas.round(4).tolist()}


# ---------------------------------------------------------------- occlusion

class ToyScorer:
    """Stand-in for a Keras model: class score = template-weighted mean of max(R - B, 0)."""

    def __init__(self, size):
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        self.template = np.exp(-((yy - size * 0.5) ** 2 + (xx - size * 0.47) ** 2) / (2 * (size * 0.2) ** 2))
        self.template /= self.template.sum()

    def predict(self, batch, verbose=0):
        img = batch[0]
        brownness = np.clip(img[..., 0] - img[..., 2], 0, None)
        score = float((self.template * brownness).sum())
        return np.array([[1 - score, score]])


def occlusion_figure():
    size, patch, stride = 224, 32, 16
    image = jpeg_roundtrip(synthetic_lesion(size))
    model = ToyScorer(size)
    batch = (image.astype(np.float64) / 255.0)[None]

    sensitivity = ExplainabilityEngine.compute_occlusion_sensitivity(model, batch, pred_index=1, patch_size=patch, stride=stride)

    # Raw grid exactly as filled inside compute_occlusion_sensitivity, for display.
    baseline = model.predict(batch)[0, 1]
    raw = np.zeros((size // stride, size // stride))
    positions = []
    for i in range(0, size - patch + 1, stride):
        for j in range(0, size - patch + 1, stride):
            occluded = batch.copy()
            occluded[0, i:i + patch, j:j + patch, :] = 0
            raw[i // stride, j // stride] = baseline - model.predict(occluded)[0, 1]
            positions.append((i, j))

    duration = 14
    scan_end = 0.75
    img_x, img_y, scale = 40, 100, 1.5
    disp = int(size * scale)

    body = []
    body.append('  <text x="30" y="40" class="t" font-size="20" font-weight="700">compute_occlusion_sensitivity: slide a black patch, measure the score drop</text>')
    body.append('  <text x="30" y="62" class="t mut" font-size="13">Real function, toy scorer (template-weighted brownness) in place of the CNN. Implemented in explainability.py but not called by the Streamlit app.</text>')
    body.append(f'  <image x="{img_x}" y="{img_y}" width="{disp}" height="{disp}" href="{data_uri(image)}"/>')

    xs = ';'.join(f'{img_x + j * scale:.1f}' for _, j in positions)
    ys = ';'.join(f'{img_y + i * scale:.1f}' for i, _ in positions)
    count = len(positions)
    key_times = ';'.join(f'{scan_end * k / count:.4f}' for k in range(count))
    body.append(f'  <rect width="{patch * scale}" height="{patch * scale}" fill="#000" stroke="#39d0d8" stroke-width="1.5" opacity="0.9">'
                f'<animate attributeName="x" dur="{duration}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{key_times}" values="{xs}"/>'
                f'<animate attributeName="y" dur="{duration}s" repeatCount="indefinite" calcMode="discrete" keyTimes="{key_times}" values="{ys}"/>'
                f'{blink(duration, 0.0, scan_end + 0.01)}</rect>')
    body.append(f'  <text x="{img_x}" y="{img_y + disp + 22}" class="t m mut" font-size="11">patch 32 px · stride 16 · {count} + 1 passes</text>')

    grid_x, grid_y, cell = 450, 100, 22
    vmax = raw.max()
    body.append(f'  <text x="{grid_x}" y="{grid_y - 10}" class="t" font-size="12">raw map: baseline − occluded score ({raw.shape[0]}×{raw.shape[1]})</text>')
    for k, (i, j) in enumerate(positions):
        r, c = i // stride, j // stride
        colour = colormap_rgb('hot', raw[r, c] / vmax)
        body.append(f'  <rect x="{grid_x + c * cell}" y="{grid_y + r * cell}" width="{cell - 1}" height="{cell - 1}" fill="{colour}" opacity="0">'
                    f'{blink(duration, scan_end * k / count)}</rect>')
    last = raw.shape[0] - 1
    for t in range(raw.shape[0]):
        body.append(f'  <rect x="{grid_x + last * cell}" y="{grid_y + t * cell}" width="{cell - 1}" height="{cell - 1}" fill="none" stroke="#d29922" stroke-dasharray="3 2"/>')
        if t < last:
            body.append(f'  <rect x="{grid_x + t * cell}" y="{grid_y + last * cell}" width="{cell - 1}" height="{cell - 1}" fill="none" stroke="#d29922" stroke-dasharray="3 2"/>')
    body.append(f'  <text x="{grid_x}" y="{grid_y + raw.shape[0] * cell + 20}" class="t" font-size="11" style="fill:#d29922">dashed: last row/column is never visited and stays 0</text>')

    final = np.uint8(255 * np.array([[colormaps["hot"](v)[:3] for v in row] for row in sensitivity]))
    fx = 800
    body.append(f'  <g opacity="0">{blink(duration, scan_end + 0.02)}'
                f'<text x="{fx}" y="{grid_y - 10}" class="t" font-size="12">cv2.resize to 224×224, then min-max normalise</text>'
                f'<image x="{fx}" y="{grid_y}" width="{disp}" height="{disp}" href="{data_uri(final)}"/></g>')
    body.append(f'  <text x="450" y="{grid_y + raw.shape[0] * cell + 44}" class="t mut" font-size="11.5">At 299 px the same loop makes 17×17 = 289 occluded passes. A patch value of 0 is black for EfficientNetB3 inputs but</text>')
    body.append(f'  <text x="450" y="{grid_y + raw.shape[0] * cell + 62}" class="t mut" font-size="11.5">mid-grey for the Inception models, whose preprocessed range is centred on 0.</text>')

    svg = svg_document(1200, 520, 'Occlusion sensitivity',
                       'A black patch scans the image in raster order while the corresponding cell of the sensitivity grid fills in; the grid is then resized and normalised.',
                       '', '\n'.join(body))
    (OUT_DIR / 'occlusion.svg').write_text(svg)
    return {'positions': count, 'grid': raw.shape}


# ---------------------------------------------------------------- ABCDE

def abcde_figure():
    image = jpeg_roundtrip(synthetic_lesion(240, seed=11))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])

    metrics = ExplainabilityEngine.analyze_color_distribution(image)
    abcde = ExplainabilityEngine.compute_abcde_criteria(image, metrics)

    duration = 14
    size = 240
    panel_y = 110
    body = []
    body.append('  <text x="30" y="40" class="t" font-size="20" font-weight="700">compute_abcde_criteria on a synthetic dark lesion</text>')
    body.append('  <text x="30" y="62" class="t mut" font-size="13">All numbers below come from the real functions in explainability.py applied to the image on the left.</text>')

    body.append(f'  <text x="40" y="{panel_y - 10}" class="t" font-size="12">1 · RGB input</text>')
    body.append(f'  <image x="40" y="{panel_y}" width="{size}" height="{size}" href="{data_uri(image)}"/>')

    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    hist /= hist.max()
    hx, hy, hw, hh = 310, panel_y + size, 256, size
    bars = ''.join(f'<rect x="{hx + i}" y="{hy - hh * v:.1f}" width="1" height="{hh * v:.1f}" fill="{"#e6edf3" if i > threshold else "#6e7681"}"/>' for i, v in enumerate(hist) if v > 0)
    body.append(f'  <g opacity="0">{blink(duration, 0.08)}<text x="{hx}" y="{panel_y - 10}" class="t" font-size="12">2 · grey histogram, Otsu T = {threshold:.0f}</text>{bars}'
                f'<line x1="{hx + threshold}" y1="{panel_y}" x2="{hx + threshold}" y2="{hy}" stroke="#f85149" stroke-width="2"/>'
                f'<text x="{hx + threshold + 6}" y="{panel_y + 16}" class="t m" font-size="11" style="fill:#f85149">gray &gt; T → 255</text>'
                f'<text x="{hx}" y="{hy + 16}" class="t m mut" font-size="10">0</text><text x="{hx + 256}" y="{hy + 16}" class="t m mut" font-size="10" text-anchor="end">255</text></g>')

    mask_rgb = np.stack([binary] * 3, axis=-1)
    mx = 600
    path = 'M' + ' L'.join(f'{mx + p[0][0]},{panel_y + p[0][1]}' for p in contour) + ' Z'
    body.append(f'  <g opacity="0">{blink(duration, 0.2)}<text x="{mx}" y="{panel_y - 10}" class="t" font-size="12">3 · mask (white = gray &gt; T), contour</text>'
                f'<image x="{mx}" y="{panel_y}" width="{size}" height="{size}" href="{data_uri(mask_rgb, fmt="PNG")}"/></g>')
    body.append(f'  <path d="{path}" fill="none" stroke="#39d0d8" stroke-width="3" pathLength="1" stroke-dasharray="1" stroke-dashoffset="1">'
                f'<animate attributeName="stroke-dashoffset" dur="{duration}s" repeatCount="indefinite" keyTimes="0;0.3;0.45;0.95;1" values="1;1;0;0;1"/></path>')
    body.append(f'  <g opacity="0">{blink(duration, 0.47)}'
                f'<rect x="{mx + x}" y="{panel_y + y}" width="{w}" height="{h}" fill="none" stroke="#d29922" stroke-dasharray="5 4" stroke-width="1.5"/>'
                f'<line x1="{mx + cx}" y1="{panel_y}" x2="{mx + cx}" y2="{panel_y + size}" stroke="#bc8cff" stroke-width="1.5"/>'
                f'<text x="{mx + cx + 4}" y="{panel_y + size - 6}" class="t m" font-size="10" style="fill:#bc8cff">cx = {cx}</text></g>')

    fg_share = (binary > 0).mean()
    rx = 870
    lines = [
        ('A - Asymmetry', f"{abcde['asymmetry']['score']:.2f} ({abcde['asymmetry']['status']})", 'mean |col j − col W−1−j| / 255, j &lt; min(cx, W−cx)'),
        ('B - Border', f"{abcde['border']['score']:.2f} ({abcde['border']['status']})", '1 − 4π·area / perimeter²'),
        ('C - Color', f"Variance {abcde['color']['variance']:.0f} ({abcde['color']['status']})", 'np.var over all R, G, B values'),
        ('D - Diameter', f"{abcde['diameter']['value_mm']:.1f}mm ({abcde['diameter']['status']})", f'max(w, h) = {max(w, h)} px ÷ 10 px/mm'),
        ('E - Evolution', 'Not assessed', 'single time point'),
    ]
    body.append(f'  <g opacity="0">{blink(duration, 0.52)}<text x="{rx}" y="{panel_y - 10}" class="t" font-size="12">4 · text produced for the report</text>')
    for k, (name, value, how) in enumerate(lines):
        yy = panel_y + 12 + k * 46
        body.append(f'<text x="{rx}" y="{yy}" class="t" font-size="12.5"><tspan font-weight="700">{name}:</tspan> {value}</text>'
                    f'<text x="{rx}" y="{yy + 17}" class="t m mut" font-size="10">{how}</text>')
    body.append('</g>')

    note_y = panel_y + size + 50
    body.append(f'  <g opacity="0">{blink(duration, 0.6)}'
                f'<rect x="30" y="{note_y - 22}" width="1140" height="74" rx="10" fill="#1c1a12" stroke="#5c4a12"/>'
                f'<text x="46" y="{note_y}" class="t" font-size="13" style="fill:#e3b341">THRESH_BINARY marks pixels brighter than T as foreground. Here the foreground is the skin ({100 * fg_share:.0f}% of the image),</text>'
                f'<text x="46" y="{note_y + 20}" class="t" font-size="13" style="fill:#e3b341">so the largest external contour runs along the image frame and D = {max(w, h)} px ÷ 10. On all ten images in media/ the measured contour also touches the border.</text>'
                f'<text x="46" y="{note_y + 40}" class="t mut" font-size="12">The scores are therefore image-level heuristics, not measurements of the lesion. See docs/09-abcde-heuristics.md.</text></g>')

    svg = svg_document(1200, 470, 'ABCDE heuristics',
                       'A synthetic lesion is converted to grey, thresholded with Otsu, its largest contour is traced and the four computable ABCDE scores are reported.',
                       '', '\n'.join(body))
    (OUT_DIR / 'abcde.svg').write_text(svg)
    return {'otsu': float(threshold), 'foreground_share': round(float(fg_share), 3),
            'A': round(abcde['asymmetry']['score'], 3), 'B': round(abcde['border']['score'], 3),
            'C': round(float(abcde['color']['variance']), 1), 'D': abcde['diameter']['value_mm']}


if __name__ == '__main__':
    print('training ', training_figure())
    print('ensemble ', ensemble_figure())
    print('gradcam  ', gradcam_figure())
    print('occlusion', occlusion_figure())
    print('abcde    ', abcde_figure())
