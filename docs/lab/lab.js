/*
 * Interactive widgets for the Skin Lesion Lab.
 *
 * Each section mirrors one part of the Python code. Constants are copied from
 * the source files named in the comments; when the browser cannot run a
 * component (CNNs, CLIP) the widget works on explicit user-provided inputs.
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ constants

  // classification.py · MelanomaClassifier.CLASS_LABELS
  const CLASSES = [
    { code: 'AKIEC', full: "Actinic keratosis / Bowen's disease", type: 'Pre-Malignant/Malignant' },
    { code: 'BCC', full: 'Basal cell carcinoma', type: 'Malignant' },
    { code: 'BKL', full: 'Benign keratosis', type: 'Benign' },
    { code: 'DF', full: 'Dermatofibroma', type: 'Benign' },
    { code: 'MEL', full: 'Melanoma', type: 'Malignant' },
    { code: 'NV', full: 'Melanocytic nevus', type: 'Benign' },
    { code: 'VASC', full: 'Vascular lesion', type: 'Malignant/Benign' }
  ];
  const ALERT_INDICES = new Set([0, 1, 4]);

  // explainability.py · generate_explanation · diagnosis_names
  const EXPLANATION_NAMES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma',
    'Melanoma', 'Melanocytic Nevus', 'Vascular Lesion'];

  // explainability.py · ExplainabilityEngine.DERMOSCOPIC_FEATURES (fields used by the report)
  const FEATURES = [
    { key: ['Rough, scaly patches', 'Pink or red base', 'Adherent scale'], signs: ['Hyperkeratosis', 'Parakeratosis'], dermoscopy: 'Scale' },
    { key: ['Arborizing vessels', 'Blue-gray ovoid nests', 'Ulceration'], signs: ['Telangiectasia', 'Rolled borders'], dermoscopy: 'Maple leaf areas' },
    { key: ['Milia-like cysts', 'Comedo-like openings', 'Fissures'], signs: ['Well-demarcated borders', 'Stuck-on appearance'], dermoscopy: 'Fingerprint patterns' },
    { key: ['Central white patch', 'Peripheral pigment network', 'Dimple sign'], signs: ['Firm nodule', 'Positive dimple sign'], dermoscopy: 'Central white scar-like patch' },
    { key: ['Asymmetry', 'Irregular borders', 'Multiple colors'], signs: ['ABCDE criteria positive', 'Evolution'], dermoscopy: 'Atypical network' },
    { key: ['Symmetric pattern', 'Regular network', 'Uniform color'], signs: ['Well-defined borders', 'Stable over time'], dermoscopy: 'Typical network' },
    { key: ['Red-purple color', 'Lacunae', 'Red-blue areas'], signs: ['Vascular proliferation', 'Soft consistency'], dermoscopy: 'Red lacunae' }
  ];

  // classification.py · MODEL_CONFIGS
  const MODELS = {
    EfficientNetB3: { accuracy: 0.85, size: 224 },
    InceptionV3: { accuracy: 0.84, size: 299 },
    InceptionResNetV2: { accuracy: 0.79, size: 299 }
  };

  // Tschandl et al. (2018), HAM10000 class counts
  const HAM_COUNTS = [327, 514, 1099, 115, 1113, 6705, 142];

  // ------------------------------------------------------------------ helpers

  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const fmt = (v, d) => Number(v).toFixed(d);

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function bindRange(id, format, onChange) {
    const input = $(id), out = $(id + 'o');
    const update = () => {
      if (out) out.textContent = format(parseFloat(input.value));
      onChange(parseFloat(input.value));
    };
    input.addEventListener('input', update);
    return update;
  }

  function bindSeg(container, attr, onChange) {
    container.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === btn));
        onChange(btn.dataset[attr], btn);
      });
    });
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  }

  /* Colormaps returning [r, g, b] in 0..255 */
  function jet(v) {
    v = clamp(v, 0, 1);
    const r = clamp(1.5 - Math.abs(4 * v - 3), 0, 1);
    const g = clamp(1.5 - Math.abs(4 * v - 2), 0, 1);
    const b = clamp(1.5 - Math.abs(4 * v - 1), 0, 1);
    return [r * 255, g * 255, b * 255];
  }

  function hot(v) {
    v = clamp(v, 0, 1);
    return [clamp(v / 0.365, 0, 1) * 255, clamp((v - 0.365) / 0.381, 0, 1) * 255, clamp((v - 0.746) / 0.254, 0, 1) * 255];
  }

  function interpolateStops(stops, v) {
    v = clamp(v, 0, 1) * (stops.length - 1);
    const i = Math.min(Math.floor(v), stops.length - 2), f = v - i;
    return stops[i].map((c, k) => c + (stops[i + 1][k] - c) * f);
  }
  const viridis = (v) => interpolateStops([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], v);
  const coolwarm = (v) => interpolateStops([[59, 76, 192], [141, 176, 254], [221, 221, 221], [244, 154, 123], [180, 4, 38]], v);

  /* Paint a 2-D array of values into a canvas whose size equals the array size */
  function paintField(canvas, values, w, h, cmap) {
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const [r, g, b] = cmap(values[i]);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  /* RGB image {width, height, data(RGB)} into a canvas of the same size */
  function paintRgb(canvas, img) {
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    const out = ctx.createImageData(img.width, img.height);
    const c = img.data.length / (img.width * img.height);
    for (let p = 0, q = 0; p < img.width * img.height; p++, q += c) {
      out.data[p * 4] = img.data[q];
      out.data[p * 4 + 1] = img.data[q + 1];
      out.data[p * 4 + 2] = img.data[q + 2];
      out.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
  }

  /* Float bilinear resize with OpenCV's half-pixel mapping (cv2.resize on float maps) */
  function resizeFloat(src, sw, sh, dw, dh) {
    const out = new Float64Array(dw * dh);
    for (let y = 0; y < dh; y++) {
      let fy = (y + 0.5) * (sh / dh) - 0.5;
      let y0 = Math.floor(fy);
      fy -= y0;
      if (y0 < 0) { y0 = 0; fy = 0; }
      if (y0 >= sh - 1) { y0 = sh - 1; fy = 0; }
      const y1 = Math.min(y0 + 1, sh - 1);
      for (let x = 0; x < dw; x++) {
        let fx = (x + 0.5) * (sw / dw) - 0.5;
        let x0 = Math.floor(fx);
        fx -= x0;
        if (x0 < 0) { x0 = 0; fx = 0; }
        if (x0 >= sw - 1) { x0 = sw - 1; fx = 0; }
        const x1 = Math.min(x0 + 1, sw - 1);
        const top = src[y0 * sw + x0] * (1 - fx) + src[y0 * sw + x1] * fx;
        const bottom = src[y1 * sw + x0] * (1 - fx) + src[y1 * sw + x1] * fx;
        out[y * dw + x] = top * (1 - fy) + bottom * fy;
      }
    }
    return out;
  }

  /*
   * Synthetic dermoscopy-like image: a dark, irregular lesion on lighter skin.
   * Same construction as synthetic_lesion() in docs/assets/generate_figures.py.
   */
  function syntheticLesion(opts) {
    const o = Object.assign({ width: 240, height: 240, radius: 0.27, irregularity: 0.16, darkness: 0.7, skin: 0.85, offset: 0, seed: 7, noise: 5 }, opts);
    const rand = mulberry32(o.seed);
    const w = o.width, h = o.height, size = Math.min(w, h);
    const data = new Uint8Array(w * h * 3);
    const cy = h * 0.5, cx = w * (0.47 + o.offset);
    const skinBase = [224 * o.skin + 30 * (1 - o.skin), 182 * o.skin + 20 * (1 - o.skin), 156 * o.skin + 15 * (1 - o.skin)];
    const lesionBase = [96 + (1 - o.darkness) * 90, 58 + (1 - o.darkness) * 80, 42 + (1 - o.darkness) * 70];
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const theta = Math.atan2(y - cy, x - cx);
        const r = Math.hypot(y - cy, x - cx);
        const boundary = size * o.radius * (1 + o.irregularity * Math.sin(3 * theta + 0.6) + o.irregularity * 0.45 * Math.sin(7 * theta));
        const inside = clamp((boundary - r) / 6, 0, 1);
        const core = clamp(1 - r / Math.max(boundary, 1), 0, 1);
        for (let k = 0; k < 3; k++) {
          const skin = skinBase[k] - 18 * (r / size);
          const lesion = lesionBase[k] - 40 * o.darkness * core;
          const v = skin * (1 - inside) + lesion * inside + o.noise * gaussian(rand);
          data[(y * w + x) * 3 + k] = clamp(Math.round(v), 0, 255);
        }
      }
    }
    return { width: w, height: h, data };
  }

  /* Decode a user file into RGB pixels at native resolution (no EXIF rotation, like PIL) */
  async function decodeFile(file) {
    let bitmap;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: 'none' });
    } catch (err) {
      bitmap = await createImageBitmap(file);
    }
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const rgb = new Uint8Array(bitmap.width * bitmap.height * 3);
    for (let p = 0, q = 0; q < rgba.length; p += 3, q += 4) {
      rgb[p] = rgba[q];
      rgb[p + 1] = rgba[q + 1];
      rgb[p + 2] = rgba[q + 2];
    }
    return { width: bitmap.width, height: bitmap.height, data: rgb };
  }

  /* Draw an image scaled into a fixed-size canvas */
  function drawScaled(canvas, img) {
    const tmp = document.createElement('canvas');
    paintRgb(tmp, img);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const s = Math.min(canvas.width / img.width, canvas.height / img.height);
    const dw = img.width * s, dh = img.height * s;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(tmp, (canvas.width - dw) / 2, (canvas.height - dh) / 2, dw, dh);
    return { scale: s, ox: (canvas.width - dw) / 2, oy: (canvas.height - dh) / 2 };
  }

  // Shared state used to assemble the database record and the report text.
  const shared = { prediction: null, abcde: null, imageSize: [240, 240] };

  // ------------------------------------------------------------------ 00 page chrome

  function initChrome() {
    const root = document.documentElement;
    const toggle = $('themeToggle');
    try {
      const saved = localStorage.getItem('slab-theme');
      if (saved) root.dataset.theme = saved;
    } catch (err) { /* storage unavailable: keep the system theme */ }
    toggle.addEventListener('click', () => {
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('slab-theme', root.dataset.theme); } catch (err) { /* ignore */ }
    });

    const links = Array.from(document.querySelectorAll('#rail a'));
    const sections = links.map((a) => document.querySelector(a.getAttribute('href')));
    const progress = $('progress');
    const onScroll = () => {
      const y = window.scrollY + 120;
      let current = 0;
      sections.forEach((s, i) => { if (s && s.offsetTop <= y) current = i; });
      links.forEach((a, i) => {
        a.classList.toggle('active', i === current);
        a.classList.toggle('done', i < current);
      });
      const max = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.width = (max > 0 ? (100 * window.scrollY) / max : 0) + '%';
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    const reveals = document.querySelectorAll('.window, figure, .callout, .card');
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
      }, { rootMargin: '0px 0px -40px 0px' });
      reveals.forEach((el) => { el.classList.add('reveal'); io.observe(el); });
    }
  }

  // ------------------------------------------------------------------ 01 labels

  function initLabels() {
    const rows = CLASSES.map((c, i) => {
      const alert = ALERT_INDICES.has(i) ? '<span class="pill bad">alert</span>' : '<span class="pill">no</span>';
      const counted = c.type.includes('Malignant') ? '<span class="pill warn">yes</span>' : '<span class="pill">no</span>';
      return `<tr><td class="num">${i}</td><td class="mono">${c.code}</td><td>${escapeHtml(c.full)}</td><td class="mono">${c.type}</td><td>${alert}</td><td>${counted}</td></tr>`;
    });
    $('classTable').innerHTML = '<tr><th>index</th><th>code</th><th>full name</th><th>lesion_type</th><th>red alert</th><th>"Malignant finding"</th></tr>' + rows.join('');

    const total = HAM_COUNTS.reduce((a, b) => a + b, 0);
    const K = HAM_COUNTS.length;
    const weights = HAM_COUNTS.map((n) => total / (K * n));
    const render = (mode) => {
      let values, digits, note;
      if (mode === 'count') {
        values = HAM_COUNTS; digits = 0;
        note = `NV alone is ${fmt((100 * HAM_COUNTS[5]) / total, 1)} % of the images; DF is ${fmt((100 * HAM_COUNTS[3]) / total, 2)} %. An unweighted model can reach high accuracy by favouring NV.`;
      } else if (mode === 'weight') {
        values = weights; digits = 2;
        note = `w_c = N / (K · N_c) with N = ${total}, K = ${K}. A DF image weighs ${fmt(weights[3] / weights[5], 0)}× an NV image. train.py computes these from the training split, so the exact values depend on it.`;
      } else {
        values = HAM_COUNTS.map((n, i) => n * weights[i]); digits = 1;
        note = `Every class now carries the same total weight N / K = ${fmt(total / K, 1)}: the loss is balanced across classes, not across images.`;
      }
      const max = Math.max(...values);
      $('hamBars').innerHTML = values.map((v, i) => `
        <div class="bar-row"><span class="lbl">${CLASSES[i].code}</span>
          <div class="bar-track"><div class="bar-fill${ALERT_INDICES.has(i) ? ' alert' : ''}" style="width:${(100 * v) / max}%"></div></div>
          <span class="val">${fmt(v, digits)}</span></div>`).join('');
      $('hamNote').textContent = note;
    };
    bindSeg(document.querySelector('#data .seg'), 'ham', render);
    render('count');
  }

  // ------------------------------------------------------------------ 02 validation

  function initValidation() {
    const dims = () => {
      const w = parseInt($('dimW').value, 10), h = parseInt($('dimH').value, 10);
      $('dimWo').textContent = w;
      $('dimHo').textContent = h;
      const v = SLH.validateDimensions(w, h);
      const el = $('dimVerdict');
      el.className = 'verdict ' + (v.ok ? 'ok' : 'no');
      el.textContent = (v.ok ? 'pass: ' : 'reject: ') + v.message;
    };
    $('dimW').addEventListener('input', dims);
    $('dimH').addEventListener('input', dims);
    dims();

    const clip = () => {
      const sm = parseFloat($('sMed').value), sn = parseFloat($('sNon').value);
      $('sMedo').textContent = fmt(sm, 3);
      $('sNono').textContent = fmt(sn, 3);
      const pImpl = 1 / (1 + Math.exp(-(sm - sn)));
      const pScaled = 1 / (1 + Math.exp(-100 * (sm - sn)));
      $('clipBars').innerHTML = `
        <div class="bar-row"><span class="lbl">as coded</span><div class="bar-track"><div class="bar-fill" style="width:${100 * pImpl}%"></div></div><span class="val">${fmt(pImpl, 4)}</span></div>
        <div class="bar-row"><span class="lbl">×100</span><div class="bar-track"><div class="bar-fill" style="width:${100 * pScaled}%;background:var(--violet)"></div></div><span class="val">${fmt(pScaled, 4)}</span></div>`;
      const ok = pImpl >= 0.5;
      const el = $('clipVerdict');
      el.className = 'verdict ' + (ok ? 'ok' : 'no');
      el.textContent = ok
        ? `accepted: "Valid skin lesion image", confidence = p_skin = ${fmt(pImpl, 4)}`
        : `rejected: "Image may not be a valid skin lesion", confidence = 1 - p_non = ${fmt(pImpl, 4)}`;
    };
    $('sMed').addEventListener('input', clip);
    $('sNon').addEventListener('input', clip);
    clip();

    const run = (img, label) => {
      drawScaled($('valCanvas'), img);
      const lines = [`${label}: ${img.width}×${img.height} px`];
      const d = SLH.validateDimensions(img.width, img.height);
      lines.push(`gate 1  ${d.ok ? 'pass' : 'REJECT'}  ${d.message}`);
      const small = img.height > 100 ? SLH.resizeBilinear(img, 100, 100) : img;
      const sctx = $('valSmall').getContext('2d');
      sctx.clearRect(0, 0, 100, 100);
      if (small.width === 100 && small.height === 100) paintRgb($('valSmall'), small);
      if (d.ok) {
        const u = SLH.countUniqueColors(img);
        const pass = u.count >= 30;
        lines.push(`gate 2  ${pass ? 'pass' : 'REJECT'}  ${u.count} unique colours${u.resized ? ' after resize to 100×100' : ' (height ≤ 100: no resize)'}`);
        if (!pass) lines.push(`        "Image appears to be a simple graphic (${u.count} unique colors)"`);
        else lines.push('gate 3  runs only in the Python app (CLIP ViT-B/32)');
      } else {
        lines.push('gate 2  not reached');
      }
      lines.push('', 'Counts use a bilinear resize equivalent to cv2.resize up to ±1 rounding per channel.');
      $('valReport').textContent = lines.join('\n');
    };
    $('valFile').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      $('valReport').textContent = 'decoding...';
      try { run(await decodeFile(file), file.name); } catch (err) { $('valReport').textContent = 'Could not decode this file: ' + err.message; }
    });
    $('valSynthetic').addEventListener('click', () => run(syntheticLesion({ width: 640, height: 480 }), 'synthetic lesion'));
    $('valIcon').addEventListener('click', () => {
      const w = 500, h = 500, data = new Uint8Array(w * h * 3);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const inTriangle = y > 100 && y < 400 && Math.abs(x - 250) < (y - 100) / 2;
        const colour = inTriangle ? [255, 255, 255] : [31, 111, 235];
        data.set(colour, (y * w + x) * 3);
      }
      run({ width: w, height: h, data }, 'flat icon (2 colours)');
    });
    $('valSynthetic').click();
  }

  // ------------------------------------------------------------------ 03 preprocessing

  function initPreprocessing() {
    // Keras EfficientNet Normalization constants (ImageNet, red channel); the
    // layer stores the standard deviation as "variance", hence sqrt(0.229).
    const MEAN_R = 0.485, STD_R = 0.229;
    let model = 'EfficientNetB3';
    const render = () => {
      const v = parseFloat($('ppVal').value);
      $('ppValo').textContent = v;
      const rows = [];
      const size = MODELS[model].size;
      rows.push(['load_img + resize', `${size}×${size}, float32`, fmt(v, 1)]);
      if (model === 'EfficientNetB3') {
        const afterDiv = v / 255;
        const inner = afterDiv / 255;
        const inference = (inner - MEAN_R) / Math.sqrt(STD_R);
        const training = inference / Math.sqrt(STD_R);
        rows.push(['efficientnet.preprocess_input', 'identity (pass-through)', fmt(v, 1)]);
        rows.push(['÷ 255 (classification.py)', '', fmt(afterDiv, 5)]);
        rows.push(['model: Rescaling(1/255)', 'built into EfficientNetB3', fmt(inner, 7)]);
        rows.push(['model: Normalization', '(x − 0.485) / √0.229, inference graph', fmt(inference, 5)]);
        rows.push(['model: Rescaling(1/√0.229)', "training graph only (weights='imagenet')", fmt(training, 5)]);
        $('ppNote').className = 'callout warn';
        $('ppNote').innerHTML = `Across the full 0 to 255 range the normalised red value spans only ${fmt((255 / 65025) / Math.sqrt(STD_R), 4)} at inference, versus about 4.37 with standard ImageNet preprocessing: the image is divided by 255 twice. Training saw the same double division, so this is consistent. The real mismatch is the last row: the inference graph is built with <code>weights=None</code> and lacks the extra <code>Rescaling</code> layer that Keras adds for <code>weights='imagenet'</code>, so inputs reach the stem convolution about 0.48× as large as in training, if the weights were trained with this repository's <code>train.py</code> under current Keras.`;
      } else {
        const pp = v / 127.5 - 1;
        rows.push([`${model === 'InceptionV3' ? 'inception_v3' : 'inception_resnet_v2'}.preprocess_input`, 'x / 127.5 − 1', fmt(pp, 5)]);
        rows.push(['÷ 255 (classification.py)', 'range [−0.0039, 0.0039]', fmt(pp / 255, 7)]);
        $('ppNote').className = 'callout';
        $('ppNote').innerHTML = 'The Inception models normally receive values in [−1, 1]; here they receive [−1/255, 1/255]. Training used the identical transform, so the weights are adapted to it, but the ImageNet initialisation of phase 1 started from inputs 255× smaller than those it was pretrained on. A patch of zeros in this space (used by occlusion sensitivity) corresponds to mid-grey.';
      }
      $('ppTable').innerHTML = '<tr><th>step</th><th>operation</th><th>value</th></tr>' +
        rows.map((r) => `<tr><td class="mono">${r[0]}</td><td>${r[1]}</td><td class="num">${r[2]}</td></tr>`).join('');
    };
    bindSeg($('ppSeg'), 'pp', (m) => { model = m; render(); });
    $('ppVal').addEventListener('input', render);
    render();
  }

  // ------------------------------------------------------------------ 04 architecture

  function initArchitecture() {
    const SPECS = {
      EfficientNetB3: { size: 224, feat: [7, 7, 1536], base: 10783535, total: 11210806, gradcam: 'top_conv (Conv2D) · 7×7×1536' },
      InceptionV3: { size: 299, feat: [8, 8, 2048], base: 21802784, total: 22361127, gradcam: 'conv2d_93 (Conv2D) · 8×8×192, one branch of mixed10' },
      InceptionResNetV2: { size: 299, feat: [8, 8, 1536], base: 54336736, total: 54764007, gradcam: 'conv_7b_ac (Activation, matched by name) · 8×8×1536' }
    };
    let model = 'EfficientNetB3';
    let selected = 1;
    const layersFor = (m) => {
      const s = SPECS[m], C = s.feat[2];
      const d256 = C * 256 + 256, d128 = 256 * 128 + 128, d7 = 128 * 7 + 7;
      return [
        { name: 'Input', shape: `${s.size}×${s.size}×3`, params: 0, info: `RGB image after preprocessing (chapter 03). Input size comes from MODEL_CONFIGS['${m}']['img_size'].` },
        { name: m, shape: s.feat.join('×'), params: s.base, info: `ImageNet backbone with include_top=False. Frozen in phase 1 (base_model.trainable = False), trainable in phase 2. Grad-CAM layer chosen automatically: ${s.gradcam}.` },
        { name: 'GlobalAveragePooling2D', shape: `${C}`, params: 0, info: `Averages each of the ${C} feature maps over its ${s.feat[0]}×${s.feat[1]} positions, giving a translation-invariant descriptor.` },
        { name: 'Dense 256 · ReLU', shape: '256', params: d256, info: `${C} × 256 weights + 256 biases = ${d256.toLocaleString('en-US')} parameters.` },
        { name: 'Dropout 0.2', shape: '256', params: 0, info: 'During fit, each unit is zeroed with probability 0.2 and the rest are scaled by 1/0.8. At inference (model.predict) it is the identity.' },
        { name: 'Dense 128 · ReLU', shape: '128', params: d128, info: `256 × 128 + 128 = ${d128.toLocaleString('en-US')} parameters.` },
        { name: 'Dropout 0.1', shape: '128', params: 0, info: 'Zeroes units with probability 0.1 during training only.' },
        { name: 'Dense 7 · softmax', shape: '7', params: d7, info: `128 × 7 + 7 = ${d7} parameters. softmax(z)_i = exp(z_i) / Σ_j exp(z_j), so the 7 outputs are positive and sum to 1. The order is AKIEC, BCC, BKL, DF, MEL, NV, VASC.` }
      ];
    };
    const render = () => {
      const layers = layersFor(model);
      $('layerStrip').innerHTML = layers.map((l, i) => `<button type="button" class="layer${i === selected ? ' on' : ''}" data-i="${i}"><div>${escapeHtml(l.name)}</div><div class="shape">${l.shape}</div></button>`).join('<span class="flow-dot" aria-hidden="true" style="align-self:center;color:var(--accent-2)">&#9656;</span>');
      $('layerStrip').querySelectorAll('.layer').forEach((b) => b.addEventListener('click', () => { selected = parseInt(b.dataset.i, 10); render(); }));
      const l = layers[selected], s = SPECS[model];
      $('layerInfo').innerHTML = `<strong>${escapeHtml(l.name)}</strong> · output <code>${l.shape}</code> · ${l.params.toLocaleString('en-US')} parameters<p style="margin:8px 0 0">${escapeHtml(l.info)}</p>
        <p class="sub" style="margin:8px 0 0">Whole model: ${s.total.toLocaleString('en-US')} parameters, of which ${(s.total - s.base).toLocaleString('en-US')} in the head.</p>`;
    };
    bindSeg($('archSeg'), 'arch', (m) => { model = m; render(); });
    render();
  }

  // ------------------------------------------------------------------ 05 training

  /*
   * Keras ImageDataGenerator.apply_transform: matrix = R · S · Sh · Z about the
   * image centre, applied to (row, col) coordinates with scipy's
   * affine_transform(order=1, mode='nearest'); flips come afterwards.
   */
  function kerasAugment(img, p) {
    const h = img.height, w = img.width;
    const mul = (a, b) => a.map((row, i) => b[0].map((_, j) => row.reduce((s, _v, k) => s + a[i][k] * b[k][j], 0)));
    let m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const theta = (p.theta * Math.PI) / 180;
    if (p.theta) m = mul(m, [[Math.cos(theta), -Math.sin(theta), 0], [Math.sin(theta), Math.cos(theta), 0], [0, 0, 1]]);
    if (p.tx || p.ty) m = mul(m, [[1, 0, p.tx], [0, 1, p.ty], [0, 0, 1]]);
    const shear = (p.shear * Math.PI) / 180;
    if (p.shear) m = mul(m, [[1, -Math.sin(shear), 0], [0, Math.cos(shear), 0], [0, 0, 1]]);
    if (p.zx !== 1 || p.zy !== 1) m = mul(m, [[p.zx, 0, 0], [0, p.zy, 0], [0, 0, 1]]);
    const ox = h / 2 - 0.5, oy = w / 2 - 0.5;
    m = mul(mul([[1, 0, ox], [0, 1, oy], [0, 0, 1]], m), [[1, 0, -ox], [0, 1, -oy], [0, 0, 1]]);

    const out = new Uint8Array(w * h * 3);
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const sr = clamp(m[0][0] * r + m[0][1] * c + m[0][2], 0, h - 1);
        const sc = clamp(m[1][0] * r + m[1][1] * c + m[1][2], 0, w - 1);
        const r0 = Math.floor(sr), c0 = Math.floor(sc);
        const r1 = Math.min(r0 + 1, h - 1), c1 = Math.min(c0 + 1, w - 1);
        const fr = sr - r0, fc = sc - c0;
        let rr = r, cc = c;
        if (p.flipV) rr = h - 1 - r;
        if (p.flipH) cc = w - 1 - c;
        for (let k = 0; k < 3; k++) {
          const v = (img.data[(r0 * w + c0) * 3 + k] * (1 - fc) + img.data[(r0 * w + c1) * 3 + k] * fc) * (1 - fr) +
                    (img.data[(r1 * w + c0) * 3 + k] * (1 - fc) + img.data[(r1 * w + c1) * 3 + k] * fc) * fr;
          out[(rr * w + cc) * 3 + k] = Math.round(v);
        }
      }
    }
    return { width: w, height: h, data: out };
  }

  function simulateCallbacks(losses, lr0) {
    let lr = lr0, bestLr = null, waitLr = 0, bestEs = null, waitEs = 0, bestEpoch = 0, stopped = -1;
    const rows = [];
    for (let e = 0; e < losses.length; e++) {
      const loss = losses[e];
      const row = { epoch: e + 1, loss, lr, action: '' };
      waitEs += 1;
      const esImproved = bestEs === null || loss < bestEs;
      if (esImproved) { bestEs = loss; bestEpoch = e; waitEs = 0; }
      const rlrImproved = bestLr === null || loss + 1e-4 < bestLr;
      if (rlrImproved) { bestLr = loss; waitLr = 0; }
      else {
        waitLr += 1;
        if (waitLr >= 5 && lr > 1e-7) {
          lr = Math.max(lr * 0.5, 1e-7);
          waitLr = 0;
          row.action = `ReduceLROnPlateau → ${lr.toExponential(2)}`;
          row.reduce = true;
        }
      }
      row.esWait = waitEs;
      row.rlrWait = waitLr;
      row.esImproved = esImproved;
      row.rlrImproved = rlrImproved;
      if (!esImproved && waitEs >= 10 && e > 0) {
        row.action = (row.action ? row.action + '; ' : '') + 'EarlyStopping';
        row.stop = true;
        stopped = e;
        rows.push(row);
        break;
      }
      rows.push(row);
    }
    return { rows, bestEpoch, stopped };
  }

  function initTraining() {
    const source = syntheticLesion({ width: 160, height: 160, seed: 3, radius: 0.24 });
    // a small dark dot makes flips and rotations visible
    for (let y = 46; y < 58; y++) for (let x = 96; x < 108; x++) if (Math.hypot(y - 52, x - 102) < 6) source.data.set([25, 14, 10], (y * 160 + x) * 3);
    paintRgb($('augSrc'), source);

    const ids = ['augRot', 'augTx', 'augTy', 'augShear', 'augZx', 'augZy'];
    const formats = { augRot: (v) => fmt(v, 1), augTx: (v) => fmt(v, 3), augTy: (v) => fmt(v, 3), augShear: (v) => fmt(v, 2), augZx: (v) => fmt(v, 2), augZy: (v) => fmt(v, 2) };
    const render = () => {
      ids.forEach((id) => { $(id + 'o').textContent = formats[id](parseFloat($(id).value)); });
      const p = {
        theta: parseFloat($('augRot').value),
        tx: parseFloat($('augTx').value) * source.height,
        ty: parseFloat($('augTy').value) * source.width,
        shear: parseFloat($('augShear').value),
        zx: parseFloat($('augZx').value),
        zy: parseFloat($('augZy').value),
        flipH: $('augFh').checked,
        flipV: $('augFv').checked
      };
      paintRgb($('augDst'), kerasAugment(source, p));
    };
    ids.forEach((id) => $(id).addEventListener('input', render));
    $('augFh').addEventListener('change', render);
    $('augFv').addEventListener('change', render);
    $('augRandom').addEventListener('click', () => {
      const u = (a, b) => a + Math.random() * (b - a);
      $('augRot').value = u(-40, 40);
      $('augTx').value = u(-0.2, 0.2);
      $('augTy').value = u(-0.2, 0.2);
      $('augShear').value = u(-0.2, 0.2);
      $('augZx').value = u(0.8, 1.2);
      $('augZy').value = u(0.8, 1.2);
      $('augFh').checked = Math.random() < 0.5;
      $('augFv').checked = Math.random() < 0.5;
      render();
    });
    $('augReset').addEventListener('click', () => {
      $('augRot').value = 0; $('augTx').value = 0; $('augTy').value = 0; $('augShear').value = 0;
      $('augZx').value = 1; $('augZy').value = 1; $('augFh').checked = false; $('augFv').checked = false;
      render();
    });
    render();

    const PRESETS = {
      plateau: '1.62 1.41 1.27 1.18 1.12 1.085 1.07 1.074 1.071 1.0702 1.076 1.072 1.031 1.012 1.005',
      steady: '1.55 1.32 1.18 1.08 1.01 0.96 0.92 0.89 0.87 0.855 0.845 0.838 0.833 0.83 0.828',
      noisy: '0.97 0.91 0.872 0.851 0.838 0.8379 0.842 0.839 0.8385 0.841 0.812 0.801 0.805 0.803 0.809 0.802 0.806 0.804 0.807 0.805 0.808 0.806 0.81 0.809 0.811 0.812 0.81 0.813 0.812 0.814'
    };
    let lr0 = 1e-4, maxEpochs = 15;
    const sim = () => {
      const losses = $('simInput').value.split(/[\s,;]+/).map(parseFloat).filter((v) => Number.isFinite(v));
      const used = losses.slice(0, maxEpochs);
      const { rows, bestEpoch, stopped } = simulateCallbacks(used, lr0);
      const body = rows.map((r) => `<tr class="${r.stop ? 'stop' : r.reduce ? 'reduce' : ''}"><td class="num">${r.epoch}</td><td class="num">${r.loss}</td>
        <td class="num">${r.lr.toExponential(1)}</td><td class="num">${r.rlrImproved ? 'yes' : 'no'} · ${r.rlrWait}/5</td>
        <td class="num">${r.esImproved ? 'yes' : 'no'} · ${r.esWait}/10</td><td>${r.action}</td></tr>`);
      for (let e = rows.length; e < used.length; e++) body.push(`<tr class="skipped"><td class="num">${e + 1}</td><td class="num">${used[e]}</td><td colspan="4">not run</td></tr>`);
      $('simTable').innerHTML = '<tr><th>epoch</th><th>val_loss</th><th>lr used</th><th>RLR improved · wait</th><th>ES improved · wait</th><th>action at epoch end</th></tr>' + body.join('');
      const extra = losses.length > maxEpochs ? ` Values beyond epoch ${maxEpochs} are ignored (the phase length).` : '';
      $('simSummary').textContent = rows.length
        ? `${stopped >= 0 ? `Training stops after epoch ${stopped + 1}.` : `All ${rows.length} epochs run.`} On train end EarlyStopping restores the weights of epoch ${bestEpoch + 1} (val_loss ${used[bestEpoch]}).${extra}`
        : 'Enter at least one number.';
    };
    bindSeg($('simPhase'), 'lr', (v, btn) => { lr0 = parseFloat(v); maxEpochs = parseInt(btn.dataset.max, 10); sim(); });
    document.querySelectorAll('[data-preset]').forEach((b) => b.addEventListener('click', () => { $('simInput').value = PRESETS[b.dataset.preset]; sim(); }));
    $('simInput').addEventListener('input', sim);
    $('simInput').value = PRESETS.plateau;
    sim();
  }

  // ------------------------------------------------------------------ 06 inference and ensemble

  function initEnsemble() {
    const SCENARIOS = {
      agree: { EfficientNetB3: [2, 3, 6, 1, 70, 17, 1], InceptionV3: [3, 4, 8, 1, 64, 19, 1], InceptionResNetV2: [3, 5, 9, 2, 58, 22, 1] },
      disagree: { EfficientNetB3: [2, 3, 10, 1, 58, 25, 1], InceptionV3: [3, 5, 15, 1, 41, 34, 1], InceptionResNetV2: [4, 6, 22, 2, 30, 35, 1] },
      uncertain: { EfficientNetB3: [12, 14, 16, 10, 18, 20, 10], InceptionV3: [14, 12, 15, 11, 17, 19, 12], InceptionResNetV2: [13, 15, 14, 12, 16, 18, 12] }
    };
    const raw = JSON.parse(JSON.stringify(SCENARIOS.disagree));
    let editing = 'EfficientNetB3';

    const selected = () => Array.from(document.querySelectorAll('[data-model]')).filter((c) => c.checked).map((c) => c.dataset.model);
    const normalise = (v) => { const s = v.reduce((a, b) => a + b, 0) || 1; return v.map((x) => x / s); };

    const renderSliders = () => {
      const models = selected();
      if (!models.includes(editing)) editing = models[0] || 'EfficientNetB3';
      $('ensEditSeg').innerHTML = models.map((m) => `<button type="button" class="${m === editing ? 'on' : ''}" data-edit="${m}">${m}</button>`).join('');
      $('ensEditSeg').querySelectorAll('button').forEach((b) => b.addEventListener('click', () => { editing = b.dataset.edit; renderSliders(); }));
      const p = normalise(raw[editing]);
      $('ensSliders').innerHTML = CLASSES.map((c, i) => `<div class="control"><label for="ens${i}">${c.code}</label><input type="range" id="ens${i}" min="0" max="100" step="0.5" value="${raw[editing][i]}"><output>${fmt(p[i], 3)}</output></div>`).join('');
      CLASSES.forEach((_, i) => $('ens' + i).addEventListener('input', (e) => {
        raw[editing][i] = parseFloat(e.target.value);
        const q = normalise(raw[editing]);
        $('ensSliders').querySelectorAll('output').forEach((o, k) => { o.textContent = fmt(q[k], 3); });
        compute();
      }));
    };

    const compute = () => {
      const models = selected();
      if (models.length === 0) {
        $('ensAlert').innerHTML = '<div class="callout warn">No model selected: the app shows nothing until at least one model is chosen.</div>';
        $('ensMetrics').innerHTML = ''; $('ensBars').innerHTML = ''; $('ensFormula').textContent = '';
        shared.prediction = null;
        updateRecord();
        return;
      }
      let scores, formula;
      if (models.length === 1) {
        scores = normalise(raw[models[0]]);
        formula = `single model: predict_single('${models[0]}') → argmax of its softmax`;
      } else {
        const total = models.reduce((s, m) => s + MODELS[m].accuracy, 0);
        scores = new Array(7).fill(0);
        models.forEach((m) => normalise(raw[m]).forEach((v, i) => { scores[i] += (v * MODELS[m].accuracy) / total; }));
        formula = 'ensemble weights: ' + models.map((m) => `${m} ${fmt(MODELS[m].accuracy / total, 4)}`).join(' · ');
      }
      const idx = scores.indexOf(Math.max(...scores));
      const cls = CLASSES[idx];
      const malignantWord = cls.type.includes('Malignant');
      $('ensAlert').innerHTML = ALERT_INDICES.has(idx)
        ? '<div class="callout bad"><strong>ALERT: Potentially Malignant or Pre-Malignant Lesion Detected.</strong> This analysis suggests a lesion that may require immediate medical attention. Please consult a qualified dermatologist as soon as possible.</div>'
        : '';
      $('ensMetrics').innerHTML = `
        <div class="metric"><div class="k">Diagnosis</div><div class="v">${escapeHtml(cls.full)}</div></div>
        <div class="metric"><div class="k">Lesion Type</div><div class="v"><span style="color:${malignantWord ? 'var(--bad)' : 'var(--good)'}">&#9679;</span> ${cls.type}</div></div>
        <div class="metric"><div class="k">Confidence</div><div class="v">${fmt(100 * scores[idx], 1)}%</div></div>`;
      const order = scores.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      $('ensBars').innerHTML = order.map(([v, i]) => `<div class="bar-row${i === idx ? ' win' : ''}"><span class="lbl">${CLASSES[i].code}</span>
        <div class="bar-track"><div class="bar-fill${CLASSES[i].code === 'MEL' ? ' alert' : ''}" style="width:${100 * v}%"></div></div><span class="val">${fmt(100 * v, 1)}%</span></div>`).join('');
      const second = order[1];
      const diff = second[0] > 0.1 ? ` · differential: ${EXPLANATION_NAMES[second[1]]} ${fmt(100 * second[0], 1)}%` : ' · no differential (second class ≤ 10%)';
      $('ensFormula').textContent = formula + diff;
      shared.prediction = { index: idx, scores, models };
      updateRecord();
    };

    document.querySelectorAll('[data-model]').forEach((c) => c.addEventListener('change', () => { renderSliders(); compute(); }));
    document.querySelectorAll('[data-scenario]').forEach((b) => b.addEventListener('click', () => {
      const s = b.dataset.scenario;
      Object.keys(MODELS).forEach((m) => {
        if (s === 'random') {
          // Dirichlet(0.6) sample: normalised Gamma draws (Marsaglia-Tsang with boost)
          raw[m] = CLASSES.map(() => {
            const a = 1.6, d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
            let x, v, u;
            do {
              do { x = gaussian(Math.random); v = 1 + c * x; } while (v <= 0);
              v = v * v * v; u = Math.random();
            } while (u > 1 - 0.0331 * x ** 4 && Math.log(u) > 0.5 * x * x + d * (1 - v + Math.log(v)));
            return 100 * d * v * Math.pow(Math.random(), 1 / 0.6);
          });
          const s2 = raw[m].reduce((p, q) => p + q, 0);
          raw[m] = raw[m].map((v) => Math.round((1000 * v) / s2) / 10);
        } else {
          raw[m] = SCENARIOS[s][m].slice();
        }
      });
      renderSliders();
      compute();
    }));
    renderSliders();
    compute();
  }

  // ------------------------------------------------------------------ 07 Grad-CAM

  function initGradcam() {
    const N = 7;
    const channels = [[], [], [], []];
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const r = Math.hypot(y - 3.2, x - 3.0);
      channels[0].push(Math.exp(-((y - 3.2) ** 2 + (x - 3.0) ** 2) / 3.0));
      channels[1].push(Math.exp(-((r - 2.2) ** 2) / 0.6));
      channels[2].push(clamp(r / 4.0, 0, 1));
      channels[3].push(y < 2 && x >= 5 ? 1 : 0);
    }
    const names = ['A¹ lesion centre', 'A² lesion border', 'A³ background skin', 'A⁴ corner artefact'];
    // pooled gradients of the toy example in docs/assets/gradcam.svg
    const DEFAULT_ALPHA = [0.8968, 0.5233, -0.5907, 0.0442];
    const image = syntheticLesion({ width: 224, height: 224 });

    $('gcSliders').innerHTML = names.map((n, k) => `<div class="control"><label for="gcA${k}">α${'¹²³⁴'[k]} ${n.slice(3)}</label><input type="range" id="gcA${k}" min="-1.5" max="1.5" step="0.01" value="${DEFAULT_ALPHA[k]}"><output id="gcA${k}o"></output></div>`).join('');
    $('gcChannels').innerHTML = names.map((n, k) => `<div class="canvas-box"><canvas id="gcC${k}" class="pixelated" width="7" height="7" style="width:96px"></canvas><span>${n}</span></div>`).join('');
    channels.forEach((ch, k) => paintField($('gcC' + k), ch, N, N, viridis));

    const render = () => {
      const alpha = DEFAULT_ALPHA.map((_, k) => parseFloat($('gcA' + k).value));
      alpha.forEach((a, k) => { $('gcA' + k + 'o').textContent = fmt(a, 2); });
      const blend = parseFloat($('gcAlpha').value);
      $('gcAlphao').textContent = fmt(blend, 2);

      const sum = new Float64Array(N * N);
      for (let i = 0; i < N * N; i++) for (let k = 0; k < 4; k++) sum[i] += alpha[k] * channels[k][i];
      const maxAbs = Math.max(...sum.map(Math.abs)) || 1;
      paintField($('gcSum'), Array.from(sum, (v) => 0.5 + v / (2 * maxAbs)), N, N, coolwarm);

      const max = Math.max(...sum);
      const verdict = $('gcVerdict');
      let heat;
      if (max > 0) {
        heat = Array.from(sum, (v) => Math.max(v, 0) / max);
        verdict.className = 'verdict ok';
        verdict.textContent = `max = ${fmt(max, 3)} > 0: heatmap = maximum(h, 0) / ${fmt(max, 3)}`;
      } else {
        heat = new Array(N * N).fill(0);
        verdict.className = 'verdict no';
        verdict.textContent = max === 0
          ? 'max = 0: 0 / 0 gives NaN in the app; nothing meaningful can be drawn'
          : `max = ${fmt(max, 3)} < 0: every cell becomes 0 / negative = 0, an all-blue overlay`;
      }
      paintField($('gcHeat'), heat, N, N, jet);

      // overlay_heatmap: resize, uint8(255 * h), COLORMAP_JET, alpha blend, clip
      const big = resizeFloat(heat, N, N, 224, 224);
      const out = new Uint8Array(224 * 224 * 3);
      for (let i = 0; i < 224 * 224; i++) {
        const [r, g, b] = jet(Math.floor(255 * big[i]) / 255);
        out[i * 3] = clamp(Math.trunc(r * blend + image.data[i * 3] * (1 - blend)), 0, 255);
        out[i * 3 + 1] = clamp(Math.trunc(g * blend + image.data[i * 3 + 1] * (1 - blend)), 0, 255);
        out[i * 3 + 2] = clamp(Math.trunc(b * blend + image.data[i * 3 + 2] * (1 - blend)), 0, 255);
      }
      paintRgb($('gcOverlay'), { width: 224, height: 224, data: out });
    };
    DEFAULT_ALPHA.forEach((_, k) => $('gcA' + k).addEventListener('input', render));
    $('gcAlpha').addEventListener('input', render);
    $('gcReset').addEventListener('click', () => { DEFAULT_ALPHA.forEach((a, k) => { $('gcA' + k).value = a; }); render(); });
    render();
  }

  // ------------------------------------------------------------------ 08 occlusion

  function initOcclusion() {
    let size = 224, timer = null;
    let image, brown, template, baseline;

    const prepare = () => {
      image = syntheticLesion({ width: size, height: size });
      brown = new Float64Array(size * size);
      template = new Float64Array(size * size);
      let tsum = 0;
      const sigma = size * 0.2;
      for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
        const i = y * size + x;
        brown[i] = Math.max(image.data[i * 3] - image.data[i * 3 + 2], 0) / 255;
        template[i] = Math.exp(-((y - size * 0.5) ** 2 + (x - size * 0.47) ** 2) / (2 * sigma * sigma));
        tsum += template[i];
      }
      baseline = 0;
      for (let i = 0; i < size * size; i++) { template[i] /= tsum; baseline += template[i] * brown[i]; }
    };

    /* Port of compute_occlusion_sensitivity; the toy score only changes inside the patch */
    const compute = (patch, stride) => {
      const gw = Math.floor(size / stride);
      const raw = new Float64Array(gw * gw);
      const positions = [];
      for (let i = 0; i <= size - patch; i += stride) {
        for (let j = 0; j <= size - patch; j += stride) {
          let drop = 0;
          for (let y = i; y < i + patch; y++) for (let x = j; x < j + patch; x++) drop += template[y * size + x] * brown[y * size + x];
          raw[(i / stride) * gw + j / stride] = drop;
          positions.push([i, j]);
        }
      }
      return { raw, gw, positions };
    };

    const drawImage = (patchPos, patch) => {
      paintRgb($('occImg'), image);
      if (patchPos) {
        const ctx = $('occImg').getContext('2d');
        ctx.fillStyle = size === 224 ? '#000' : 'rgb(128,128,128)';
        ctx.fillRect(patchPos[1], patchPos[0], patch, patch);
        ctx.strokeStyle = '#39d0d8';
        ctx.lineWidth = 2;
        ctx.strokeRect(patchPos[1], patchPos[0], patch, patch);
      }
    };

    const drawRaw = (res, upto) => {
      const { raw, gw, positions } = res;
      const canvas = $('occRaw');
      canvas.width = gw * 16;
      canvas.height = gw * 16;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0b0f14';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const max = Math.max(...raw) || 1;
      const visited = new Set(positions.map(([i, j]) => `${i}/${j}`));
      for (let k = 0; k < Math.min(upto, positions.length); k++) {
        const [i, j] = positions[k];
        const [r, g, b] = hot(raw[(i / res.stride) * gw + j / res.stride] / max);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect((j / res.stride) * 16, (i / res.stride) * 16, 15, 15);
      }
      ctx.strokeStyle = '#d29922';
      ctx.setLineDash([3, 2]);
      for (let r = 0; r < gw; r++) for (let c = 0; c < gw; c++) {
        if (!visited.has(`${r * res.stride}/${c * res.stride}`)) ctx.strokeRect(c * 16 + 1, r * 16 + 1, 13, 13);
      }
      ctx.setLineDash([]);
    };

    const drawFinal = (res) => {
      const big = resizeFloat(res.raw, res.gw, res.gw, size, size);
      let lo = Infinity, hi = -Infinity;
      big.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
      paintField($('occMap'), Array.from(big, (v) => (v - lo) / (hi - lo + 1e-8)), size, size, hot);
    };

    const run = (animate) => {
      if (timer) { cancelAnimationFrame(timer); timer = null; }
      const patch = parseInt($('occPatch').value, 10), stride = parseInt($('occStride').value, 10);
      $('occPatcho').textContent = patch;
      $('occStrideo').textContent = stride;
      const res = compute(patch, stride);
      res.stride = stride;
      const n = Math.sqrt(res.positions.length);
      $('occStats').className = 'verdict';
      $('occStats').textContent = `${n}×${n} positions → ${res.positions.length} occluded + 1 baseline forward passes; grid ${res.gw}×${res.gw}, ${res.gw - n} unvisited row(s)/column(s) stay 0. Patch fill: ${size === 224 ? 'black (EfficientNetB3 input 0)' : 'mid-grey (Inception input 0)'}.`;
      if (!animate) {
        drawImage(null, patch);
        drawRaw(res, res.positions.length);
        drawFinal(res);
        return;
      }
      let k = 0;
      const ctx = $('occMap').getContext('2d');
      ctx.clearRect(0, 0, $('occMap').width, $('occMap').height);
      const step = () => {
        k = Math.min(k + Math.max(1, Math.floor(res.positions.length / 160)), res.positions.length);
        drawImage(res.positions[k - 1], patch);
        drawRaw(res, k);
        if (k < res.positions.length) timer = requestAnimationFrame(step);
        else { drawImage(null, patch); drawFinal(res); timer = null; }
      };
      step();
    };

    bindSeg($('occSize'), 'size', (v) => {
      size = parseInt(v, 10);
      $('occImg').width = size; $('occImg').height = size;
      $('occMap').width = size; $('occMap').height = size;
      prepare();
      run(false);
    });
    $('occPatch').addEventListener('input', () => run(false));
    $('occStride').addEventListener('input', () => run(false));
    $('occRun').addEventListener('click', () => run(true));
    prepare();
    run(false);
  }

  // ------------------------------------------------------------------ 09 ABCDE

  function initAbcde() {
    let current = null, currentLabel = '';

    const synth = () => syntheticLesion({
      radius: parseFloat($('abRadius').value),
      irregularity: parseFloat($('abIrreg').value),
      darkness: parseFloat($('abDark').value),
      skin: parseFloat($('abSkin').value),
      offset: parseFloat($('abOffset').value),
      seed: 11
    });

    const describeContour = (contour, img) => {
      if (!contour) return 'no foreground pixels';
      const r = contour.rect;
      const touches = r.x === 0 || r.y === 0 || r.x + r.w === img.width || r.y + r.h === img.height;
      return `area ${fmt(contour.area, 1)} px², perimeter ${fmt(contour.perimeter, 1)} px, bounding box ${r.w}×${r.h} at (${r.x}, ${r.y})${touches ? ', touches the image border' : ''}`;
    };

    const analyse = () => {
      const img = current;
      const res = SLH.computeAbcde(img);
      const seg = res.segmentation;
      shared.abcde = res;
      shared.imageSize = [img.width, img.height];

      drawScaled($('abImg'), img);

      // mask with contour, bounding box and cx line
      const maskImg = { width: img.width, height: img.height, data: new Uint8Array(img.width * img.height * 3) };
      for (let i = 0; i < seg.mask.length; i++) { const v = seg.mask[i] ? 255 : 0; maskImg.data[i * 3] = v; maskImg.data[i * 3 + 1] = v; maskImg.data[i * 3 + 2] = v; }
      const m = drawScaled($('abMask'), maskImg);
      const ctx = $('abMask').getContext('2d');
      if (seg.contour) {
        ctx.strokeStyle = '#39d0d8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        seg.contour.points.forEach(([x, y], i) => { const px = m.ox + (x + 0.5) * m.scale, py = m.oy + (y + 0.5) * m.scale; if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
        ctx.closePath();
        ctx.stroke();
        const r = seg.contour.rect;
        ctx.strokeStyle = '#d29922';
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(m.ox + r.x * m.scale, m.oy + r.y * m.scale, r.w * m.scale, r.h * m.scale);
        ctx.setLineDash([]);
        if (res.asymmetry.cx !== null) {
          ctx.strokeStyle = '#bc8cff';
          ctx.beginPath();
          ctx.moveTo(m.ox + res.asymmetry.cx * m.scale, m.oy);
          ctx.lineTo(m.ox + res.asymmetry.cx * m.scale, m.oy + img.height * m.scale);
          ctx.stroke();
        }
      }

      // histogram with threshold
      const hc = $('abHist'), hctx = hc.getContext('2d');
      hctx.clearRect(0, 0, hc.width, hc.height);
      const hist = seg.histogram, hmax = Math.max(...hist);
      for (let i = 0; i < 256; i++) {
        const hgt = (hc.height - 14) * (hist[i] / hmax);
        hctx.fillStyle = i > seg.threshold ? '#c9d1d9' : '#6e7681';
        hctx.fillRect(i * 2, hc.height - 12 - hgt, 2, hgt);
      }
      hctx.strokeStyle = '#f85149';
      hctx.lineWidth = 2;
      hctx.beginPath(); hctx.moveTo(seg.threshold * 2 + 1, 0); hctx.lineTo(seg.threshold * 2 + 1, hc.height - 12); hctx.stroke();
      hctx.fillStyle = '#8b949e';
      hctx.font = '11px ui-monospace, monospace';
      hctx.fillText(`grey histogram · Otsu T = ${seg.threshold} · light bars (gray > T) become foreground`, 4, hc.height - 1);

      const fg = seg.mask.reduce((a, b) => a + b, 0) / seg.mask.length;
      const lines = [
        `${currentLabel}: ${img.width}×${img.height} px`,
        `Otsu threshold T = ${seg.threshold}; foreground (gray > T) = ${fmt(100 * fg, 1)} % of pixels`,
        `largest external contour: ${describeContour(seg.contour, img)}`,
        `centroid column cx = ${res.asymmetry.cx}; asymmetry compares columns j and W-1-j for j < ${res.asymmetry.band}`,
        '',
        'ABCDE Criteria Analysis (as printed by the app):',
        `  A - Asymmetry: ${fmt(res.asymmetry.score, 2)} (${res.asymmetry.status})`,
        `  B - Border: ${fmt(res.border.score, 2)} (${res.border.status})${res.border.circularity !== null ? `   circularity ${fmt(res.border.circularity, 3)}` : ''}`,
        `  C - Color: Variance ${fmt(res.color.variance, 0)} (${res.color.status})`,
        `  D - Diameter: ${fmt(res.diameter.value_mm, 1)}mm (${res.diameter.status})   ${res.diameter.pixels} px at 10 px/mm`,
        `  E - Evolution: ${res.evolution.status}`,
        '',
        'Printed only when the prediction is MEL:',
        `  Blue-white areas: ${fmt(100 * res.colorMetrics.blue_white_presence, 1)}%   (pixels with B > 200)`,
        `  Darkness ratio: ${fmt(100 * res.colorMetrics.darkness_ratio, 1)}%   (OpenCV L < 50, approx.)`
      ];
      $('abReport').textContent = lines.join('\n');

      const inv = $('abInvert').checked;
      $('abInvReport').style.display = inv ? 'block' : 'none';
      if (inv) {
        const invMask = seg.mask.map((v) => 1 - v);
        const c = SLH.largestExternalContour(invMask, img.width, img.height);
        if (!c) {
          $('abInvReport').textContent = 'Hypothetical (not in the code): no dark region found.';
        } else {
          const circ = (4 * Math.PI * c.area) / (c.perimeter * c.perimeter + 1e-10);
          $('abInvReport').textContent = [
            'Hypothetical measurement of the DARK region (THRESH_BINARY_INV); the application does not do this:',
            `  contour: ${describeContour(c, img)}`,
            `  B would be ${fmt(clamp(1 - circ, 0, 1), 2)} (${1 - circ > 0.4 ? 'Irregular' : 'Regular'}), D would be ${fmt(Math.max(c.rect.w, c.rect.h) / 10, 1)}mm (${Math.max(c.rect.w, c.rect.h) / 10 > 6 ? '>6mm' : '≤6mm'})`
          ].join('\n');
          const ictx = $('abMask').getContext('2d');
          ictx.strokeStyle = '#f85149';
          ictx.lineWidth = 2;
          ictx.beginPath();
          c.points.forEach(([x, y], i) => { const px = m.ox + (x + 0.5) * m.scale, py = m.oy + (y + 0.5) * m.scale; if (i) ictx.lineTo(px, py); else ictx.moveTo(px, py); });
          ictx.closePath();
          ictx.stroke();
        }
      }
      updateRecord();
    };

    const fromSynth = () => {
      $('abSynthControls').style.opacity = 1;
      ['abRadius', 'abIrreg', 'abDark', 'abSkin', 'abOffset'].forEach((id) => { $(id + 'o').textContent = fmt(parseFloat($(id).value), 2); });
      current = synth();
      currentLabel = 'synthetic lesion';
      analyse();
    };
    ['abRadius', 'abIrreg', 'abDark', 'abSkin', 'abOffset'].forEach((id) => $(id).addEventListener('input', fromSynth));
    $('abSynth').addEventListener('click', fromSynth);
    $('abInvert').addEventListener('change', () => current && analyse());
    $('abFile').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      $('abReport').textContent = 'decoding and analysing...';
      try {
        const img = await decodeFile(file);
        if (img.width * img.height > 16e6) { $('abReport').textContent = 'Image larger than 16 megapixels: too large for the in-browser port.'; return; }
        current = img;
        currentLabel = file.name;
        $('abSynthControls').style.opacity = 0.45;
        setTimeout(analyse, 20);
      } catch (err) {
        $('abReport').textContent = 'Could not decode this file: ' + err.message;
      }
    });
    fromSynth();
  }

  // ------------------------------------------------------------------ 10 storage and report

  function buildExplanation(pred, abcde) {
    const i = pred.index, scores = pred.scores, f = FEATURES[i];
    const confidence = scores[i] * 100;
    const out = [];
    out.push(`**Diagnostic Analysis for ${EXPLANATION_NAMES[i]}**`, '');
    if (ALERT_INDICES.has(i)) out.push('[warning icon] **ALERT: Potentially Malignant Lesion Detected**', '');
    out.push(`**Confidence Level:** ${fmt(confidence, 1)}%`, '');
    out.push('**Check for these diagnostic features:**');
    f.key.forEach((k) => out.push(`• ${k}`));
    out.push('', '**Clinical Correlation:**');
    f.signs.forEach((s) => out.push(`• ${s}`));
    out.push('', '**ABCDE Criteria Analysis:**');
    out.push(`• **A - Asymmetry:** ${fmt(abcde.asymmetry.score, 2)} (${abcde.asymmetry.status})`);
    out.push(`• **B - Border:** ${fmt(abcde.border.score, 2)} (${abcde.border.status})`);
    out.push(`• **C - Color:** Variance ${fmt(abcde.color.variance, 0)} (${abcde.color.status})`);
    out.push(`• **D - Diameter:** ${fmt(abcde.diameter.value_mm, 1)}mm (${abcde.diameter.status})`);
    out.push(`• **E - Evolution:** ${abcde.evolution.status}`);
    if (i === 4) {
      out.push('', '**Additional Quantitative Details:**');
      out.push(`• Blue-white areas: ${fmt(abcde.colorMetrics.blue_white_presence * 100, 1)}%`);
      if (abcde.colorMetrics.blue_white_presence > 0.1) out.push('  → Blue-white veil present (melanoma-specific sign)');
      out.push(`• Darkness ratio: ${fmt(abcde.colorMetrics.darkness_ratio * 100, 1)}%`);
    }
    const order = scores.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]);
    if (order[1][0] > 0.1) {
      out.push('', '**Differential Consideration:**');
      out.push(`• ${EXPLANATION_NAMES[order[1][1]]}: ${fmt(order[1][0] * 100, 1)}%`);
    }
    out.push('', '**[warning icon] Important Reminder:**');
    out.push('• This tool must NOT be used for self-diagnosis');
    out.push('• Always consult a qualified dermatologist for proper evaluation');
    out.push('• Only dermatologists are responsible for diagnosis and treatment decisions');
    out.push('• This is a decision support tool, not a diagnostic device');
    out.push('', '**Model Reasoning Transparency:**');
    out.push(`• Decision based on pattern recognition of ${f.dermoscopy}`);
    out.push(`• Confidence distribution indicates ${confidence > 70 ? 'high certainty' : 'moderate certainty'}`);
    return out.join('\n');
  }

  function updateRecord() {
    const target = $('recordJson');
    if (!target) return;
    const pred = shared.prediction, abcde = shared.abcde;
    if (!pred) {
      target.textContent = '// select at least one model in chapter 06';
      $('explainText').textContent = '';
      return;
    }
    const scores = {};
    CLASSES.forEach((c, k) => { scores[c.code] = Number(pred.scores[k].toFixed(6)); });
    const cls = CLASSES[pred.index];
    const explanation = abcde ? buildExplanation(pred, abcde) : '';
    const record = {
      filename: 'example.jpg',
      file_path: 'user_uploads/<user_id>/<uuid>_example.jpg',
      diagnosis: cls.full,
      lesion_type: cls.type,
      confidence_scores: JSON.stringify(scores),
      model_used: pred.models.join(', '),
      is_valid_skin_image: true,
      explanation_text: explanation ? explanation.slice(0, 60) + '... (' + explanation.length + ' characters)' : '',
      heatmap_path: 'heatmaps/<uuid>_heatmap.npy',
      processing_time: '<seconds of the classification call>',
      image_width: shared.imageSize[0],
      image_height: shared.imageSize[1]
    };
    target.textContent = JSON.stringify(record, null, 2);
    $('explainText').innerHTML = escapeHtml(explanation).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  }

  function initStorage() {
    const hash = '$2b$12$w/APCAloGFYKdem.XRRzceB85ZWZP3m63L5q6.DMwrquekm6Wf5Cy';
    const parts = [
      { text: '$2b$', info: '"2b": the bcrypt algorithm version identifier.' },
      { text: '12$', info: 'Cost factor 12, the bcrypt.gensalt() default: the key setup is repeated 2^12 = 4096 times, which makes each guess deliberately slow.' },
      { text: hash.slice(7, 29), info: '22 characters: the 128-bit random salt (bcrypt base-64). A new salt per user means equal passwords produce different hashes.' },
      { text: hash.slice(29), info: '31 characters: the 184-bit hash output. checkpw re-derives it from the candidate password and the stored salt and compares the results.' }
    ];
    const colours = ['var(--violet)', 'var(--warn)', 'var(--accent-2)', 'var(--good)'];
    $('bcryptHash').innerHTML = parts.map((p, i) => `<span data-i="${i}" style="color:${colours[i]};cursor:help;border-bottom:1px dotted">${escapeHtml(p.text)}</span>`).join('');
    $('bcryptHash').querySelectorAll('span').forEach((s) => {
      const show = () => { $('bcryptPart').textContent = parts[parseInt(s.dataset.i, 10)].info; };
      s.addEventListener('mouseenter', show);
      s.addEventListener('click', show);
    });
    updateRecord();
  }

  // ------------------------------------------------------------------ boot

  document.addEventListener('DOMContentLoaded', () => {
    initChrome();
    initLabels();
    initValidation();
    initPreprocessing();
    initArchitecture();
    initTraining();
    initEnsemble();
    initGradcam();
    initOcclusion();
    initAbcde();
    initStorage();
  });
})();
