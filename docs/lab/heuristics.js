/*
 * Browser port of the image heuristics implemented in the Python code.
 *
 * Every function mirrors a specific Python function so that the interactive
 * documentation shows exactly what the application computes:
 *
 *   rgbToGray              cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 *   otsuThreshold          cv2.threshold(gray, 0, 255, THRESH_BINARY + THRESH_OTSU)
 *   largestExternalContour cv2.findContours(RETR_EXTERNAL) + max(contourArea)
 *   detectAsymmetry        ExplainabilityEngine.detect_asymmetry
 *   detectBorder           ExplainabilityEngine.detect_border_irregularity
 *   computeDiameter        ExplainabilityEngine.compute_diameter
 *   colorMetrics           ExplainabilityEngine.analyze_color_distribution (subset)
 *   computeAbcde           ExplainabilityEngine.compute_abcde_criteria
 *   validateDimensions     ImageValidator.validate_image_dimensions
 *   countUniqueColors      unique-colour gate inside ImageValidator.validate_image
 *
 * Images are passed as { width, height, data } where data is a Uint8Array or
 * Uint8ClampedArray holding RGB (3 channels) or RGBA (4 channels) pixels.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.SLH = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FLT_EPSILON = 1.1920928955078125e-7;

  function channelsOf(img) {
    return img.data.length / (img.width * img.height);
  }

  /* OpenCV 4.x fixed-point luma, verified exhaustively over all 2^24 RGB triples:
     (R*9798 + G*19235 + B*3735 + 16384) >> 15 */
  function rgbToGray(img) {
    const n = img.width * img.height;
    const c = channelsOf(img);
    const gray = new Uint8Array(n);
    for (let p = 0, q = 0; p < n; p++, q += c) {
      gray[p] = (img.data[q] * 9798 + img.data[q + 1] * 19235 + img.data[q + 2] * 3735 + 16384) >> 15;
    }
    return gray;
  }

  function histogram(gray) {
    const hist = new Float64Array(256);
    for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
    return hist;
  }

  /* Literal port of getThreshVal_Otsu_8u from OpenCV imgproc/thresh.cpp */
  function otsuThreshold(gray) {
    const hist = histogram(gray);
    const scale = 1 / gray.length;
    let mu = 0;
    for (let i = 0; i < 256; i++) mu += i * hist[i];
    mu *= scale;

    let mu1 = 0, q1 = 0, maxSigma = 0, maxVal = 0;
    for (let i = 0; i < 256; i++) {
      const pi = hist[i] * scale;
      mu1 *= q1;
      q1 += pi;
      const q2 = 1 - q1;
      if (Math.min(q1, q2) < FLT_EPSILON || Math.max(q1, q2) > 1 - FLT_EPSILON) continue;
      mu1 = (mu1 + i * pi) / q1;
      const mu2 = (mu - q1 * mu1) / q2;
      const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
      if (sigma > maxSigma) {
        maxSigma = sigma;
        maxVal = i;
      }
    }
    return { threshold: maxVal, histogram: hist };
  }

  /* THRESH_BINARY: pixels strictly brighter than the threshold become foreground */
  function binarize(gray, threshold) {
    const mask = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i++) mask[i] = gray[i] > threshold ? 1 : 0;
    return mask;
  }

  /* Clockwise neighbour order on screen (y grows downwards): E, SE, S, SW, W, NW, N, NE */
  const DR = [0, 1, 1, 1, 0, -1, -1, -1];
  const DC = [1, 1, 0, -1, -1, -1, 0, 1];

  function directionOf(dr, dc) {
    for (let d = 0; d < 8; d++) if (DR[d] === dr && DC[d] === dc) return d;
    return -1;
  }

  /*
   * Suzuki-Abe outer border following (the algorithm behind cv2.findContours),
   * started at the first raster pixel of a component, whose west neighbour is
   * background by construction.
   */
  function traceOuterBorder(mask, width, height, startRow, startCol) {
    const at = (r, c) => (r >= 0 && r < height && c >= 0 && c < width ? mask[r * width + c] : 0);
    const points = [];

    let first = -1;
    for (let k = 0; k < 8; k++) {
      const d = (4 + k) % 8;
      if (at(startRow + DR[d], startCol + DC[d])) {
        first = d;
        break;
      }
    }
    if (first < 0) {
      points.push([startCol, startRow]);
      return points;
    }

    const r1 = startRow + DR[first], c1 = startCol + DC[first];
    let r2 = r1, c2 = c1, r3 = startRow, c3 = startCol;
    const guard = 8 * width * height + 16;
    for (let iter = 0; iter < guard; iter++) {
      let d = directionOf(r2 - r3, c2 - c3);
      let r4 = -1, c4 = -1;
      for (let k = 0; k < 8; k++) {
        d = (d + 7) % 8;
        if (at(r3 + DR[d], c3 + DC[d])) {
          r4 = r3 + DR[d];
          c4 = c3 + DC[d];
          break;
        }
      }
      points.push([c3, r3]);
      if (r4 === startRow && c4 === startCol && r3 === r1 && c3 === c1) break;
      r2 = r3; c2 = c3; r3 = r4; c3 = c4;
    }
    return points;
  }

  function contourArea(points) {
    if (points.length < 3) return 0;
    let a = 0;
    for (let i = 0; i < points.length; i++) {
      const [x0, y0] = points[i];
      const [x1, y1] = points[(i + 1) % points.length];
      a += x0 * y1 - x1 * y0;
    }
    return Math.abs(a) / 2;
  }

  function arcLength(points) {
    if (points.length < 2) return 0;
    let p = 0;
    for (let i = 0; i < points.length; i++) {
      const [x0, y0] = points[i];
      const [x1, y1] = points[(i + 1) % points.length];
      p += Math.hypot(x1 - x0, y1 - y0);
    }
    return p;
  }

  /* Polygon moments as in cv2.moments(contour); returns null when m00 == 0 */
  function contourCentroid(points) {
    let a00 = 0, a10 = 0, a01 = 0;
    for (let i = 0; i < points.length; i++) {
      const [xp, yp] = points[i === 0 ? points.length - 1 : i - 1];
      const [x, y] = points[i];
      const cross = xp * y - x * yp;
      a00 += cross;
      a10 += cross * (xp + x);
      a01 += cross * (yp + y);
    }
    if (Math.abs(a00) <= FLT_EPSILON) return null;
    const m00 = a00 / 2;
    return { m00: Math.abs(m00), cx: a10 / 6 / m00, cy: a01 / 6 / m00 };
  }

  function boundingRect(points) {
    let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
    for (const [x, y] of points) {
      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    return { x: xmin, y: ymin, w: xmax - xmin + 1, h: ymax - ymin + 1 };
  }

  /*
   * Equivalent to max(cv2.findContours(mask, RETR_EXTERNAL, ...), key=contourArea).
   * A contour nested inside a hole always encloses less area than its
   * container, so taking the maximum over all components equals taking it
   * over the external ones.
   */
  function largestExternalContour(mask, width, height) {
    const labels = new Int32Array(width * height);
    const stack = new Int32Array(width * height);
    let best = null;
    let label = 0;

    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const idx = r * width + c;
        if (!mask[idx] || labels[idx]) continue;

        label++;
        let top = 0;
        stack[top++] = idx;
        labels[idx] = label;
        let pixelCount = 0;
        while (top > 0) {
          const cur = stack[--top];
          pixelCount++;
          const cr = (cur / width) | 0, cc = cur - cr * width;
          for (let d = 0; d < 8; d++) {
            const nr = cr + DR[d], nc = cc + DC[d];
            if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
            const nidx = nr * width + nc;
            if (mask[nidx] && !labels[nidx]) {
              labels[nidx] = label;
              stack[top++] = nidx;
            }
          }
        }

        const points = traceOuterBorder(mask, width, height, r, c);
        const area = contourArea(points);
        if (best === null || area > best.area) {
          best = { points, area, pixelCount, label };
        }
      }
    }
    if (best) {
      best.perimeter = arcLength(best.points);
      best.rect = boundingRect(best.points);
      best.labels = labels;
    }
    return best;
  }

  function segment(img) {
    const gray = rgbToGray(img);
    const otsu = otsuThreshold(gray);
    const mask = binarize(gray, otsu.threshold);
    const contour = largestExternalContour(mask, img.width, img.height);
    return { gray, threshold: otsu.threshold, histogram: otsu.histogram, mask, contour };
  }

  /* ExplainabilityEngine.detect_asymmetry */
  function detectAsymmetry(img, seg) {
    seg = seg || segment(img);
    const contour = seg.contour;
    if (!contour) return { score: 0, cx: null, band: 0 };
    const centroid = contourCentroid(contour.points);
    if (!centroid) return { score: 0, cx: null, band: 0 };

    const w = img.width, h = img.height, gray = seg.gray;
    const cx = Math.trunc(centroid.cx);
    const band = Math.min(cx, w - cx);
    if (band <= 0) return { score: 1, cx, band: 0 };

    /* left[:, :band] is compared with fliplr(img[:, cx:])[:, :band],
       i.e. column j against column (w - 1 - j) */
    let sum = 0;
    for (let r = 0; r < h; r++) {
      const row = r * w;
      for (let j = 0; j < band; j++) sum += Math.abs(gray[row + j] - gray[row + w - 1 - j]);
    }
    const score = Math.min(sum / (h * band) / 255, 1);
    return { score, cx, band };
  }

  /* ExplainabilityEngine.detect_border_irregularity */
  function detectBorder(seg) {
    const contour = seg.contour;
    if (!contour || contour.area === 0) return { score: 0, circularity: null };
    const circularity = (4 * Math.PI * contour.area) / (contour.perimeter * contour.perimeter + 1e-10);
    return { score: Math.min(Math.max(1 - circularity, 0), 1), circularity };
  }

  /* ExplainabilityEngine.compute_diameter with the default pixels_per_mm = 10 */
  function computeDiameter(seg, pixelsPerMm) {
    pixelsPerMm = pixelsPerMm || 10;
    if (!seg.contour) return { mm: 0, pixels: 0 };
    const pixels = Math.max(seg.contour.rect.w, seg.contour.rect.h);
    return { mm: pixels / pixelsPerMm, pixels };
  }

  function srgbToLinear(v) {
    v /= 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  }

  /*
   * analyze_color_distribution: the metrics consumed by the ABCDE text.
   * color_variance is np.var over all R, G and B values together.
   * darkness_ratio uses the 8-bit OpenCV L channel (L* scaled by 255/100) < 50;
   * it is computed here in floating point, so pixels on the boundary may differ.
   */
  function colorMetrics(img) {
    const n = img.width * img.height;
    const c = channelsOf(img);
    let sum = 0, sumSq = 0, blue = 0, dark = 0;
    let sr = 0, sg = 0, sb = 0;
    for (let p = 0, q = 0; p < n; p++, q += c) {
      const r = img.data[q], g = img.data[q + 1], b = img.data[q + 2];
      sum += r + g + b;
      sumSq += r * r + g * g + b * b;
      sr += r; sg += g; sb += b;
      if (b > 200) blue++;
      const y = 0.212671 * srgbToLinear(r) + 0.71516 * srgbToLinear(g) + 0.072169 * srgbToLinear(b);
      const lStar = y > 0.008856 ? 116 * Math.cbrt(y) - 16 : 903.3 * y;
      if (Math.round((lStar * 255) / 100) < 50) dark++;
    }
    const count = 3 * n;
    const mean = sum / count;
    return {
      mean_red: sr / n,
      mean_green: sg / n,
      mean_blue: sb / n,
      color_variance: sumSq / count - mean * mean,
      blue_white_presence: blue / n,
      darkness_ratio: dark / n
    };
  }

  /* ExplainabilityEngine.compute_abcde_criteria */
  function computeAbcde(img) {
    const seg = segment(img);
    const asym = detectAsymmetry(img, seg);
    const border = detectBorder(seg);
    const diameter = computeDiameter(seg);
    const color = colorMetrics(img);
    return {
      segmentation: seg,
      asymmetry: { score: asym.score, status: asym.score > 0.3 ? 'High' : 'Low', cx: asym.cx, band: asym.band },
      border: { score: border.score, status: border.score > 0.4 ? 'Irregular' : 'Regular', circularity: border.circularity },
      color: {
        score: Math.min(color.color_variance / 2000, 1),
        variance: color.color_variance,
        status: color.color_variance > 1000 ? 'Multiple colors' : 'Uniform'
      },
      diameter: { value_mm: diameter.mm, pixels: diameter.pixels, status: diameter.mm > 6 ? '>6mm' : '≤6mm' },
      evolution: { status: 'Not assessed (single timepoint)' },
      colorMetrics: color
    };
  }

  /* ImageValidator.validate_image_dimensions */
  function validateDimensions(width, height) {
    if (height < 100 || width < 100) return { ok: false, message: 'Image too small (minimum 100x100 pixels required)' };
    if (height > 10000 || width > 10000) return { ok: false, message: 'Image too large (maximum 10000x10000 pixels)' };
    const ratio = Math.max(width, height) / Math.min(width, height);
    if (ratio > 5) return { ok: false, message: 'Unusual aspect ratio (' + ratio.toFixed(1) + ':1), likely not a medical image' };
    return { ok: true, message: 'Dimensions valid' };
  }

  /*
   * Bilinear resize with OpenCV's half-pixel mapping (cv2.resize, INTER_LINEAR).
   * OpenCV uses 11-bit fixed-point weights, so single channel values may differ by 1.
   */
  function resizeBilinear(img, outW, outH) {
    const c = channelsOf(img);
    const out = new Uint8Array(outW * outH * 3);
    const sx = img.width / outW, sy = img.height / outH;
    for (let dy = 0; dy < outH; dy++) {
      let fy = (dy + 0.5) * sy - 0.5;
      let y0 = Math.floor(fy);
      fy -= y0;
      if (y0 < 0) { y0 = 0; fy = 0; }
      if (y0 >= img.height - 1) { y0 = img.height - 1; fy = 0; }
      const y1 = Math.min(y0 + 1, img.height - 1);
      for (let dx = 0; dx < outW; dx++) {
        let fx = (dx + 0.5) * sx - 0.5;
        let x0 = Math.floor(fx);
        fx -= x0;
        if (x0 < 0) { x0 = 0; fx = 0; }
        if (x0 >= img.width - 1) { x0 = img.width - 1; fx = 0; }
        const x1 = Math.min(x0 + 1, img.width - 1);
        for (let k = 0; k < 3; k++) {
          const v00 = img.data[(y0 * img.width + x0) * c + k];
          const v01 = img.data[(y0 * img.width + x1) * c + k];
          const v10 = img.data[(y1 * img.width + x0) * c + k];
          const v11 = img.data[(y1 * img.width + x1) * c + k];
          const top = v00 * (1 - fx) + v01 * fx;
          const bottom = v10 * (1 - fx) + v11 * fx;
          out[(dy * outW + dx) * 3 + k] = Math.round(top * (1 - fy) + bottom * fy);
        }
      }
    }
    return { width: outW, height: outH, data: out };
  }

  /* Unique-colour gate: the image is resized to 100x100 only when its height exceeds 100 */
  function countUniqueColors(img) {
    const src = img.height > 100 ? resizeBilinear(img, 100, 100) : img;
    const c = channelsOf(src);
    const seen = new Set();
    for (let q = 0; q < src.data.length; q += c) {
      seen.add((src.data[q] << 16) | (src.data[q + 1] << 8) | src.data[q + 2]);
    }
    return { count: seen.size, resized: img.height > 100 };
  }

  return {
    rgbToGray,
    otsuThreshold,
    binarize,
    traceOuterBorder,
    contourArea,
    arcLength,
    contourCentroid,
    boundingRect,
    largestExternalContour,
    segment,
    detectAsymmetry,
    detectBorder,
    computeDiameter,
    colorMetrics,
    computeAbcde,
    validateDimensions,
    resizeBilinear,
    countUniqueColors
  };
});
