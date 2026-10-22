/*
 * Compare docs/lab/heuristics.js with the reference produced by make_reference.py.
 * Usage: node docs/lab/verify/compare.js <directory written by make_reference.py>
 */
'use strict';

const fs = require('fs');
const path = require('path');
const SLH = require('../heuristics.js');

const dir = process.argv[2] || 'slh_verify';
const reference = JSON.parse(fs.readFileSync(path.join(dir, 'reference.json'), 'utf8'));

const TOLERANCE = 1e-6;
let failures = 0;

for (const [name, ref] of Object.entries(reference)) {
  const data = new Uint8Array(fs.readFileSync(path.join(dir, `${name}.bin`)));
  const img = { width: ref.w, height: ref.h, data };
  const result = SLH.computeAbcde(img);
  const seg = result.segmentation;
  const rect = seg.contour.rect;

  const checks = {
    gray: seg.gray.reduce((s, v) => s + v, 0) === ref.gray_sum,
    otsu: seg.threshold === ref.otsu,
    area: Math.abs(seg.contour.area - ref.area) < TOLERANCE,
    perimeter: Math.abs(seg.contour.perimeter - ref.perimeter) < 1e-3,
    rect: [rect.x, rect.y, rect.w, rect.h].join() === ref.rect.join(),
    A: Math.abs(result.asymmetry.score - ref.A) < TOLERANCE,
    B: Math.abs(result.border.score - ref.B) < TOLERANCE,
    C: Math.abs(result.color.variance - ref.C) < 1e-6 * Math.max(1, ref.C),
    D: Math.abs(result.diameter.value_mm - ref.D) < TOLERANCE,
    dark: Math.abs(result.colorMetrics.darkness_ratio - ref.dark) < 0.002
  };
  const unique = SLH.countUniqueColors(img).count;
  const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([k]) => k);
  if (failed.length) failures++;
  console.log(`${name.padEnd(28)} ${failed.length ? 'FAIL ' + failed.join(',') : 'ok'}   unique colours ${unique} (python ${ref.unique})`);
}

console.log(failures ? `\n${failures} image(s) differ` : '\nall images match');
process.exitCode = failures ? 1 : 0;
