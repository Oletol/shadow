// Turn a 16 kHz recording (+ optional wav2vec2 log-probs) into an analysis object
// that scoring.js can compare. Pure functions.

import { analysePitch } from './pitch.js';
import { phraseToLabels, ctcForcedAlign, wordsFromAlignment, greedyDecode } from './align.js';

export const SR = 16000;

/** Find where speech starts and ends using frame energy (returns seconds). */
export function speechSpan(x, sr = SR) {
  const hop = Math.round(sr * 0.01), win = Math.round(sr * 0.025);
  const n = Math.max(1, Math.floor((x.length - win) / hop) + 1);
  const db = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let e = 0;
    for (let j = 0; j < win; j++) { const v = x[f * hop + j] || 0; e += v * v; }
    db[f] = 10 * Math.log10(e / win + 1e-12);
  }
  const sorted = Array.from(db).sort((a, b) => a - b);
  const floor = sorted[Math.floor(n * 0.1)];
  const peak = sorted[n - 1];
  const thr = Math.max(floor + 12, peak - 40);
  let a = 0, b = n - 1;
  // require 3 consecutive frames above threshold to start/end
  while (a < n - 3 && !(db[a] > thr && db[a + 1] > thr && db[a + 2] > thr)) a++;
  while (b > 2 && !(db[b] > thr && db[b - 1] > thr && db[b - 2] > thr)) b--;
  if (b <= a) { a = 0; b = n - 1; }
  const start = Math.max(0, a * hop / sr - 0.04);
  const end = Math.min(x.length / sr, (b * hop + win) / sr + 0.04);
  return [start, end];
}

/** Peak-normalise to -1 dBFS without changing the shape. */
export function normalisePeak(x) {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  if (!m) return Float32Array.from(x);
  const g = 0.89 / m;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * g;
  return out;
}

/**
 * @param {Float32Array} samples 16 kHz mono
 * @param {string} text target phrase
 * @param {{logp: Float32Array, T: number, V: number}|null} ctc log-probabilities from wav2vec2
 */
export function buildAnalysis(samples, text, ctc) {
  const x = normalisePeak(samples);
  const duration = x.length / SR;
  let span = speechSpan(x, SR);
  const pitch = analysePitch(x, SR);

  let words = null, acoustic = null, alignOk = false;
  if (ctc && ctc.T > 0) {
    const frameSec = duration / ctc.T;
    acoustic = greedyDecode(ctc.logp, ctc.T, ctc.V);
    const lab = phraseToLabels(text);
    const al = ctcForcedAlign(ctc.logp, ctc.T, ctc.V, lab.labels);
    if (al) {
      words = wordsFromAlignment(ctc.logp, ctc.T, ctc.V, lab, al, frameSec);
      if (words.every((w) => Number.isFinite(w.start) && Number.isFinite(w.end))) {
        alignOk = true;
        // tighten the speech span with the word boundaries
        // (energy span limited to a small margin around the aligned words, so that
        // breaths and clicks before/after the phrase are ignored, but the pitch
        // tail of the last word is kept)
        const first = words[0].start, last = words[words.length - 1].end;
        span = [
          Math.min(Math.max(span[0], first - 0.15), first),
          Math.max(Math.min(span[1], last + 0.35), last)
        ];
      } else words = null;
    }
  }
  return { samples: x, duration, span, pitch, words, acoustic, alignOk };
}
