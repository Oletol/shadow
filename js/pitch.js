// Pitch (F0) extraction with the YIN algorithm plus post-processing.
// Pure functions, no browser APIs: runs in the page, a worker or Node tests.

export const PITCH_HOP_S = 0.01; // 10 ms frames

/**
 * YIN fundamental-frequency tracker.
 * @param {Float32Array} x  mono samples
 * @param {number} sr       sample rate
 * @returns {{f0: Float32Array, conf: Float32Array, rms: Float32Array, hop: number}}
 *   f0 in Hz (0 = unvoiced), conf = 1 - aperiodicity, rms per frame, hop in seconds
 */
export function yinTrack(x, sr, opts = {}) {
  const fmin = opts.fmin || 65;
  const fmax = opts.fmax || 500;
  const hop = Math.round(sr * (opts.hop || PITCH_HOP_S));
  const W = opts.window || Math.round(sr * 0.032); // integration window
  const maxLag = Math.ceil(sr / fmin);
  const minLag = Math.floor(sr / fmax);
  const thr = opts.threshold || 0.15;
  const nFrames = Math.max(0, Math.floor((x.length - W - maxLag) / hop) + 1);

  const f0 = new Float32Array(nFrames);
  const conf = new Float32Array(nFrames);
  const rms = new Float32Array(nFrames);
  const d = new Float32Array(maxLag + 2);
  const dn = new Float32Array(maxLag + 2);

  // Frame energy (for the voicing gate)
  for (let f = 0; f < nFrames; f++) {
    const s = f * hop;
    let e = 0;
    for (let j = 0; j < W; j++) { const v = x[s + j]; e += v * v; }
    rms[f] = Math.sqrt(e / W);
  }
  let peakRms = 0;
  for (let f = 0; f < nFrames; f++) if (rms[f] > peakRms) peakRms = rms[f];
  const gate = Math.max(1e-4, peakRms * Math.pow(10, -38 / 20)); // -38 dB re: loudest frame

  for (let f = 0; f < nFrames; f++) {
    if (rms[f] < gate) continue;
    const s = f * hop;
    // difference function
    for (let tau = 1; tau <= maxLag; tau++) {
      let sum = 0;
      for (let j = 0; j < W; j++) {
        const diff = x[s + j] - x[s + j + tau];
        sum += diff * diff;
      }
      d[tau] = sum;
    }
    // cumulative mean normalised difference
    dn[0] = 1;
    let running = 0;
    for (let tau = 1; tau <= maxLag; tau++) {
      running += d[tau];
      dn[tau] = running > 0 ? d[tau] * tau / running : 1;
    }
    // absolute threshold, then walk to the local minimum
    let tauEst = -1;
    for (let tau = minLag; tau <= maxLag; tau++) {
      if (dn[tau] < thr) {
        while (tau + 1 <= maxLag && dn[tau + 1] < dn[tau]) tau++;
        tauEst = tau;
        break;
      }
    }
    if (tauEst < 0) {
      // no dip under threshold: take the global minimum if it is still fairly periodic
      let best = minLag;
      for (let tau = minLag + 1; tau <= maxLag; tau++) if (dn[tau] < dn[best]) best = tau;
      if (dn[best] < 0.3) tauEst = best; else continue;
    }
    // parabolic interpolation
    let betterTau = tauEst;
    if (tauEst > 1 && tauEst < maxLag) {
      const a = dn[tauEst - 1], b = dn[tauEst], c = dn[tauEst + 1];
      const den = a - 2 * b + c;
      if (Math.abs(den) > 1e-9) betterTau = tauEst + 0.5 * (a - c) / den;
    }
    const hz = sr / betterTau;
    if (hz >= fmin && hz <= fmax) {
      f0[f] = hz;
      conf[f] = Math.max(0, 1 - dn[tauEst]);
    }
  }
  return { f0, conf, rms, hop: hop / sr };
}

/** Median of the positive values of an array. */
export function voicedMedian(arr) {
  const v = [];
  for (let i = 0; i < arr.length; i++) if (arr[i] > 0) v.push(arr[i]);
  if (!v.length) return 0;
  v.sort((a, b) => a - b);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/**
 * Clean a raw F0 track: octave-error correction against the speaker median,
 * removal of very short voiced islands, 5-point median filter inside voiced runs,
 * and removal of isolated jumps.
 */
export function cleanPitch(raw, opts = {}) {
  const minRun = opts.minRun || 4; // frames (40 ms)
  const out = Float32Array.from(raw);
  const med = voicedMedian(out);
  if (!med) return out;

  // 1. octave correction: fold values that sit ~an octave away from the median
  for (let i = 0; i < out.length; i++) {
    const v = out[i];
    if (!v) continue;
    const st = 12 * Math.log2(v / med);
    if (st > 9.5) out[i] = v / 2;
    else if (st < -10.5) out[i] = v * 2;
  }

  // 2. local jump correction: compare with neighbourhood median
  const tmp = Float32Array.from(out);
  for (let i = 0; i < out.length; i++) {
    if (!tmp[i]) continue;
    const nb = [];
    for (let k = -3; k <= 3; k++) {
      const j = i + k;
      if (k && j >= 0 && j < tmp.length && tmp[j] > 0) nb.push(tmp[j]);
    }
    if (nb.length < 3) continue;
    nb.sort((a, b) => a - b);
    const m = nb[nb.length >> 1];
    const st = 12 * Math.log2(tmp[i] / m);
    if (Math.abs(st) > 9) {
      // try octave fold, otherwise mark unvoiced
      const up = 12 * Math.log2((tmp[i] * 2) / m), dn = 12 * Math.log2((tmp[i] / 2) / m);
      if (Math.abs(up) < 3) out[i] = tmp[i] * 2;
      else if (Math.abs(dn) < 3) out[i] = tmp[i] / 2;
      else out[i] = 0;
    }
  }

  // 3. drop voiced islands shorter than minRun
  let i = 0;
  while (i < out.length) {
    if (!out[i]) { i++; continue; }
    let j = i;
    while (j < out.length && out[j] > 0) j++;
    if (j - i < minRun) for (let k = i; k < j; k++) out[k] = 0;
    i = j;
  }

  // 4. 5-point median filter inside voiced runs
  const src = Float32Array.from(out);
  for (let k = 0; k < out.length; k++) {
    if (!src[k]) continue;
    const w = [];
    for (let m = -2; m <= 2; m++) {
      const j = k + m;
      if (j >= 0 && j < src.length && src[j] > 0) w.push(src[j]);
    }
    w.sort((a, b) => a - b);
    out[k] = w[w.length >> 1];
  }
  return out;
}

/** Convert Hz to semitones relative to the speaker's own median (0 = unvoiced -> NaN). */
export function toSemitones(f0, ref) {
  const med = ref || voicedMedian(f0);
  const st = new Float32Array(f0.length);
  for (let i = 0; i < f0.length; i++) st[i] = f0[i] > 0 && med > 0 ? 12 * Math.log2(f0[i] / med) : NaN;
  return st;
}

/** Fill short unvoiced gaps (<= maxGap frames) by linear interpolation. */
export function bridgeGaps(st, maxGap = 15) {
  const out = Float32Array.from(st);
  let last = -1;
  for (let i = 0; i < out.length; i++) {
    if (Number.isNaN(out[i])) continue;
    if (last >= 0 && i - last > 1 && i - last - 1 <= maxGap) {
      for (let k = last + 1; k < i; k++) {
        const t = (k - last) / (i - last);
        out[k] = out[last] * (1 - t) + out[i] * t;
      }
    }
    last = i;
  }
  return out;
}

/** Full pitch analysis used by the app. */
export function analysePitch(samples, sr) {
  const tr = yinTrack(samples, sr);
  const f0 = cleanPitch(tr.f0);
  const median = voicedMedian(f0);
  const st = toSemitones(f0, median);
  const stBridged = bridgeGaps(st, 15);
  let voiced = 0;
  for (let i = 0; i < f0.length; i++) if (f0[i] > 0) voiced++;
  // frame times are window centres
  const t0 = (Math.round(sr * 0.032) / 2) / sr;
  return {
    f0, st, stBridged, rms: tr.rms, hop: tr.hop, t0, median,
    voicedRatio: f0.length ? voiced / f0.length : 0
  };
}

/** Sample a semitone track at time t (seconds); NaN if unvoiced there. */
export function sampleTrack(track, hop, t0, t) {
  const pos = (t - t0) / hop;
  const i0 = Math.floor(pos);
  if (i0 < 0 || i0 + 1 >= track.length) {
    const i = Math.round(pos);
    return i >= 0 && i < track.length ? track[i] : NaN;
  }
  const a = track[i0], b = track[i0 + 1];
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.isNaN(a) ? b : a;
  const fr = pos - i0;
  return a * (1 - fr) + b * fr;
}
