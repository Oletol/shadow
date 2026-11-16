// Comparison of a student recording with the model recording.
// Intonation is compared frame by frame on the model's time axis after
// word-anchored time alignment; sounds are scored by the phoneme model (phonemes.js),
// the wav2vec2 letter model only gives the word boundaries.

import { sampleTrack } from './pitch.js';
import { buildTimeMap, alignWordLists, splitWords, envelopeTimeMap } from './align.js';

// Version of the scoring method. It is saved with every attempt, so that results
// collected under different versions are never mixed in an analysis.
// Change it whenever a formula, a weight or a threshold changes.
export const SCORING_VERSION = '2026.11.15b';

export const STRICTNESS = {
  lenient:  { sigma: 4.5, label: 'Lenient' },
  standard: { sigma: 3.2, label: 'Standard' },
  strict:   { sigma: 2.3, label: 'Strict' }
};

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const mean = (a) => a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN;
const round = (v) => Math.round(v);

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  if (n < 3) return NaN;
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return da && db ? num / Math.sqrt(da * db) : NaN;
}

function percentile(arr, p) {
  const v = arr.filter(Number.isFinite).sort((x, y) => x - y);
  if (!v.length) return NaN;
  const idx = clamp((v.length - 1) * p, 0, v.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

function smooth(arr, r = 2) {
  const out = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    let s = 0, c = 0;
    for (let k = -r; k <= r; k++) {
      const v = arr[i + k];
      if (Number.isFinite(v)) { s += v; c++; }
    }
    out[i] = c ? s / c : NaN;
  }
  return out;
}

/**
 * Classify a pitch movement (semitone values over time) into a nuclear-tone label.
 * Returns {tone, start, end, min, max, span}.
 */
export function classifyTone(values) {
  const v = smooth(values.filter(Number.isFinite), 2);
  if (v.length < 6) return { tone: 'unclear', span: 0 };
  const k = Math.max(2, Math.round(v.length * 0.2));
  const s = mean(v.slice(0, k)), e = mean(v.slice(-k));
  let mn = Infinity, mx = -Infinity, pmn = 0, pmx = 0;
  v.forEach((x, i) => {
    if (x < mn) { mn = x; pmn = i / (v.length - 1); }
    if (x > mx) { mx = x; pmx = i / (v.length - 1); }
  });
  const span = mx - mn;
  let tone;
  if (span < 1.5) tone = 'level';
  else if (pmn > 0.2 && pmn < 0.8 && s - mn >= 1.2 && e - mn >= 1.2) tone = 'fall-rise';
  else if (pmx > 0.2 && pmx < 0.8 && mx - s >= 1.2 && mx - e >= 1.2) tone = 'rise-fall';
  else if (e - s <= -1.2) tone = 'fall';
  else if (e - s >= 1.2) tone = 'rise';
  else tone = 'level';
  return { tone, start: s, end: e, min: mn, max: mx, span };
}

function toneAgreement(a, b) {
  if (a === 'unclear' || b === 'unclear') return 50;
  if (a === b) return 100;
  const ending = (t) => (t === 'fall' || t === 'rise-fall') ? 'down' : (t === 'rise' || t === 'fall-rise') ? 'up' : 'flat';
  if (ending(a) === ending(b)) return 60;
  if (ending(a) === 'flat' || ending(b) === 'flat') return 30;
  return 0;
}

function movementClass(vals) {
  const v = vals.filter(Number.isFinite);
  if (v.length < 6) return null;
  const k = Math.max(2, Math.floor(v.length / 3));
  const d = mean(v.slice(-k)) - mean(v.slice(0, k));
  return { delta: d, cls: d > 1.5 ? 'rise' : d < -1.5 ? 'fall' : 'level' };
}

function frameTimes(p) {
  return (i) => p.t0 + i * p.hop;
}

function dbTrack(rms) {
  const out = new Float32Array(rms.length);
  for (let i = 0; i < rms.length; i++) out[i] = 20 * Math.log10(rms[i] + 1e-6);
  return out;
}

/**
 * Syllable nuclei of a recording, found from the loudness curve of the voiced
 * stretches: every loudness peak that is separated from its neighbour by a dip of
 * at least 3 dB (or by a voiceless stretch) is one syllable. No speech model is
 * needed, so this also works for humming and before the models are loaded.
 * Returns [{start, end, peak, peakDb, dur}] in seconds.
 */
export function findSyllables(p, span) {
  const n = p.st.length;
  const i0 = Math.max(0, Math.floor((span[0] - p.t0) / p.hop));
  const i1 = Math.min(n - 1, Math.ceil((span[1] - p.t0) / p.hop));
  if (i1 - i0 < 4) return [];
  const db = smooth(Array.from(dbTrack(p.rms)), 2);
  // voiced frames, with gaps of up to 3 frames closed
  const voiced = new Uint8Array(n);
  for (let i = i0; i <= i1; i++) voiced[i] = Number.isFinite(p.st[i]) ? 1 : 0;
  for (let i = i0; i <= i1; i++) {
    if (voiced[i]) continue;
    let j = i;
    while (j <= i1 && !voiced[j]) j++;
    if (i > i0 && j <= i1 && j - i <= 3) for (let k = i; k < j; k++) voiced[k] = 1;
    i = j;
  }
  const out = [];
  let top = -Infinity;
  for (let i = i0; i <= i1; i++) if (voiced[i]) top = Math.max(top, db[i]);
  const push = (a, b) => {
    if (b - a + 1 < 4) return;
    let pk = a;
    for (let i = a; i <= b; i++) if (db[i] > db[pk]) pk = i;
    if (db[pk] < top - 25) return;
    out.push({ start: p.t0 + a * p.hop, end: p.t0 + b * p.hop, peak: p.t0 + pk * p.hop, peakDb: db[pk], dur: (b - a + 1) * p.hop });
  };
  let i = i0;
  while (i <= i1) {
    if (!voiced[i]) { i++; continue; }
    let j = i;
    while (j + 1 <= i1 && voiced[j + 1]) j++;
    // split the voiced stretch i..j at loudness dips
    let segStart = i, peak = i, valley = -1;
    for (let k = i + 1; k <= j; k++) {
      if (valley < 0) {
        if (db[k] > db[peak]) peak = k;
        else if (db[peak] - db[k] >= 3) valley = k;
      } else {
        if (db[k] < db[valley]) valley = k;
        else if (db[k] - db[valley] >= 3 && k - peak >= 8) {
          push(segStart, valley - 1);
          segStart = valley; peak = k; valley = -1;
        }
      }
    }
    push(segStart, j);
    i = j + 1;
  }
  return out;
}

// Overall slope of the phrase, in semitones from its first to its last compared frame.
// It is measured on the frames where both the speaker and the student have a pitch
// value (the same frames as the rest of the comparison, so a recording compared with
// itself gives exactly 0), as the slope of a line through the DIFFERENCE between the
// two pitch lines: `tilt`. A melody repeated in another voice, at another speed or
// with a delay has a tilt near 0; a rise of 10 semitones laid over the phrase has a
// tilt of about 8-10. The line is the median of the slopes between pairs of frames
// (Theil-Sen), so single wrong pitch values do not move it.
// A tilt up to `free` semitones costs nothing (an alignment error of 80 ms alone gives
// up to about 3); beyond it the part falls with tolerance `tol` x the strictness
// sigma. The part also limits the Melody score: the lower it is, the closer the limit
// comes to `capBase`; a phrase that goes the other way than the speaker's (each by
// `level` semitones or more) cannot get more than `capOpposite`, whatever the
// frame-by-frame tolerance lets through.
export const TREND = { minFrames: 8, minSeconds: 0.2, free: 3, tol: 1.15, weight: 0.15, capBase: 50, capOpposite: 60, level: 2 };

/** Robust slope of y over t: the median of the slopes between pairs of points. */
function robustSlope(t, y) {
  const step = Math.max(1, Math.floor(t.length / 150));
  const slopes = [];
  for (let i = 0; i < t.length; i += step) {
    for (let j = i + step; j < t.length; j += step) {
      const dt = t[j] - t[i];
      if (dt >= 0.08) slopes.push((y[j] - y[i]) / dt);
    }
  }
  return slopes.length ? percentile(slopes, 0.5) : NaN;
}

/** z-scores of a list (0 when all values are equal). */
function zscores(arr) {
  const m = mean(arr), sd = Math.sqrt(mean(arr.map((v) => (v - m) ** 2)));
  return arr.map((v) => sd > 1e-9 ? (v - m) / sd : 0);
}

/**
 * The nucleus is the last prominent syllable of the phrase. Prominence combines
 * loudness, length and the size of the pitch movement on the syllable.
 * Returns the index in `syl`, or -1.
 */
function findNucleus(syl) {
  const ok = syl.filter((s) => s.n >= 3);
  if (!ok.length) return -1;
  const zl = zscores(ok.map((s) => s.db)), zd = zscores(ok.map((s) => s.dur)), zp = zscores(ok.map((s) => s.pitchSpan));
  ok.forEach((s, k) => { s.prominence = 0.5 * zl[k] + 0.5 * zd[k] + zp[k]; });
  for (let k = ok.length - 1; k >= 0; k--) if (ok[k].prominence >= 0) return syl.indexOf(ok[k]);
  return syl.indexOf(ok[ok.length - 1]);
}

/** Intonation comparison. ref/stu: {pitch, span, words|null}. */
/** How far (seconds) a pitch movement may be early or late without being counted as an error. */
export const TIME_TOLERANCE_S = 0.07;

export function compareIntonation(ref, stu, map, strictness = 'standard') {
  const sigma0 = (STRICTNESS[strictness] || STRICTNESS.standard).sigma;
  const rp = ref.pitch, sp = stu.pitch;
  const tR = frameTimes(rp);
  const iFrom = Math.max(0, Math.floor((ref.span[0] - rp.t0) / rp.hop));
  const iTo = Math.min(rp.st.length - 1, Math.ceil((ref.span[1] - rp.t0) / rp.hop));

  // refVals / stuValsRaw are what the chart draws; refScored leaves out the frames
  // where the speaker's pitch is not reliable (creaky voice, octave corrections).
  const refVals = [], refScored = [], stuValsRaw = [], times = [];
  let refVoiced = 0, refUnreliable = 0;
  const sampleStu = (t) => sampleTrack(sp.stBridged, sp.hop, sp.t0, map(t));
  for (let i = iFrom; i <= iTo; i++) {
    const t = tR(i);
    const r = rp.st[i];
    const trusted = !rp.reliable || rp.reliable[i] === 1;
    times.push(t);
    refVals.push(r);
    refScored.push(trusted ? r : NaN);
    stuValsRaw.push(sampleStu(t));
    if (Number.isFinite(r)) { refVoiced++; if (!trusted) refUnreliable++; }
  }
  const firstR = [], firstS = [];
  refScored.forEach((r, k) => {
    if (Number.isFinite(r) && Number.isFinite(stuValsRaw[k])) { firstR.push(r); firstS.push(stuValsRaw[k]); }
  });
  const scoredVoiced = refVoiced - refUnreliable;
  const coverage = scoredVoiced ? firstR.length / scoredVoiced : 0;
  if (firstR.length < 8) {
    return { ok: false, score: null, coverage, times, refVals, stuVals: stuValsRaw.map(() => NaN),
      reason: 'Too little voiced speech could be matched between the two recordings.' };
  }
  const offset = mean(firstS.map((s, k) => s - firstR[k]));
  const stuVals = stuValsRaw.map((s) => Number.isFinite(s) ? s - offset : NaN);

  // Timing tolerance: a movement made slightly early or late is the same melody.
  // Each speaker frame is compared with the closest student value within +-70 ms.
  const shifts = [];
  for (let d = -TIME_TOLERANCE_S; d <= TIME_TOLERANCE_S + 1e-9; d += 0.01) shifts.push(d);
  const stuNear = refScored.map((r, k) => {
    if (!Number.isFinite(r)) return NaN;
    let best = NaN, bd = Infinity;
    for (const d of shifts) {
      const v = d === 0 ? stuValsRaw[k] : sampleStu(times[k] + d);
      if (!Number.isFinite(v)) continue;
      const diff = Math.abs(v - offset - r);
      if (diff < bd) { bd = diff; best = v - offset; }
    }
    return best;
  });
  const pairsR = [], pairsS = [];
  refScored.forEach((r, k) => {
    if (Number.isFinite(r) && Number.isFinite(stuNear[k])) { pairsR.push(r); pairsS.push(stuNear[k]); }
  });
  const diffs = pairsS.map((s, k) => s - pairsR[k]);
  const rmse = Math.sqrt(mean(diffs.map((d) => d * d)));
  const rmseExact = Math.sqrt(mean(firstS.map((s, k) => (s - offset - firstR[k]) ** 2)));
  const r = pearson(pairsR, pairsS);

  // The student is measured on the same frames as the speaker, so that a recording
  // compared with itself gives identical numbers on both sides.
  const stuScored = stuVals.map((v, k) => Number.isFinite(refScored[k]) ? v : NaN);
  const rangeRef = percentile(pairsR, 0.9) - percentile(pairsR, 0.1);
  const rangeStu = percentile(pairsS, 0.9) - percentile(pairsS, 0.1);
  const rangeRatio = rangeRef > 0 && rangeStu > 0 ? Math.min(rangeRef, rangeStu) / Math.max(rangeRef, rangeStu) : 0;
  // A speaker with a wide pitch range is harder to follow to the semitone:
  // the tolerance grows with the range (x1 at 7 semitones, at most x1.6).
  // On a phrase with little pitch movement the tolerance shrinks (down to x0.5),
  // otherwise any melody, even the opposite one, would fit inside it.
  const rangeFactor = clamp(rangeRef / 7, 0.5, 1.6);
  const sigma = sigma0 * rangeFactor;

  // Overall slope of the phrase (see TREND), on the frames scored on both sides.
  let trendRef = NaN, trendStu = NaN, tilt = NaN, sTrend = null, trendOpposite = false;
  {
    const tt = [], rr = [], dd = [];
    refScored.forEach((r, k) => {
      if (Number.isFinite(r) && Number.isFinite(stuVals[k])) { tt.push(times[k]); rr.push(r); dd.push(stuVals[k] - r); }
    });
    if (tt.length >= TREND.minFrames && percentile(tt, 0.9) - percentile(tt, 0.1) >= TREND.minSeconds) {
      const len = tt[tt.length - 1] - tt[0];
      trendRef = robustSlope(tt, rr) * len;
      tilt = robustSlope(tt, dd) * len;
    }
    if (Number.isFinite(tilt) && Number.isFinite(trendRef)) {
      trendStu = trendRef + tilt;
      const d = Math.max(0, Math.abs(tilt) - TREND.free);
      sTrend = 100 * Math.exp(-((d / (TREND.tol * sigma0)) ** 2));
      trendOpposite = trendRef * trendStu < 0 && Math.min(Math.abs(trendRef), Math.abs(trendStu)) >= TREND.level;
    }
  }

  // per word
  const words = [];
  if (ref.words) {
    ref.words.forEach((w, wi) => {
      const rv = [], sv = [], svTrusted = [];
      const rs = [], sn = [];
      times.forEach((t, k) => {
        if (t >= w.start && t <= w.end) {
          rv.push(refVals[k]); sv.push(stuVals[k]); rs.push(refScored[k]); sn.push(stuNear[k]);
          // a jump of the pitch tracker in the student's recording must not pass for the highest point
          const j = Math.round((map(t) - sp.t0) / sp.hop);
          svTrusted.push(!sp.reliable || sp.reliable[j] === 1 ? stuVals[k] : NaN);
        }
      });
      const mr = movementClass(rs), ms = movementClass(sv);
      const both = rs.map((x, k) => [x, sn[k]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      const wRmse = both.length >= 3 ? Math.sqrt(mean(both.map(([x, y]) => (x - y) ** 2))) : NaN;
      const lvlR = mean(rv.filter(Number.isFinite)), lvlS = mean(sv.filter(Number.isFinite));
      words.push({
        index: wi, word: w.word,
        refMove: mr, stuMove: ms,
        levelDiff: Number.isFinite(lvlR) && Number.isFinite(lvlS) ? lvlS - lvlR : NaN,
        rmse: wRmse,
        peakRef: rv.some(Number.isFinite) ? Math.max(...rv.filter(Number.isFinite)) : NaN,
        peakStu: svTrusted.some(Number.isFinite) ? Math.max(...svTrusted.filter(Number.isFinite)) : NaN,
        score: Number.isFinite(wRmse) ? round(100 * Math.exp(-((wRmse / sigma) ** 2))) : null
      });
    });
  }

  // Syllables of the speaker; the student's values are read at the same places
  // through the time map.
  const finite = (a) => a.filter(Number.isFinite);
  const third = (a) => { const k = Math.max(1, Math.floor(a.length / 3)); return mean(a.slice(-k)) - mean(a.slice(0, k)); };
  const stuDbAt = (t) => { const v = sampleTrack(sp.rms, sp.hop, sp.t0, map(t)); return Number.isFinite(v) ? 20 * Math.log10(v + 1e-6) : NaN; };
  const syllables = findSyllables(rp, ref.span).map((sy) => {
    const r = [], sN = [], sAll = [];
    times.forEach((t, k) => {
      if (t < sy.start || t > sy.end) return;
      if (Number.isFinite(refScored[k]) && Number.isFinite(stuNear[k])) { r.push(refScored[k]); sN.push(stuNear[k]); }
      if (Number.isFinite(stuVals[k])) sAll.push(stuVals[k]);
    });
    const rAll = finite(times.map((t, k) => (t >= sy.start && t <= sy.end ? refScored[k] : NaN)));
    const word = ref.words ? ref.words.findIndex((w) => sy.peak >= w.start && sy.peak <= w.end) : -1;
    return {
      start: sy.start, end: sy.end, peak: sy.peak, word, n: r.length,
      levelRef: r.length ? percentile(r, 0.5) : NaN, levelStu: sN.length ? percentile(sN, 0.5) : NaN,
      moveRef: r.length >= 6 ? third(r) : NaN, moveStu: sN.length >= 6 ? third(sN) : NaN,
      ref: { db: sy.peakDb, dur: sy.dur, pitchSpan: rAll.length ? Math.max(...rAll) - Math.min(...rAll) : 0, n: rAll.length },
      stu: { db: stuDbAt(sy.peak), dur: map(sy.end) - map(sy.start), pitchSpan: sAll.length ? Math.max(...sAll) - Math.min(...sAll) : 0, n: sAll.length }
    };
  });
  // Melody by syllables: level of every syllable (and its movement when the syllable
  // is long enough) against the speaker's. Small shifts in time do not change it.
  const sylUsed = syllables.filter((sy) => sy.n >= 3);
  let sylRmse = NaN;
  if (sylUsed.length >= 2) {
    const wsum = sylUsed.reduce((a, sy) => a + sy.n, 0);
    const off = sylUsed.reduce((a, sy) => a + sy.n * (sy.levelStu - sy.levelRef), 0) / wsum;
    sylRmse = Math.sqrt(sylUsed.reduce((a, sy) => {
      const d = sy.levelStu - sy.levelRef - off;
      const m = Number.isFinite(sy.moveRef) && Number.isFinite(sy.moveStu) ? sy.moveStu - sy.moveRef : 0;
      sy.diff = d; sy.moveDiff = m;
      return a + sy.n * (d * d + 0.5 * m * m);
    }, 0) / wsum);
  }

  // Nucleus: the last prominent syllable. The nuclear tone runs from the start of
  // that syllable to the end of the phrase (nucleus and tail).
  const refSyl = syllables.map((sy) => ({ ...sy.ref }));
  const stuSyl = syllables.map((sy) => ({ ...sy.stu }));
  const nucIdx = findNucleus(refSyl), nucIdxStu = findNucleus(stuSyl);
  let nucStart = ref.span[0] + (ref.span[1] - ref.span[0]) * 0.6;
  let nucWord = null, nucBy = 'position';
  const framesFrom = (t0) => times.reduce((c, t, k) => c + (t >= t0 && Number.isFinite(refScored[k]) ? 1 : 0), 0);
  if (nucIdx >= 0 && framesFrom(syllables[nucIdx].start) >= 6) {
    nucStart = syllables[nucIdx].start; nucBy = 'syllable';
    if (ref.words && syllables[nucIdx].word >= 0) nucWord = ref.words[syllables[nucIdx].word].word;
  } else if (ref.words) {
    for (let wi = ref.words.length - 1; wi >= 0; wi--) {
      const w = ref.words[wi];
      const n = times.reduce((c, t, k) => c + (t >= w.start && t <= w.end && Number.isFinite(refScored[k]) ? 1 : 0), 0);
      if (n >= 8) { nucStart = w.start; nucWord = w.word; nucBy = 'word'; break; }
    }
  }
  // Both sides are classified on the same frames (those heard in both recordings).
  const nr = [], ns = [];
  times.forEach((t, k) => {
    if (t >= nucStart && Number.isFinite(refScored[k]) && Number.isFinite(stuNear[k])) { nr.push(refScored[k]); ns.push(stuNear[k]); }
  });
  const toneRef = classifyTone(nr), toneStu = classifyTone(ns);
  // A nuclear movement that stays within one semitone of the speaker's is the same
  // tone, even if the two labels fall on different sides of a threshold.
  let nucRmse = NaN;
  if (nr.length >= 6) {
    const off = mean(ns.map((v, k) => v - nr[k]));
    nucRmse = Math.sqrt(mean(ns.map((v, k) => (v - nr[k] - off) ** 2)));
    if (nucRmse <= 1 && toneStu.tone !== toneRef.tone) toneStu.tone = toneRef.tone;
  }
  const nucWordStu = ref.words && nucIdxStu >= 0 && syllables[nucIdxStu].word >= 0 ? ref.words[syllables[nucIdxStu].word].word : null;

  // highest-pitched word (main accent / prominence by pitch)
  let peakWordRef = null, peakWordStu = null;
  if (words.length) {
    const byR = words.filter((w) => Number.isFinite(w.peakRef)).sort((a, b) => b.peakRef - a.peakRef);
    const byS = words.filter((w) => Number.isFinite(w.peakStu)).sort((a, b) => b.peakStu - a.peakStu);
    peakWordRef = byR[0] ? byR[0].index : null;
    peakWordStu = byS[0] ? byS[0].index : null;
  }

  const withMoves = words.filter((w) => w.refMove && w.stuMove);
  const moveAgree = withMoves.length ? withMoves.filter((w) => w.refMove.cls === w.stuMove.cls).length / withMoves.length : NaN;

  const sRmse = 100 * Math.exp(-((rmse / sigma) ** 2));
  const sShape = Number.isFinite(r) ? 100 * clamp((r - 0.1) / 0.8, 0, 1) : 50;
  // When the tone cannot be classified on either side there is nothing to compare:
  // the tone part is then left out instead of counting as half right.
  const toneKnown = !(toneRef.tone === 'unclear' && toneStu.tone === 'unclear');
  const sTone = toneAgreement(toneRef.tone, toneStu.tone);
  const sMove = Number.isFinite(moveAgree) ? 100 * moveAgree : sShape;
  const sRange = 100 * rangeRatio;
  const sSyll = Number.isFinite(sylRmse) ? 100 * Math.exp(-((sylRmse / sigma) ** 2)) : null;
  const partsW = sSyll != null
    ? [[sRmse, 0.20], [sSyll, 0.20], [sShape, 0.18], [sTone, toneKnown ? 0.22 : 0], [sMove, 0.10], [sRange, 0.10]]
    : [[sRmse, 0.38], [sShape, 0.20], [sTone, toneKnown ? 0.22 : 0], [sMove, 0.10], [sRange, 0.10]];
  if (sTrend != null) partsW.push([sTrend, TREND.weight]);
  let score = partsW.reduce((a, [v, w]) => a + v * w, 0) / partsW.reduce((a, [, w]) => a + w, 0);
  // a phrase that as a whole goes the other way than the speaker's is not the same melody
  let cappedByTrend = false;
  if (sTrend != null) {
    const cap = Math.min(TREND.capBase + (100 - TREND.capBase) * sTrend / 100, trendOpposite ? TREND.capOpposite : 100);
    if (score > cap) { score = cap; cappedByTrend = true; }
  }
  if (coverage < 0.35) score = Math.min(score, 60);

  return {
    ok: true, score: round(score), coverage, rmse, rmseExact, correlation: r, offset,
    rangeFactor, unreliableShare: refVoiced ? refUnreliable / refVoiced : 0,
    rangeRef, rangeStu, rangeRatio, trendRef, trendStu, tilt, trendOpposite, cappedByTrend,
    toneRef, toneStu, nucWord, nucStart, nucBy, nucWordStu,
    syllables, sylRmse, nucRmse, nucleusSyllable: nucIdx, nucleusSyllableStu: nucIdxStu,
    peakWordRef, peakWordStu, moveAgree,
    parts: { contour: round(sRmse), syllables: sSyll != null ? round(sSyll) : null, shape: round(sShape), tone: toneKnown ? sTone : null, movement: round(sMove), range: round(sRange), trend: sTrend != null ? round(sTrend) : null },
    words, times, refVals, stuVals
  };
}

/** Timing, pauses and tempo. */
export function compareRhythm(ref, stu, expectedTempo = 1) {
  const refDur = ref.span[1] - ref.span[0];
  const stuDur = stu.span[1] - stu.span[0];
  // expectedTempo > 1 when the student shadows the speaker played slowly
  const tempo = stuDur / refDur / expectedTempo;
  const sTempo = 100 * Math.exp(-((Math.log(tempo) / 0.35) ** 2));
  if (!ref.words || !stu.words) {
    return { score: round(sTempo), tempo, refDur, stuDur, words: [], pauses: [] };
  }
  const n = Math.min(ref.words.length, stu.words.length);
  const dR = [], dS = [];
  for (let i = 0; i < n; i++) {
    dR.push(Math.max(0.03, ref.words[i].end - ref.words[i].start));
    dS.push(Math.max(0.03, stu.words[i].end - stu.words[i].start));
  }
  const totR = dR.reduce((a, b) => a + b, 0), totS = dS.reduce((a, b) => a + b, 0);
  const words = dR.map((d, i) => {
    const ratio = (dS[i] / totS) / (d / totR); // relative length, tempo removed
    return { index: i, word: ref.words[i].word, refDur: d, stuDur: dS[i], ratio };
  });
  const mLog = mean(words.map((w) => Math.abs(Math.log(w.ratio))));
  const sDur = 100 * Math.exp(-((mLog / 0.3) ** 2));

  const pauses = [];
  for (let i = 0; i < n - 1; i++) {
    const gr = ref.words[i + 1].start - ref.words[i].end;
    const gs = stu.words[i + 1].start - stu.words[i].end;
    pauses.push({ after: ref.words[i].word, ref: gr > 0.15, stu: gs > 0.15 * tempo, refGap: gr, stuGap: gs });
  }
  const sPause = pauses.length ? 100 * pauses.filter((p) => p.ref === p.stu).length / pauses.length : 100;
  const score = 0.5 * sDur + 0.3 * sTempo + 0.2 * sPause;
  return { score: round(score), tempo, refDur, stuDur, words, pauses, parts: { durations: round(sDur), tempo: round(sTempo), pauses: round(sPause) } };
}

/** Word prominence (energy + pitch) and stress position inside longer words. */
export function compareStress(ref, stu, map) {
  if (!ref.words || !stu.words) return { score: null, words: [] };
  function prominence(side, words) {
    const p = side.pitch;
    const db = dbTrack(p.rms);
    return words.map((w) => {
      const i0 = Math.max(0, Math.floor((w.start - p.t0) / p.hop));
      const i1 = Math.min(db.length - 1, Math.ceil((w.end - p.t0) / p.hop));
      let best = -Infinity, bestPos = 0.5, maxSt = -Infinity;
      const curve = [];
      for (let i = i0; i <= i1; i++) {
        const st = Number.isFinite(p.st[i]) ? p.st[i] : 0;
        curve.push(db[i] + 0.8 * st);
        if (Number.isFinite(p.st[i])) maxSt = Math.max(maxSt, p.st[i]);
      }
      const sm = smooth(curve, 3);
      sm.forEach((v, k) => { if (v > best) { best = v; bestPos = sm.length > 1 ? k / (sm.length - 1) : 0.5; } });
      return { level: best, peakPos: bestPos, dur: w.end - w.start };
    });
  }
  const pr = prominence(ref, ref.words), ps = prominence(stu, stu.words);
  const n = Math.min(pr.length, ps.length);
  const zs = (arr) => {
    const m = mean(arr), sd = Math.sqrt(mean(arr.map((v) => (v - m) ** 2))) || 1;
    return arr.map((v) => (v - m) / sd);
  };
  const zr = zs(pr.slice(0, n).map((x) => x.level)), zt = zs(ps.slice(0, n).map((x) => x.level));
  const strongR = zr.indexOf(Math.max(...zr)), strongS = zt.indexOf(Math.max(...zt));
  const words = [];
  for (let i = 0; i < n; i++) {
    const long = pr[i].dur > 0.28 && ref.words[i].word.length >= 5;
    words.push({
      index: i, word: ref.words[i].word, zRef: zr[i], zStu: zt[i],
      peakPosRef: pr[i].peakPos, peakPosStu: ps[i].peakPos,
      internalMismatch: long && Math.abs(pr[i].peakPos - ps[i].peakPos) > 0.3
    });
  }
  const r = pearson(zr, zt);
  const sCorr = Number.isFinite(r) ? 100 * clamp((r + 0.1) / 0.9, 0, 1) : 60;
  const sMain = strongR === strongS ? 100 : 30;
  const longWords = words.filter((w) => pr[w.index].dur > 0.28 && w.word.length >= 5);
  const sInternal = longWords.length ? 100 * longWords.filter((w) => !w.internalMismatch).length / longWords.length : 100;
  const score = n >= 3 ? 0.4 * sCorr + 0.35 * sMain + 0.25 * sInternal : 0.6 * sMain + 0.4 * sInternal;
  return { score: round(score), words, strongRef: strongR, strongStu: strongS, correlation: r };
}

/**
 * Sounds. The score comes from the phoneme model alone: goodness of pronunciation
 * measured against the speaker, plus the map of expected errors (see phonemes.js).
 * The letter model only gives the word boundaries; its confidences are not scored.
 * Without the phoneme comparison there is no Sounds score at all.
 */
export function comparePronunciation(ref, stu, phon) {
  if (!ref.words) return { score: null, words: [], phonemes: false };
  const have = !!(phon && phon.score != null);
  const words = ref.words.map((rw, i) => {
    const pw = have ? phon.words[i] : null;
    const w = { index: i, word: rw.word, score: pw && pw.score != null ? pw.score : null, issues: pw ? pw.issues : [] };
    if (pw) { w.ipaRef = pw.ref.join(''); w.ipaStu = pw.stu.join(''); w.phonItems = pw.items; }
    return w;
  });
  return { score: have ? round(phon.score) : null, words, phonemes: have };
}

/** Did the student say the right words? Uses the Whisper transcript. */
export function compareWords(target, whisperText, acousticText) {
  const t = splitWords(target);
  const w = whisperText != null ? alignWordLists(t, splitWords(whisperText)) : null;
  const a = acousticText != null ? alignWordLists(t, splitWords(acousticText)) : null;
  // Only the recogniser (with its language model) decides whether words are missing;
  // the letter-by-letter acoustic transcript is shown for diagnosis only.
  return { target: t, whisper: w, acoustic: a, accuracy: w ? w.accuracy : null };
}

export function compareAll(ref, stu, opts = {}) {
  const melodyOnly = opts.mode === 'melody';
  const haveWords = !melodyOnly && ref.words && stu.words;
  const map = haveWords
    ? buildTimeMap(ref.words, stu.words, ref.span, stu.span)
    : envelopeTimeMap(ref.pitch, stu.pitch, ref.span, stu.span);
  const refW = haveWords ? ref : { ...ref, words: melodyOnly ? null : ref.words };
  const stuW = haveWords ? stu : { ...stu, words: null };
  const intonation = compareIntonation(melodyOnly ? { ...ref, words: ref.words } : refW, melodyOnly ? { ...stu, words: null } : stuW, map, opts.strictness);
  const rhythm = compareRhythm(haveWords ? ref : { ...ref, words: null }, stuW, opts.expectedTempo || 1);
  const empty = { score: null, words: [] };
  const stress = melodyOnly ? empty : compareStress(ref, stu, map);
  const pron = melodyOnly ? empty : comparePronunciation(ref, stu, opts.phonemes);
  const words = melodyOnly ? compareWords(opts.target || '', null, null) : compareWords(opts.target || '', opts.whisperText, opts.acousticText);

  const parts = melodyOnly
    ? [['intonation', intonation.score, 0.8], ['rhythm', rhythm.score, 0.2]]
    : [['intonation', intonation.score, 0.40], ['pronunciation', pron.score, 0.30], ['rhythm', rhythm.score, 0.15], ['stress', stress.score, 0.15]];
  const used = parts.filter(([, s]) => s != null);
  const wsum = used.reduce((s, p) => s + p[2], 0) || 1;
  let overall = used.reduce((s, p) => s + p[1] * p[2], 0) / wsum;
  let capped = false, cappedByMelody = false;
  // The goal is to reproduce the model's melody: a clearly different intonation
  // cannot be compensated by good sounds.
  if (!melodyOnly && intonation.score != null && overall > intonation.score + 20) { overall = intonation.score + 20; cappedByMelody = true; }
  if (!melodyOnly && words.accuracy != null && words.accuracy < 0.6) { overall = Math.min(overall, 45); capped = true; }
  return { overall: round(overall), capped, cappedByMelody, mode: melodyOnly ? 'melody' : 'speech', intonation, rhythm, stress, pronunciation: pron, words, map };
}

// ---------------------------------------------------------------------------
// Feedback in plain English for students
// ---------------------------------------------------------------------------

const TONE_TEXT = {
  fall: 'a fall', rise: 'a rise', 'fall-rise': 'a fall-rise', 'rise-fall': 'a rise-fall',
  level: 'a level tone', unclear: 'an unclear movement'
};

export function buildFeedback(res) {
  const items = [];
  const push = (level, area, text, word) => items.push({ level, area, text, word });
  const I = res.intonation, P = res.pronunciation, R = res.rhythm, S = res.stress, W = res.words;

  // Words
  if (W.whisper) {
    const wrong = W.whisper.ops.filter((o) => o.type !== 'ok');
    if (!wrong.length) push('good', 'Words', 'All words of the phrase were recognised.');
    else {
      wrong.slice(0, 4).forEach((o) => {
        if (o.type === 'del') push('bad', 'Words', `The word “${o.target}” was not heard. Make sure you say every word.`, o.target);
        else if (o.type === 'sub') push('bad', 'Words', `“${o.target}” sounded like “${o.heard}”.`, o.target);
        else push('warn', 'Words', `An extra word was heard: “${o.heard}”.`);
      });
    }
  }

  // Intonation
  if (I.ok) {
    if (I.toneRef.tone !== I.toneStu.tone && I.toneRef.tone !== 'unclear') {
      push('bad', 'Intonation',
        `Final tone${I.nucWord ? ` on “${I.nucWord}”` : ''}: the speaker uses ${TONE_TEXT[I.toneRef.tone]}, you use ${TONE_TEXT[I.toneStu.tone]}.`,
        I.nucWord);
    } else if (I.toneRef.tone !== 'unclear') {
      push('good', 'Intonation', `Your final tone matches the model (${TONE_TEXT[I.toneRef.tone]}${I.nucWord ? ` on “${I.nucWord}”` : ''}).`);
    }
    if (I.nucBy === 'syllable' && I.nucWord && I.nucWordStu && I.nucWord !== I.nucWordStu) {
      push('warn', 'Intonation',
        `The main pitch movement of the model starts on “${I.nucWord}”; yours starts on “${I.nucWordStu}”.`, I.nucWord);
    }
    if (I.peakWordRef != null && I.peakWordStu != null && I.peakWordRef !== I.peakWordStu) {
      push('warn', 'Intonation',
        `The highest pitch in the model is on “${I.words[I.peakWordRef].word}”; in your version it is on “${I.words[I.peakWordStu].word}”.`,
        I.words[I.peakWordRef].word);
    }
    if (I.rangeRef > 2 && I.rangeStu < I.rangeRef * 0.6) {
      push('warn', 'Intonation', `Your melody is flatter than the model: about ${I.rangeStu.toFixed(1)} semitones of movement against ${I.rangeRef.toFixed(1)}. Let your voice go higher and lower.`);
    } else if (I.rangeStu > I.rangeRef * 1.7 && I.rangeStu - I.rangeRef > 2) {
      push('warn', 'Intonation', `Your pitch moves much more than the model (${I.rangeStu.toFixed(1)} vs ${I.rangeRef.toFixed(1)} semitones). Keep the movements smaller.`);
    }
    if (I.parts.trend != null && (I.parts.trend < 70 || I.trendOpposite)) {
      const say = (v) => v <= -TREND.level ? 'goes down' : v >= TREND.level ? 'goes up' : 'stays at about the same height';
      const a = say(I.trendRef), b = say(I.trendStu);
      push(I.trendOpposite || I.parts.trend < 40 ? 'bad' : 'warn', 'Intonation',
        a !== b ? `Over the whole phrase the speaker’s voice ${a}; yours ${b}. Listen to where the phrase starts and where it ends.`
          : `Over the whole phrase your voice ${b} ${Math.abs(I.trendStu) > Math.abs(I.trendRef) ? 'more' : 'less'} than the speaker’s.`);
    }
    I.words.forEach((w) => {
      if (!w.refMove || !w.stuMove || w.refMove.cls === w.stuMove.cls) return;
      const m = { rise: 'goes up', fall: 'goes down', level: 'stays level' };
      push('warn', 'Intonation', `On “${w.word}” the model’s voice ${m[w.refMove.cls]}, yours ${m[w.stuMove.cls]}.`, w.word);
    });
    if (I.rmse < 1.5) push('good', 'Intonation', 'Your pitch line follows the model very closely.');
  } else if (I.reason) {
    push('warn', 'Intonation', I.reason + ' Record again in a quiet place, closer to the microphone.');
  }

  // Pronunciation
  if (P.phonemes) {
    let n = 0;
    P.words.forEach((w) => (w.issues || []).forEach((iss) => {
      if (n++ < 5) push(iss.kind === 'vowel' || iss.kind.startsWith('vowel') ? 'warn' : 'bad', 'Sounds', `“${w.word}”: ${iss.text}`, w.word);
    }));
  }
  if (P.score != null) {
    // a low word without a named error: point to the IPA column
    const weak = P.words.filter((w) => w.score != null && w.score < 60 && !(w.issues && w.issues.length)).sort((a, b) => a.score - b.score);
    weak.slice(0, 3).forEach((w) => {
      push('bad', 'Sounds', `“${w.word}” is not clear yet (${w.score}/100). Compare its sounds with the speaker’s in the IPA column and listen to the word again.`, w.word);
    });
    const anyIssues = P.words.some((w) => w.issues && w.issues.length);
    if (!weak.length && !anyIssues) push('good', 'Sounds', 'Every word was pronounced clearly.');
  }

  // Rhythm
  if (R.tempo > 1.25) push('warn', 'Rhythm', `You are ${Math.round((R.tempo - 1) * 100)}% slower than the model. Try to keep up with the speaker.`);
  else if (R.tempo < 0.8) push('warn', 'Rhythm', `You are ${Math.round((1 - R.tempo) * 100)}% faster than the model. Do not rush.`);
  (R.words || []).forEach((w) => {
    if (w.ratio > 1.6) push('warn', 'Rhythm', `“${w.word}” is too long compared with the rest of the phrase.`, w.word);
    else if (w.ratio < 0.6) push('warn', 'Rhythm', `“${w.word}” is too short: give it more time.`, w.word);
  });
  (R.pauses || []).forEach((p) => {
    if (p.ref && !p.stu) push('warn', 'Rhythm', `The model pauses after “${p.after}”; you do not.`);
    if (!p.ref && p.stu) push('warn', 'Rhythm', `You pause after “${p.after}”; the model links straight on.`);
  });

  // Stress
  if (S.score != null && S.words.length) {
    if (S.strongRef !== S.strongStu) {
      push('bad', 'Stress', `The most prominent word in the model is “${S.words[S.strongRef].word}”; in your version it is “${S.words[S.strongStu].word}”.`, S.words[S.strongRef].word);
    }
    S.words.filter((w) => w.internalMismatch).forEach((w) => {
      const where = (p) => p < 0.34 ? 'the beginning' : p > 0.66 ? 'the end' : 'the middle';
      push('warn', 'Stress', `In “${w.word}” the model’s strongest part is near ${where(w.peakPosRef)}; yours is near ${where(w.peakPosStu)}.`, w.word);
    });
  }

  const order = { bad: 0, warn: 1, good: 2 };
  items.sort((a, b) => order[a.level] - order[b.level]);
  return items;
}

export function verdict(score) {
  if (score == null) return '';
  if (score >= 88) return 'Almost identical to the model';
  if (score >= 75) return 'Close to the model';
  if (score >= 60) return 'Partly matches the model';
  if (score >= 40) return 'Noticeably different from the model';
  return 'Very different from the model';
}
