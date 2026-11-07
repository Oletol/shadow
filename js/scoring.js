// Comparison of a student recording with the model recording.
// Intonation is compared frame by frame on the model's time axis after
// word-anchored time alignment; pronunciation uses wav2vec2 CTC letter posteriors.

import { sampleTrack } from './pitch.js';
import { buildTimeMap, alignWordLists, splitWords } from './align.js';

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

/** Intonation comparison. ref/stu: {pitch, span, words|null}. */
export function compareIntonation(ref, stu, map, strictness = 'standard') {
  const sigma = (STRICTNESS[strictness] || STRICTNESS.standard).sigma;
  const rp = ref.pitch, sp = stu.pitch;
  const tR = frameTimes(rp);
  const iFrom = Math.max(0, Math.floor((ref.span[0] - rp.t0) / rp.hop));
  const iTo = Math.min(rp.st.length - 1, Math.ceil((ref.span[1] - rp.t0) / rp.hop));

  const refVals = [], stuValsRaw = [], times = [];
  let refVoiced = 0;
  for (let i = iFrom; i <= iTo; i++) {
    const t = tR(i);
    const r = rp.st[i];
    const s = sampleTrack(sp.stBridged, sp.hop, sp.t0, map(t));
    times.push(t);
    refVals.push(r);
    stuValsRaw.push(s);
    if (Number.isFinite(r)) refVoiced++;
  }
  const pairsR = [], pairsS = [];
  refVals.forEach((r, k) => {
    if (Number.isFinite(r) && Number.isFinite(stuValsRaw[k])) { pairsR.push(r); pairsS.push(stuValsRaw[k]); }
  });
  const coverage = refVoiced ? pairsR.length / refVoiced : 0;
  if (pairsR.length < 8) {
    return { ok: false, score: null, coverage, times, refVals, stuVals: stuValsRaw.map(() => NaN),
      reason: 'Too little voiced speech could be matched between the two recordings.' };
  }
  const offset = mean(pairsS.map((s, k) => s - pairsR[k]));
  const stuVals = stuValsRaw.map((s) => Number.isFinite(s) ? s - offset : NaN);
  const diffs = pairsS.map((s, k) => s - offset - pairsR[k]);
  const rmse = Math.sqrt(mean(diffs.map((d) => d * d)));
  const r = pearson(pairsR, pairsS);

  const rangeRef = percentile(refVals, 0.9) - percentile(refVals, 0.1);
  const rangeStu = percentile(stuVals, 0.9) - percentile(stuVals, 0.1);
  const rangeRatio = rangeRef > 0 && rangeStu > 0 ? Math.min(rangeRef, rangeStu) / Math.max(rangeRef, rangeStu) : 0;

  // per word
  const words = [];
  if (ref.words && stu.words) {
    ref.words.forEach((w, wi) => {
      const rv = [], sv = [];
      times.forEach((t, k) => {
        if (t >= w.start && t <= w.end) { rv.push(refVals[k]); sv.push(stuVals[k]); }
      });
      const mr = movementClass(rv), ms = movementClass(sv);
      const both = rv.map((x, k) => [x, sv[k]]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
      const wRmse = both.length >= 3 ? Math.sqrt(mean(both.map(([x, y]) => (x - y) ** 2))) : NaN;
      const lvlR = mean(rv.filter(Number.isFinite)), lvlS = mean(sv.filter(Number.isFinite));
      words.push({
        index: wi, word: w.word,
        refMove: mr, stuMove: ms,
        levelDiff: Number.isFinite(lvlR) && Number.isFinite(lvlS) ? lvlS - lvlR : NaN,
        rmse: wRmse,
        peakRef: rv.some(Number.isFinite) ? Math.max(...rv.filter(Number.isFinite)) : NaN,
        peakStu: sv.some(Number.isFinite) ? Math.max(...sv.filter(Number.isFinite)) : NaN,
        score: Number.isFinite(wRmse) ? round(100 * Math.exp(-((wRmse / sigma) ** 2))) : null
      });
    });
  }

  // nucleus: from the last word that has enough voiced reference frames to the end
  let nucStart = ref.span[0] + (ref.span[1] - ref.span[0]) * 0.6;
  let nucWord = null;
  if (ref.words) {
    for (let wi = ref.words.length - 1; wi >= 0; wi--) {
      const w = ref.words[wi];
      const n = times.reduce((c, t, k) => c + (t >= w.start && t <= w.end && Number.isFinite(refVals[k]) ? 1 : 0), 0);
      if (n >= 8) { nucStart = w.start; nucWord = w.word; break; }
    }
  }
  const nr = [], ns = [];
  times.forEach((t, k) => { if (t >= nucStart) { nr.push(refVals[k]); ns.push(stuVals[k]); } });
  const toneRef = classifyTone(nr), toneStu = classifyTone(ns);

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
  const sTone = toneAgreement(toneRef.tone, toneStu.tone);
  const sMove = Number.isFinite(moveAgree) ? 100 * moveAgree : sShape;
  const sRange = 100 * rangeRatio;
  let score = 0.38 * sRmse + 0.20 * sShape + 0.22 * sTone + 0.10 * sMove + 0.10 * sRange;
  if (coverage < 0.35) score = Math.min(score, 60);

  return {
    ok: true, score: round(score), coverage, rmse, correlation: r, offset,
    rangeRef, rangeStu, rangeRatio,
    toneRef, toneStu, nucWord, nucStart,
    peakWordRef, peakWordStu, moveAgree,
    parts: { contour: round(sRmse), shape: round(sShape), tone: sTone, movement: round(sMove), range: round(sRange) },
    words, times, refVals, stuVals
  };
}

/** Timing, pauses and tempo. */
export function compareRhythm(ref, stu) {
  const refDur = ref.span[1] - ref.span[0];
  const stuDur = stu.span[1] - stu.span[0];
  const tempo = stuDur / refDur;
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

/** Pronunciation quality from wav2vec2 letter posteriors, calibrated by the model speaker. */
export function comparePronunciation(ref, stu) {
  if (!ref.words || !stu.words) return { score: null, words: [] };
  const words = ref.words.map((rw, i) => {
    const sw = stu.words[i];
    const base = Math.max(0.45, rw.conf || 0);
    const ratio = clamp((sw.conf || 0) / base, 0, 1);
    const weak = [];
    sw.letters.forEach((l, k) => {
      const rl = rw.letters[k];
      const refC = rl ? rl.conf : 0.9;
      if (l.conf < 0.3 && refC > 0.5) weak.push(l.ch);
    });
    return {
      index: i, word: rw.word, score: round(100 * Math.pow(ratio, 1.3)),
      conf: sw.conf, refConf: rw.conf, heard: sw.heard, weakLetters: weak
    };
  });
  const totalLetters = words.reduce((s, w) => s + w.word.length, 0) || 1;
  const score = words.reduce((s, w) => s + w.score * w.word.length, 0) / totalLetters;
  return { score: round(score), words };
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
  const map = (ref.words && stu.words)
    ? buildTimeMap(ref.words, stu.words, ref.span, stu.span)
    : (t) => stu.span[0] + (t - ref.span[0]) * (stu.span[1] - stu.span[0]) / (ref.span[1] - ref.span[0]);
  const intonation = compareIntonation(ref, stu, map, opts.strictness);
  const rhythm = compareRhythm(ref, stu);
  const stress = compareStress(ref, stu, map);
  const pron = comparePronunciation(ref, stu);
  const words = compareWords(opts.target || '', opts.whisperText, opts.acousticText);

  const parts = [
    ['intonation', intonation.score, 0.40],
    ['pronunciation', pron.score, 0.30],
    ['rhythm', rhythm.score, 0.15],
    ['stress', stress.score, 0.15]
  ].filter(([, s]) => s != null);
  const wsum = parts.reduce((s, p) => s + p[2], 0) || 1;
  let overall = parts.reduce((s, p) => s + p[1] * p[2], 0) / wsum;
  let capped = false;
  // The goal is to reproduce the model's melody: a clearly different intonation
  // cannot be compensated by good sounds.
  if (intonation.score != null && overall > intonation.score + 20) overall = intonation.score + 20;
  if (words.accuracy != null && words.accuracy < 0.6) { overall = Math.min(overall, 45); capped = true; }
  return { overall: round(overall), capped, intonation, rhythm, stress, pronunciation: pron, words, map };
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
  if (P.score != null) {
    const weak = P.words.filter((w) => w.score < 60).sort((a, b) => a.score - b.score);
    weak.slice(0, 3).forEach((w) => {
      const letters = w.weakLetters.length ? ` Listen to the sounds spelled ${w.weakLetters.map((c) => `“${c.toLowerCase()}”`).join(', ')}.` : '';
      const heard = w.heard && w.heard !== w.word ? ` The acoustic model heard “${w.heard}”.` : '';
      push('bad', 'Pronunciation', `“${w.word}” is not clear yet (${w.score}/100).${heard}${letters}`, w.word);
    });
    if (!weak.length) push('good', 'Pronunciation', 'Every word was pronounced clearly.');
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
