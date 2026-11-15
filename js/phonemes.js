// Phoneme (IPA) comparison between the speaker and the student.
// Input: CTC log-probabilities from a phoneme recogniser
// (wav2vec2-lv-60-espeak-cv-ft) for both recordings. Pure functions.

import { ctcForcedAlign } from './align.js';

/** Greedy CTC decoding with timing: [{id, sym, t, conf}] (t in frames, at the peak). */
export function greedyPhones(logp, T, V, id2sym, blank = 0) {
  const out = [];
  let prev = -1, cur = null;
  for (let t = 0; t < T; t++) {
    let best = 0, bv = -Infinity;
    for (let v = 0; v < V; v++) if (logp[t * V + v] > bv) { bv = logp[t * V + v]; best = v; }
    if (best !== blank && best !== prev) {
      const sym = id2sym[best];
      if (sym && !/^<.*>$/.test(sym) && sym !== '|' && sym.trim()) {
        cur = { id: best, sym, t, conf: Math.exp(bv) };
        out.push(cur);
      } else cur = null;
    } else if (best !== blank && best === prev && cur && Math.exp(bv) > cur.conf) {
      cur.conf = Math.exp(bv); cur.t = t;
    }
    if (best === blank) cur = null;
    prev = best;
  }
  return out;
}

// --- symbol classes --------------------------------------------------------

const strip = (s) => String(s || '').replace(/[ːˑ̃ʰʲʷ̩̯]/g, '');
const GROUPS = [
  ['ə', 'ɐ', 'ᵻ', 'ɨ', 'ɘ'],
  ['ɪ', 'ᵻ', 'ɨ', 'i'],
  ['i'],
  ['u', 'ʉ', 'ɯ'],
  ['ʊ', 'u'],
  ['ɔ', 'o'],
  ['ɑ', 'a'],
  ['ɒ', 'ɔ'],
  ['e', 'ɛ'],
  ['ɜ', 'ɞ'],
  ['ʌ', 'ɐ'],
  ['ɡ', 'g'],
  ['ɹ', 'r', 'ɻ'],
  ['ʔ', 't'],
  ['ɫ', 'l']
];
const RHOTIC = new Set(['ɹ', 'ɚ', 'ɝ', 'ɻ', 'r', 'ɹ̩']);
const VOWEL_RE = /[aeiouæɑɒɔəɚɛɜɝɪʊʌɐᵻɨʉɘɞɤɯøœy]/;

export function isVowel(sym) { return VOWEL_RE.test(sym || ''); }
export function isRhotic(sym) { return RHOTIC.has(sym) || /[ɚɝ˞]/.test(sym || ''); }

/** 1 = same, 0.75 = close variant (length, near vowel), 0 = different. */
export function phoneSimilarity(a, b) {
  if (a === b) return 1;
  const sa = strip(a), sb = strip(b);
  if (sa === sb) return 0.85;           // only length/diacritic differs
  // diphthongs: compare by first element when both are diphthongs of 2+ chars
  if (sa.length > 1 && sb.length > 1 && sa[0] === sb[0] && isVowel(sa) && isVowel(sb)) return 0.6;
  for (const g of GROUPS) if (g.includes(sa) && g.includes(sb)) return 0.75;
  // typical learner substitutions: related but different sounds
  for (const g of NEAR) if (g.includes(sa) && g.includes(sb)) return 0.35;
  return 0;
}

const NEAR = [['t', 'd', 'ɾ', 'ʔ'], ['θ', 'f', 's', 't'], ['ð', 'v', 'z', 'd'], ['l', 'ɫ', 'ɹ', 'r'], ['v', 'w'], ['ʃ', 's'], ['ʒ', 'z'], ['n', 'ŋ']];

/** Cost of aligning two phones: never pair a vowel with a consonant. */
function subCost(a, b) {
  if (isVowel(strip(a)) !== isVowel(strip(b))) return 2.5;
  return 1.2 * (1 - phoneSimilarity(a, b));
}

/** Levenshtein alignment of two phone sequences with graded substitution cost. */
export function alignPhones(ref, stu) {
  const n = ref.length, m = stu.length;
  const D = Array.from({ length: n + 1 }, () => new Float32Array(m + 1));
  for (let i = 0; i <= n; i++) D[i][0] = i;
  for (let j = 0; j <= m; j++) D[0][j] = j;
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) {
    D[i][j] = Math.min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + subCost(ref[i - 1].sym, stu[j - 1].sym));
  }
  const ops = [];
  let i = n, j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0) {
      const sim = phoneSimilarity(ref[i - 1].sym, stu[j - 1].sym);
      if (Math.abs(D[i][j] - (D[i - 1][j - 1] + subCost(ref[i - 1].sym, stu[j - 1].sym))) < 1e-6) {
        ops.push({ type: sim >= 0.75 ? 'match' : sim > 0 ? 'near' : 'sub', ri: i - 1, si: j - 1, sim });
        i--; j--; continue;
      }
    }
    if (i > 0 && Math.abs(D[i][j] - (D[i - 1][j] + 1)) < 1e-6) { ops.push({ type: 'del', ri: i - 1, si: -1, sim: 0 }); i--; }
    else { ops.push({ type: 'ins', ri: -1, si: j - 1, sim: 0 }); j--; }
  }
  return ops.reverse();
}


// --- accent of the speaker ---------------------------------------------------

/**
 * The phoneme model was trained mostly on American speech and tends to write an r
 * after vowels even for a speaker who does not pronounce it. For a lecture marked
 * "accent": "non-rhotic" the speaker's transcription is corrected by rule:
 *   r after a vowel is removed unless a vowel follows (linking r is kept),
 *   ɚ becomes ə, ɝ becomes ɜː.
 */
export function applyAccent(phones, accent, id2sym) {
  if (accent !== 'non-rhotic') return phones;
  const sym2id = {};
  Object.entries(id2sym).forEach(([i, sym]) => { sym2id[sym] = Number(i); });
  const swap = (p, sym) => (sym in sym2id ? { ...p, sym, id: sym2id[sym] } : p);
  const isR = (x) => /^[ɹɻr]$/.test(strip(x));
  const out = [];
  phones.forEach((p, k) => {
    const prev = phones[k - 1], next = phones[k + 1];
    if (isR(p.sym) && prev && isVowel(strip(prev.sym)) && !(next && isVowel(strip(next.sym)))) return;
    if (p.sym === 'ɚ') out.push(swap(p, 'ə'));
    else if (p.sym === 'ɝ') out.push(swap(swap(p, 'ɜ'), 'ɜː'));
    else out.push(p);
  });
  return out;
}

// --- goodness of pronunciation (GOP) ---------------------------------------

/**
 * Parameters of the GOP score. They are meant to be calibrated against human
 * ratings (tools/calibrate.py writes them to data/settings.json -> "gop").
 *  offset, slope: logistic curve over the log-likelihood ratio
 *  presence: the expected sound counts as fully present once its probability
 *            reaches this share of the speaker's own probability for that sound
 *  refCap: the speaker's own likelihood ratio at which the model is taken to hear
 *          the sound reliably. Below it the student is judged against the speaker's
 *          ratio instead of an absolute one, and the sound weighs less in the word.
 *  minWeight: weight of a sound that the model cannot tell apart even for the speaker
 *  unsureBelow: reliability (0..1) under which a sound is shown as "not judged"
 */
export const GOP = { offset: -0.5, slope: 1, presence: 0.5, refCap: 2, minWeight: 0.3, unsureBelow: 0.5 };
export function setGopParams(p) {
  if (!p) return;
  Object.keys(GOP).forEach((k) => { if (Number.isFinite(p[k])) GOP[k] = p[k]; });
}

const isSpecial = (sym) => !sym || /^<.*>$/.test(sym) || sym === '|' || !String(sym).trim();

/**
 * For every speaker phone k (aligned on the student's posteriors):
 *   pT = probability mass of the expected sound and its close variants (length mark,
 *        neighbouring vowel symbol) at its best frame;
 *   pC = highest probability of any other sound inside the phone's own frames;
 *   lr = log(pT) - log(pC): > 0 the expected sound wins, < 0 another sound wins.
 * score = logistic(lr) * presence, 0..1. A dropped sound has pT near 0 and scores 0.
 *
 * refLr (optional): the same ratio measured on the speaker's own recording. The speaker
 * is the point of reference: where the model is unsure even about the speaker
 * (refLr < GOP.refCap), the student's ratio is raised by the missing amount, and the
 * result carries reliability (0..1) and weight for the word score.
 */
export function gopScores(stu, refPhones, spans, id2sym, blank = 0, refLr = null) {
  const { logp, T, V } = stu;
  const eqCache = new Map();
  const equivalents = (id) => {
    if (!eqCache.has(id)) {
      const set = new Uint8Array(V);
      for (let v = 0; v < V; v++) {
        if (v === blank || isSpecial(id2sym[v])) continue;
        if (v === id || phoneSimilarity(id2sym[id], id2sym[v]) >= 0.75) set[v] = 1;
      }
      eqCache.set(id, set);
    }
    return eqCache.get(id);
  };
  const EPS = 1e-4;
  return refPhones.map((p, k) => {
    const sp = spans[k];
    const rl = refLr && Number.isFinite(refLr[k]) ? Math.min(GOP.refCap, Math.max(0, refLr[k])) : GOP.refCap;
    const reliability = GOP.refCap > 0 ? rl / GOP.refCap : 1;
    const weight = GOP.minWeight + (1 - GOP.minWeight) * reliability;
    if (!sp || sp.start < 0) return { score: 0, lr: -Infinity, lrRel: -Infinity, pT: 0, pC: 0, heard: null, reliability, weight };
    const eq = equivalents(p.id);
    // the expected sound may peak one frame outside its Viterbi span, but never inside a neighbour
    let lo = Math.max(0, sp.start - 1), hi = Math.min(T, sp.end + 1);
    const prev = spans[k - 1], next = spans[k + 1];
    if (prev && prev.start >= 0) lo = Math.max(lo, Math.min(prev.end, sp.start));
    if (next && next.start >= 0) hi = Math.min(hi, Math.max(next.start, sp.end));
    let pT = 0;
    for (let t = lo; t < hi; t++) {
      let m = 0;
      for (let v = 0; v < V; v++) if (eq[v]) m += Math.exp(logp[t * V + v]);
      if (m > pT) pT = m;
    }
    let pC = 0, heard = null;
    for (let t = sp.start; t < sp.end; t++) {
      for (let v = 0; v < V; v++) {
        if (v === blank || eq[v] || isSpecial(id2sym[v])) continue;
        const q = Math.exp(logp[t * V + v]);
        if (q > pC) { pC = q; heard = id2sym[v]; }
      }
    }
    pT = Math.min(1, pT);
    const lr = Math.log(pT + EPS) - Math.log(pC + EPS);
    const lrRel = lr + (GOP.refCap - rl);
    const logistic = 1 / (1 + Math.exp(-(lrRel - GOP.offset) / GOP.slope));
    const base = Math.max(0.3, p.conf || 0.5);
    const presence = Math.min(1, pT / (GOP.presence * base));
    return { score: logistic * presence, lr, lrRel, pT, pC, heard, reliability, weight };
  });
}

// --- map of expected learner errors ------------------------------------------

/**
 * Substitutions that are typical for a group of learners. Each rule names the sound
 * of the speaker, the sounds that tend to replace it, and what to tell the student.
 * A rule is checked directly on the probabilities of the model: the expected sound
 * against the listed replacements only (a "pair ratio"), so a known error is found
 * even when some third sound is the model's first guess.
 *   final: the rule applies only to the last sound of a word.
 * 'zh' collects the errors described for speakers of Chinese; most of them are common
 * to other learners too, and the texts do not mention the first language.
 */
export const ERROR_MAPS = {
  zh: [
    { kind: 'th', ref: ['θ'], heard: ['s', 'f', 't'], tip: 'Put the tip of the tongue between the teeth and blow: /θ/.' },
    { kind: 'th', ref: ['ð'], heard: ['z', 'd', 'l'], tip: 'Put the tip of the tongue between the teeth and use your voice: /ð/.' },
    { kind: 'vw', ref: ['v'], heard: ['w'], tip: 'Touch the lower lip with the upper teeth; do not round the lips.' },
    { kind: 'vf', ref: ['v'], heard: ['f'], tip: 'Keep the voice on: /v/ is /f/ with voice.' },
    { kind: 'vw', ref: ['w'], heard: ['v'], tip: 'Round the lips; the teeth do not touch the lip.' },
    { kind: 'ln', ref: ['l', 'ɫ'], heard: ['n'], tip: 'For /l/ the air goes out through the mouth, round the sides of the tongue, not through the nose.' },
    { kind: 'ln', ref: ['n'], heard: ['l', 'ɫ'], tip: 'For /n/ the air goes out through the nose; the tongue closes the mouth completely.' },
    { kind: 'lr', ref: ['l', 'ɫ'], heard: ['ɹ', 'r', 'ɻ'], tip: 'For /l/ the tip of the tongue touches the ridge behind the upper teeth.' },
    { kind: 'lr', ref: ['ɹ'], heard: ['l', 'ɫ', 'w'], tip: 'For /r/ the tongue tip touches nothing; curl it back slightly and round the lips a little.' },
    { kind: 'ng', ref: ['ŋ'], heard: ['n'], tip: 'Keep the back of the tongue up, as at the end of “sing”; the tip stays down.' },
    { kind: 'ng', ref: ['n'], heard: ['ŋ'], final: true, tip: 'End on /n/: the tongue tip goes up behind the teeth.' },
    { kind: 'sh', ref: ['ʃ'], heard: ['s', 'ɕ', 'ʂ'], tip: 'Round the lips and pull the tongue back a little: /ʃ/.' },
    { kind: 'zh', ref: ['ʒ'], heard: ['ʃ', 'z', 'ɹ', 'ʐ', 'j'], tip: '/ʒ/ is /ʃ/ with voice, as in “measure”.' },
    { kind: 'ch', ref: ['tʃ'], heard: ['tɕ', 'ts', 'ʈʂ'], tip: 'Round the lips for /tʃ/, as in “chair”.' },
    { kind: 'j', ref: ['dʒ'], heard: ['tʃ', 'tɕ', 'dʑ', 'ts', 'dz', 'ʒ'], tip: 'Start with /d/ and keep the voice on: /dʒ/, as in “job”.' },
    { kind: 'h', ref: ['h'], heard: ['x', 'χ'], tip: 'English /h/ is only breath; do not raise the back of the tongue.' },
    { kind: 'zs', ref: ['z'], heard: ['s', 'ts', 'dz'], tip: 'Keep the voice on: /z/ is /s/ with voice.' },
    { kind: 'devoice', ref: ['b'], heard: ['p'], final: true, tip: 'Keep the final /b/ voiced and make the vowel before it longer.' },
    { kind: 'devoice', ref: ['d'], heard: ['t'], final: true, tip: 'Keep the final /d/ voiced and make the vowel before it longer.' },
    { kind: 'devoice', ref: ['ɡ', 'g'], heard: ['k'], final: true, tip: 'Keep the final /ɡ/ voiced and make the vowel before it longer.' },
    { kind: 'dark-l', ref: ['l', 'ɫ'], heard: ['o', 'ɔ', 'ʊ', 'u', 'ɤ', 'w'], final: true, tip: 'Finish the word with the tongue tip touching the ridge behind the upper teeth: the final /l/ must be heard.' },
    { kind: 'tense', ref: ['ɪ'], heard: ['i'], tip: 'This vowel is short and relaxed, as in “sit”, not as in “seat”.' },
    { kind: 'tense', ref: ['i'], heard: ['ɪ'], long: true, tip: 'This vowel is long and tense, as in “seat”, not as in “sit”.' },
    { kind: 'tense', ref: ['ʊ'], heard: ['u'], tip: 'This vowel is short and relaxed, as in “book”, not as in “boot”.' },
    { kind: 'tense', ref: ['u'], heard: ['ʊ'], long: true, tip: 'This vowel is long, as in “boot”, not as in “book”.' },
    { kind: 'ash', ref: ['æ'], heard: ['ɛ', 'e', 'a', 'ɑ'], tip: 'Open the mouth wider and spread the lips: /æ/ as in “cat”, not as in “bed”.' },
    { kind: 'diph', ref: ['eɪ'], heard: ['ɛ', 'e'], tip: 'This is a glide of two parts: start at /e/ and move towards /ɪ/, as in “day”.' },
    { kind: 'diph', ref: ['əʊ', 'oʊ'], heard: ['ɔ', 'o', 'ɒ'], tip: 'This is a glide of two parts: move towards /ʊ/ at the end, as in “go”.' },
    { kind: 'diph', ref: ['aɪ'], heard: ['a', 'ɑ', 'æ'], tip: 'This is a glide of two parts: finish with a short /ɪ/, as in “my”.' },
    { kind: 'diph', ref: ['aʊ'], heard: ['a', 'ɑ', 'ɔ'], tip: 'This is a glide of two parts: finish with a short /ʊ/, as in “now”.' }
  ]
};

/**
 * refMin: the speaker's own pair ratio must be at least this, otherwise the model
 *         cannot tell the two sounds apart on this recording and the rule is skipped;
 * margin: the replacement must win in the student's recording by at least this.
 */
export const ERROR_MAP = { name: 'zh', refMin: 1, margin: 0.5 };
export function setErrorMap(name, params) {
  ERROR_MAP.name = name && ERROR_MAPS[name] ? name : null;
  if (params) ['refMin', 'margin'].forEach((k) => { if (Number.isFinite(params[k])) ERROR_MAP[k] = params[k]; });
}

const hasLength = (sym) => /ː/.test(sym || '');

/** log ratio between the expected sound and a set of replacement sounds inside a span. */
function pairRatio(post, span, expIds, confIds, id2sym) {
  const { logp, T, V } = post;
  if (!span || span.start < 0) return null;
  const lo = Math.max(0, span.start - 1), hi = Math.min(T, span.end + 1);
  let pE = 0, pC = 0, heard = null;
  for (let t = lo; t < hi; t++) {
    let e = 0;
    for (const v of expIds) e += Math.exp(logp[t * V + v]);
    if (e > pE) pE = e;
    for (const v of confIds) {
      const q = Math.exp(logp[t * V + v]);
      if (q > pC) { pC = q; heard = id2sym[v]; }
    }
  }
  return { lr: Math.log(Math.min(1, pE) + 1e-4) - Math.log(pC + 1e-4), heard };
}

/**
 * Check the rules of the active error map for every speaker phone.
 * Returns an array (one entry per phone): null or {kind, heard, tip, lr, refLr}.
 */
export function checkErrorMap(ref, stu, refPhones, refSpans, stuSpans, id2sym, blank, isFinal) {
  const rules = ERROR_MAP.name ? ERROR_MAPS[ERROR_MAP.name] : null;
  if (!rules || !refSpans || !stuSpans) return refPhones.map(() => null);
  const V = stu.V;
  const idsOf = (pred) => { const out = []; for (let v = 0; v < V; v++) { const sym = id2sym[v]; if (v !== blank && !isSpecial(sym) && pred(sym)) out.push(v); } return out; };
  return refPhones.map((p, k) => {
    const base = strip(p.sym);
    for (const rule of rules) {
      if (!rule.ref.includes(base) && !rule.ref.includes(p.sym)) continue;
      if (rule.final && !isFinal(k)) continue;
      if (rule.long && !hasLength(p.sym)) continue;
      const confIds = idsOf((sym) => rule.heard.includes(strip(sym)) && strip(sym) !== base);
      if (!confIds.length) continue;
      const expIds = idsOf((sym) => strip(sym) === base);
      const r = pairRatio(ref, refSpans[k], expIds, confIds, id2sym);
      if (!r || r.lr < ERROR_MAP.refMin) continue;      // the model does not separate the pair on the speaker
      const s = pairRatio(stu, stuSpans[k], expIds, confIds, id2sym);
      if (s && s.lr <= -ERROR_MAP.margin) return { kind: rule.kind, heard: s.heard, tip: rule.tip, lr: s.lr, refLr: r.lr };
    }
    return null;
  });
}

// --- descriptions for students ----------------------------------------------

export const IPA_HINTS = {
  'ɒ': 'short rounded o, as in British “hot”',
  'ɑː': 'long open a, as in “father”, “car”',
  'ɑ': 'open a, as in “father”',
  'æ': 'a as in “cat”',
  'ʌ': 'short u as in “cup”',
  'ɔː': 'long o as in “law”, “four”',
  'ɔ': 'o as in “law”',
  'ə': 'weak vowel (schwa), as in “a”, “the”',
  'ɚ': 'American r-coloured vowel, as in US “water”',
  'ɝ': 'American r-coloured vowel, as in US “bird”',
  'ɜː': 'long vowel as in British “bird” (no r)',
  'ɪ': 'short i as in “sit”',
  'iː': 'long ee as in “see”',
  'ʊ': 'short oo as in “book”',
  'uː': 'long oo as in “too”',
  'ɛ': 'e as in “bed”',
  'e': 'e as in “bed”',
  'eɪ': 'diphthong as in “day”',
  'aɪ': 'diphthong as in “my”',
  'ɔɪ': 'diphthong as in “boy”',
  'aʊ': 'diphthong as in “now”',
  'oʊ': 'American o as in US “go”',
  'əʊ': 'British o as in “go”',
  'ɪə': 'diphthong as in British “here”',
  'eə': 'diphthong as in British “wear”',
  'ɛə': 'diphthong as in British “wear”',
  'ʊə': 'diphthong as in British “tour”',
  'θ': 'th as in “think” (tongue between the teeth, voiceless)',
  'ð': 'th as in “that” (tongue between the teeth, voiced)',
  'ɹ': 'English r',
  'ɾ': 'quick tap like a soft d, as in US “water”',
  'ʔ': 'glottal stop',
  'ŋ': 'ng as in “sing”',
  'ʃ': 'sh', 'ʒ': 's as in “measure”', 'tʃ': 'ch', 'dʒ': 'j',
  'j': 'y as in “yes”', 'w': 'w', 'v': 'v', 'l': 'l', 'h': 'h'
};

export function describe(sym) {
  return IPA_HINTS[sym] || IPA_HINTS[strip(sym)] || null;
}

// --- main comparison ---------------------------------------------------------

/**
 * @param ref {logp,T,V,frameSec} speaker phoneme posteriors
 * @param stu {logp,T,V,frameSec} student phoneme posteriors
 * @param id2sym id -> IPA symbol
 * @param refWords words of the speaker with start/end (s) from the letter alignment
 * @param opts {accent} accent of the speaker, from lecture.json ("non-rhotic")
 */
export function comparePhonemes(ref, stu, id2sym, refWords, blank = 0, opts = {}) {
  const rp = applyAccent(greedyPhones(ref.logp, ref.T, ref.V, id2sym, blank), opts.accent, id2sym);
  const sp = greedyPhones(stu.logp, stu.T, stu.V, id2sym, blank);
  if (!rp.length) return null;

  // phone -> word of the speaker by time
  const wordOfRef = rp.map((p) => nearestWord(p.t * ref.frameSec, refWords));

  // Goodness of pronunciation: the speaker's phones are force-aligned on the student's
  // posteriors; each phone gets a graded score from the likelihood ratio between the
  // expected sound and the best competing sound (see gopScores).
  // The same measure is first taken on the speaker's own recording: it shows how well
  // the model hears each sound when it is pronounced correctly.
  const ids = rp.map((p) => p.id);
  const alRef = ctcForcedAlign(ref.logp, ref.T, ref.V, ids, blank);
  const refLr = alRef ? gopScores(ref, rp, alRef.tokenSpans, id2sym, blank).map((g) => g.lr) : null;
  const al = ctcForcedAlign(stu.logp, stu.T, stu.V, ids, blank);
  const gop = al ? gopScores(stu, rp, al.tokenSpans, id2sym, blank, refLr) : null;
  // Known learner errors, checked sound by sound against the speaker.
  const isFinal = (k) => k === rp.length - 1 || wordOfRef[k + 1] !== wordOfRef[k];
  const known = alRef && al ? checkErrorMap(ref, stu, rp, alRef.tokenSpans, al.tokenSpans, id2sym, blank, isFinal) : rp.map(() => null);

  const ops = alignPhones(rp, sp);
  const words = (refWords || [{ word: '' }]).map((w) => ({ word: w.word, ref: [], stu: [], items: [], issues: [] }));
  let lastWord = 0;
  ops.forEach((op, k) => {
    const wi = op.ri >= 0 ? wordOfRef[op.ri] : attachInsertion(ops, k, wordOfRef, lastWord);
    lastWord = wi;
    const W = words[wi];
    if (!W) return;
    const r = op.ri >= 0 ? rp[op.ri] : null;
    const s = op.si >= 0 ? sp[op.si] : null;
    if (r) W.ref.push(r.sym);
    if (s) W.stu.push(s.sym);
    const g = gop && op.ri >= 0 ? gop[op.ri] : null;
    let score;
    if (g) score = g.score;                      // graded, independent of the greedy decoding
    else if (op.type === 'match') score = 1;     // fallback when forced alignment is impossible
    else if (op.type === 'near') score = 0.6;
    else score = 0;
    W.items.push({
      type: op.type, ref: r ? r.sym : null, stu: s ? s.sym : null, score, ins: op.type === 'ins',
      gop: g ? g.lr : null, gopRel: g ? g.lrRel : null, heard: g && g.lrRel < 0 ? g.heard : null,
      weight: g ? g.weight : 1, reliability: g ? g.reliability : 1,
      unsure: !!(g && g.reliability < GOP.unsureBelow),  // the model does not hear this sound reliably even from the speaker
      known: op.ri >= 0 ? known[op.ri] : null             // a rule of the error map that fired on this sound
    });
  });

  words.forEach((W, wi) => {
    const scored = W.items.filter((it) => !it.ins);
    const insPenalty = W.items.filter((it) => it.ins).length * 0.5;
    const wsum = scored.reduce((a, b) => a + b.weight, 0);
    W.score = scored.length && wsum > 0 ? Math.round(100 * Math.max(0, (scored.reduce((a, b) => a + b.score * b.weight, 0) - insPenalty) / wsum)) : null;
    W.issues = findIssues(W, wi === words.length - 1);
  });
  const all = words.filter((w) => w.score != null);
  const weight = (w) => Math.max(1, w.ref.length);
  const score = all.length ? Math.round(all.reduce((a, w) => a + w.score * weight(w), 0) / all.reduce((a, w) => a + weight(w), 0)) : null;
  return { score, words, refPhones: rp.map((p) => p.sym), stuPhones: sp.map((p) => p.sym) };
}

function nearestWord(t, words) {
  if (!words || !words.length) return 0;
  let best = 0, bd = Infinity;
  words.forEach((w, i) => {
    const d = t < w.start ? w.start - t : t > w.end ? t - w.end : 0;
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}

function attachInsertion(ops, k, wordOfRef, lastWord) {
  // an inserted phone belongs to the word of the previous speaker phone
  for (let j = k - 1; j >= 0; j--) if (ops[j].ri >= 0) return wordOfRef[ops[j].ri];
  for (let j = k + 1; j < ops.length; j++) if (ops[j].ri >= 0) return wordOfRef[ops[j].ri];
  return lastWord;
}

/** Turn phone differences inside a word into named, teachable issues. */
export function findIssues(W) {
  const issues = [];
  const add = (kind, text) => { if (!issues.some((x) => x.kind === kind)) issues.push({ kind, text }); };
  const items = W.items;
  // rules of the error map come first: they name the sound and say how to make it
  items.forEach((it) => {
    if (!it.known || !it.ref) return;
    add(it.known.kind, `/${it.ref}/ sounds like /${it.known.heard}/. ${it.known.tip}`);
  });
  items.forEach((it, k) => {
    const r = it.ref, s = it.stu;
    if (it.type === 'match' || it.unsure) return;   // no remark where the reference itself is unreliable
    // r after a vowel where the speaker has none (rhoticity)
    if (s && isRhotic(s) && !(r && isRhotic(r))) {
      const prevIsVowel = k > 0 && items[k - 1].stu && isVowel(items[k - 1].stu);
      if (prevIsVowel || /[ɚɝ]/.test(s)) add('rhotic', `You pronounce an r after the vowel. In RP, r is silent here: say /${W.ref.join('')}/.`);
    }
    // flapped t
    if (s && /^[ɾd]$/.test(s) && r && /^[tʔ]$/.test(r)) add('flap', 'Your t sounds like a quick American d. In RP, keep a clear /t/ (or a glottal stop).');
    // th sounds
    if (r && /^[θð]$/.test(r) && s && !/^[θð]$/.test(s)) add('th', `“th” became /${s}/. Put the tip of the tongue between the teeth: /${r}/.`);
    if (r && /^[θð]$/.test(r) && it.type === 'del') add('th', `The “th” sound /${r}/ is missing.`);
    // l / r
    if (r && /^[lɫ]$/.test(r) && s && isRhotic(s)) add('lr', 'You say r instead of l.');
    if (r === 'ɹ' && s && /^[lɫ]$/.test(s)) add('lr', 'You say l instead of r.');
    // v / w
    if (r === 'v' && s === 'w') add('vw', 'You say w instead of v: touch the lower lip with the upper teeth.');
    if (r === 'w' && s === 'v') add('vw', 'You say v instead of w: round the lips, no teeth.');
    // vowels
    if (r && s && isVowel(r) && isVowel(s) && !isRhotic(s) && it.type !== 'match') {
      const dr = describe(r), ds = describe(s);
      add('vowel:' + r, `Vowel: the speaker says /${r}/${dr ? ' (' + dr + ')' : ''}, you say /${s}/${ds ? ' (' + ds + ')' : ''}.`);
    }
  });
  // extra vowel at the end of the word (common for speakers of Chinese and other syllable-timed languages)
  const last = items[items.length - 1];
  const beforeLast = items[items.length - 2];
  if (last && last.ins && last.stu && isVowel(last.stu) && beforeLast && beforeLast.ref && !isVowel(beforeLast.ref)) {
    add('epenthesis', `You add a vowel /${last.stu}/ after the last consonant. End the word cleanly on /${beforeLast.ref}/.`);
  }
  // final consonant dropped
  const lastRef = [...items].reverse().find((it) => it.ref);
  if (lastRef && lastRef.type === 'del' && !lastRef.unsure && !isVowel(lastRef.ref)) add('final', `The final sound /${lastRef.ref}/ is missing.`);
  // generic consonant substitutions not covered above
  items.forEach((it) => {
    if (it.type === 'sub' && !it.unsure && it.ref && it.stu && !isVowel(it.ref) && !isVowel(it.stu)
      && !issues.length) add('cons:' + it.ref, `The speaker says /${it.ref}/, you say /${it.stu}/.`);
  });
  return issues;
}
