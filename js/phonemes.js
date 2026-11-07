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
 * @param stuWords words of the student (optional, used to attach insertions)
 */
export function comparePhonemes(ref, stu, id2sym, refWords, blank = 0) {
  const rp = greedyPhones(ref.logp, ref.T, ref.V, id2sym, blank);
  const sp = greedyPhones(stu.logp, stu.T, stu.V, id2sym, blank);
  if (!rp.length) return null;

  // phone -> word of the speaker by time
  const wordOfRef = rp.map((p) => nearestWord(p.t * ref.frameSec, refWords));

  // forced alignment of the speaker's phones on the student's posteriors (goodness of pronunciation)
  const al = ctcForcedAlign(stu.logp, stu.T, stu.V, rp.map((p) => p.id), blank);
  const forcedConf = rp.map((p, k) => {
    if (!al) return null;
    const sp2 = al.tokenSpans[k];
    if (!sp2 || sp2.start < 0) return 0;
    let best = -Infinity;
    for (let t = sp2.start; t < sp2.end; t++) best = Math.max(best, stu.logp[t * stu.V + p.id]);
    return Math.exp(best);
  });

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
    const fc = op.ri >= 0 ? forcedConf[op.ri] : null;
    const refConf = r ? r.conf : null;
    let score;
    if (op.type === 'match') score = fc == null ? 1 : 0.6 + 0.4 * Math.min(1, fc / Math.max(0.3, refConf || 0.5));
    else if (op.type === 'near') score = 0.6;
    else score = 0;
    W.items.push({ type: op.type, ref: r ? r.sym : null, stu: s ? s.sym : null, score, ins: op.type === 'ins' });
  });

  words.forEach((W, wi) => {
    const scored = W.items.filter((it) => !it.ins);
    const insPenalty = W.items.filter((it) => it.ins).length * 0.5;
    W.score = scored.length ? Math.round(100 * Math.max(0, (scored.reduce((a, b) => a + b.score, 0) - insPenalty) / scored.length)) : null;
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
  items.forEach((it, k) => {
    const r = it.ref, s = it.stu;
    if (it.type === 'match') return;
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
  if (lastRef && lastRef.type === 'del' && !isVowel(lastRef.ref)) add('final', `The final sound /${lastRef.ref}/ is missing.`);
  // generic consonant substitutions not covered above
  items.forEach((it) => {
    if (it.type === 'sub' && it.ref && it.stu && !isVowel(it.ref) && !isVowel(it.stu)
      && !issues.length) add('cons:' + it.ref, `The speaker says /${it.ref}/, you say /${it.stu}/.`);
  });
  return issues;
}
