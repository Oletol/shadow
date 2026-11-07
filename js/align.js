// Text normalisation, CTC forced alignment and word-level comparison.
// Pure functions (no browser APIs).

// Vocabulary of facebook/wav2vec2-base-960h (Xenova/wav2vec2-base-960h)
export const W2V_VOCAB = {
  "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
  E: 5, T: 6, A: 7, O: 8, N: 9, I: 10, H: 11, S: 12, R: 13, D: 14, L: 15, U: 16,
  M: 17, W: 18, C: 19, F: 20, G: 21, Y: 22, P: 23, B: 24, V: 25, K: 26, "'": 27,
  X: 28, J: 29, Q: 30, Z: 31
};
export const W2V_BLANK = 0;
export const W2V_ID2TOK = Object.fromEntries(Object.entries(W2V_VOCAB).map(([k, v]) => [v, k]));

const NUM_WORDS = {
  0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven',
  8: 'eight', 9: 'nine', 10: 'ten', 20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty'
};

/** Normalise text for comparison: lower case, plain apostrophes, letters/digits only. */
export function normaliseText(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[‘’ʼ`´]/g, "'")
    .replace(/\d+/g, (m) => NUM_WORDS[m] ? ' ' + NUM_WORDS[m] + ' ' : ' ' + m + ' ')
    .replace(/[^a-z0-9'\s]/g, ' ')
    .replace(/(^|\s)'+|'+(\s|$)/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function splitWords(text) {
  const n = normaliseText(text);
  return n ? n.split(' ') : [];
}

/**
 * Turn the target phrase into CTC labels.
 * Returns {labels:number[], wordOf:number[], words:string[]} where wordOf[k] is the word
 * index of label k (-1 for the word separator "|").
 */
export function phraseToLabels(text) {
  const words = splitWords(text);
  const labels = [], wordOf = [];
  words.forEach((w, wi) => {
    if (wi > 0) { labels.push(W2V_VOCAB['|']); wordOf.push(-1); }
    for (const ch of w.toUpperCase()) {
      const id = W2V_VOCAB[ch];
      if (id === undefined) continue;
      labels.push(id); wordOf.push(wi);
    }
  });
  return { labels, wordOf, words };
}

/** Row-wise log-softmax of logits [T, V] stored flat. */
export function logSoftmax(flat, T, V) {
  const out = new Float32Array(T * V);
  for (let t = 0; t < T; t++) {
    let m = -Infinity;
    for (let v = 0; v < V; v++) if (flat[t * V + v] > m) m = flat[t * V + v];
    let s = 0;
    for (let v = 0; v < V; v++) s += Math.exp(flat[t * V + v] - m);
    const lse = m + Math.log(s);
    for (let v = 0; v < V; v++) out[t * V + v] = flat[t * V + v] - lse;
  }
  return out;
}

/** Greedy CTC decoding (no language model): what the acoustics alone say. */
export function greedyDecode(logp, T, V, from = 0, to = T) {
  let prev = -1, out = '';
  for (let t = from; t < to; t++) {
    let best = 0, bv = -Infinity;
    for (let v = 0; v < V; v++) if (logp[t * V + v] > bv) { bv = logp[t * V + v]; best = v; }
    if (best !== prev && best !== W2V_BLANK) {
      const tok = W2V_ID2TOK[best];
      if (tok === '|') out += ' ';
      else if (tok && tok.length === 1) out += tok;
    }
    prev = best;
  }
  return out.replace(/\s+/g, ' ').trim().toLowerCase();
}

/**
 * Viterbi CTC forced alignment (same algorithm as torchaudio.functional.forced_align).
 * @returns {null | {path:number[], tokenSpans:{start:number,end:number,label:number}[]}}
 *   path[t] = index into labels or -1 for blank; spans are [start, end) in frames.
 */
export function ctcForcedAlign(logp, T, V, labels, blank = W2V_BLANK) {
  const L = labels.length;
  if (!L) return null;
  const S = 2 * L + 1;
  // minimal frames needed: L + number of repeated neighbours
  let need = L;
  for (let k = 1; k < L; k++) if (labels[k] === labels[k - 1]) need++;
  if (T < need) return null;

  const NEG = -1e30;
  const ext = new Int32Array(S);
  for (let s = 0; s < S; s++) ext[s] = s % 2 === 0 ? blank : labels[(s - 1) >> 1];

  let prev = new Float64Array(S).fill(NEG);
  let cur = new Float64Array(S);
  const back = new Int8Array(T * S); // 0: stay, 1: from s-1, 2: from s-2

  prev[0] = logp[ext[0]];
  prev[1] = logp[ext[1]];
  for (let t = 1; t < T; t++) {
    const row = t * V;
    for (let s = 0; s < S; s++) {
      let best = prev[s], how = 0;
      if (s >= 1 && prev[s - 1] > best) { best = prev[s - 1]; how = 1; }
      if (s >= 2 && ext[s] !== blank && ext[s] !== ext[s - 2] && prev[s - 2] > best) {
        best = prev[s - 2]; how = 2;
      }
      cur[s] = best <= NEG / 2 ? NEG : best + logp[row + ext[s]];
      back[t * S + s] = how;
    }
    const sw = prev; prev = cur; cur = sw;
  }
  let s = prev[S - 1] >= prev[S - 2] ? S - 1 : S - 2;
  if (prev[s] <= NEG / 2) return null;

  const stateAt = new Int32Array(T);
  for (let t = T - 1; t >= 0; t--) {
    stateAt[t] = s;
    if (t > 0) {
      const how = back[t * S + s];
      s -= how;
    }
  }
  const path = new Array(T);
  for (let t = 0; t < T; t++) path[t] = stateAt[t] % 2 === 0 ? -1 : (stateAt[t] - 1) >> 1;

  const tokenSpans = labels.map((label) => ({ start: -1, end: -1, label }));
  for (let t = 0; t < T; t++) {
    const k = path[t];
    if (k < 0) continue;
    if (tokenSpans[k].start < 0) tokenSpans[k].start = t;
    tokenSpans[k].end = t + 1;
  }
  return { path, tokenSpans };
}

/**
 * Word-level result of forced alignment: timing and acoustic confidence per word.
 * Confidence of a letter = highest posterior of that letter inside its span;
 * confidence of a word = mean letter confidence.
 */
export function wordsFromAlignment(logp, T, V, lab, align, frameSec) {
  const { labels, wordOf, words } = lab;
  const out = words.map((w) => ({ word: w, start: Infinity, end: -Infinity, letters: [] }));
  align.tokenSpans.forEach((sp, k) => {
    const wi = wordOf[k];
    if (wi < 0 || sp.start < 0) return;
    let best = -Infinity;
    for (let t = sp.start; t < sp.end; t++) best = Math.max(best, logp[t * V + labels[k]]);
    out[wi].letters.push({ ch: W2V_ID2TOK[labels[k]], conf: Math.exp(best), start: sp.start, end: sp.end });
    out[wi].start = Math.min(out[wi].start, sp.start);
    out[wi].end = Math.max(out[wi].end, sp.end);
  });
  for (let i = 0; i < out.length; i++) {
    const w = out[i];
    const confs = w.letters.map((l) => l.conf);
    w.conf = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0;
    w.minConf = confs.length ? Math.min(...confs) : 0;
    // what the acoustic model hears inside the word (+-1 frame), no language model
    const a = Math.max(0, w.start - 1), b = Math.min(T, w.end + 1);
    w.heard = Number.isFinite(w.start) ? greedyDecode(logp, T, V, a, b).replace(/\s/g, '') : '';
    w.startFrame = w.start; w.endFrame = w.end;
    w.start = Number.isFinite(w.start) ? w.start * frameSec : NaN;
    w.end = Number.isFinite(w.end) ? w.end * frameSec : NaN;
  }
  // Extend word ends towards the next word (CTC spikes are short): split the gap
  // only when it is short; long gaps are pauses and stay unassigned.
  for (let i = 0; i < out.length - 1; i++) {
    const gap = out[i + 1].start - out[i].end;
    if (gap > 0 && gap < 0.12) { out[i].end += gap / 2; out[i + 1].start -= gap / 2; }
    else if (gap >= 0.12) { out[i].end += 0.06; out[i].pauseAfter = gap - 0.06; }
  }
  if (out.length) out[out.length - 1].end += 0.06;
  return out;
}

/** Word-level Levenshtein alignment between target and recognised word lists. */
export function alignWordLists(target, heard) {
  const n = target.length, m = heard.length;
  const D = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = 0; i <= n; i++) D[i][0] = i;
  for (let j = 0; j <= m; j++) D[0][j] = j;
  const sim = (a, b) => a === b || charSimilarity(a, b) >= 0.8;
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) {
    D[i][j] = Math.min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (sim(target[i - 1], heard[j - 1]) ? 0 : 1));
  }
  const ops = [];
  let i = n, j = m;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && D[i][j] === D[i - 1][j - 1] + (sim(target[i - 1], heard[j - 1]) ? 0 : 1)) {
      ops.push({ type: sim(target[i - 1], heard[j - 1]) ? 'ok' : 'sub', target: target[i - 1], heard: heard[j - 1], ti: i - 1 });
      i--; j--;
    } else if (i > 0 && D[i][j] === D[i - 1][j] + 1) {
      ops.push({ type: 'del', target: target[i - 1], heard: '', ti: i - 1 }); i--;
    } else {
      ops.push({ type: 'ins', target: '', heard: heard[j - 1], ti: -1 }); j--;
    }
  }
  ops.reverse();
  const errors = ops.filter((o) => o.type !== 'ok').length;
  const accuracy = n ? Math.max(0, 1 - errors / n) : 0;
  return { ops, accuracy, errors };
}

/** Character-level similarity 0..1 (1 - normalised edit distance). */
export function charSimilarity(a, b) {
  a = String(a || ''); b = String(b || '');
  if (!a.length && !b.length) return 1;
  const m = a.length, n = b.length;
  let prev = new Int32Array(n + 1), cur = new Int32Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    cur[0] = i;
    for (let j = 1; j <= n; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    const s = prev; prev = cur; cur = s;
  }
  return 1 - prev[n] / Math.max(m, n);
}

/**
 * Build a monotonic piecewise-linear time map from reference time to student time,
 * anchored at word boundaries of both recordings.
 */
export function buildTimeMap(refWords, stuWords, refSpan, stuSpan) {
  const anchors = [[refSpan[0], stuSpan[0]]];
  const n = Math.min(refWords.length, stuWords.length);
  for (let i = 0; i < n; i++) {
    const r = refWords[i], s = stuWords[i];
    if ([r.start, r.end, s.start, s.end].every(Number.isFinite)) {
      anchors.push([r.start, s.start]);
      anchors.push([r.end, s.end]);
    }
  }
  anchors.push([refSpan[1], stuSpan[1]]);
  anchors.sort((a, b) => a[0] - b[0]);
  // enforce monotonicity in both axes
  const clean = [];
  for (const a of anchors) {
    const last = clean[clean.length - 1];
    if (!last || (a[0] > last[0] + 1e-4 && a[1] > last[1] + 1e-4)) clean.push(a);
  }
  return function map(t) {
    if (clean.length < 2) return t;
    if (t <= clean[0][0]) return clean[0][1] + (t - clean[0][0]);
    for (let k = 1; k < clean.length; k++) {
      if (t <= clean[k][0]) {
        const [r0, s0] = clean[k - 1], [r1, s1] = clean[k];
        return s0 + (t - r0) * (s1 - s0) / (r1 - r0);
      }
    }
    const last = clean[clean.length - 1];
    return last[1] + (t - last[0]);
  };
}
