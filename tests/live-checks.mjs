// Checks on real recordings: `node tests/live-checks.mjs [--all] [--json]`
//
// 1. The speaker's own recordings are changed in ways that do NOT change the melody
//    (shift in time, tempo, transposition, noise, telephone band, echo, loudness)
//    and must keep a high Melody score.
// 2. The same recordings are given a DIFFERENT melody (monotone, mirrored, a rise
//    laid over the phrase) by resampling at a varying rate, and must score low.
// 3. Other phrases and second takes are listed for information only: two short
//    phrases can share the same tune.
// Needs ffmpeg on the PATH (macOS: `brew install ffmpeg`). Only the melody and
// rhythm are checked here: the neural models run in the browser, not in Node.

import { execFileSync } from 'node:child_process';
import { readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildAnalysis, SR } from '../js/analysis.js';
import { compareAll, compareIntonation } from '../js/scoring.js';
import { analysePitch } from '../js/pitch.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const ALL = process.argv.includes('--all');
const JSON_OUT = process.argv.includes('--json');

function haveFfmpeg() {
  try { execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' }); return true; } catch (_) { return false; }
}
if (!haveFfmpeg()) {
  console.log('ffmpeg was not found, the checks on real recordings were skipped.');
  process.exit(0);
}

function decode(file, filters = '') {
  const af = ['aresample=' + SR, filters].filter(Boolean).join(',');
  const buf = execFileSync('ffmpeg', ['-v', 'error', '-i', file, '-af', af, '-ac', '1', '-ar', String(SR), '-f', 'f32le', '-'],
    { maxBuffer: 1 << 28 });
  return new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4)).slice();
}

function addNoise(x, snrDb) {
  let e = 0;
  for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  const sd = Math.sqrt(e / x.length) / Math.pow(10, snrDb / 20);
  let seed = 4711;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const g = Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
    out[i] = x[i] + sd * g;
  }
  return out;
}

const transpose = (st) => {
  const r = Math.pow(2, st / 12);
  return `asetrate=${Math.round(SR * r)},aresample=${SR},atempo=${(1 / r).toFixed(5)}`;
};

// name, ffmpeg filter (or function on samples), lowest acceptable Melody score
export const SAME_MELODY = [
  ['identical', '', 97],
  ['80 ms later', 'adelay=80:all=1', 90],
  ['120 ms later', 'adelay=120:all=1', 85],
  ['10% slower', 'atempo=0.9', 85],
  ['10% faster', 'atempo=1.1', 85],
  ['25% slower', 'atempo=0.75', 80],
  ['3 semitones higher', transpose(3), 85],
  ['4 semitones lower', transpose(-4), 85],
  ['much quieter', 'volume=0.15', 90],
  ['telephone band', 'highpass=f=300,lowpass=f=3400', 80],
  ['room echo', 'aecho=0.8:0.7:40:0.3', 80],
  ['noise, SNR 20 dB', (x) => addNoise(x, 20), 80],
  ['noise, SNR 10 dB', (x) => addNoise(x, 10), 65]
];

/**
 * Change the melody of a recording and keep everything else: the signal is resampled
 * at a rate that varies in time, so the pitch at every moment moves by delta(st, pos)
 * semitones (st: pitch there relative to the speaker's median, pos: 0..1 in the phrase).
 */
function reshape(x, delta) {
  const p = analysePitch(x, SR);
  const n = p.st.length;
  const st = Array.from(p.st);
  let first = st.findIndex(Number.isFinite);
  if (first < 0) return x;
  for (let i = 0; i < first; i++) st[i] = st[first];
  for (let i = first + 1, last = first; i < n; i++) {
    if (!Number.isFinite(st[i])) continue;
    for (let k = last + 1; k < i; k++) st[k] = st[last] + (st[i] - st[last]) * (k - last) / (i - last);
    last = i;
    if (i === n - 1 || !st.slice(i + 1).some(Number.isFinite)) { for (let k = i + 1; k < n; k++) st[k] = st[i]; break; }
  }
  const sm = st.map((_, i) => { let a = 0, c = 0; for (let k = Math.max(0, i - 3); k <= Math.min(n - 1, i + 3); k++) { a += st[k]; c++; } return a / c; });
  const out = [];
  let pos = 0;
  while (pos < x.length - 1) {
    const f = Math.max(0, Math.min(n - 1, Math.round((pos / SR - p.t0) / p.hop)));
    const i = Math.floor(pos), fr = pos - i;
    out.push(x[i] * (1 - fr) + x[i + 1] * fr);
    pos += Math.pow(2, delta(sm[f], pos / x.length) / 12);
  }
  return Float32Array.from(out);
}

// name, change of the pitch in semitones, highest acceptable Melody score
export const OTHER_MELODY = [
  ['monotone', (st) => -st, 60],
  ['mirrored melody', (st) => -2 * st, 40],
  ['rise of 10 semitones laid over the phrase', (st, pos) => -5 + 10 * pos, 70],
  ['fall of 10 semitones laid over the phrase', (st, pos) => 5 - 10 * pos, 70]
];
/** Monotone speech is only a different melody when the speaker's own pitch moves. */
const MIN_RANGE_FOR_MONOTONE = 4;

const lectures = readdirSync(join(ROOT, 'assets'), { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
const rows = [];
let failed = 0;

for (const lec of lectures) {
  const dir = join(ROOT, 'assets', lec);
  let files = readdirSync(dir).filter((f) => /\.mp3$/i.test(f) && !f.startsWith('._')).sort();
  if (!files.length) continue;
  const items = files.map((f) => {
    const samples = decode(join(dir, f));
    return { file: f, text: f.replace(/^\d+\s*/, '').replace(/\.mp3$/i, ''), samples, dur: samples.length / SR };
  });
  const pick = ALL ? items : items.filter((_, i) => i % 3 === 0);
  for (const it of pick) {
    const ref = buildAnalysis(it.samples, it.text, null);
    const run = (samples) => compareAll(ref, buildAnalysis(samples, it.text, null), { target: it.text, mode: 'melody' }).intonation;
    for (const [name, change, min] of SAME_MELODY) {
      const samples = typeof change === 'function' ? change(it.samples) : change ? decode(join(dir, it.file), change) : it.samples;
      const I = run(samples);
      const ok = I.ok && I.score >= min;
      if (!ok) failed++;
      rows.push({ lecture: lec, phrase: it.file, check: name, kind: 'same', score: I.score, limit: min, ok, parts: I.parts, tones: I.ok ? I.toneRef.tone + '/' + I.toneStu.tone : '' });
    }
    // the pitch arrives late against a perfect alignment (as after an alignment error)
    for (const [ms, min] of [[80, 90], [120, 80]]) {
      const I = compareIntonation(ref, ref, (t) => t - ms / 1000);
      const ok = I.ok && I.score >= min;
      if (!ok) failed++;
      rows.push({ lecture: lec, phrase: it.file, check: `alignment off by ${ms} ms`, kind: 'same', score: I.score, limit: min, ok, parts: I.parts, tones: I.ok ? I.toneRef.tone + '/' + I.toneStu.tone : '' });
    }
    // a different melody on the same words
    const self = run(it.samples);
    for (const [name, delta, max] of OTHER_MELODY) {
      if (name === 'monotone' && !(self.rangeRef >= MIN_RANGE_FOR_MONOTONE)) continue;
      const I = run(reshape(it.samples, delta));
      const ok = !I.ok || I.score <= max;
      if (!ok) failed++;
      rows.push({ lecture: lec, phrase: it.file, check: name, kind: 'different', score: I.score, limit: max, ok, parts: I.parts, tones: I.ok ? I.toneRef.tone + '/' + I.toneStu.tone : '', range: self.rangeRef });
    }
    // other phrases of a similar length, for information only
    const others = items.filter((o) => o !== it && o.text !== it.text && o.dur > it.dur * 0.7 && o.dur < it.dur * 1.4).slice(0, 3);
    for (const o of others) {
      const I = run(o.samples);
      rows.push({ lecture: lec, phrase: it.file, check: 'other phrase: ' + o.file.slice(0, 28), kind: 'other', score: I.score, limit: null, ok: true, parts: I.parts, tones: I.ok ? I.toneRef.tone + '/' + I.toneStu.tone : '' });
    }
    // a second take of the same text by the speaker, shown for information only
    for (const o of items.filter((o) => o !== it && o.text === it.text)) {
      const I = run(o.samples);
      rows.push({ lecture: lec, phrase: it.file, check: 'second take of the same phrase', kind: 'info', score: I.score, limit: null, ok: true, parts: I.parts, tones: I.ok ? I.toneRef.tone + '/' + I.toneStu.tone : '' });
    }
  }
}

if (JSON_OUT) console.log(JSON.stringify(rows));
else {
  const by = new Map();
  rows.forEach((r) => {
    const key = r.kind === 'other' ? 'other phrase (information)' : r.kind === 'info' ? r.check + ' (information)' : r.check;
    if (!by.has(key)) by.set(key, []);
    by.get(key).push(r);
  });
  console.log('check'.padEnd(44) + 'n'.padStart(4) + 'min'.padStart(6) + 'mean'.padStart(6) + 'max'.padStart(6) + '  limit  failed');
  by.forEach((list, key) => {
    const sc = list.map((r) => r.score == null ? 0 : r.score);
    const lim = list[0].limit == null ? '' : (list[0].kind === 'different' ? '<=' : '>=') + list[0].limit;
    console.log(key.padEnd(44) + String(list.length).padStart(4) + String(Math.min(...sc)).padStart(6)
      + String(Math.round(sc.reduce((a, b) => a + b, 0) / sc.length)).padStart(6) + String(Math.max(...sc)).padStart(6)
      + '  ' + lim.padEnd(6) + ' ' + list.filter((r) => !r.ok).length);
  });
  const bad = rows.filter((r) => !r.ok);
  if (bad.length) {
    console.log('\nFailed:');
    bad.forEach((r) => console.log(`  ${r.score}  ${r.check}  |  ${r.phrase}  ${r.tones}  ${r.range ? 'speaker range ' + r.range.toFixed(1) + ' st  ' : ''}${JSON.stringify(r.parts)}`));
  }
  console.log(failed ? `\n${failed} check(s) outside the limits` : '\nAll checks inside the limits');
}
process.exit(failed ? 1 : 0);
