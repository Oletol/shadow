// Shadowing practice: page logic.
import { fetchAudio16k, decodeTo16k, Recorder, playSamples, audioContext, SR } from './audio.js';
import { buildAnalysis } from './analysis.js';
import { compareAll, buildFeedback, verdict, STRICTNESS } from './scoring.js';
import { drawStave, drawWave, drawSpectrogram } from './charts.js';
import { splitWords } from './align.js';
import { yinFrameHz } from './pitch.js';
import { comparePhonemes } from './phonemes.js';
import { inScorm, startReporting, report, phraseKey } from './report.js';

// Practice steps (a sequence: melody first, then shadowing, then alone)
const MODES = {
  melody: { record: 'Hum the melody', play: false, rate: 1, scoring: 'melody',
    tip: 'Listen two or three times, then hum the tune on “mm” or “da-da”. Only the melody is scored here.' },
  slow: { record: 'Shadow at ¾ speed', play: true, rate: 0.75, scoring: 'speech',
    tip: 'The speaker plays at three-quarter speed while you speak at the same time. Wear headphones.' },
  shadow: { record: 'Shadow', play: true, rate: 1, scoring: 'speech',
    tip: 'The speaker plays at normal speed while you speak at the same time. Wear headphones.' },
  own: { record: 'Record', play: false, rate: 1, scoring: 'speech',
    tip: 'Say the phrase on your own, without the speaker. Then check your sounds in IPA.' }
};
const MODE_ORDER = ['melody', 'slow', 'shadow', 'own'];
const PASS = 80;

const $ = (id) => document.getElementById(id);
const LS = 'shadowing.v3.';
const store = {
  get: (k, d) => { try { const v = localStorage.getItem(LS + k); return v == null ? d : v; } catch (_) { return d; } },
  set: (k, v) => { try { localStorage.setItem(LS + k, v); } catch (_) {} }
};

const S = {
  cfg: null,
  idx: 0,
  token: 0,
  ref: null,          // { idx, samples, base, full, whisper }
  stu: null,          // { samples, analysis, whisper, url }
  result: null,
  highlight: null,
  playhead: NaN,
  strictness: store.get('strictness', null),
  refCache: new Map(),
  lecture: null,     // current lecture { id, title, full: [{title, audio}], phrases }
  lectureIdx: 0,
  mode: store.get('mode', 'melody'),
  live: null,         // live pitch points while recording
  loop: null,         // stop() of the running loop
  attempts: 0,        // recordings made on this page (numbers the attempts sent to Moodle)
  listens: 0          // times the student played the speaker since the last recording
};
if (!MODES[S.mode]) S.mode = 'melody';

// ---------------------------------------------------------------------------
// Model worker
// ---------------------------------------------------------------------------
function deferred() {
  let resolve, reject;
  const p = new Promise((a, b) => { resolve = a; reject = b; });
  p.catch(() => {});
  return { p, resolve, reject, state: 'pending' };
}
const models = { ctc: deferred(), asr: deferred(), phon: deferred() };
let worker = null, msgId = 0;
const pending = new Map();

function setModelStatus(which, state, text) {
  $('dot-' + which).className = 'dot ' + state;
  $('st-' + which).textContent = text;
}

function startWorker(cfg) {
  try {
    worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  } catch (e) {
    ['ctc', 'asr'].forEach((m) => { models[m].state = 'failed'; models[m].reject(e); setModelStatus(m, 'failed', 'not available in this browser'); });
    return;
  }
  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'progress') {
      const pct = m.total ? Math.round(100 * m.loaded / m.total) : 0;
      setModelStatus(m.model, 'loading', `downloading ${pct}%`);
      if (m.model === 'ctc' && S.waitingForModel) showProgress('Preparing the sound analysis model (first time only)', pct);
      if (m.model === 'phon' && S.waitingForPhon) showProgress('Downloading the IPA sound model (first time only)', pct);
    } else if (m.type === 'ready') {
      models[m.model].state = 'ready';
      models[m.model].resolve(true);
      setModelStatus(m.model, 'ready', 'ready');
      if (m.model === 'phon') store.set('phonLoaded', '1');
    } else if (m.type === 'load-error') {
      models[m.model].state = 'failed';
      models[m.model].reject(new Error(m.error));
      setModelStatus(m.model, 'failed', 'could not load');
      console.warn(m.model, m.error);
      report('problem', { what: 'model-' + m.model, detail: String(m.error).slice(0, 120) });
    } else if (m.type === 'result' || m.type === 'error') {
      const p = pending.get(m.id);
      if (!p) return;
      pending.delete(m.id);
      m.type === 'result' ? p.resolve(m.result) : p.reject(new Error(m.error));
    }
  };
  worker.onerror = (e) => {
    console.error(e);
    ['ctc', 'asr', 'phon'].forEach((k) => {
      if (models[k].state === 'pending') { models[k].state = 'failed'; models[k].reject(new Error('worker failed')); setModelStatus(k, 'failed', 'could not start'); }
    });
  };
  worker.postMessage({ type: 'config', config: {
    ctcModel: cfg.models && cfg.models.alignment,
    asrModel: cfg.models && cfg.models.recognition,
    modelHost: cfg.models && cfg.models.host
  } });
  setModelStatus('ctc', 'loading', 'starting');
  setModelStatus('asr', 'loading', 'waiting');
  worker.postMessage({ type: 'load' });
  // the IPA model is large: load it only on request, or straight away if this
  // browser already has it (or the teacher switched it on for everyone)
  if (cfg.phonemes === 'auto' || store.get('phonLoaded', '0') === '1') requestPhon();
}

let phonRequested = false;
function requestPhon() {
  if (phonRequested || !worker) return;
  phonRequested = true;
  setModelStatus('phon', 'loading', 'starting');
  worker.postMessage({ type: 'load-phon' });
}
async function runPhon(samples) {
  requestPhon();
  await models.phon.p;
  const r = await callWorker('phon', samples);
  return { ...r, frameSec: samples.length / SR / r.T };
}

function callWorker(type, audio) {
  return new Promise((resolve, reject) => {
    if (!worker) return reject(new Error('Models are not available.'));
    const id = ++msgId;
    pending.set(id, { resolve, reject });
    const copy = Float32Array.from(audio);
    worker.postMessage({ type, id, audio: copy }, [copy.buffer]);
  });
}

async function runCtc(samples) {
  await models.ctc.p;
  return callWorker('ctc', samples);
}
async function runAsr(samples, span) {
  await models.asr.p;
  const a = Math.max(0, Math.floor((span[0] - 0.25) * SR)), b = Math.min(samples.length, Math.ceil((span[1] + 0.25) * SR));
  const r = await callWorker('asr', samples.subarray(a, b));
  return r.text;
}

// ---------------------------------------------------------------------------
// Status line
// ---------------------------------------------------------------------------
function status(text, kind) {
  const el = $('status');
  el.className = 'status' + (kind ? ' ' + kind : '');
  el.textContent = text || '';
}
function showProgress(text, pct) {
  const el = $('status');
  el.className = 'status';
  el.innerHTML = '';
  el.append(text);
  const bar = document.createElement('span');
  bar.className = 'bar';
  const fill = document.createElement('i');
  fill.style.width = (pct || 0) + '%';
  bar.append(fill);
  el.append(bar);
}

// ---------------------------------------------------------------------------
// Phrase loading
// ---------------------------------------------------------------------------
const modelAudio = new Audio();
modelAudio.crossOrigin = 'anonymous';
modelAudio.preload = 'auto';
try { modelAudio.preservesPitch = true; modelAudio.mozPreservesPitch = true; modelAudio.webkitPreservesPitch = true; } catch (_) {}

function phrase() { return S.lecture.phrases[S.idx]; }

async function selectPhrase(idx) {
  const n = S.lecture.phrases.length;
  S.idx = ((idx % n) + n) % n;
  store.set('idx.' + S.lecture.id, String(S.idx));
  const token = ++S.token;
  const p = phrase();
  S.listens = 0;
  report('phrase', { lecture: S.lecture.id, phrase: S.idx + 1, text: p.text });
  $('phrase-select').value = String(S.idx);
  $('phrase').textContent = p.text;
  $('phrase-count').textContent = `Phrase ${S.idx + 1} of ${n}`;
  $('ref-note').hidden = true;
  modelAudio.pause();
  modelAudio.src = p.audio;
  clearStudent();
  S.ref = null;
  $('stave-msg').textContent = 'Loading the speaker’s recording…';
  renderStave();
  renderSteps();
  status('');

  try {
    let ref = S.refCache.get(p.audio);
    if (!ref) {
      const samples = await fetchAudio16k(p.audio);
      ref = { text: p.text, samples, base: buildAnalysis(samples, p.text, null), full: null, whisper: null };
      S.refCache.set(p.audio, ref);
    }
    if (token !== S.token) return;
    S.ref = ref;
    $('stave-msg').textContent = ref.base.pitch.voicedRatio < 0.05 ? 'No melody could be found in this recording.' : '';
    renderStave();
    prepareReferenceModels(ref, token);
  } catch (e) {
    if (token !== S.token) return;
    $('stave-msg').textContent = 'The speaker’s recording could not be loaded.';
    status(e.message, 'error');
  }
}

async function prepareReferenceModels(ref, token) {
  const p = { text: ref.text };
  if (!ref.full) {
    try {
      const ctc = await runCtc(ref.samples);
      ref.full = buildAnalysis(ref.samples, p.text, ctc);
    } catch (e) {
      console.warn('Reference alignment unavailable:', e);
      return;
    }
  }
  if (token === S.token) renderStave();
  if (!S.cfg.showReferenceCheck) return;
  if (ref.whisper == null) {
    try { ref.whisper = await runAsr(ref.samples, refAnalysis(ref).span); } catch (_) { return; }
  }
  if (token !== S.token) return;
  const target = splitWords(p.text).join(' ');
  const heard = splitWords(ref.whisper).join(' ');
  if (heard && heard !== target) {
    const note = $('ref-note');
    note.textContent = `Teacher check: the recogniser hears “${ref.whisper}” in the speaker’s recording. If the text above does not match the audio, rename the audio file.`;
    note.hidden = false;
  }
}

function refAnalysis(ref = S.ref) { return ref ? (ref.full || ref.base) : null; }

// ---------------------------------------------------------------------------
// Stave
// ---------------------------------------------------------------------------
function staveData() {
  const A = refAnalysis();
  if (!A) return null;
  const P = A.pitch;
  const times = [], vals = [];
  for (let i = 0; i < P.st.length; i++) {
    const t = P.t0 + i * P.hop;
    if (t < A.span[0] || t > A.span[1]) continue;
    times.push(t); vals.push(P.st[i]);
  }
  const d = { times, ref: vals, span: A.span, words: A.words, highlight: S.highlight, playhead: S.playhead };
  if (S.live) { d.live = S.live; return d; }
  const R = S.result;
  if (R && R.intonation && R.intonation.ok) {
    // student line, already on the speaker's time axis
    const map = new Map(R.intonation.times.map((t, k) => [t.toFixed(4), R.intonation.stuVals[k]]));
    d.stu = times.map((t) => { const v = map.get(t.toFixed(4)); return v === undefined ? NaN : v; });
    d.wordColors = (A.words || []).map((_, i) => colorForScore(wordScore(i), true));
  }
  return d;
}

function renderStave() {
  const d = staveData();
  const canvas = $('stave');
  if (!d) { drawStave(canvas, { times: [], ref: null, span: [0, 1] }); canvas.classList.remove('clickable'); return; }
  drawStave(canvas, d);
  canvas.classList.toggle('clickable', !!d.words);
}

function wordAtX(x) {
  const A = refAnalysis();
  if (!A || !A.words) return -1;
  const w = $('stave').getBoundingClientRect().width;
  const t = A.span[0] + (x - 8) / (w - 16) * (A.span[1] - A.span[0]);
  let best = -1, bd = Infinity;
  A.words.forEach((wd, i) => {
    const d = t < wd.start ? wd.start - t : t > wd.end ? t - wd.end : 0;
    if (d < bd) { bd = d; best = i; }
  });
  return bd < 0.25 ? best : -1;
}

// ---------------------------------------------------------------------------
// Listening
// ---------------------------------------------------------------------------
let rafId = 0;
function animatePlayhead() {
  cancelAnimationFrame(rafId);
  const step = () => {
    if (modelAudio.paused) { S.playhead = NaN; renderStave(); return; }
    S.playhead = modelAudio.currentTime;
    renderStave();
    rafId = requestAnimationFrame(step);
  };
  rafId = requestAnimationFrame(step);
}

function listen(rate) {
  audioContext();
  modelAudio.pause();
  modelAudio.currentTime = 0;
  modelAudio.playbackRate = rate;
  const p = modelAudio.play();
  if (p && p.catch) p.catch((e) => status('The recording could not start: ' + e.message, 'error'));
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) animatePlayhead();
}

let stopWordPlayback = null;
function playWord(i) {
  const A = refAnalysis();
  if (!A || !A.words || !A.words[i]) return;
  if (stopWordPlayback) stopWordPlayback();
  stopLoop();
  S.highlight = i;
  S.loopWord = i;
  updateLoopLabel();
  renderStave();
  const w = A.words[i];
  const stuA = S.stu && S.stu.analysis;
  const sw = stuA && stuA.words && stuA.words[i];
  let cancelled = false;
  let stopCur = playSamples(A.samples, Math.max(0, w.start - 0.06), w.end + 0.06, () => {
    if (cancelled) return;
    if (sw && Number.isFinite(sw.start)) {
      setTimeout(() => {
        if (cancelled) return;
        stopCur = playSamples(stuA.samples, Math.max(0, sw.start - 0.06), sw.end + 0.06, () => { S.highlight = null; renderStave(); });
      }, 250);
    } else { S.highlight = null; renderStave(); }
  });
  stopWordPlayback = () => { cancelled = true; stopCur(); };
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------
const recorder = new Recorder(onMicFrame);
let autoStopTimer = 0;
let liveState = null;

function onMicFrame(lvl, data, sr) {
  $('level').style.width = Math.round(lvl * 100) + '%';
  if (!liveState || !recorder.recording || !data) return;
  // downsample to about 16 kHz and estimate the pitch of this frame
  const f = Math.max(1, Math.round(sr / 16000));
  const n = Math.floor(data.length / f);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s2 = 0; for (let k = 0; k < f; k++) s2 += data[i * f + k]; x[i] = s2 / f; }
  const hz = yinFrameHz(x, sr / f, 70, 420);
  if (!hz) { renderStave(); return; }
  const A = refAnalysis();
  const now = performance.now();
  let t;
  if (MODES[liveState.mode].play) t = modelAudio.currentTime;            // speaker's timeline
  else {
    if (liveState.start == null) liveState.start = now;
    t = (A ? A.span[0] : 0) + (now - liveState.start) / 1000;
  }
  if (!liveState.median) {
    liveState.first.push(hz);
    if (liveState.first.length < 12) return;
    const v = [...liveState.first].sort((a2, b2) => a2 - b2);
    liveState.median = v[v.length >> 1];
  }
  const st = 12 * Math.log2(hz / liveState.median);
  const prev = S.live[S.live.length - 1];
  // ignore octave jumps of single frames
  if (prev && Math.abs(st - prev.st) > 7 && t - prev.t < 0.05) return;
  S.live.push({ t, st });
  renderStave();
}

async function toggleRecording() {
  if (recorder.recording) return stopRecording();
  if (!Recorder.supported()) {
    status('This browser cannot record from the microphone. Open the page over HTTPS in Chrome, Edge, Firefox or Safari.', 'error');
    return;
  }
  fullAudio().pause();
  stopLoop();
  try {
    audioContext();
    await recorder.start();
  } catch (e) {
    status('Microphone access was blocked. Allow the microphone for this site and try again.', 'error');
    report('problem', { what: 'microphone', detail: e && e.name });
    return;
  }
  const mode = S.mode;
  const M = MODES[mode];
  S.recMode = mode;
  S.result = null;
  S.live = [];
  const savedMedian = parseFloat(store.get('voiceMedian', '0'));
  liveState = { mode, start: null, first: [], median: savedMedian > 50 ? savedMedian : 0 };
  $('results').hidden = true;
  $('btn-record').classList.add('is-recording');
  $('record-label').textContent = 'Stop';
  status(M.play ? 'Recording… speak together with the speaker.' : mode === 'melody' ? 'Recording… hum the melody, then press Stop.' : 'Recording… say the phrase, then press Stop.');
  clearTimeout(autoStopTimer);
  if (M.play) {
    listen(M.rate);
    modelAudio.onended = () => { clearTimeout(autoStopTimer); autoStopTimer = setTimeout(stopRecording, 1200); };
  } else {
    modelAudio.onended = null;
  }
  autoStopTimer = setTimeout(() => recorder.recording && stopRecording(), 20000);
}

async function stopRecording() {
  clearTimeout(autoStopTimer);
  modelAudio.onended = null;
  const blob = await recorder.stop();
  if (MODES[S.recMode] && MODES[S.recMode].play) modelAudio.pause();
  liveState = null;
  $('btn-record').classList.remove('is-recording');
  $('record-label').textContent = MODES[S.mode].record;
  $('level').style.width = '0';
  if (!blob || !blob.size) { status('Nothing was recorded. Check the microphone.', 'error'); return; }
  try {
    const samples = await decodeTo16k(await blob.arrayBuffer());
    if (S.stu && S.stu.url) URL.revokeObjectURL(S.stu.url);
    S.stu = { samples, analysis: null, whisper: null, phon: null, mode: S.recMode, url: URL.createObjectURL(blob),
      attempt: ++S.attempts, listens: S.listens, duration: samples.length / SR };
    S.listens = 0;
    await analyseStudent();
  } catch (e) {
    console.error(e);
    status('Your recording could not be read: ' + e.message, 'error');
  }
}

function clearStudent() {
  if (S.stu && S.stu.url) URL.revokeObjectURL(S.stu.url);
  S.stu = null;
  S.result = null;
  S.highlight = null;
  S.live = null;
  S.loopWord = null;
  stopLoop();
  updateLoopLabel();
  $('results').hidden = true;
  $('record-label').textContent = MODES[S.mode].record;
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------
async function analyseStudent() {
  const token = S.token;
  const ref = S.ref;
  const p = phrase();
  if (!ref) { status('The speaker’s recording is not ready yet.', 'error'); return; }

  // quick check: is there any voice?
  const quick = buildAnalysis(S.stu.samples, p.text, null);
  if (quick.pitch.voicedRatio < 0.03 || quick.span[1] - quick.span[0] < 0.25) {
    S.live = null; renderStave();
    status('No voice was heard in your recording. Check that the right microphone is selected and speak a little louder.', 'error');
    return;
  }
  store.set('voiceMedian', String(Math.round(quick.pitch.median)));

  // humming: only the melody is compared, no speech models needed
  if (S.stu.mode === 'melody') {
    S.stu.analysis = quick;
    S.live = null;
    computeAndRender();
    status('Done. When the melody reaches ' + PASS + ', move on to step 2.');
    return;
  }

  // the alignment model gives word timing and sound quality
  let stuA = quick;
  if (models.ctc.state !== 'failed') {
    S.waitingForModel = models.ctc.state !== 'ready';
    if (S.waitingForModel) showProgress('Preparing the sound analysis model (first time only)', 0);
    try {
      await models.ctc.p;
      S.waitingForModel = false;
      status('Analysing your recording…');
      if (!ref.full) {
        ref.full = buildAnalysis(ref.samples, ref.text, await runCtc(ref.samples));
      }
      const ctc = await runCtc(S.stu.samples);
      stuA = buildAnalysis(S.stu.samples, p.text, ctc);
    } catch (e) {
      S.waitingForModel = false;
      console.warn('alignment failed', e);
    }
  }
  if (token !== S.token || !S.stu) return;
  S.stu.analysis = stuA;
  S.live = null;
  computeAndRender();

  const partial = !(refAnalysis().words && stuA.words);
  status(partial
    ? 'Melody and timing were compared. Word-level results need the sound analysis model, which is not available right now.'
    : 'Done. Checking the words with the speech recogniser…');

  // Whisper runs last; results are updated when it finishes
  if (models.asr.state !== 'failed') {
    try {
      const text = await runAsr(S.stu.samples, stuA.span);
      if (token !== S.token || !S.stu) return;
      S.stu.whisper = text;
      computeAndRender();
      status('Done. Select a word in the melody or in the table to compare it with the speaker.');
    } catch (e) {
      if (!partial) status('Done. The speech recogniser is not available, so words were checked by sound only.');
    }
  }
  // IPA comparison runs by itself once its model is in this browser
  if (!partial && models.phon.state === 'ready') runIpa(true);
}

async function runIpa(quiet) {
  const token = S.token;
  const ref = S.ref, stu = S.stu;
  const refA = refAnalysis();
  if (!ref || !stu || !stu.analysis || !refA || !refA.words) return;
  const btn = $('btn-ipa');
  btn.disabled = true;
  try {
    S.waitingForPhon = models.phon.state !== 'ready';
    if (S.waitingForPhon) showProgress('Downloading the IPA sound model (first time only)', 0);
    else if (!quiet) status('Comparing your sounds with the speaker…');
    if (!ref.phon) ref.phon = await runPhon(ref.samples);
    S.waitingForPhon = false;
    if (!quiet) status('Comparing your sounds with the speaker…');
    const sp = await runPhon(stu.samples);
    if (token !== S.token || stu !== S.stu) return;
    stu.phon = comparePhonemes(ref.phon, sp, sp.id2sym, refA.words, sp.blank);
    computeAndRender();
    if (stu.phon) report('ipa', { lecture: S.lecture.id, phrase: S.idx + 1 });
    status(stu.phon ? 'Done. The table now shows the sounds of each word in IPA.' : 'The speaker’s sounds could not be recognised in IPA.');
  } catch (e) {
    S.waitingForPhon = false;
    console.warn(e);
    status('The IPA sound check is not available right now: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
}

function computeAndRender() {
  const refA = refAnalysis();
  if (!refA || !S.stu || !S.stu.analysis) return;
  const strictness = S.strictness || S.cfg.strictness || 'standard';
  const M = MODES[S.stu.mode] || MODES.own;
  S.result = compareAll(refA, S.stu.analysis, {
    strictness,
    mode: M.scoring,
    expectedTempo: M.play ? 1 / M.rate : 1,
    target: phrase().text,
    whisperText: S.stu.whisper,
    acousticText: S.stu.analysis.acoustic,
    phonemes: S.stu.phon
  });
  saveBest(S.stu.mode, S.result.overall);
  reportAttempt(strictness);
  renderResults();
  renderStave();
}

// One attempt = one recording. It is sent again with the same number when the
// result is refined (speech recogniser, IPA check, strictness), so the latest
// numbers replace the earlier ones.
function reportAttempt(strictness) {
  const R = S.result, st = S.stu, p = phrase();
  if (!R || !st || R.overall == null) return;
  const sc = (x) => (x && x.score != null ? x.score : null);
  const speech = R.mode !== 'melody';   // humming: only melody and timing are scored
  report('attempt', {
    id: st.attempt, lecture: S.lecture.id, phrase: S.idx + 1, text: p.text, key: phraseKey(p),
    step: st.mode, overall: R.overall, passed: R.overall >= PASS,
    melody: sc(R.intonation), rhythm: sc(R.rhythm),
    sounds: speech ? sc(R.pronunciation) : null, stress: speech ? sc(R.stress) : null,
    words: speech && R.words && R.words.accuracy != null ? Math.round(R.words.accuracy * 100) : null,
    listens: st.listens, duration: Math.round(st.duration * 10) / 10,
    strictness, ipa: !!st.phon
  });
}

// ---------------------------------------------------------------------------
// Practice steps and best scores
// ---------------------------------------------------------------------------
// In the Moodle package the best scores belong to the student's Moodle record
// (they follow the student to any computer); elsewhere they stay in this browser.
const scormBest = new Map();
let scormState = null;       // set when the package has sent what Moodle saved
const bestKey = (mode, p = phrase()) => 'best.' + p.audio + '.' + mode;
function getBest(mode, p) {
  const k = bestKey(mode, p);
  if (inScorm) return scormBest.has(k) ? scormBest.get(k) : -1;
  return parseInt(store.get(k, '-1'), 10);
}
function saveBest(mode, score) {
  if (score == null || !S.lecture) return;
  const prev = getBest(mode);
  if (score > prev) {
    if (inScorm) { scormBest.set(bestKey(mode), score); sendProgress(); }
    else store.set(bestKey(mode), String(score));
  }
  renderSteps();
}

// What Moodle keeps: best score per step for every phrase (restored next time),
// the grade = average best "On your own" score over all phrases of the package,
// and "completed" once every phrase has passed "On your own".
function sendProgress() {
  if (!inScorm || !scormState) return;
  const b = {};
  let sum = 0, n = 0, all = true;
  for (const L of S.cfg.lectures) {
    for (const p of L.phrases) {
      const row = MODE_ORDER.map((m) => getBest(m, p));
      if (row.some((v) => v >= 0)) b[phraseKey(p)] = row;
      n++;
      sum += Math.max(0, row[3]);
      if (row[3] < PASS) all = false;
    }
  }
  report('progress', { suspend: JSON.stringify({ v: 1, b }), score: n ? Math.round(sum / n) : 0, completed: n > 0 && all, phrases: n });
}

function restoreFromMoodle(state) {
  scormState = state;
  let saved = null;
  try { saved = state.suspend ? JSON.parse(state.suspend) : null; } catch (_) {}
  const b = saved && saved.v === 1 && saved.b ? saved.b : {};
  for (const L of S.cfg.lectures) {
    for (const p of L.phrases) {
      const row = b[phraseKey(p)];
      if (!Array.isArray(row)) continue;
      MODE_ORDER.forEach((m, i) => {
        const v = Number(row[i]), k = bestKey(m, p);
        if (Number.isFinite(v) && v > (scormBest.has(k) ? scormBest.get(k) : -1)) scormBest.set(k, v);
      });
    }
  }
  renderSteps();
  if (scormBest.size) sendProgress();
}

function renderSteps() {
  if (!S.lecture) return;
  document.querySelectorAll('.step').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === S.mode)));
  document.querySelectorAll('[data-best]').forEach((el) => {
    const v = getBest(el.dataset.best);
    el.textContent = v >= 0 ? String(v) : '';
    el.classList.toggle('passed', v >= PASS);
    el.title = v >= 0 ? 'Your best score for this step' : '';
  });
  $('step-tip').textContent = MODES[S.mode].tip;
  if (!recorder.recording) $('record-label').textContent = MODES[S.mode].record;
}
function setMode(mode) {
  if (!MODES[mode] || recorder.recording) return;
  S.mode = mode;
  store.set('mode', mode);
  renderSteps();
}

// ---------------------------------------------------------------------------
// Loop a word or the ending of the phrase (speaker only)
// ---------------------------------------------------------------------------
function loopRange() {
  const A = refAnalysis();
  if (!A) return null;
  if (A.words && S.loopWord != null && A.words[S.loopWord]) {
    const w = A.words[S.loopWord];
    return [Math.max(0, w.start - 0.06), w.end + 0.08];
  }
  const R = S.result && S.result.intonation;
  let from = A.span[0] + (A.span[1] - A.span[0]) * 0.55;
  if (A.words && A.words.length) from = A.words[A.words.length - 1].start - 0.05;
  if (R && Number.isFinite(R.nucStart)) from = R.nucStart - 0.05;
  return [Math.max(0, from), A.span[1]];
}
function updateLoopLabel() {
  const A = refAnalysis();
  const w = A && A.words && S.loopWord != null && A.words[S.loopWord];
  $('loop-label').textContent = S.loop ? 'Stop the loop' : w ? `Loop “${w.word}”` : 'Loop the ending';
  $('btn-loop').setAttribute('aria-pressed', String(!!S.loop));
}
function stopLoop() {
  if (S.loop) { S.loop(); S.loop = null; }
  if ($('btn-loop')) updateLoopLabel();
}
function toggleLoop() {
  if (S.loop) return stopLoop();
  const A = refAnalysis();
  const r = loopRange();
  if (!A || !r) return;
  modelAudio.pause();
  let stopped = false, stopCur = () => {}, timer = 0;
  const play = () => {
    if (stopped) return;
    stopCur = playSamples(A.samples, r[0], r[1], () => { if (!stopped) timer = setTimeout(play, 700); });
  };
  play();
  S.loop = () => { stopped = true; clearTimeout(timer); stopCur(); };
  updateLoopLabel();
}

// ---------------------------------------------------------------------------
// Results UI
// ---------------------------------------------------------------------------
function level(score) { return score == null ? 'none' : score >= 75 ? 'good' : score >= 55 ? 'warn' : 'bad'; }
function colorForScore(score, onDark) {
  const l = level(score);
  if (l === 'none') return null;
  return onDark ? { good: '#7fe0b0', warn: '#ffc46b', bad: '#ff8a75' }[l] : `var(--${l})`;
}
function wordScore(i) {
  const R = S.result;
  if (!R) return null;
  const parts = [];
  const pw = R.pronunciation.words[i];
  if (pw) parts.push(pw.score);
  const iw = R.intonation.words && R.intonation.words[i];
  if (iw && iw.score != null) parts.push(iw.score);
  return parts.length ? Math.min(...parts) : null;
}

const MOVE_SVG = {
  rise: '<path d="M3 13 L13 9 L23 3"/>',
  fall: '<path d="M3 3 L13 7 L23 13"/>',
  level: '<path d="M3 8 L23 8"/>'
};
const MOVE_WORD = { rise: 'rises', fall: 'falls', level: 'level' };
function moveHtml(m, who) {
  if (!m) return `<span class="move ${who}">–</span>`;
  return `<span class="move ${who}" title="${who === 'model' ? 'Speaker' : 'You'}: ${MOVE_WORD[m.cls]}"><svg viewBox="0 0 26 16" aria-hidden="true">${MOVE_SVG[m.cls]}</svg>${MOVE_WORD[m.cls]}</span>`;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function renderResults() {
  const R = S.result;
  if (!R) return;
  $('results').hidden = false;
  $('overall-num').textContent = R.overall;
  $('verdict').textContent = verdict(R.overall);
  const melodyOnly = R.mode === 'melody';
  const mi = MODE_ORDER.indexOf(S.stu.mode);
  const nextStep = mi >= 0 && mi < MODE_ORDER.length - 1 ? MODE_ORDER[mi + 1] : null;
  const stepName = { melody: 'Melody', slow: 'Shadow slowly', shadow: 'Shadow', own: 'On your own' };
  let sub = 'Melody carries the most weight in this score, then the clarity of sounds.';
  if (melodyOnly) sub = 'Only the melody and timing of your humming were compared.';
  if (R.capped) sub = 'Some words of the phrase were missing or different, so the score is limited. Say the whole phrase.';
  else if (R.cappedByMelody) sub = 'Your melody is different from the speaker’s, so good sounds cannot raise the score further.';
  if (R.overall >= PASS && nextStep) sub += ` Step passed: go on to “${stepName[nextStep]}”.`;
  else if (R.overall < PASS) sub += ` Aim for ${PASS} before the next step.`;
  $('verdict-sub').textContent = sub;

  // meters
  const I = R.intonation;
  const tempoNote = R.rhythm.tempo > 1.08 ? `${Math.round((R.rhythm.tempo - 1) * 100)}% slower than the speaker`
    : R.rhythm.tempo < 0.92 ? `${Math.round((1 - R.rhythm.tempo) * 100)}% faster than the speaker` : 'Same speed as the speaker';
  const meters = [
    ['Melody', I.score, I.ok ? `Your pitch line is on average ${I.rmse.toFixed(1)} semitones from the speaker’s` : 'Not enough voice to compare', true],
    ['Sounds', R.pronunciation.score, melodyOnly ? 'Not checked when humming' : R.pronunciation.score == null ? 'Needs the sound analysis model'
      : R.pronunciation.phonemes ? 'Each sound compared with the speaker in IPA' : 'How clearly each sound came through; check in IPA for detail'],
    ['Rhythm', R.rhythm.score, tempoNote],
    ['Stress', R.stress.score, melodyOnly ? 'Not checked when humming' : R.stress.score != null ? 'Which words stand out and where' : 'Needs the sound analysis model']
  ];
  $('meters').innerHTML = meters.map(([name, score, note, main]) => `
    <div class="m is-${level(score)}${main ? ' is-main' : ''}">
      <div class="m-head"><span>${name}</span><span class="m-val">${score == null ? '–' : score}</span></div>
      <p class="m-note">${esc(note)}</p>
    </div>`).join('');

  // word table
  const A = refAnalysis();
  const rows = (A.words || splitWords(phrase().text).map((w) => ({ word: w }))).map((w, i) => {
    const pw = R.pronunciation.words[i];
    const iw = I.words && I.words[i];
    const rw = R.rhythm.words && R.rhythm.words[i];
    const sounds = pw ? `<span class="pill ${level(pw.score)}">${pw.score}</span>${pw.heard && pw.heard !== w.word.replace(/'/g, "'") && pw.score < 75 ? `<span class="heard-small">heard as “${esc(pw.heard)}”</span>` : ''}` : '<span class="pill none">–</span>';
    const mismatch = iw && iw.refMove && iw.stuMove && iw.refMove.cls !== iw.stuMove.cls;
    const melody = iw ? `<span class="moves">${moveHtml(iw.refMove, 'model')}<span class="${mismatch ? 'move mismatch' : 'move'}">${moveHtml(iw.stuMove, 'you')}</span></span>` : '–';
    let len = '–';
    if (rw) {
      const lbl = rw.ratio > 1.4 ? ['too long', 'warn'] : rw.ratio < 0.7 ? ['too short', 'warn'] : ['fine', 'good'];
      len = `<span class="pill ${lbl[1]}">${lbl[0]}</span>`;
    }
    const btn = A.words ? `<button class="word-btn" type="button" data-word="${i}"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>${esc(w.word)}</button>` : esc(w.word);
    const ipa = pw && pw.phonItems ? ipaHtml(pw.phonItems) : '–';
    return `<tr><td>${btn}</td><td class="col-sounds">${sounds}</td><td class="col-ipa">${ipa}</td><td>${melody}</td><td>${len}</td></tr>`;
  });
  $('word-rows').innerHTML = rows.join('');
  const table = $('word-rows').closest('table');
  table.classList.toggle('no-ipa', !R.pronunciation.phonemes);
  table.classList.toggle('melody-only', melodyOnly);
  const hasWords = !!(A.words && (melodyOnly || S.stu.analysis.words));
  document.querySelectorAll('.word-part').forEach((el) => { el.hidden = !hasWords; });
  $('ipa-offer').hidden = melodyOnly || !!R.pronunciation.phonemes || !(A.words && S.stu.analysis.words) || models.phon.state === 'failed';

  // feedback
  const fb = buildFeedback(R);
  $('feedback').innerHTML = fb.slice(0, 8).map((f) => `<li class="${f.level}"><span class="fdot" aria-hidden="true"></span><span class="area">${f.area}</span><span class="text">${esc(f.text)}</span></li>`).join('');

  // transcripts
  const W = R.words;
  const heard = [];
  heard.push(['Expected', esc(phrase().text)]);
  if (W.whisper) {
    const parts = W.whisper.ops.map((o) => o.type === 'ok' ? esc(o.heard)
      : o.type === 'del' ? `<span class="miss">${esc(o.target)}</span>`
      : `<span class="sub">${esc(o.heard)}</span>`);
    heard.push(['Speech recogniser', parts.join(' ') || '<em>nothing</em>']);
  } else {
    heard.push(['Speech recogniser', models.asr.state === 'failed' ? '<em>not available</em>'
      : S.stu.whisper == null ? '<em>working…</em>' : '<em>nothing</em>']);
  }
  if (S.stu.analysis.acoustic != null) heard.push(['Sound by sound', esc(S.stu.analysis.acoustic || '–')]);
  $('heard').innerHTML = heard.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  // numbers
  const refP = A.pitch, stuP = S.stu.analysis.pitch;
  const nums = [];
  if (I.ok) {
    nums.push(['Average pitch difference', `${I.rmse.toFixed(2)} semitones`]);
    nums.push(['Shape correlation', Number.isFinite(I.correlation) ? I.correlation.toFixed(2) : '–']);
    nums.push(['Final tone (speaker / you)', `${I.toneRef.tone} / ${I.toneStu.tone}`]);
    nums.push(['Pitch range (speaker / you)', `${I.rangeRef.toFixed(1)} / ${I.rangeStu.toFixed(1)} st`]);
    nums.push(['Voice frames compared', `${Math.round(I.coverage * 100)}%`]);
  }
  nums.push(['Speaking time (speaker / you)', `${R.rhythm.refDur.toFixed(2)} / ${R.rhythm.stuDur.toFixed(2)} s`]);
  nums.push(['Typical pitch (speaker / you)', `${Math.round(refP.median)} / ${Math.round(stuP.median)} Hz`]);
  nums.push(['Scoring', (STRICTNESS[S.strictness || S.cfg.strictness] || STRICTNESS.standard).label]);
  $('numbers').innerHTML = nums.map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join('');
  if ($('results').querySelector('details.more').open) drawPictures();
}

/** Speaker and student IPA with the differing sounds marked. */
function ipaHtml(items) {
  const refPart = items.filter((it) => it.ref).map((it) =>
    it.type === 'match' ? esc(it.ref) : `<span class="${it.type === 'del' ? 'gone' : 'x'}">${esc(it.ref)}</span>`).join('');
  const stuPart = items.filter((it) => it.stu).map((it) =>
    it.type === 'match' ? esc(it.stu) : `<span class="x">${esc(it.stu)}</span>`).join('');
  return `<span class="ipa"><span title="Speaker">/${refPart}/</span><span class="you" title="You">/${stuPart || '–'}/</span></span>`;
}

function drawPictures() {
  const A = refAnalysis();
  if (!A || !S.stu || !S.stu.analysis) return;
  drawWave($('wave-ref'), A.samples, SR, { words: A.words, color: '#0541f0' });
  drawSpectrogram($('spec-ref'), A.samples, SR);
  drawWave($('wave-stu'), S.stu.analysis.samples, SR, { words: S.stu.analysis.words, color: '#d96d0b' });
  drawSpectrogram($('spec-stu'), S.stu.analysis.samples, SR);
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------
async function loadConfig() {
  // data/lectures.json is generated from the assets/ folders (tools/build_index.py);
  // data/phrases.json is the older single-lecture format.
  const get = async (url) => { const r = await fetch(url, { cache: 'no-cache' }); if (!r.ok) throw new Error(url); return r.json(); };
  let cfg;
  try { cfg = await get('data/lectures.json'); }
  catch (_) {
    cfg = await get('data/phrases.json');
    cfg.lectures = [{ id: 'default', title: cfg.lecture && cfg.lecture.title, phrases: cfg.phrases }];
  }
  try {
    const settings = await get('data/settings.json');
    cfg = { ...cfg, ...settings, lectures: cfg.lectures };
  } catch (_) {}
  cfg.lectures = (cfg.lectures || []).filter((l) => l.phrases && l.phrases.length);
  if (!cfg.lectures.length) throw new Error('empty');
  return cfg;
}

function selectLecture(li, phraseIdx) {
  const n = S.cfg.lectures.length;
  S.lectureIdx = ((li % n) + n) % n;
  S.lecture = S.cfg.lectures[S.lectureIdx];
  store.set('lecture', S.lecture.id);
  $('lecture-select').value = String(S.lectureIdx);
  $('lecture-title').textContent = S.lecture.title || '';

  setupFullLecture();

  const sel = $('phrase-select');
  sel.innerHTML = '';
  S.lecture.phrases.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `${i + 1}. ${p.text}`;
    sel.append(o);
  });
  const saved = parseInt(store.get('idx.' + S.lecture.id, '0'), 10);
  selectPhrase(phraseIdx != null ? phraseIdx : (Number.isFinite(saved) ? saved : 0));
}

// ---------------------------------------------------------------------------
// Full lecture audio (files in assets/<lecture>/full/)
// ---------------------------------------------------------------------------
const fullAudio = () => $('full-audio');

function setupFullLecture() {
  const parts = S.lecture.full || [];
  const box = $('lecture');
  const a = fullAudio();
  a.pause();
  box.hidden = !parts.length;
  const sel = $('part-select');
  sel.innerHTML = '';
  parts.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = p.title;
    sel.append(o);
  });
  $('part-field').hidden = parts.length < 2;
  if (!parts.length) { a.removeAttribute('src'); return; }
  const saved = parseInt(store.get('part.' + S.lecture.id, '0'), 10);
  selectPart(Number.isFinite(saved) && saved < parts.length ? saved : 0);
}

function selectPart(i) {
  const parts = S.lecture.full || [];
  if (!parts[i]) return;
  const a = fullAudio();
  $('part-select').value = String(i);
  store.set('part.' + S.lecture.id, String(i));
  const key = 'pos.' + parts[i].audio;
  a.src = parts[i].audio;
  // continue where the student stopped last time
  a.onloadedmetadata = () => {
    const pos = parseFloat(store.get(key, '0'));
    if (pos > 3 && pos < a.duration - 3) a.currentTime = pos;
  };
  let last = 0;
  a.ontimeupdate = () => {
    if (Math.abs(a.currentTime - last) > 5) { last = a.currentTime; store.set(key, String(Math.floor(a.currentTime))); }
  };
}

async function init() {
  try {
    S.cfg = await loadConfig();
  } catch (e) {
    $('stave-msg').textContent = 'The list of phrases could not be loaded. Open the page through a web server, not as a file, and check data/lectures.json.';
    return;
  }
  const cfg = S.cfg;
  // ?lecture=<folder name> shows only that lecture (one Moodle page per lecture)
  const only = new URLSearchParams(location.search).get('lecture');
  if (only) {
    const one = cfg.lectures.filter((l) => l.id.toLowerCase() === only.toLowerCase());
    if (one.length) cfg.lectures = one;
  }
  if (cfg.title) { $('app-title').textContent = cfg.title; document.title = cfg.title; }
  if (!S.strictness) S.strictness = cfg.strictness || 'standard';
  $('strictness').value = S.strictness;

  const lsel = $('lecture-select');
  cfg.lectures.forEach((l, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = l.title || l.id;
    lsel.append(o);
  });
  $('lecture-field').hidden = cfg.lectures.length < 2;
  lsel.addEventListener('change', () => selectLecture(parseInt(lsel.value, 10)));
  $('part-select').addEventListener('change', (e) => selectPart(parseInt(e.target.value, 10)));
  // only one thing plays at a time
  fullAudio().addEventListener('play', () => {
    modelAudio.pause();
    const part = (S.lecture.full || [])[parseInt($('part-select').value, 10) || 0];
    report('full-lecture', { lecture: S.lecture.id, part: part ? part.title : '' });
  });
  modelAudio.addEventListener('play', () => fullAudio().pause());

  const sel = $('phrase-select');
  sel.addEventListener('change', () => selectPhrase(parseInt(sel.value, 10)));
  $('prev').addEventListener('click', () => selectPhrase(S.idx - 1));
  $('next').addEventListener('click', () => selectPhrase(S.idx + 1));
  $('btn-listen').addEventListener('click', () => { stopLoop(); S.listens++; listen(1); });
  $('btn-slow').addEventListener('click', () => { stopLoop(); S.listens++; listen(0.75); });
  $('btn-record').addEventListener('click', toggleRecording);
  $('btn-loop').addEventListener('click', toggleLoop);
  $('btn-ipa').addEventListener('click', () => runIpa(false));
  document.querySelectorAll('.step').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('btn-mine').addEventListener('click', () => { if (S.stu) playSamples(S.stu.samples); });
  $('strictness').addEventListener('change', (e) => {
    S.strictness = e.target.value;
    store.set('strictness', S.strictness);
    computeAndRender();
  });
  $('word-rows').addEventListener('click', (e) => {
    const b = e.target.closest('[data-word]');
    if (b) playWord(parseInt(b.dataset.word, 10));
  });
  $('stave').addEventListener('click', (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const i = wordAtX(e.clientX - rect.left);
    if (i >= 0) playWord(i);
  });
  $('results').querySelector('details.more').addEventListener('toggle', (e) => { if (e.target.open) drawPictures(); });
  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderStave, 120); });
  document.fonts && document.fonts.ready.then(renderStave);

  startWorker(cfg);
  startReporting(cfg, restoreFromMoodle);
  const savedLecture = store.get('lecture', null);
  const li = Math.max(0, cfg.lectures.findIndex((l) => l.id === savedLecture));
  selectLecture(li);
}

init();
