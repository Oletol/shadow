// Usage data. Two optional channels; both carry numbers only, never audio.
//
//  * Moodle. When the page runs inside the SCORM package (scorm/index.html opens
//    it with ?scorm=1), every attempt and the best scores go to the package by
//    postMessage, and the package writes them to Moodle (journal, attempts report).
//  * GoatCounter. When data/settings.json has "goatcounter", anonymous counts of
//    visits and events (attempts, passed steps, problems) go to that GoatCounter site.
//  * Research statistics. When data/settings.json has "statsEndpoint" (the address of
//    a Google Apps Script web app, see tools/stats-apps-script.gs), every attempt is
//    saved there as one row: scores, parts of the scores, word-by-word detail and a
//    random anonymous code of the browser. No names, no audio. The visitor is told
//    about it under the page and can switch it off.

const params = new URLSearchParams(location.search);
export const inScorm = params.has('scorm') && window.parent !== window;

function post(msg) {
  try { window.parent.postMessage({ source: 'shadow', ...msg }, '*'); } catch (_) {}
}

// Stable short key of a phrase (its audio file), used in the data saved in Moodle.
export function phraseKey(p) {
  let h = 0x811c9dc5;
  const s = p.audio || p.text || '';
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(36);
}

// onState({ connected, suspend }) is called once, when the SCORM package answers.
export function startReporting(cfg, onState) {
  if (typeof cfg.goatcounter === 'string' && /^https:\/\//.test(cfg.goatcounter)) loadGoat(cfg.goatcounter);
  startStats(cfg);
  if (!inScorm) return;
  let done = false;
  window.addEventListener('message', (ev) => {
    if (ev.source !== window.parent || done) return;
    const m = ev.data;
    if (!m || m.source !== 'shadow-scorm' || m.type !== 'state') return;
    done = true;
    onState({ connected: !!m.connected, suspend: typeof m.suspend === 'string' ? m.suspend : '' });
  });
  post({ type: 'hello', lectures: cfg.lectures.map((l) => l.id) });
}

export function report(type, data = {}) {
  if (inScorm) post({ type, ...data });
  goatEvent(type, data);
  if (type === 'attempt') statsAttempt(data);
}

// ---------------------------------------------------------------------------
// Research statistics (anonymous, one row per attempt)
// ---------------------------------------------------------------------------
const LS_STATS = 'shadow.';
const ls = {
  get: (k) => { try { return localStorage.getItem(LS_STATS + k); } catch (_) { return null; } },
  set: (k, v) => { try { localStorage.setItem(LS_STATS + k, v); } catch (_) {} }
};
let statsUrl = null;
const statsPending = new Map();   // attempt id -> {data, timer}

export function statsAvailable() { return !!statsUrl; }
export function statsEnabled() { return !!statsUrl && ls.get('statsOff') !== '1'; }
export function setStatsEnabled(on) {
  ls.set('statsOff', on ? '0' : '1');
  if (!on) { statsPending.forEach((p) => clearTimeout(p.timer)); statsPending.clear(); }
}

/** Random code of this browser: links the attempts of one person without knowing who it is. */
function anonId() {
  let id = ls.get('anon');
  if (!id) {
    const a = new Uint8Array(9);
    crypto.getRandomValues(a);
    id = Array.from(a, (b) => b.toString(36).padStart(2, '0')).join('').slice(0, 12);
    ls.set('anon', id);
  }
  return id;
}

function startStats(cfg) {
  if (typeof cfg.statsEndpoint !== 'string' || !/^https:\/\//.test(cfg.statsEndpoint)) return;
  statsUrl = cfg.statsEndpoint;
  // whatever has not been sent yet goes out when the page is closed or hidden
  window.addEventListener('pagehide', () => statsFlush(true));
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') statsFlush(true); });
}

// An attempt is refined for a few seconds after the recording (speech recogniser,
// IPA check), so it is sent once it has been quiet for 8 seconds.
function statsAttempt(d) {
  if (!statsEnabled() || d.id == null) return;
  const prev = statsPending.get(d.id);
  if (prev) clearTimeout(prev.timer);
  statsPending.set(d.id, { data: d, timer: setTimeout(() => statsSend(d.id, false), 8000) });
}

function statsFlush(closing) { [...statsPending.keys()].forEach((id) => statsSend(id, closing)); }

function statsSend(id, closing) {
  const p = statsPending.get(id);
  if (!p) return;
  clearTimeout(p.timer);
  statsPending.delete(id);
  if (!statsEnabled()) return;
  const d = p.data;
  const body = JSON.stringify({
    anon: anonId(), time: new Date().toISOString(), source: inScorm ? 'moodle' : 'web',
    version: d.version, audio: d.lecture, phrase: d.phrase, text: d.text, step: d.step, attempt: d.id,
    overall: d.overall, melody: d.melody, sounds: d.sounds, rhythm: d.rhythm, stress: d.stress, words: d.words,
    contour: d.contour, syllables: d.syllables, trend: d.trend, tone: d.tone, errors: d.errors,
    listens: d.listens, duration: d.duration, strictness: d.strictness, ipa: d.ipa,
    detail: d.detail || null
  });
  // text/plain keeps the request "simple": Apps Script does not answer CORS preflights
  try {
    if (closing && navigator.sendBeacon) navigator.sendBeacon(statsUrl, body);
    else fetch(statsUrl, { method: 'POST', mode: 'no-cors', keepalive: true, headers: { 'Content-Type': 'text/plain;charset=utf-8' }, body }).catch(() => {});
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// GoatCounter
// ---------------------------------------------------------------------------
let goatOn = false, goatReady = false;
const goatQueue = [];
const counted = new Set();

function loadGoat(endpoint) {
  goatOn = true;
  const s = document.createElement('script');
  s.async = true;
  s.src = 'https://gc.zgo.at/count.js';
  s.dataset.goatcounter = endpoint;
  // the page is usually shown inside Moodle, so count it inside a frame too
  s.dataset.goatcounterSettings = JSON.stringify({ allow_frame: true });
  s.onload = () => { goatReady = true; goatQueue.splice(0).forEach(goatSend); };
  document.head.append(s);
}

function goatSend(e) {
  try { window.goatcounter.count({ path: e.path, title: e.title || '', event: true }); } catch (_) {}
}

function goatCount(path, title, once) {
  if (!goatOn) return;
  if (once) { if (counted.has(once)) return; counted.add(once); }
  const e = { path, title };
  goatReady ? goatSend(e) : goatQueue.push(e);
}

function goatEvent(type, d) {
  if (!goatOn) return;
  if (type === 'attempt') {
    goatCount(`attempt/${d.step}`, d.lecture, 'a' + d.id);
    if (d.passed) goatCount(`pass/${d.step}`, d.lecture, 'p' + d.id);
  } else if (type === 'phrase') {
    goatCount(`phrase/${d.lecture}/${String(d.phrase).padStart(2, '0')}`, d.text);
  } else if (type === 'problem') {
    goatCount(`problem/${d.what}`, d.detail || '', 'x' + d.what);
  } else if (type === 'ipa') {
    goatCount('ipa', d.lecture);
  } else if (type === 'full-lecture') {
    goatCount(`full-lecture/${d.lecture}`, d.part, 'f' + d.lecture + d.part);
  }
}
