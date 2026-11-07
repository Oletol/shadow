// Usage data. Two optional channels; both carry numbers only, never audio.
//
//  * Moodle. When the page runs inside the SCORM package (scorm/index.html opens
//    it with ?scorm=1), every attempt and the best scores go to the package by
//    postMessage, and the package writes them to Moodle (journal, attempts report).
//  * GoatCounter. When data/settings.json has "goatcounter", anonymous counts of
//    visits and events (attempts, passed steps, problems) go to that GoatCounter site.

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
