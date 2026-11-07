// Web Worker that runs the neural speech models with Transformers.js, so the page
// stays responsive. Models are downloaded once and then served from the browser cache.
//
//  - wav2vec2 (CTC, letters, no language model): forced alignment of the target phrase,
//    word timings and letter-level pronunciation confidence.
//  - Whisper (sequence-to-sequence ASR): what words a listener would hear.
//  - wav2vec2 espeak (CTC, IPA phonemes, loaded on request): sound-by-sound comparison.

import { AutoProcessor, AutoModelForCTC, pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';
import { logSoftmax } from './align.js';

env.allowLocalModels = false;
env.useBrowserCache = true;
try { env.backends.onnx.wasm.numThreads = 1; } catch (_) {}

let cfg = {
  ctcModel: 'Xenova/wav2vec2-base-960h',
  asrModel: 'Xenova/whisper-base.en',
  phonModel: 'onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX',
  modelHost: null
};

let ctcPromise = null, asrPromise = null, phonPromise = null;

function progress(model) {
  const files = new Map();
  return (p) => {
    if (p.status === 'progress' && p.file) {
      files.set(p.file, { loaded: p.loaded || 0, total: p.total || 0 });
      let loaded = 0, total = 0;
      files.forEach((f) => { loaded += f.loaded; total += f.total; });
      self.postMessage({ type: 'progress', model, loaded, total });
    } else if (p.status === 'done' && p.file) {
      const f = files.get(p.file);
      if (f) f.loaded = f.total;
    }
  };
}

function loadCtc() {
  if (!ctcPromise) {
    ctcPromise = (async () => {
      const opts = { progress_callback: progress('ctc') };
      const processor = await AutoProcessor.from_pretrained(cfg.ctcModel, opts);
      const model = await AutoModelForCTC.from_pretrained(cfg.ctcModel, { ...opts, dtype: 'q8', device: 'wasm' });
      self.postMessage({ type: 'ready', model: 'ctc' });
      return { processor, model };
    })().catch((e) => { ctcPromise = null; throw e; });
  }
  return ctcPromise;
}

function loadAsr() {
  if (!asrPromise) {
    asrPromise = (async () => {
      const asr = await pipeline('automatic-speech-recognition', cfg.asrModel, {
        progress_callback: progress('asr'),
        dtype: { encoder_model: 'fp32', decoder_model_merged: 'q8' },
        device: 'wasm'
      });
      self.postMessage({ type: 'ready', model: 'asr' });
      return asr;
    })().catch((e) => { asrPromise = null; throw e; });
  }
  return asrPromise;
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ': ' + r.status);
  return r.json();
}

/** id -> IPA symbol for the phoneme model. */
async function phonVocab() {
  const host = (cfg.modelHost || 'https://huggingface.co/').replace(/\/?$/, '/');
  const base = host + cfg.phonModel + '/resolve/main/';
  let vocab = null;
  try { vocab = await fetchJson(base + 'vocab.json'); } catch (_) {}
  if (!vocab) {
    try { const tj = await fetchJson(base + 'tokenizer.json'); vocab = tj.model && tj.model.vocab; } catch (_) {}
  }
  if (!vocab) throw new Error('The phoneme list of the model could not be loaded.');
  const id2sym = {};
  if (Array.isArray(vocab)) vocab.forEach((t, i) => { id2sym[i] = Array.isArray(t) ? t[0] : t; });
  else Object.entries(vocab).forEach(([t, i]) => { id2sym[i] = t; });
  let blank = 0;
  Object.entries(id2sym).forEach(([i, t]) => { if (t === '<pad>') blank = Number(i); });
  return { id2sym, blank };
}

function loadPhon() {
  if (!phonPromise) {
    phonPromise = (async () => {
      const opts = { progress_callback: progress('phon') };
      const processor = await AutoProcessor.from_pretrained(cfg.phonModel, opts);
      let model = null, lastErr = null;
      for (const dtype of ['q8', 'int8', 'uint8', 'q4']) {
        try { model = await AutoModelForCTC.from_pretrained(cfg.phonModel, { ...opts, dtype, device: 'wasm' }); break; }
        catch (e) { lastErr = e; }
      }
      if (!model) throw lastErr || new Error('Phoneme model could not be loaded.');
      const vocab = await phonVocab();
      self.postMessage({ type: 'ready', model: 'phon' });
      return { processor, model, vocab };
    })().catch((e) => { phonPromise = null; throw e; });
  }
  return phonPromise;
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  try {
    if (msg.type === 'config') {
      cfg = { ...cfg, ...msg.config };
      if (cfg.modelHost) env.remoteHost = cfg.modelHost.endsWith('/') ? cfg.modelHost : cfg.modelHost + '/';
      return;
    }
    if (msg.type === 'load') {
      // load sequentially: the alignment model is needed first
      await loadCtc().catch((e) => self.postMessage({ type: 'load-error', model: 'ctc', error: String(e && e.message || e) }));
      await loadAsr().catch((e) => self.postMessage({ type: 'load-error', model: 'asr', error: String(e && e.message || e) }));
      return;
    }
    if (msg.type === 'load-phon') {
      await loadPhon().catch((e) => self.postMessage({ type: 'load-error', model: 'phon', error: String(e && e.message || e) }));
      return;
    }
    if (msg.type === 'phon') {
      const { processor, model, vocab } = await loadPhon();
      const inputs = await processor(msg.audio);
      const out = await model(inputs);
      const [, T, V] = out.logits.dims;
      const logp = logSoftmax(out.logits.data, T, V);
      self.postMessage({ type: 'result', id: msg.id, result: { logp, T, V, id2sym: vocab.id2sym, blank: vocab.blank } }, [logp.buffer]);
      return;
    }
    if (msg.type === 'ctc') {
      const { processor, model } = await loadCtc();
      const inputs = await processor(msg.audio);
      const out = await model(inputs);
      const [, T, V] = out.logits.dims;
      const logp = logSoftmax(out.logits.data, T, V);
      self.postMessage({ type: 'result', id: msg.id, result: { logp, T, V } }, [logp.buffer]);
      return;
    }
    if (msg.type === 'asr') {
      const asr = await loadAsr();
      const r = await asr(msg.audio, { return_timestamps: false, chunk_length_s: 30 });
      const text = Array.isArray(r) ? r.map((x) => x.text).join(' ') : (r && r.text) || '';
      self.postMessage({ type: 'result', id: msg.id, result: { text: text.trim() } });
      return;
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: msg.id, error: String(e && e.message || e) });
  }
};
