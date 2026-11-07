// Web Worker that runs the neural speech models with Transformers.js, so the page
// stays responsive. Models are downloaded once and then served from the browser cache.
//
//  - wav2vec2 (CTC, letters, no language model): forced alignment of the target phrase,
//    word timings and letter-level pronunciation confidence.
//  - Whisper (sequence-to-sequence ASR): what words a listener would hear.

import { AutoProcessor, AutoModelForCTC, pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';
import { logSoftmax } from './align.js';

env.allowLocalModels = false;
env.useBrowserCache = true;
try { env.backends.onnx.wasm.numThreads = 1; } catch (_) {}

let cfg = {
  ctcModel: 'Xenova/wav2vec2-base-960h',
  asrModel: 'Xenova/whisper-base.en',
  modelHost: null
};

let ctcPromise = null, asrPromise = null;

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
