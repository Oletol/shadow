// Browser audio helpers: decoding/resampling to 16 kHz, recording, playback.

export const SR = 16000;
let ctx = null;

export function audioContext() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('This browser cannot process audio (Web Audio API is missing).');
    ctx = new AC();
  }
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  return ctx;
}

/** Decode any supported audio file and resample it to 16 kHz mono. */
export async function decodeTo16k(arrayBuffer) {
  const ac = audioContext();
  const buf = await new Promise((resolve, reject) => {
    // callback form for older Safari
    const p = ac.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
    if (p && p.then) p.then(resolve, reject);
  });
  const length = Math.max(1, Math.ceil(buf.duration * SR));
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const off = new OAC(1, length, SR);
  const src = off.createBufferSource();
  src.buffer = buf; // multichannel input is down-mixed to the mono destination
  src.connect(off.destination);
  src.start(0);
  const rendered = await off.startRendering();
  return Float32Array.from(rendered.getChannelData(0));
}

export async function fetchAudio16k(url) {
  const r = await fetch(url, { mode: 'cors' });
  if (!r.ok) throw new Error(`The model recording could not be downloaded (HTTP ${r.status}).`);
  return decodeTo16k(await r.arrayBuffer());
}

/** Microphone recorder with a live level callback. */
export class Recorder {
  constructor(onLevel) {
    this.onLevel = onLevel;
    this.chunks = [];
  }

  static supported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,   // removes the model voice when shadowing through speakers
        noiseSuppression: true,
        autoGainControl: false    // AGC distorts loudness, which we use for stress
      }
    });
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
    const mimeType = types.find((t) => window.MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t));
    this.mr = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.chunks = [];
    this.mr.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };

    // level meter
    const ac = audioContext();
    this.srcNode = ac.createMediaStreamSource(this.stream);
    this.analyser = ac.createAnalyser();
    this.analyser.fftSize = 1024;
    this.srcNode.connect(this.analyser);
    const data = new Float32Array(this.analyser.fftSize);
    const tick = () => {
      if (!this.analyser) return;
      this.analyser.getFloatTimeDomainData(data);
      let s = 0;
      for (let i = 0; i < data.length; i++) s += data[i] * data[i];
      const db = 20 * Math.log10(Math.sqrt(s / data.length) + 1e-6);
      if (this.onLevel) this.onLevel(Math.max(0, Math.min(1, (db + 60) / 50)));
      this.raf = requestAnimationFrame(tick);
    };
    tick();
    this.startedAt = performance.now();
    this.mr.start();
  }

  stop() {
    return new Promise((resolve) => {
      if (!this.mr || this.mr.state === 'inactive') return resolve(null);
      this.mr.onstop = () => {
        cancelAnimationFrame(this.raf);
        try { this.srcNode.disconnect(); } catch (_) {}
        this.analyser = null;
        this.stream.getTracks().forEach((t) => t.stop());
        const type = (this.chunks[0] && this.chunks[0].type) || this.mr.mimeType || 'audio/webm';
        resolve(new Blob(this.chunks, { type }));
      };
      this.mr.stop();
    });
  }

  get recording() { return !!(this.mr && this.mr.state === 'recording'); }
}

/** Play a segment of a 16 kHz Float32Array. Returns a stop() function. */
export function playSamples(samples, from = 0, to = null, onEnd) {
  const ac = audioContext();
  const buf = ac.createBuffer(1, samples.length, SR);
  buf.copyToChannel(samples, 0);
  const src = ac.createBufferSource();
  src.buffer = buf;
  src.connect(ac.destination);
  const dur = to != null ? Math.max(0.05, to - from) : undefined;
  src.onended = () => onEnd && onEnd();
  src.start(0, Math.max(0, from), dur);
  return () => { try { src.stop(); } catch (_) {} };
}

/** Encode 16 kHz mono samples as a WAV blob (for download/playback). */
export function toWavBlob(samples) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true);
  return new Blob([buf], { type: 'audio/wav' });
}
