// Canvas drawing: the melody stave (hero), waveforms and spectrograms.

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, rect.width), h = Math.max(1, rect.height);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

/** Smooth a NaN-gapped track for drawing. */
function smoothForDraw(vals, r = 2) {
  return vals.map((v, i) => {
    if (!Number.isFinite(v)) return NaN;
    let s = 0, c = 0;
    for (let k = -r; k <= r; k++) { const x = vals[i + k]; if (Number.isFinite(x)) { s += x; c++; } }
    return s / c;
  });
}

function drawLine(g, pts, color, width, dashedGaps = true) {
  // pts: [{x,y}|null]
  g.lineCap = 'round'; g.lineJoin = 'round';
  // solid voiced runs
  g.strokeStyle = color; g.lineWidth = width; g.setLineDash([]);
  g.beginPath();
  let pen = false;
  for (const p of pts) {
    if (!p) { pen = false; continue; }
    if (!pen) { g.moveTo(p.x, p.y); pen = true; } else g.lineTo(p.x, p.y);
  }
  g.stroke();
  if (!dashedGaps) return;
  // faint dotted bridges over short unvoiced gaps
  g.lineWidth = Math.max(1, width / 2.5); g.setLineDash([2, 5]); g.globalAlpha = 0.5;
  g.beginPath();
  let last = null, gap = 0;
  for (const p of pts) {
    if (!p) { gap++; continue; }
    if (last && gap > 0 && gap < 25) { g.moveTo(last.x, last.y); g.lineTo(p.x, p.y); }
    last = p; gap = 0;
  }
  g.stroke();
  g.setLineDash([]); g.globalAlpha = 1;
}

/**
 * The stave: five lines = -8, -4, 0, +4, +8 semitones around the speaker's median.
 * @param {HTMLCanvasElement} canvas
 * @param {{times:number[], ref:number[], stu?:number[], span:[number,number], words?:{start,end,word}[], playhead?:number, highlight?:number}} d
 */
export function drawStave(canvas, d) {
  const { g, w, h } = setup(canvas);
  const padX = 8, top = 14, bottom = d.words ? 34 : 12;
  const lo = -10, hi = 10;
  const [t0, t1] = d.span;
  const X = (t) => padX + (t - t0) / (t1 - t0) * (w - 2 * padX);
  const Y = (st) => top + (hi - Math.max(lo, Math.min(hi, st))) / (hi - lo) * (h - top - bottom);

  const lineCol = css('--stave-line', 'rgba(255,255,255,0.16)');
  const refCol = css('--model-line', '#8fb3ff');
  const stuCol = css('--you-line', '#f5851c');

  // word bands
  if (d.words) {
    d.words.forEach((wd, i) => {
      if (!Number.isFinite(wd.start)) return;
      const x0 = X(wd.start), x1 = X(wd.end);
      if (d.highlight === i) {
        g.fillStyle = 'rgba(255,255,255,0.08)';
        g.fillRect(x0, top - 10, x1 - x0, h - top - bottom + 20);
      }
      g.strokeStyle = 'rgba(255,255,255,0.10)';
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0 + 0.5, h - bottom + 6); g.lineTo(x0 + 0.5, h - 6); g.stroke();
    });
  }
  // stave lines
  g.strokeStyle = lineCol; g.lineWidth = 1;
  [-8, -4, 0, 4, 8].forEach((st) => {
    const y = Math.round(Y(st)) + 0.5;
    g.beginPath(); g.moveTo(padX, y); g.lineTo(w - padX, y); g.stroke();
  });

  const toPts = (vals) => smoothForDraw(vals).map((v, i) => Number.isFinite(v) ? { x: X(d.times[i]), y: Y(v) } : null);
  if (d.ref) {
    // soft glow under the model line
    g.save(); g.shadowColor = refCol; g.shadowBlur = 10;
    drawLine(g, toPts(d.ref), refCol, 4);
    g.restore();
  }
  if (d.stu) drawLine(g, toPts(d.stu), stuCol, 3);

  // live pitch while recording: {t, st} points, broken where the voice stops
  if (d.live && d.live.length) {
    const pts = [];
    let prevT = -Infinity;
    for (const p of d.live) {
      if (p.t - prevT > 0.09) pts.push(null);
      if (p.t >= t0 - 0.3 && p.t <= t1 + 0.5) pts.push({ x: X(Math.min(p.t, t1 + 0.4)), y: Y(p.st) });
      prevT = p.t;
    }
    drawLine(g, pts, stuCol, 3, false);
    const last = d.live[d.live.length - 1];
    if (last && last.t >= t0 - 0.3) {
      g.fillStyle = stuCol;
      g.beginPath(); g.arc(X(Math.min(last.t, t1 + 0.4)), Y(last.st), 5, 0, Math.PI * 2); g.fill();
    }
  }

  // word labels under their stretch of melody
  if (d.words) {
    g.textBaseline = 'alphabetic';
    g.textAlign = 'center';
    let lastRight = -Infinity;
    d.words.forEach((wd, i) => {
      if (!Number.isFinite(wd.start)) return;
      const x0 = X(wd.start), x1 = X(wd.end);
      g.font = `${d.highlight === i ? 600 : 500} 13px Onest, system-ui, sans-serif`;
      const tw = g.measureText(wd.word).width;
      let cx = Math.min(w - tw / 2 - 2, Math.max(tw / 2 + 2, (x0 + x1) / 2));
      if (cx - tw / 2 < lastRight + 6) cx = lastRight + 6 + tw / 2; // avoid overlapping labels
      lastRight = cx + tw / 2;
      const col = (d.wordColors && d.wordColors[i]) || 'rgba(255,255,255,0.85)';
      g.fillStyle = col;
      g.fillText(wd.word, cx, h - 10);
      if (d.wordColors && d.wordColors[i]) {
        g.fillRect(cx - 9, h - 5, 18, 2);
      }
    });
    g.textAlign = 'start';
  }

  if (Number.isFinite(d.playhead)) {
    const x = X(d.playhead);
    g.strokeStyle = 'rgba(255,255,255,0.7)'; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x, top - 8); g.lineTo(x, h - bottom + 8); g.stroke();
  }
  return { X, Y };
}

/** Plain waveform with optional word bands. */
export function drawWave(canvas, samples, sr, opts = {}) {
  const { g, w, h } = setup(canvas);
  const color = opts.color || css('--blue', '#0541f0');
  if (opts.words) {
    opts.words.forEach((wd, i) => {
      if (!Number.isFinite(wd.start)) return;
      g.fillStyle = i % 2 ? 'rgba(80,85,105,0.05)' : 'rgba(80,85,105,0.10)';
      const x0 = wd.start * sr / samples.length * w, x1 = wd.end * sr / samples.length * w;
      g.fillRect(x0, 0, x1 - x0, h);
    });
  }
  g.fillStyle = color;
  const step = samples.length / w;
  for (let x = 0; x < w; x++) {
    let mn = 1, mx = -1;
    const a = Math.floor(x * step), b = Math.min(samples.length, Math.floor((x + 1) * step));
    for (let i = a; i < b; i++) { const v = samples[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mx < mn) continue;
    const y0 = (1 - mx) * h / 2, y1 = (1 - mn) * h / 2;
    g.fillRect(x, y0, 1, Math.max(1, y1 - y0));
  }
  if (opts.words) {
    g.font = '600 12px Onest, system-ui, sans-serif';
    opts.words.forEach((wd) => {
      if (!Number.isFinite(wd.start)) return;
      const x0 = wd.start * sr / samples.length * w;
      const tw = g.measureText(wd.word).width;
      g.fillStyle = 'rgba(255,255,255,0.9)';
      g.fillRect(x0 + 1, 2, tw + 6, 16);
      g.fillStyle = css('--ink', '#0a1e64');
      g.fillText(wd.word, x0 + 4, 14);
    });
  }
}

// radix-2 FFT (in place)
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

/** Wide-band spectrogram up to 5 kHz. */
export function drawSpectrogram(canvas, samples, sr) {
  const { g, w, h } = setup(canvas);
  const N = 512, hop = Math.max(64, Math.floor(samples.length / Math.max(200, w)));
  const frames = Math.max(1, Math.floor((samples.length - N) / hop));
  const maxBin = Math.floor(5000 / (sr / N));
  const win = new Float32Array(N).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)));
  const mags = new Float32Array(frames * maxBin);
  let peak = -Infinity;
  const re = new Float32Array(N), im = new Float32Array(N);
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < N; i++) { re[i] = (samples[f * hop + i] || 0) * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < maxBin; k++) {
      const db = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-10);
      mags[f * maxBin + k] = db;
      if (db > peak) peak = db;
    }
  }
  const img = g.createImageData(Math.max(1, frames), maxBin);
  for (let f = 0; f < frames; f++) for (let k = 0; k < maxBin; k++) {
    const v = Math.max(0, Math.min(1, (mags[f * maxBin + k] - (peak - 70)) / 70));
    const idx = ((maxBin - 1 - k) * frames + f) * 4;
    // navy -> blue -> white ramp
    img.data[idx] = Math.round(10 + 245 * Math.pow(v, 1.6));
    img.data[idx + 1] = Math.round(30 + 225 * Math.pow(v, 1.3));
    img.data[idx + 2] = Math.round(100 + 155 * Math.pow(v, 0.7));
    img.data[idx + 3] = 255;
  }
  const tmp = document.createElement('canvas');
  tmp.width = Math.max(1, frames); tmp.height = maxBin;
  tmp.getContext('2d').putImageData(img, 0, 0);
  g.imageSmoothingEnabled = true;
  g.drawImage(tmp, 0, 0, w, h);
}
