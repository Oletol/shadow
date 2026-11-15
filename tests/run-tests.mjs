// Node tests for the analysis core: `node tests/run-tests.mjs`
import { yinTrack, analysePitch, voicedMedian } from '../js/pitch.js';
import { phraseToLabels, ctcForcedAlign, greedyDecode, logSoftmax, normaliseText, alignWordLists, W2V_VOCAB } from '../js/align.js';
import { buildAnalysis, speechSpan, SR } from '../js/analysis.js';
import { compareAll, buildFeedback, classifyTone } from '../js/scoring.js';
import { comparePhonemes, gopScores, greedyPhones, setErrorMap, applyAccent } from '../js/phonemes.js';

let failed = 0;
function check(name, cond, info = '') {
  console.log((cond ? 'PASS ' : 'FAIL ') + name + (info ? '  ' + info : ''));
  if (!cond) failed++;
}

// Harmonic "voice": F0 contour function f(t), amplitude envelope a(t)
function synth(dur, f0fn, ampFn, sr = SR, noise = 0.003) {
  const n = Math.round(dur * sr);
  const x = new Float32Array(n);
  let ph = 0;
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const f = f0fn(t);
    const a = ampFn(t);
    ph += 2 * Math.PI * f / sr;
    let s = 0;
    if (f > 0 && a > 0) for (let h = 1; h <= 8; h++) s += Math.sin(h * ph) / h;
    x[i] = 0.3 * a * s + noise * rnd();
  }
  return x;
}

// 1. YIN accuracy on a glide 110 -> 220 Hz
{
  const f = (t) => 110 + 110 * t;
  const x = synth(1, f, () => 1);
  const tr = yinTrack(x, SR);
  let err = 0, n = 0;
  for (let i = 5; i < tr.f0.length - 5; i++) {
    const t = i * tr.hop + 0.016;
    if (tr.f0[i] > 0) { err += Math.abs(1200 * Math.log2(tr.f0[i] / f(t))); n++; }
  }
  check('YIN glide mean error < 25 cents', n > 80 && err / n < 25, `(${(err / n).toFixed(1)} cents, ${n} frames)`);
}
{
  const x = synth(0.6, () => 230, () => 1);
  const tr = yinTrack(x, SR);
  check('YIN female-range 230 Hz', Math.abs(voicedMedian(tr.f0) - 230) < 3, voicedMedian(tr.f0).toFixed(1));
}

// 2. Text normalisation and CTC alignment on synthetic log-probs
check('normalise apostrophes', normaliseText('It’s who you are!') === "it's who you are");
check('numbers to words', normaliseText('over the next 40 minutes') === 'over the next forty minutes');

function fakeLogits(text, frames, wordsAt, V = 32) {
  // wordsAt: [startFrame] per word; letters emitted every 3 frames
  const lab = phraseToLabels(text);
  const T = frames;
  const flat = new Float32Array(T * V).fill(-5);
  for (let t = 0; t < T; t++) flat[t * V + 0] = 5; // blank everywhere
  const words = lab.words;
  let k = 0;
  words.forEach((w, wi) => {
    let f = wordsAt[wi];
    for (const ch of w.toUpperCase()) {
      const id = W2V_VOCAB[ch];
      flat[f * V + id] = 8; flat[f * V + 0] = -5;
      f += 3;
    }
    if (wi < words.length - 1) { flat[(f) * V + 4] = 8; flat[f * V + 0] = -5; }
    k++;
  });
  return { logp: logSoftmax(flat, T, V), T, V };
}
{
  const text = 'What was that?';
  const ctc = fakeLogits(text, 120, [20, 50, 80]);
  check('greedy decode', greedyDecode(ctc.logp, ctc.T, ctc.V) === 'what was that', greedyDecode(ctc.logp, ctc.T, ctc.V));
  const lab = phraseToLabels(text);
  const al = ctcForcedAlign(ctc.logp, ctc.T, ctc.V, lab.labels);
  check('forced alignment found', !!al);
  check('first letter at frame 20', al.tokenSpans[0].start === 20, String(al.tokenSpans[0].start));
  const sThat = al.tokenSpans[lab.wordOf.indexOf(2)].start;
  check('word 3 starts at frame 80', sThat === 80, String(sThat));
  // repeated letters need a blank between them
  const lab2 = phraseToLabels('all');
  check('impossible alignment returns null', ctcForcedAlign(new Float32Array(2 * 32), 2, 32, lab2.labels) === null);
}
check('word list alignment', alignWordLists(['what', 'was', 'that'], ['what', 'is', 'that']).errors === 1);

// 3. Tone classifier
check('tone fall', classifyTone([4, 4, 3.5, 3, 2, 1, 0, -1, -2, -3]).tone === 'fall');
check('tone rise', classifyTone([-3, -3, -2, -1, 0, 1, 2, 3, 4, 4]).tone === 'rise');
check('tone fall-rise', classifyTone([2, 1, 0, -1, -2, -2, -1, 0, 1, 2]).tone === 'fall-rise');

// 4. End-to-end comparison with synthetic phrases
// Model: 3 "words", melody high-fall on last word. Durations: 0.2-0.45, 0.55-0.8, 0.9-1.4 s
function phrase(f0fn, words, total, tempo = 1, shiftSt = 0) {
  const amp = (t) => words.some(([a, b]) => t >= a * tempo && t <= b * tempo) ? 1 : 0;
  const f = (t) => f0fn(t / tempo) * Math.pow(2, shiftSt / 12);
  return synth(total * tempo, f, amp);
}
const W = [[0.2, 0.45], [0.55, 0.8], [0.9, 1.4]];
const modelMelody = (t) => t < 0.5 ? 120 + 20 * (t - 0.2) / 0.3 : t < 0.85 ? 125 : 170 - 80 * (t - 0.9) / 0.5; // high fall
const risingMelody = (t) => t < 0.5 ? 120 : t < 0.85 ? 118 : 110 + 70 * (t - 0.9) / 0.5; // rise at the end
const text = 'what was that';

function ctcFor(words, tempo, dur) {
  const T = Math.round(dur * tempo / 0.02);
  const starts = words.map(([a]) => Math.round((a * tempo + 0.02) / 0.02));
  return fakeLogits(text, T, starts);
}
const ref = buildAnalysis(phrase(modelMelody, W, 1.6), text, ctcFor(W, 1, 1.6));
check('reference words aligned', ref.alignOk, ref.words && ref.words.map((w) => `${w.word}@${w.start.toFixed(2)}-${w.end.toFixed(2)}`).join(' '));
check('speech span', Math.abs(ref.span[0] - 0.2) < 0.2 && Math.abs(ref.span[1] - 1.4) < 0.2, ref.span.map((v) => v.toFixed(2)).join('-'));

// Same melody, deeper voice (-7 st), slower 1.15x
const good = buildAnalysis(phrase(modelMelody, W, 1.6, 1.15, -7), text, ctcFor(W, 1.15, 1.6));
const resGood = compareAll(ref, good, { target: text, whisperText: 'What was that?' });
// Wrong melody: rise at the end
const bad = buildAnalysis(phrase(risingMelody, W, 1.6, 1.0, 0), text, ctcFor(W, 1, 1.6));
const resBad = compareAll(ref, bad, { target: text, whisperText: 'What was that?' });

console.log('  good: intonation', resGood.intonation.score, 'rmse', resGood.intonation.rmse.toFixed(2),
  'tone', resGood.intonation.toneRef.tone, '/', resGood.intonation.toneStu.tone, 'overall', resGood.overall);
console.log('  bad:  intonation', resBad.intonation.score, 'rmse', resBad.intonation.rmse.toFixed(2),
  'tone', resBad.intonation.toneRef.tone, '/', resBad.intonation.toneStu.tone, 'overall', resBad.overall);
check('same melody in another voice scores high', resGood.intonation.score >= 85);
check('rising ending scores low', resBad.intonation.score <= 55);
check('model tone detected as fall', resGood.intonation.toneRef.tone === 'fall');
check('wrong tone detected as rise', resBad.intonation.toneStu.tone === 'rise');
check('tempo detected', Math.abs(resGood.rhythm.tempo - 1.15) < 0.08, resGood.rhythm.tempo.toFixed(2));
const fb = buildFeedback(resBad);
check('feedback mentions final tone', fb.some((f) => /Final tone/.test(f.text)), fb.map((f) => f.text).join(' | '));
const resWords = compareAll(ref, good, { target: text, whisperText: 'What is it?' });
check('wrong words cap the score', resWords.capped && resWords.overall <= 45);

// 5. Without the model (fallback path)
const refNo = buildAnalysis(phrase(modelMelody, W, 1.6), text, null);
const stuNo = buildAnalysis(phrase(modelMelody, W, 1.6, 1.1, 4), text, null);
const resNo = compareAll(refNo, stuNo, { target: text });
check('fallback without models still scores intonation', resNo.intonation.ok && resNo.intonation.score > 70, String(resNo.intonation.score));

// 6. Melody-only mode (humming): no words needed, envelope DTW alignment
{
  const hum = buildAnalysis(phrase(modelMelody, W, 1.6, 1.2, -5), text, null);
  const resHum = compareAll(ref, hum, { target: text, mode: 'melody' });
  check('melody mode: same tune scores high', resHum.intonation.score >= 85, String(resHum.intonation.score));
  check('melody mode: no pronunciation score', resHum.pronunciation.score == null);
  const humBad = buildAnalysis(phrase(risingMelody, W, 1.6, 1.0, 0), text, null);
  const resHumBad = compareAll(ref, humBad, { target: text, mode: 'melody' });
  check('melody mode: wrong tune scores low', resHumBad.intonation.score <= 55, String(resHumBad.intonation.score));
  const slow = buildAnalysis(phrase(modelMelody, W, 1.6, 1.33, 0), text, ctcFor(W, 1.33, 1.6));
  const resSlow = compareAll(ref, slow, { target: text, expectedTempo: 1 / 0.75 });
  check('slow shadowing: tempo judged against 0.75 speed', Math.abs(resSlow.rhythm.tempo - 1) < 0.1, resSlow.rhythm.tempo.toFixed(2));
}

// 7. Phoneme comparison (IPA)
{
  const syms = ['<pad>', '<s>', '</s>', '<unk>', 'w', 'ɛ', 'ə', 'ɹ', 'ð', 'z', 'æ', 't', 'ɒ', 'ɑː', 'ɾ', 'iː'];
  const V = syms.length;
  const id = Object.fromEntries(syms.map((s, i) => [s, i]));
  const id2sym = Object.fromEntries(syms.map((s, i) => [i, s]));
  function phonLogits(seq, T) {
    const flat = new Float32Array(T * V).fill(-6);
    for (let t = 0; t < T; t++) flat[t * V] = 6;
    seq.forEach(([sym, f]) => { flat[f * V + id[sym]] = 9; flat[f * V] = -6; });
    return { logp: logSoftmax(flat, T, V), T, V, frameSec: 0.02 };
  }
  // speaker: "that wear" RP  /ð æ t/  /w ɛ ə/
  const refPh = phonLogits([['ð', 5], ['æ', 9], ['t', 13], ['w', 30], ['ɛ', 34], ['ə', 38]], 50);
  // student: /z æ ɾ/ /w ɛ ɹ/  (th -> z, flapped t, rhotic r)
  const stuPh = phonLogits([['z', 5], ['æ', 9], ['ɾ', 13], ['w', 30], ['ɛ', 34], ['ɹ', 38]], 50);
  const refWords = [{ word: 'that', start: 0.08, end: 0.3 }, { word: 'wear', start: 0.58, end: 0.8 }];
  const res = comparePhonemes(refPh, stuPh, id2sym, refWords);
  const kinds = res.words.flatMap((w) => w.issues.map((i) => i.kind));
  check('IPA: speaker phones per word', res.words[0].ref.join('') === 'ðæt' && res.words[1].ref.join('') === 'wɛə', res.words.map((w) => w.ref.join('') + '/' + w.stu.join('')).join(' '));
  check('IPA: th substitution found', kinds.includes('th'));
  check('IPA: flapped t found', kinds.includes('flap'));
  check('IPA: rhotic r found', kinds.includes('rhotic'), kinds.join(','));
  const same = comparePhonemes(refPh, refPh, id2sym, refWords);
  check('IPA: identical speech scores 100', same.score === 100 && same.words.every((w) => !w.issues.length), String(same.score));
  // extra vowel after a final consonant
  const epi = phonLogits([['ð', 5], ['æ', 9], ['t', 13], ['ə', 17], ['w', 30], ['ɛ', 34], ['ə', 38]], 50);
  const resE = comparePhonemes(refPh, epi, id2sym, refWords);
  check('IPA: epenthetic vowel found', resE.words[0].issues.some((i) => i.kind === 'epenthesis'), JSON.stringify(resE.words[0].items.map((i) => i.type)));

  // 8. Goodness of pronunciation: graded score from the likelihood ratio
  {
    // frame with chosen probabilities: [[sym, prob], ...], the rest goes to blank
    function probLogits(framesSpec, T) {
      const flat = new Float32Array(T * V).fill(Math.log(1e-6));
      for (let t = 0; t < T; t++) flat[t * V] = Math.log(0.999);
      Object.entries(framesSpec).forEach(([f, list]) => {
        let rest = 1;
        list.forEach(([sym, p]) => { flat[f * V + id[sym]] = Math.log(p); rest -= p; });
        flat[f * V] = Math.log(Math.max(1e-6, rest));
      });
      return { logp: flat, T, V, frameSec: 0.02 };
    }
    const refG = probLogits({ 5: [['ð', 0.95]], 9: [['æ', 0.95]], 13: [['t', 0.95]] }, 20);
    const rp = greedyPhones(refG.logp, refG.T, refG.V, id2sym, 0);
    const run = (spec) => {
      const stu = probLogits(spec, 20);
      const res = comparePhonemes(refG, stu, id2sym, [{ word: 'that', start: 0.08, end: 0.3 }]);
      return res.words[0].items.filter((it) => it.ref);
    };
    const clear = run({ 5: [['ð', 0.9]], 9: [['æ', 0.9]], 13: [['t', 0.9]] });
    const unsure = run({ 5: [['ð', 0.45], ['z', 0.5]], 9: [['æ', 0.9]], 13: [['t', 0.9]] });
    const wrong = run({ 5: [['z', 0.95], ['ð', 0.02]], 9: [['æ', 0.9]], 13: [['t', 0.9]] });
    const dropped = run({ 5: [['ð', 0.9]], 9: [['æ', 0.9]] });
    check('GOP: clear sound scores near 1', clear[0].score > 0.95, clear[0].score.toFixed(2));
    check('GOP: uncertain sound gets a middle score', unsure[0].score > 0.35 && unsure[0].score < 0.8, unsure[0].score.toFixed(2));
    check('GOP: wrong sound scores near 0 and names what was heard', wrong[0].score < 0.1 && wrong[0].heard === 'z', wrong[0].score.toFixed(2) + ' ' + wrong[0].heard);
    check('GOP: ordering clear > unsure > wrong', clear[0].score > unsure[0].score && unsure[0].score > wrong[0].score);
    check('GOP: dropped final sound scores 0', dropped[2].score < 0.05, dropped[2].score.toFixed(3));
    check('GOP: neighbours of an error keep their score', wrong[1].score > 0.95 && wrong[2].score > 0.95);
    // a length variant of the expected vowel counts as the expected vowel
    const refV = probLogits({ 5: [['ɑː', 0.95]] }, 12);
    const stuV = probLogits({ 5: [['ɑː', 0.4], ['ɒ', 0.1]] }, 12);
    const g = gopScores(stuV, greedyPhones(refV.logp, 12, V, id2sym, 0), [{ start: 5, end: 6 }], id2sym, 0);
    check('GOP: weak but mostly uncontested sound is largely accepted', g[0].score > 0.65 && g[0].score < 0.9, g[0].score.toFixed(2));
    check('GOP: speaker phones decoded', rp.map((p) => p.sym).join('') === 'ðæt');

    // 9. The speaker as the point of reference: a sound the model hears badly even
    //    from the speaker is not held against the student.
    const word = [{ word: 'that', start: 0.08, end: 0.3 }];
    const firstOf = (refSpec, stuSpec) => comparePhonemes(probLogits(refSpec, 20), probLogits(stuSpec, 20), id2sym, word).words[0];
    const stuSpec = { 5: [['z', 0.6], ['ð', 0.3]], 9: [['æ', 0.9]], 13: [['t', 0.9]] };
    const clearRef = firstOf({ 5: [['ð', 0.95]], 9: [['æ', 0.95]], 13: [['t', 0.95]] }, stuSpec);
    const vagueRef = firstOf({ 5: [['ð', 0.5], ['z', 0.4]], 9: [['æ', 0.95]], 13: [['t', 0.95]] }, stuSpec);
    const cItem = clearRef.items.find((it) => it.ref === 'ð'), vItem = vagueRef.items.find((it) => it.ref === 'ð');
    check('GOP ref: clear speaker sound is judged in full', !cItem.unsure && cItem.weight === 1 && clearRef.issues.some((i) => i.kind === 'th'), cItem.score.toFixed(2));
    check('GOP ref: sound unclear in the speaker is marked and not reported', vItem.unsure && vItem.weight < 0.5 && !vagueRef.issues.length, 'weight ' + vItem.weight.toFixed(2) + ' score ' + vItem.score.toFixed(2));
    check('GOP ref: word score is not pulled down by the unclear sound', vagueRef.score > clearRef.score + 15, vagueRef.score + ' vs ' + clearRef.score);
    check('GOP ref: other sounds keep full weight', vagueRef.items.filter((it) => it.ref && it.ref !== 'ð').every((it) => it.weight === 1 && !it.unsure));

    // 10. Map of expected learner errors
    // th -> z where a third sound (ɾ) is the model's first guess: the pair check still finds it
    const hidden = firstOf({ 5: [['ð', 0.95]], 9: [['æ', 0.95]], 13: [['t', 0.95]] }, { 5: [['ɾ', 0.5], ['z', 0.4], ['ð', 0.05]], 9: [['æ', 0.9]], 13: [['t', 0.9]] });
    const hk = hidden.items.find((it) => it.ref === 'ð').known;
    check('error map: th -> z found behind another first guess', !!hk && hk.kind === 'th' && hk.heard === 'z', JSON.stringify(hk));
    check('error map: the remark names the sound and gives a tip', hidden.issues.some((i) => i.kind === 'th' && /tongue/.test(i.text)), hidden.issues.map((i) => i.text).join(' | '));
    check('error map: correct speech raises nothing', firstOf({ 5: [['ð', 0.95]], 9: [['æ', 0.95]], 13: [['t', 0.95]] }, { 5: [['ð', 0.9]], 9: [['æ', 0.9]], 13: [['t', 0.9]] }).items.every((it) => !it.known));
    check('error map: skipped when the speaker pair is not separated by the model', !vagueRef.items.some((it) => it.known));
    // long/short vowel: iː -> ɪ is invisible to the graded score (close variants) but is a rule of the map
    const symsV = ['<pad>', 'ʃ', 'iː', 'ɪ', 'p'];
    const idV = Object.fromEntries(symsV.map((x, i) => [x, i])), id2V = Object.fromEntries(symsV.map((x, i) => [i, x]));
    const mk = (spec) => {
      const flat = new Float32Array(20 * symsV.length).fill(Math.log(1e-6));
      for (let t = 0; t < 20; t++) flat[t * symsV.length] = Math.log(0.999);
      Object.entries(spec).forEach(([f, list]) => { let rest = 1; list.forEach(([sym, pr]) => { flat[f * symsV.length + idV[sym]] = Math.log(pr); rest -= pr; }); flat[f * symsV.length] = Math.log(Math.max(1e-6, rest)); });
      return { logp: flat, T: 20, V: symsV.length, frameSec: 0.02 };
    };
    const sheep = comparePhonemes(mk({ 5: [['ʃ', 0.95]], 9: [['iː', 0.95]], 13: [['p', 0.95]] }), mk({ 5: [['ʃ', 0.9]], 9: [['ɪ', 0.85], ['iː', 0.05]], 13: [['p', 0.9]] }), id2V, [{ word: 'sheep', start: 0.08, end: 0.3 }]).words[0];
    check('error map: long vowel said short (sheep -> ship)', sheep.issues.some((i) => i.kind === 'tense'), sheep.issues.map((i) => i.text).join(' | '));
    // 11. Non-rhotic speaker: r after a vowel is taken out of the model's transcription
    {
      const sy = ['<pad>', 'ɹ', 'ɔː', 'ɚ', 'z', 'ə', 'j', 'k', 'm', 'n', 'ɪ', 'ŋ', 'ɐ', 'aɪ', 'v', 'd'];
      const i2 = Object.fromEntries(sy.map((x, i) => [i, x]));
      const ph = (str) => str.split(' ').map((x) => ({ sym: x, id: sy.indexOf(x), t: 0, conf: 1 }));
      const fix = (str) => applyAccent(ph(str), 'non-rhotic', i2).map((x) => x.sym).join('');
      check('accent: roarers', fix('ɹ ɔː ɹ ɹ ɚ z') === 'ɹɔːɹəz', fix('ɹ ɔː ɹ ɹ ɚ z'));
      check('accent: morning, york', fix('m ɔː ɹ n ɪ ŋ') === 'mɔːnɪŋ' && fix('j ɔː ɹ k') === 'jɔːk');
      check('accent: r before a vowel is kept', fix('ɐ ɹ aɪ v d') === 'ɐɹaɪvd');
      check('accent: ids follow the symbols', applyAccent(ph('ɚ'), 'non-rhotic', i2)[0].id === sy.indexOf('ə'));
      check('accent: rhotic speaker is left as is', applyAccent(ph('j ɔː ɹ k'), undefined, i2).length === 4);
    }
    setErrorMap(null);
    check('error map: can be switched off', !firstOf({ 5: [['ð', 0.95]], 9: [['æ', 0.95]], 13: [['t', 0.95]] }, stuSpec).items.some((it) => it.known));
    setErrorMap('zh');
  }
}

console.log(failed ? `\n${failed} test(s) failed` : '\nAll tests passed');
process.exit(failed ? 1 : 0);
