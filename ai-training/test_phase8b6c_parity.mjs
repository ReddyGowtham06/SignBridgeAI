/**
 * SIGNBRIDGE AI — Phase 8B.6C TypeScript Inference Parity Audit Runner
 * ====================================================================
 * Evaluates the Phase 8B.6B Champion Model (JSON) across all 142 held-out
 * test split samples using the production TypeScript inference engine:
 * `frontend/src/utils/signClassifier.ts`.
 *
 * Compares:
 * - Feature standardization (scalerMean, scalerScale)
 * - Raw logits
 * - Softmax probabilities
 * - Top-1 class predictions
 * - Top-3 candidates ranking & scores
 * - In-browser execution latency (microseconds per sample)
 * - Gesture rejection gate functionality
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import {
  parseModelData,
  classifySign,
  TemporalDebouncer,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
} from '../frontend/src/utils/signClassifier.ts';

const PROJECT_ROOT = 'D:/SignBridgeAI';
const CHAMPION_JSON_PATH = path.join(PROJECT_ROOT, 'ai-training/models/phase8b6b/phase8b6b_champion.json');
const TEST_REF_PATH = path.join(PROJECT_ROOT, 'ai-training/data/phase8b6c_audit/test_samples_py_reference.json');
const OUT_REPORT_PATH = path.join(PROJECT_ROOT, 'ai-training/data/phase8b6c_audit/phase8b6c_typescript_parity_report.json');

function runParityAudit() {
  console.log('='.repeat(76));
  console.log('PHASE 8B.6C: TYPESCRIPT INFERENCE PARITY AUDIT (142 HELD-OUT SAMPLES)');
  console.log('='.repeat(76));

  // 1. Load and parse model
  console.log(`Loading Champion JSON: ${CHAMPION_JSON_PATH}`);
  const rawModelJson = JSON.parse(fs.readFileSync(CHAMPION_JSON_PATH, 'utf-8'));
  const tParse0 = performance.now();
  const model = parseModelData(rawModelJson);
  const tParse = performance.now() - tParse0;
  console.log(`Model parsed in ${tParse.toFixed(3)} ms (Classes: ${model.numClasses}, Dims: ${model.featureDim})\n`);

  // 2. Load test samples
  console.log(`Loading 142 Test Samples with Python References: ${TEST_REF_PATH}`);
  const testRefData = JSON.parse(fs.readFileSync(TEST_REF_PATH, 'utf-8'));
  const samples = testRefData.samples;
  const totalSamples = samples.length;
  console.log(`Loaded ${totalSamples} samples.\n`);

  let maxLogitDiff = 0.0;
  let logitDiffSum = 0.0;
  let maxProbDiff = 0.0;
  let probDiffSum = 0.0;
  let maxScaledDiff = 0.0;

  let top1Matches = 0;
  let top3Matches = 0;
  const mismatches = [];
  const latenciesUs = [];

  for (let i = 0; i < totalSamples; i++) {
    const s = samples[i];
    const features = new Float32Array(s.features);

    // Verify standardization parity directly
    for (let f = 0; f < 758; f++) {
      const z_ts = (features[f] - model.scalerMean[f]) / model.scalerScale[f];
      const diffZ = Math.abs(z_ts - s.pyScaled[f]);
      if (diffZ > maxScaledDiff) maxScaledDiff = diffZ;
    }

    // Benchmark in-browser execution latency
    const t0 = performance.now();
    const res = classifySign(features, model, { applyRejectionGates: false });
    const t1 = performance.now();
    latenciesUs.push((t1 - t0) * 1000.0);

    // Compare logits
    for (let c = 0; c < 49; c++) {
      const diffL = Math.abs(res.logits[c] - s.pyLogits[c]);
      const diffP = Math.abs(res.probabilities[c] - s.pyProbs[c]);
      if (diffL > maxLogitDiff) maxLogitDiff = diffL;
      if (diffP > maxProbDiff) maxProbDiff = diffP;
      logitDiffSum += diffL;
      probDiffSum += diffP;
    }

    // Check Top-1 match
    const isTop1Match = (res.predictedClass === s.pyPred);
    if (isTop1Match) {
      top1Matches += 1;
    } else {
      mismatches.push({
        sampleId: s.sampleId,
        trueLabel: s.trueLabel,
        pyPred: s.pyPred,
        tsPred: res.predictedClass,
      });
    }

    // Check Top-3 match
    let isTop3Match = true;
    for (let k = 0; k < 3; k++) {
      const tsCand = res.topCandidates[k];
      const pyCand = s.pyTop3[k];
      if (tsCand.classIndex !== pyCand.classIndex || Math.abs(tsCand.score - pyCand.score) > 1e-4) {
        isTop3Match = false;
        break;
      }
    }
    if (isTop3Match) {
      top3Matches += 1;
    }
  }

  const meanLogitDiff = logitDiffSum / (totalSamples * 49);
  const meanProbDiff = probDiffSum / (totalSamples * 49);
  latenciesUs.sort((a, b) => a - b);
  const meanLatencyUs = latenciesUs.reduce((a, b) => a + b, 0) / totalSamples;
  const p50LatencyUs = latenciesUs[Math.floor(totalSamples * 0.50)];
  const p95LatencyUs = latenciesUs[Math.floor(totalSamples * 0.95)];
  const p99LatencyUs = latenciesUs[Math.floor(totalSamples * 0.99)];

  console.log('PARITY AUDIT RESULTS:');
  console.log(`  • Samples Evaluated          : ${totalSamples}`);
  console.log(`  • Max Scaler Diff            : ${maxScaledDiff.toExponential(2)}`);
  console.log(`  • Max Logit Diff             : ${maxLogitDiff.toExponential(2)}`);
  console.log(`  • Mean Logit Diff            : ${meanLogitDiff.toExponential(2)}`);
  console.log(`  • Max Probability Diff       : ${maxProbDiff.toExponential(2)}`);
  console.log(`  • Mean Probability Diff      : ${meanProbDiff.toExponential(2)}`);
  console.log(`  • Top-1 Class Agreement      : ${top1Matches}/${totalSamples} (${((top1Matches / totalSamples) * 100).toFixed(2)}%)`);
  console.log(`  • Top-3 Candidates Agreement : ${top3Matches}/${totalSamples} (${((top3Matches / totalSamples) * 100).toFixed(2)}%)`);
  console.log(`  • Mean Latency               : ${meanLatencyUs.toFixed(1)} µs (${(meanLatencyUs / 1000).toFixed(4)} ms)`);
  console.log(`  • P95 Latency                : ${p95LatencyUs.toFixed(1)} µs (${(p95LatencyUs / 1000).toFixed(4)} ms)`);
  console.log(`  • P99 Latency                : ${p99LatencyUs.toFixed(1)} µs (${(p99LatencyUs / 1000).toFixed(4)} ms)\n`);

  // 3. Test Rejection Gates
  console.log('Testing Multi-Stage Gesture Rejection Gates:');
  // Gate 1: Presence
  const dummyLowPresence = new Float32Array(758); // All zeros -> presence = 0
  const gate1Res = classifySign(dummyLowPresence, model, { applyRejectionGates: true });
  const gate1Pass = (gate1Res.status === 'rejected' && gate1Res.rejectionReason === 'presence_too_low');
  console.log(`  • Gate 1 (Presence < 0.20)     : ${gate1Pass ? 'PASSED (Rejected correctly)' : 'FAILED'}`);

  // Gate 2: Motion
  const dummyStationary = new Float32Array(758);
  dummyStationary[0] = 1.0; // Left hand present
  dummyStationary[379] = 1.0; // Right hand present
  // velocity indices (253-315 and 632-694) remain 0.0 -> motion = 0
  const gate2Res = classifySign(dummyStationary, model, { applyRejectionGates: true });
  const gate2Pass = (gate2Res.status === 'rejected' && gate2Res.rejectionReason === 'motion_stationary');
  console.log(`  • Gate 2 (Motion < 0.05)       : ${gate2Pass ? 'PASSED (Rejected correctly)' : 'FAILED'}`);

  // Gate 3: Confidence
  const highMotionFeatures = new Float32Array(samples[0].features);
  highMotionFeatures[253] = 0.5; // Ensure motion
  const gate3Res = classifySign(highMotionFeatures, model, { applyRejectionGates: true, confidenceThreshold: 0.999 });
  const gate3Pass = (gate3Res.status === 'rejected' && gate3Res.rejectionReason === 'confidence_low');
  console.log(`  • Gate 3 (Confidence < Thr)    : ${gate3Pass ? 'PASSED (Rejected correctly)' : 'FAILED'}`);

  // Gate 4: Temporal Debouncer
  const debouncer = new TemporalDebouncer({ minConsecutiveFrames: 2 });
  const mockAccepted1 = { status: 'accepted', predictedClass: 10, predictedLabel: '19.girl', confidence: 0.85, rejectionReason: null };
  const mockAccepted2 = { status: 'accepted', predictedClass: 10, predictedLabel: '19.girl', confidence: 0.88, rejectionReason: null };
  const mockRejected = { status: 'rejected', predictedClass: null, predictedLabel: null, confidence: 0.0, rejectionReason: 'presence_too_low' };

  const d1 = debouncer.update(mockAccepted1);
  const d2 = debouncer.update(mockAccepted2);
  const d3 = debouncer.update(mockRejected);
  const gate4Pass = (!d1.isConfirmed && d2.isConfirmed && d2.confirmedClassIndex === 10 && !d3.isConfirmed);
  console.log(`  • Gate 4 (Temporal Debouncer)  : ${gate4Pass ? 'PASSED (Confirmed streak correctly)' : 'FAILED'}\n`);

  const auditPassed = (
    maxLogitDiff < 1e-4 &&
    maxProbDiff < 1e-4 &&
    top1Matches === totalSamples &&
    top3Matches === totalSamples &&
    gate1Pass &&
    gate2Pass &&
    gate3Pass &&
    gate4Pass
  );

  console.log('='.repeat(76));
  console.log(`TYPESCRIPT PARITY & INTEGRATION STATUS: ${auditPassed ? 'PASSED (100% PARITY & COMPATIBLE)' : 'FAILED'}`);
  console.log('='.repeat(76));

  const reportPayload = {
    auditTimestamp: new Date().toISOString(),
    totalSamplesEvaluated: totalSamples,
    maxLogitDiff,
    meanLogitDiff,
    maxProbDiff,
    meanProbDiff,
    maxScaledDiff,
    top1Matches,
    top1AgreementPercent: (top1Matches / totalSamples) * 100.0,
    top3Matches,
    top3AgreementPercent: (top3Matches / totalSamples) * 100.0,
    mismatchesCount: mismatches.length,
    mismatches,
    latencyBenchmarking: {
      unit: 'microseconds',
      meanLatencyUs: Math.round(meanLatencyUs * 10) / 10,
      p50LatencyUs: Math.round(p50LatencyUs * 10) / 10,
      p95LatencyUs: Math.round(p95LatencyUs * 10) / 10,
      p99LatencyUs: Math.round(p99LatencyUs * 10) / 10,
      meanLatencyMs: Math.round((meanLatencyUs / 1000) * 1000) / 1000,
      p95LatencyMs: Math.round((p95LatencyUs / 1000) * 1000) / 1000,
    },
    rejectionGatesVerification: {
      gate1_presence: gate1Pass,
      gate2_motion: gate2Pass,
      gate3_confidence: gate3Pass,
      gate4_temporal_debouncer: gate4Pass,
    },
    overallVerdict: auditPassed ? 'PASS' : 'FAIL',
  };

  fs.writeFileSync(OUT_REPORT_PATH, JSON.stringify(reportPayload, null, 2), 'utf-8');
  console.log(`Parity report saved to: ${OUT_REPORT_PATH}`);
}

runParityAudit();
