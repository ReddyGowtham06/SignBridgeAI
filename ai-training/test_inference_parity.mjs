/**
 * SIGNBRIDGE AI — TypeScript Inference Parity & Rejection Gate Test Runner
 * =========================================================================
 * Loads the exported model, evaluates real dataset feature vectors,
 * validates numerical parity against Python checkpoint reference outputs,
 * and tests all four gesture rejection gates.
 */

import * as fs from 'node:fs';
import {
  parseModelData,
  classifySign,
  TemporalDebouncer,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
} from '../frontend/src/utils/signClassifier.ts';

function main() {
  const args = process.argv.slice(2);
  const modelPath = args[0] || 'frontend/public/models/sign_classifier.json';
  const testDataPath = args[1]; // JSON containing test samples & Python reference outputs
  const outputPath = args[2];

  // 1. Load model JSON
  const rawModelJson = JSON.parse(fs.readFileSync(modelPath, 'utf-8'));
  const model = parseModelData(rawModelJson);

  let parityResults = null;

  if (testDataPath && fs.existsSync(testDataPath)) {
    const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf-8'));
    const samples = testData.samples; // Array of { features, pyLogits, pyProbs, pyPred, pyTop3 }

    let maxLogitDiff = 0;
    let maxProbDiff = 0;
    let logitDiffSum = 0;
    let probDiffSum = 0;
    let predMatches = 0;
    let top3Matches = 0;
    const totalSamples = samples.length;
    const sampleBenchmarks = [];

    for (let i = 0; i < totalSamples; i++) {
      const s = samples[i];
      const features = new Float32Array(s.features);

      // Measure inference time
      const t0 = performance.now();
      // Bypass rejection gates to compare pure raw model projections against Python
      const res = classifySign(features, model, { applyRejectionGates: false });
      const t1 = performance.now();
      sampleBenchmarks.push((t1 - t0) * 1000); // microseconds

      // Compare logits
      for (let c = 0; c < 49; c++) {
        const diffL = Math.abs(res.logits[c] - s.pyLogits[c]);
        const diffP = Math.abs(res.probabilities[c] - s.pyProbs[c]);
        if (diffL > maxLogitDiff) maxLogitDiff = diffL;
        if (diffP > maxProbDiff) maxProbDiff = diffP;
        logitDiffSum += diffL;
        probDiffSum += diffP;
      }

      // Check Top-1
      if (res.predictedClass === s.pyPred) {
        predMatches += 1;
      }

      // Check Top-3 ordering and classes
      let t3Match = true;
      for (let k = 0; k < 3; k++) {
        if (
          res.topCandidates[k].classIndex !== s.pyTop3[k].classIndex ||
          Math.abs(res.topCandidates[k].score - s.pyTop3[k].score) > 1e-4
        ) {
          t3Match = false;
        }
      }
      if (t3Match) {
        top3Matches += 1;
      }
    }

    const meanLogitDiff = logitDiffSum / (totalSamples * 49);
    const meanProbDiff = probDiffSum / (totalSamples * 49);
    const meanTimeUs = sampleBenchmarks.reduce((a, b) => a + b, 0) / totalSamples;

    parityResults = {
      totalSamples,
      maxLogitDiff,
      meanLogitDiff,
      maxProbDiff,
      meanProbDiff,
      predMatches,
      predAgreementPercent: (predMatches / totalSamples) * 100,
      top3Matches,
      top3AgreementPercent: (top3Matches / totalSamples) * 100,
      meanInferenceTimeUs: Math.round(meanTimeUs * 10) / 10,
      meanInferenceTimeMs: Math.round((meanTimeUs / 1000) * 1000) / 1000,
    };
  }

  // 2. Unit Tests for Rejection Gates
  const gateResults = {};

  // Gate 1: Presence Rejection
  {
    const dummyFeatures = new Float32Array(758);
    // Fill with moderate motion and dummy spatial values
    for (let i = 0; i < 758; i++) dummyFeatures[i] = 0.5;
    // Set presence below threshold: Left = 0.10, Right = 0.05
    dummyFeatures[0] = 0.10;
    dummyFeatures[379] = 0.05;

    const res = classifySign(dummyFeatures, model, {
      presenceThreshold: DEFAULT_PRESENCE_THRESHOLD,
    });
    gateResults.gate1_presence = {
      tested: true,
      passed: res.status === 'rejected' && res.rejectionReason === 'presence_too_low',
      status: res.status,
      reason: res.rejectionReason,
    };
  }

  // Gate 2: Motion (Stationary Hands) Rejection
  {
    const dummyFeatures = new Float32Array(758);
    for (let i = 0; i < 758; i++) dummyFeatures[i] = 0.5;
    // Presence is high: 1.0 on both hands
    dummyFeatures[0] = 1.0;
    dummyFeatures[379] = 1.0;
    // Set velocity to near-zero (0.001 < 0.05)
    for (let i = 253; i < 316; i++) dummyFeatures[i] = 0.001;
    for (let i = 632; i < 695; i++) dummyFeatures[i] = 0.001;

    const res = classifySign(dummyFeatures, model, {
      motionThreshold: DEFAULT_MOTION_THRESHOLD,
    });
    gateResults.gate2_motion = {
      tested: true,
      passed: res.status === 'rejected' && res.rejectionReason === 'motion_stationary',
      status: res.status,
      reason: res.rejectionReason,
    };
  }

  // Gate 3: Confidence Rejection
  {
    // Find a real test sample where the model's confidence is genuinely below 0.45
    let lowConfSample = null;
    if (testDataPath && fs.existsSync(testDataPath)) {
      const testData = JSON.parse(fs.readFileSync(testDataPath, 'utf-8'));
      for (const s of testData.samples) {
        const maxP = Math.max(...s.pyProbs);
        if (maxP < DEFAULT_CONFIDENCE_THRESHOLD) {
          lowConfSample = s;
          break;
        }
      }
    }

    if (lowConfSample) {
      const feat = new Float32Array(lowConfSample.features);
      const res = classifySign(feat, model, {
        confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
      });

      gateResults.gate3_confidence = {
        tested: true,
        passed:
          res.status === 'rejected' &&
          res.rejectionReason === 'confidence_low' &&
          res.confidence < DEFAULT_CONFIDENCE_THRESHOLD &&
          res.topCandidates.length === 3,
        status: res.status,
        reason: res.rejectionReason,
        topConfidence: res.confidence,
        threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        sampleIndex: lowConfSample.sampleIndex,
      };
    } else {
      // Fallback synthetic check
      const dummyFeatures = new Float32Array(758);
      dummyFeatures[0] = 0.9;
      dummyFeatures[379] = 0.9;
      for (let i = 253; i < 316; i++) dummyFeatures[i] = 0.1;
      for (let i = 632; i < 695; i++) dummyFeatures[i] = 0.1;
      const res = classifySign(dummyFeatures, model, { confidenceThreshold: 0.999 });
      gateResults.gate3_confidence = {
        tested: true,
        passed: res.status === 'rejected' && res.rejectionReason === 'confidence_low',
        status: res.status,
        reason: res.rejectionReason,
        topConfidence: res.confidence,
      };
    }
  }

  // Gate 4: Temporal Debounce Tests
  {
    const debouncer = new TemporalDebouncer({ minConsecutiveFrames: 2 });
    const steps = [];

    // Step 1: Candidate A (streak = 1, not confirmed)
    const p1 = { status: 'accepted', predictedClass: 10, predictedLabel: '10.youareperfect' };
    const r1 = debouncer.update(p1);
    steps.push({ step: 1, isConfirmed: r1.isConfirmed, count: r1.consecutiveCount });

    // Step 2: Candidate A again (streak = 2, confirmed!)
    const p2 = { status: 'accepted', predictedClass: 10, predictedLabel: '10.youareperfect' };
    const r2 = debouncer.update(p2);
    steps.push({
      step: 2,
      isConfirmed: r2.isConfirmed,
      count: r2.consecutiveCount,
      confirmed: r2.confirmedLabel,
    });

    // Step 3: Candidate A third time (streak = 3, remains confirmed)
    const p3 = { status: 'accepted', predictedClass: 10, predictedLabel: '10.youareperfect' };
    const r3 = debouncer.update(p3);
    steps.push({ step: 3, isConfirmed: r3.isConfirmed, count: r3.consecutiveCount });

    // Step 4: New Candidate B arrives (streak resets to 1, not confirmed)
    const p4 = { status: 'accepted', predictedClass: 25, predictedLabel: '25.NO' };
    const r4 = debouncer.update(p4);
    steps.push({ step: 4, isConfirmed: r4.isConfirmed, count: r4.consecutiveCount });

    // Step 5: Interrupted by rejection (e.g. uncertain or missing hand)
    const p5 = { status: 'rejected', predictedClass: null, predictedLabel: null, rejectionReason: 'confidence_low' };
    const r5 = debouncer.update(p5);
    steps.push({ step: 5, isConfirmed: r5.isConfirmed, count: r5.consecutiveCount });

    // Step 6: Candidate B again after interruption (streak = 1, not confirmed)
    const p6 = { status: 'accepted', predictedClass: 25, predictedLabel: '25.NO' };
    const r6 = debouncer.update(p6);
    steps.push({ step: 6, isConfirmed: r6.isConfirmed, count: r6.consecutiveCount });

    // Step 7: Candidate B second time (streak = 2, confirmed!)
    const p7 = { status: 'accepted', predictedClass: 25, predictedLabel: '25.NO' };
    const r7 = debouncer.update(p7);
    steps.push({
      step: 7,
      isConfirmed: r7.isConfirmed,
      count: r7.consecutiveCount,
      confirmed: r7.confirmedLabel,
    });

    const debouncerPassed =
      !steps[0].isConfirmed &&
      steps[1].isConfirmed &&
      steps[2].isConfirmed &&
      !steps[3].isConfirmed &&
      !steps[4].isConfirmed &&
      !steps[5].isConfirmed &&
      steps[6].isConfirmed;

    gateResults.gate4_temporal_debounce = {
      tested: true,
      passed: debouncerPassed,
      steps,
    };
  }

  const finalOutput = {
    parity: parityResults,
    rejectionGates: gateResults,
  };

  const outputJson = JSON.stringify(finalOutput, null, 2);
  if (outputPath) {
    fs.writeFileSync(outputPath, outputJson, 'utf-8');
  } else {
    process.stdout.write(outputJson);
  }
}

main();
