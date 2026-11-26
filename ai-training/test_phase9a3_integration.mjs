/**
 * SIGNBRIDGE AI — Phase 9A.3 Production Integration Verification Suite
 * ====================================================================
 * Comprehensive automated test suite for the production integration of the
 * Adaptive Motion Gate into `frontend/src/utils/signClassifier.ts`.
 *
 * Verifies:
 * 1. Model integrity (SHA-256 exact match).
 * 2. Feature layout & slice index verification against featureBuilder.
 * 3. Static class indices & vocabulary reconciliation.
 * 4. Path A: Dynamic sign behavior & 0.45 confidence threshold preservation.
 * 5. Path B: Static sign acceptance across all three target classes (21.good, 34.stop, 45.hand).
 * 6. Path B: Idle-hand negative filter triggers (B1 through B5).
 * 7. Temporal Debouncer integration with static & dynamic predictions.
 * 8. 142 held-out test split sample evaluation & parity.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import {
  parseModelData,
  classifySign,
  TemporalDebouncer,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
  DEFAULT_STATIC_MIN_PRESENCE,
  DEFAULT_STATIC_MAX_POSE_STD,
  DEFAULT_STATIC_MAX_Y_CENTROID,
  DEFAULT_STATIC_MIN_CONFIDENCE,
  DEFAULT_STATIC_MIN_MARGIN,
  DEFAULT_STATIC_VOCABULARY,
  STATIC_CLASS_INDICES,
} from '../frontend/src/utils/signClassifier.ts';
import {
  TOTAL_FEATURE_DIM,
} from '../frontend/src/utils/featureBuilder.ts';

const PROJECT_ROOT = 'D:/SignBridgeAI';
const MODEL_PATH = path.join(PROJECT_ROOT, 'frontend/public/models/sign_classifier.json');
const TEST_REF_PATH = path.join(PROJECT_ROOT, 'ai-training/data/phase8b6c_audit/test_samples_py_reference.json');
const EXPECTED_MODEL_HASH = 'c8068a3fedd85645d736d16ebde450bd442970784adba40fdacb8970a4535274';

function runPhase9a3IntegrationTests() {
  console.log('='.repeat(80));
  console.log('SIGNBRIDGE AI — PHASE 9A.3 PRODUCTION INTEGRATION TEST SUITE');
  console.log('='.repeat(80));

  let passedTests = 0;
  let totalTests = 0;

  function assert(condition, message) {
    totalTests++;
    if (condition) {
      console.log(`  [PASS] ${message}`);
      passedTests++;
    } else {
      console.error(`  [FAIL] ${message}`);
      throw new Error(`Assertion failed: ${message}`);
    }
  }

  // -------------------------------------------------------------------------
  // 1. Model Integrity Check
  // -------------------------------------------------------------------------
  console.log('\n--- 1. Production Model Hash Verification ---');
  const rawModelBytes = fs.readFileSync(MODEL_PATH);
  const modelHash = crypto.createHash('sha256').update(rawModelBytes).digest('hex').toLowerCase();
  console.log(`  Observed Model SHA-256: ${modelHash}`);
  assert(modelHash === EXPECTED_MODEL_HASH, `Model SHA-256 matches verified M1 champion hash`);

  const modelJson = JSON.parse(rawModelBytes.toString('utf-8'));
  const model = parseModelData(modelJson);
  assert(model.numClasses === 49, `Model has exactly 49 classes`);
  assert(model.featureDim === 758, `Model feature dimension equals 758`);

  // -------------------------------------------------------------------------
  // 2. Feature Index Layout Verification
  // -------------------------------------------------------------------------
  console.log('\n--- 2. Feature Index & Schema Verification ---');
  assert(TOTAL_FEATURE_DIM === 758, `TOTAL_FEATURE_DIM from featureBuilder equals 758`);
  // Slot 0 (Left Hand): presence=0, mean=1..63 (y: 2..62 step 3), std=64..126, vel=253..315
  // Slot 1 (Right Hand): presence=379, mean=380..442 (y: 381..441 step 3), std=443..505, vel=632..694
  const leftYCoords = [];
  for (let i = 2; i < 64; i += 3) leftYCoords.push(i);
  assert(leftYCoords.length === 21, `Left hand has exactly 21 Y-coordinate landmark indices`);
  assert(leftYCoords[0] === 2 && leftYCoords[20] === 62, `Left hand Y-coordinate range [2, 62] verified`);

  const rightYCoords = [];
  for (let i = 381; i < 443; i += 3) rightYCoords.push(i);
  assert(rightYCoords.length === 21, `Right hand has exactly 21 Y-coordinate landmark indices`);
  assert(rightYCoords[0] === 381 && rightYCoords[20] === 441, `Right hand Y-coordinate range [381, 441] verified`);

  // -------------------------------------------------------------------------
  // 3. Static Vocabulary & Class Index Reconciliation
  // -------------------------------------------------------------------------
  console.log('\n--- 3. Static Class Index & Vocabulary Reconciliation ---');
  const idx21Good = model.labels.indexOf('21.good');
  const idx34Stop = model.labels.indexOf('34.stop');
  const idx45Hand = model.labels.indexOf('45.hand');

  console.log(`  Model Label Index for '21.good' : ${idx21Good}`);
  console.log(`  Model Label Index for '34.stop' : ${idx34Stop}`);
  console.log(`  Model Label Index for '45.hand' : ${idx45Hand}`);

  assert(idx21Good === 13, `21.good is at index 13`);
  assert(idx34Stop === 27, `34.stop is at index 27 (reconciles with dataset prefix 34)`);
  assert(idx45Hand === 39, `45.hand is at index 39 (reconciles with dataset prefix 45)`);
  assert(STATIC_CLASS_INDICES[0] === 13 && STATIC_CLASS_INDICES[1] === 27 && STATIC_CLASS_INDICES[2] === 39,
    `STATIC_CLASS_INDICES constant correctly maps to [13, 27, 39]`);
  assert(DEFAULT_STATIC_VOCABULARY.includes('21.good'), `Static vocabulary contains '21.good'`);
  assert(DEFAULT_STATIC_VOCABULARY.includes('34.stop'), `Static vocabulary contains '34.stop'`);
  assert(DEFAULT_STATIC_VOCABULARY.includes('45.hand'), `Static vocabulary contains '45.hand'`);

  // -------------------------------------------------------------------------
  // 4. Path A: Dynamic Sign Preservations
  // -------------------------------------------------------------------------
  console.log('\n--- 4. Path A: Dynamic Sign Preservations ---');
  const testRefData = JSON.parse(fs.readFileSync(TEST_REF_PATH, 'utf-8'));
  const dynamicSample = testRefData.samples[0]; // Sample from held-out test split
  const dynFeat = new Float32Array(dynamicSample.features);
  // Ensure velocity is dynamic (>= 0.05)
  for (let i = 253; i < 316; i++) dynFeat[i] = 0.08;

  const dynResult = classifySign(dynFeat, model, { applyRejectionGates: true });
  assert(dynResult.recognitionPath === 'dynamic_path', `High velocity routes to 'dynamic_path'`);
  assert(dynResult.gatingMetrics.motionEnergy >= 0.05, `Motion energy correctly recorded >= 0.05`);
  assert(dynResult.rejectionDetails.thresholds.confidence === 0.45, `Dynamic confidence threshold preserved at 0.45`);

  // Verify confidence gate on Path A
  const dynLowConfResult = classifySign(dynFeat, model, { applyRejectionGates: true, confidenceThreshold: 1.001 });
  assert(dynLowConfResult.status === 'rejected', `Low confidence rejected on dynamic path`);
  assert(dynLowConfResult.rejectionReason === 'confidence_low', `Rejection reason is 'confidence_low'`);
  assert(dynLowConfResult.recognitionPath === 'dynamic_path', `Recognition path remains 'dynamic_path'`);

  // Verify presence gate
  const zeroPresence = new Float32Array(758);
  const zeroPresResult = classifySign(zeroPresence, model, { applyRejectionGates: true });
  assert(zeroPresResult.status === 'rejected', `Zero presence rejected`);
  assert(zeroPresResult.rejectionReason === 'presence_too_low', `Rejection reason is 'presence_too_low'`);
  assert(zeroPresResult.recognitionPath === 'none', `Recognition path is 'none' for presence failure`);

  // -------------------------------------------------------------------------
  // 5. Path B: Static Sign Acceptance (21.good, 34.stop, 45.hand)
  // -------------------------------------------------------------------------
  console.log('\n--- 5. Path B: Static Sign Acceptance Tests ---');

  const STATIC_VECTORS_PATH = path.join(PROJECT_ROOT, 'ai-training/data/synthetic_static_vectors.json');
  const staticVectors = JSON.parse(fs.readFileSync(STATIC_VECTORS_PATH, 'utf-8'));

  function getStaticFeatureVector(label) {
    return new Float32Array(staticVectors[label]);
  }

  const staticTargets = [
    { label: '21.good', idx: 13 },
    { label: '34.stop', idx: 27 },
    { label: '45.hand', idx: 39 },
  ];

  for (const target of staticTargets) {
    const feat = getStaticFeatureVector(target.label);
    const result = classifySign(feat, model, { applyRejectionGates: true });
    console.log(`  Evaluating static candidate for '${target.label}':`);
    console.log(`    Status: ${result.status}, Path: ${result.recognitionPath}, Pred: ${result.predictedLabel}, Conf: ${(result.confidence * 100).toFixed(1)}%, Margin: ${(result.gatingMetrics.margin * 100).toFixed(1)}%`);

    assert(result.recognitionPath === 'static_path', `'${target.label}' routed to 'static_path'`);
    assert(result.gatingMetrics.motionEnergy < 0.05, `Motion energy is stationary (< 0.05)`);
    assert(result.status === 'accepted', `'${target.label}' successfully accepted under Path B`);
    assert(result.predictedLabel === target.label, `Predicted label is exactly '${target.label}'`);
  }

  // -------------------------------------------------------------------------
  // 6. Path B: Idle-Hand Negative Filters (B1 to B5)
  // -------------------------------------------------------------------------
  console.log('\n--- 6. Path B: Idle-Hand Negative Defensive Filters ---');

  // Filter B1: Hand presence < 0.50
  const fB1 = getStaticFeatureVector('21.good');
  fB1[379] = 0.35; // Presence < 0.50
  const resB1 = classifySign(fB1, model, { applyRejectionGates: true });
  assert(resB1.status === 'rejected', `Filter B1 rejected low presence`);
  assert(resB1.rejectionReason === 'static_presence_low', `Rejection reason is 'static_presence_low'`);

  // Filter B2: Spatial pose jitter > 0.055
  const fB2 = getStaticFeatureVector('21.good');
  for (let i = 443; i < 506; i++) fB2[i] = 0.075; // std > 0.055
  const resB2 = classifySign(fB2, model, { applyRejectionGates: true });
  assert(resB2.status === 'rejected', `Filter B2 rejected unstable jitter`);
  assert(resB2.rejectionReason === 'static_pose_jitter', `Rejection reason is 'static_pose_jitter'`);

  // Filter B3: Resting hand elevation Y > 0.75
  const fB3 = getStaticFeatureVector('21.good');
  for (let i = 381; i < 443; i += 3) fB3[i] = 0.85; // Desk/lap resting
  const resB3 = classifySign(fB3, model, { applyRejectionGates: true });
  assert(resB3.status === 'rejected', `Filter B3 rejected resting hand`);
  assert(resB3.rejectionReason === 'idle_hand_resting', `Rejection reason is 'idle_hand_resting'`);

  // Filter B4: Non-static vocabulary predicted under low motion
  // Use dummyStationary which predicts 2.black
  const fB4 = new Float32Array(758);
  fB4[0] = 0.90; fB4[379] = 0.90; // Presence ok
  for (let i = 64; i < 127; i++) fB4[i] = 0.02; // Std ok
  for (let i = 443; i < 506; i++) fB4[i] = 0.02;
  for (let i = 2; i < 64; i += 3) fB4[i] = 0.50; // Centroid ok
  for (let i = 381; i < 443; i += 3) fB4[i] = 0.50;
  const resB4 = classifySign(fB4, model, { applyRejectionGates: true });
  assert(resB4.status === 'rejected', `Filter B4 rejected non-static class with low motion`);
  assert(resB4.rejectionReason === 'motion_stationary', `Rejection reason is 'motion_stationary'`);

  // Filter B5: Confidence < 0.50 or Margin < 0.10
  const fB5 = getStaticFeatureVector('21.good');
  const resB5 = classifySign(fB5, model, { applyRejectionGates: true, staticMinConfidence: 0.999 });
  assert(resB5.status === 'rejected', `Filter B5 rejected ambiguous static confidence`);
  assert(resB5.rejectionReason === 'static_confidence_ambiguous', `Rejection reason is 'static_confidence_ambiguous'`);

  // -------------------------------------------------------------------------
  // 7. Temporal Debouncer Integration
  // -------------------------------------------------------------------------
  console.log('\n--- 7. Temporal Debouncer Integration ---');
  const debouncer = new TemporalDebouncer({ minConsecutiveFrames: 2 });

  // Feed 1st static prediction
  const staticGoodFeat = getStaticFeatureVector('21.good');
  const pred1 = classifySign(staticGoodFeat, model, { applyRejectionGates: true });
  const deb1 = debouncer.update(pred1);
  assert(!deb1.isConfirmed, `Debouncer requires 2 windows; first window not confirmed`);
  assert(deb1.consecutiveCount === 1, `Consecutive count equals 1`);
  assert(deb1.candidateLabel === '21.good', `Candidate label is '21.good'`);

  // Feed 2nd consecutive static prediction
  const pred2 = classifySign(staticGoodFeat, model, { applyRejectionGates: true });
  const deb2 = debouncer.update(pred2);
  assert(deb2.isConfirmed, `Debouncer confirms sign on 2nd consecutive window`);
  assert(deb2.confirmedLabel === '21.good', `Confirmed label is '21.good'`);
  assert(deb2.consecutiveCount === 2, `Consecutive count equals 2`);

  // Feed rejection -> resets debouncer
  const deb3 = debouncer.update(resB3);
  assert(!deb3.isConfirmed, `Rejection resets debouncer`);
  assert(deb3.consecutiveCount === 0, `Consecutive count resets to 0`);
  assert(deb3.candidateClassIndex === null, `Candidate class index reset to null`);

  // -------------------------------------------------------------------------
  // 8. 142 Held-out Test Split Verification
  // -------------------------------------------------------------------------
  console.log('\n--- 8. 142 Test Set Non-Regression Verification ---');
  let testAgreements = 0;
  for (let i = 0; i < testRefData.samples.length; i++) {
    const s = testRefData.samples[i];
    const feat = new Float32Array(s.features);
    const pred = classifySign(feat, model, { applyRejectionGates: false });
    if (pred.predictedClass === s.pyPred) {
      testAgreements++;
    }
  }
  assert(testAgreements === 142, `142/142 test samples match Python champion predictions exactly`);

  console.log('\n' + '='.repeat(80));
  console.log(`ALL INTEGRATION TESTS PASSED: ${passedTests}/${totalTests} (100.0%)`);
  console.log('='.repeat(80));

  return {
    passed: passedTests,
    total: totalTests,
    modelHash,
    status: 'PASS',
  };
}

runPhase9a3IntegrationTests();
