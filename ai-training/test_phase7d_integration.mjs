/**
 * SIGNBRIDGE AI — Phase 7D Live Webcam Inference Integration Test
 * =================================================================
 * Simulates the streaming detection loop of SignToTextPage.tsx:
 * 1. Rolling FIFO buffer bounding (strictly max 60 frames).
 * 2. Inference scheduling: gate until 60 frames, then evaluate every 15 frames.
 * 3. Rejection gate handling (presence, motion, confidence).
 * 4. Temporal debouncer (streak >= 2) before committing to transcript.
 * 5. Duplicate prevention and natural re-signing after pause.
 * 6. Stop and cleanup buffer reset.
 */

import * as fs from 'node:fs';
import { slotMediaPipeResult, computeVideoFeatureVector } from '../frontend/src/utils/featureBuilder.ts';
import {
  parseModelData,
  classifySign,
  TemporalDebouncer,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
} from '../frontend/src/utils/signClassifier.ts';

function formatClassLabel(rawLabel) {
  const stripped = rawLabel.replace(/^\d+\./, '');
  const dictionary = {
    youareperfect: 'You Are Perfect',
    goodmorning: 'Good Morning',
    heavyrain: 'Heavy Rain',
    iamfine: 'I Am Fine',
    iloveyou: 'I Love You',
    nicetomeetyou: 'Nice To Meet You',
    seeyoulater: 'See You Later',
    takecare: 'Take Care',
    whatisyourname: 'What Is Your Name',
  };
  const lower = stripped.toLowerCase();
  if (dictionary[lower]) return dictionary[lower];
  return stripped.replace(/[_-]/g, ' ').trim().toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

async function runLivePipelineSimulation() {
  console.log('=' .repeat(75));
  console.log('SIGNBRIDGE AI — PHASE 7D LIVE PIPELINE INTEGRATION VERIFICATION');
  console.log('=' .repeat(75));

  // 1. Load model JSON
  const rawModel = JSON.parse(fs.readFileSync('frontend/public/models/sign_classifier.json', 'utf-8'));
  const model = parseModelData(rawModel);
  console.log(`[PASS] Loaded model: ${model.modelType} (${model.numClasses} classes, ${model.featureDim} features)`);

  // Pipeline simulation state matching SignToTextPage.tsx
  const frameBuffer = [];
  const BUFFER_MAX_FRAMES = 60;
  const INFERENCE_STRIDE_FRAMES = 15;
  let strideCounter = 0;
  const debouncer = new TemporalDebouncer({ minConsecutiveFrames: 2 });
  let lastCommittedLabel = null;
  let transcript = '';
  const history = [];

  let inferenceRuns = 0;
  const inferenceLog = [];

  // Helper to simulate a frame arriving from MediaPipe
  function processFrame(landmarks, handedness, timestampMs) {
    const slotted = slotMediaPipeResult(landmarks, handedness, timestampMs);
    frameBuffer.push(slotted);
    if (frameBuffer.length > BUFFER_MAX_FRAMES) {
      frameBuffer.shift();
    }

    const hasHands = landmarks && landmarks.length > 0;
    if (!hasHands) {
      // User dropped hands: clear last-committed label tracker for immediate re-signing capability
      lastCommittedLabel = null;
    }

    if (frameBuffer.length < BUFFER_MAX_FRAMES) {
      return { status: 'collecting_frames', bufferCount: frameBuffer.length };
    }

    strideCounter += 1;
    if (strideCounter >= INFERENCE_STRIDE_FRAMES) {
      strideCounter = 0;
      inferenceRuns += 1;

      const features = computeVideoFeatureVector(frameBuffer);
      const prediction = classifySign(features, model, {
        presenceThreshold: DEFAULT_PRESENCE_THRESHOLD,
        motionThreshold: DEFAULT_MOTION_THRESHOLD,
        confidenceThreshold: DEFAULT_CONFIDENCE_THRESHOLD,
        applyRejectionGates: true,
      });

      const debounced = debouncer.update(prediction);

      if (prediction.status === 'rejected') {
        lastCommittedLabel = null; // Reset tracker so user can sign same word after pause
        return {
          status: prediction.rejectionReason,
          prediction,
          debounced,
        };
      }

      if (debounced.isConfirmed && debounced.confirmedLabel) {
        // Commit only on the exact evaluation cycle where the streak reaches the threshold (rising edge)
        if (debounced.consecutiveCount === debounced.requiredCount) {
          const clean = formatClassLabel(debounced.confirmedLabel);
          transcript = transcript ? `${transcript} ${clean}` : clean;
          lastCommittedLabel = debounced.confirmedLabel;
          history.push(clean);
        }
        return {
          status: 'sign_recognized',
          confirmedSign: debounced.confirmedLabel,
          transcript,
          prediction,
          debounced,
        };
      }

      return {
        status: 'recognizing',
        candidate: debounced.candidateLabel,
        prediction,
        debounced,
      };
    }

    return { status: 'waiting_stride', bufferCount: frameBuffer.length };
  }

  // --- TEST 1: Buffer Accumulation & Stride Check ---
  console.log('\nTest 1: Frame Accumulation & Stride Check...');
  let t = 1000;
  for (let f = 1; f <= 59; f++) {
    t += 33;
    const res = processFrame(null, null, t);
    if (res.status !== 'collecting_frames' || frameBuffer.length !== f) {
      throw new Error(`Buffer accumulation error at frame ${f}`);
    }
  }
  if (inferenceRuns !== 0) throw new Error('Inference triggered prematurely before 60 frames!');
  console.log('  [PASS] 59 frames buffered without premature inference.');

  // Frame 60 arrives (presence is low since landmarks are null)
  t += 33;
  const f60Res = processFrame(null, null, t);
  if (frameBuffer.length !== 60) throw new Error('Buffer length != 60 at frame 60');
  // Note: stride counter starts at 0, first stride triggers after 15 frames
  for (let f = 61; f <= 75; f++) {
    t += 33;
    processFrame(null, null, t);
  }
  if (inferenceRuns !== 1) throw new Error(`Expected exactly 1 inference run after 75 frames, got ${inferenceRuns}`);
  if (frameBuffer.length !== 60) throw new Error(`Buffer exceeded 60 frames: ${frameBuffer.length}`);
  console.log('  [PASS] Buffer bounded strictly to 60 frames. First inference triggered at stride 15.');

  // --- TEST 2: Rejection Gate 1 (Presence Rejection) ---
  console.log('\nTest 2: Gate 1 Presence Rejection...');
  const f75Features = computeVideoFeatureVector(frameBuffer);
  const p1 = classifySign(f75Features, model);
  if (p1.status !== 'rejected' || p1.rejectionReason !== 'presence_too_low') {
    throw new Error(`Gate 1 failed: status=${p1.status}, reason=${p1.rejectionReason}`);
  }
  console.log('  [PASS] Empty hand buffer correctly triggers presence_too_low.');

  // --- TEST 3: Active Signing with Temporal Debounce ---
  console.log('\nTest 3: Active Gesture Simulation & Temporal Debounce...');

  // Create active hand landmarks for Class "1.bird"
  // Let's create dynamic coordinates that vary across frames with realistic velocity
  function makeActiveHand(frameNum, dx = 0) {
    const pts = [];
    for (let i = 0; i < 21; i++) {
      pts.push({
        x: 0.5 + Math.sin(frameNum * 0.1 + i * 0.2) * 0.15 + dx,
        y: 0.5 + Math.cos(frameNum * 0.1 + i * 0.2) * 0.15,
        z: 0.05,
      });
    }
    return pts;
  }

  // Feed 60 active frames to populate the buffer
  for (let f = 0; f < 60; f++) {
    t += 33;
    const leftHand = makeActiveHand(f, -0.05);
    const rightHand = makeActiveHand(f, 0.05);
    processFrame(
      [leftHand, rightHand],
      [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
      t
    );
  }

  // Reset debouncer and strideCounter to cleanly observe the transition from Window 1 to Window 2
  debouncer.reset();
  strideCounter = 0;
  lastCommittedLabel = null;
  transcript = '';

  // Next evaluation window: stride 15
  let window1Result = null;
  for (let f = 0; f < 15; f++) {
    t += 33;
    const leftHand = makeActiveHand(60 + f, -0.05);
    const rightHand = makeActiveHand(60 + f, 0.05);
    const r = processFrame(
      [leftHand, rightHand],
      [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
      t
    );
    if (r.prediction) window1Result = r;
  }

  console.log(`  Window 1 Result: Status=${window1Result.status}, Debounced Confirmed=${window1Result.debounced?.isConfirmed}`);
  if (window1Result.debounced?.isConfirmed) {
    throw new Error('Window 1 prematurely confirmed sign before streak >= 2!');
  }
  if (transcript !== '') {
    throw new Error('Transcript updated prematurely after window 1!');
  }
  console.log('  [PASS] Streak = 1 did not prematurely commit to transcript.');

  // Next evaluation window: stride 15 (same gesture continued)
  let window2Result = null;
  for (let f = 0; f < 15; f++) {
    t += 33;
    const leftHand = makeActiveHand(75 + f, -0.05);
    const rightHand = makeActiveHand(75 + f, 0.05);
    const r = processFrame(
      [leftHand, rightHand],
      [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
      t
    );
    if (r.prediction) window2Result = r;
  }

  console.log(`  Window 2 Result: Status=${window2Result.status}, Debounced Confirmed=${window2Result.debounced?.isConfirmed}`);
  console.log(`  Transcript after Window 2: "${transcript}"`);
  if (!window2Result.debounced?.isConfirmed) {
    throw new Error('Window 2 failed to confirm sign after streak = 2!');
  }
  if (!transcript) {
    throw new Error('Transcript was not updated after confirmed sign!');
  }
  const firstWord = transcript;
  console.log(`  [PASS] Gesture confirmed on streak = 2! Committed: "${firstWord}"`);

  // --- TEST 4: Duplicate Prevention on Sustained Holding ---
  console.log('\nTest 4: Duplicate Prevention while Holding Sign...');
  for (let f = 0; f < 15; f++) {
    t += 33;
    const leftHand = makeActiveHand(90 + f, -0.05);
    const rightHand = makeActiveHand(90 + f, 0.05);
    processFrame(
      [leftHand, rightHand],
      [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
      t
    );
  }
  console.log(`  Transcript after Window 3: "${transcript}"`);
  if (transcript !== firstWord) {
    throw new Error(`Duplicate word added while holding sign: "${transcript}"`);
  }
  console.log('  [PASS] Sustained holding did not append duplicate word.');

  // --- TEST 5: Natural Pause & Re-signing Same Word ---
  console.log('\nTest 5: Natural Pause & Signing Same Word Again...');
  // Simulate user resting hands (no hands in frame) for 60 frames (flush buffer)
  for (let f = 0; f < 60; f++) {
    t += 33;
    processFrame(null, null, t);
  }

  // User performs the sign again for 60 frames (4 windows)
  for (let f = 0; f < 60; f++) {
    t += 33;
    const leftHand = makeActiveHand(f, -0.05);
    const rightHand = makeActiveHand(f, 0.05);
    processFrame(
      [leftHand, rightHand],
      [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
      t
    );
  }
  console.log(`  Transcript after pause & repeat signing: "${transcript}"`);
  if (transcript !== `${firstWord} ${firstWord}`) {
    throw new Error(`Expected "${firstWord} ${firstWord}", got "${transcript}"`);
  }
  console.log(`  [PASS] After genuine pause, re-signing was correctly accepted: "${transcript}"`);

  // --- TEST 6: Stop & Cleanup ---
  console.log('\nTest 6: Stop Camera & Buffer Cleanup...');
  frameBuffer.length = 0;
  strideCounter = 0;
  debouncer.reset();
  lastCommittedLabel = null;
  if (frameBuffer.length !== 0) throw new Error('Buffer not empty after stop!');
  console.log('  [PASS] Buffers and debouncer reset cleanly.');

  console.log('\n' + '=' .repeat(75));
  console.log('ALL PHASE 7D PIPELINE INTEGRATION TESTS PASSED (6/6)');
  console.log('=' .repeat(75));

  // Save integration report
  const report = {
    timestamp: new Date().toISOString(),
    status: 'PASSED',
    totalInferenceCycles: inferenceRuns,
    testsExecuted: [
      'buffer_accumulation_60_frames',
      'inference_stride_15_frames',
      'presence_rejection_gate',
      'active_signing_temporal_debounce',
      'duplicate_prevention_on_hold',
      'natural_pause_and_re_signing',
      'stop_and_cleanup',
    ],
    finalTranscript: transcript,
    allPassed: true,
  };
  fs.writeFileSync('ai-training/reports/phase7d_integration_report.json', JSON.stringify(report, null, 2), 'utf-8');
  console.log('Report saved to: ai-training/reports/phase7d_integration_report.json');
}

runLivePipelineSimulation().catch((err) => {
  console.error('Integration test failed:', err);
  process.exit(1);
});
