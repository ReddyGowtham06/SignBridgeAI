/**
 * SIGNBRIDGE AI — Phase 8B.7 Live Pipeline & Classifier Latency Benchmarking
 * =========================================================================
 * Rigorously measures computational latency and throughput for:
 * 1. Classifier Model Loading & Memory Allocation
 * 2. 758-Dimensional Feature Extraction (computeVideoFeatureVector on 60-frame buffer)
 * 3. Matrix-Vector Softmax Classification (classifySign on 758 features)
 * 4. Temporal Debouncer & Rejection Gate Processing
 * 5. End-to-End Computational Cadence (excluding camera sensor capture)
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
import {
  computeVideoFeatureVector,
  slotMediaPipeResult,
  TOTAL_FEATURE_DIM,
} from '../frontend/src/utils/featureBuilder.ts';

const MODEL_PATH = 'frontend/public/models/sign_classifier.json';
const OUTPUT_METRICS_PATH = 'ai-training/reports/phase8b7/phase8b7_latency_metrics.json';

// Utility for percentile calculation
function calculatePercentiles(sortedValues) {
  const n = sortedValues.length;
  const getP = (p) => {
    const idx = (p / 100) * (n - 1);
    const low = Math.floor(idx);
    const high = Math.ceil(idx);
    const weight = idx - low;
    return sortedValues[low] * (1 - weight) + sortedValues[high] * weight;
  };

  return {
    min: sortedValues[0],
    p50: getP(50),
    median: getP(50),
    p90: getP(90),
    p95: getP(95),
    p99: getP(99),
    max: sortedValues[n - 1],
    mean: sortedValues.reduce((a, b) => a + b, 0) / n,
  };
}

async function runBenchmark() {
  console.log('='.repeat(76));
  console.log('SIGNBRIDGE AI — PHASE 8B.7 PIPELINE LATENCY & PERFORMANCE BENCHMARK');
  console.log('='.repeat(76));

  // 1. Model Load & Memory Allocation
  const rawModelJson = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf-8'));
  const tLoad0 = performance.now();
  const model = parseModelData(rawModelJson);
  const tLoadMs = performance.now() - tLoad0;

  const weightsByteSize = model.WTransposed.byteLength;
  const biasByteSize = model.b.byteLength;
  const scalerMeanByteSize = model.scalerMean.byteLength;
  const scalerScaleByteSize = model.scalerScale.byteLength;
  const totalTypedArrayMemoryBytes = weightsByteSize + biasByteSize + scalerMeanByteSize + scalerScaleByteSize;

  console.log(`[PASS] Model Loaded: ${model.modelType}`);
  console.log(`  - Classes: ${model.numClasses}, Feature Dims: ${model.featureDim}`);
  console.log(`  - Model Parse Latency: ${tLoadMs.toFixed(3)} ms`);
  console.log(`  - In-Memory Parameter Buffer Size: ${(totalTypedArrayMemoryBytes / 1024).toFixed(2)} KB\n`);

  // 2. Synthesize Realistic 60-Frame Rolling Buffer
  console.log('Generating 60-frame rolling buffers for feature extraction benchmark...');
  const syntheticBuffer = [];
  const syntheticTimestamps = [];
  let baseTime = 1000.0;

  for (let i = 0; i < 60; i++) {
    baseTime += 33.333; // 30 FPS
    syntheticTimestamps.push(baseTime);

    // Realistic landmark coordinates: 2 hands, 21 points each
    const leftPoints = [];
    const rightPoints = [];
    for (let pt = 0; pt < 21; pt++) {
      leftPoints.push({
        x: 0.35 + Math.sin(i * 0.1 + pt * 0.2) * 0.05,
        y: 0.55 + Math.cos(i * 0.1 + pt * 0.2) * 0.05,
        z: Math.sin(i * 0.05) * 0.02,
      });
      rightPoints.push({
        x: 0.65 + Math.cos(i * 0.1 + pt * 0.2) * 0.05,
        y: 0.55 + Math.sin(i * 0.1 + pt * 0.2) * 0.05,
        z: Math.cos(i * 0.05) * 0.02,
      });
    }

    const landmarks = [leftPoints, rightPoints];
    const handedness = [
      [{ categoryName: 'Left', displayName: 'Left', score: 0.95 }],
      [{ categoryName: 'Right', displayName: 'Right', score: 0.95 }],
    ];

    const slotted = slotMediaPipeResult(landmarks, handedness, baseTime);
    syntheticBuffer.push(slotted);
  }

  // 3. Benchmark Feature Builder (computeVideoFeatureVector)
  console.log('Benchmarking Feature Builder over 1,000 evaluations...');
  const featureLatencies = [];
  // Warmup
  for (let w = 0; w < 50; w++) {
    computeVideoFeatureVector(syntheticBuffer);
  }

  for (let iter = 0; iter < 1000; iter++) {
    const t0 = performance.now();
    const vec = computeVideoFeatureVector(syntheticBuffer);
    const t1 = performance.now();
    featureLatencies.push((t1 - t0) * 1000); // microseconds
  }

  featureLatencies.sort((a, b) => a - b);
  const featureStats = calculatePercentiles(featureLatencies);
  console.log(`  - Feature Builder Mean Latency   : ${(featureStats.mean / 1000).toFixed(4)} ms (${featureStats.mean.toFixed(1)} µs)`);
  console.log(`  - Feature Builder Median (P50)   : ${(featureStats.median / 1000).toFixed(4)} ms (${featureStats.median.toFixed(1)} µs)`);
  console.log(`  - Feature Builder P90 Latency    : ${(featureStats.p90 / 1000).toFixed(4)} ms (${featureStats.p90.toFixed(1)} µs)`);
  console.log(`  - Feature Builder P95 Latency    : ${(featureStats.p95 / 1000).toFixed(4)} ms (${featureStats.p95.toFixed(1)} µs)`);
  console.log(`  - Feature Builder Max Latency    : ${(featureStats.max / 1000).toFixed(4)} ms (${featureStats.max.toFixed(1)} µs)\n`);

  // 4. Benchmark Classifier Inference (classifySign)
  console.log('Benchmarking Classifier Inference over 5,000 evaluations...');
  const sampleVector = computeVideoFeatureVector(syntheticBuffer);
  const classifierLatencies = [];

  // Warmup
  for (let w = 0; w < 100; w++) {
    classifySign(sampleVector, model, { applyRejectionGates: true });
  }

  for (let iter = 0; iter < 5000; iter++) {
    const t0 = performance.now();
    classifySign(sampleVector, model, { applyRejectionGates: true });
    const t1 = performance.now();
    classifierLatencies.push((t1 - t0) * 1000); // microseconds
  }

  classifierLatencies.sort((a, b) => a - b);
  const classifierStats = calculatePercentiles(classifierLatencies);
  console.log(`  - Classifier Inference Mean      : ${(classifierStats.mean / 1000).toFixed(4)} ms (${classifierStats.mean.toFixed(1)} µs)`);
  console.log(`  - Classifier Inference Median    : ${(classifierStats.median / 1000).toFixed(4)} ms (${classifierStats.median.toFixed(1)} µs)`);
  console.log(`  - Classifier Inference P90       : ${(classifierStats.p90 / 1000).toFixed(4)} ms (${classifierStats.p90.toFixed(1)} µs)`);
  console.log(`  - Classifier Inference P95       : ${(classifierStats.p95 / 1000).toFixed(4)} ms (${classifierStats.p95.toFixed(1)} µs)`);
  console.log(`  - Classifier Throughput Capacity : ${Math.round(1000000 / classifierStats.mean)} evals/sec\n`);

  // 5. Benchmark Temporal Debouncer Latency
  console.log('Benchmarking Temporal Debouncer...');
  const debouncer = new TemporalDebouncer({ minConsecutiveFrames: 2 });
  const samplePred = classifySign(sampleVector, model, { applyRejectionGates: true });
  const debouncerLatencies = [];

  for (let iter = 0; iter < 5000; iter++) {
    const t0 = performance.now();
    debouncer.update(samplePred);
    const t1 = performance.now();
    debouncerLatencies.push((t1 - t0) * 1000);
  }

  debouncerLatencies.sort((a, b) => a - b);
  const debouncerStats = calculatePercentiles(debouncerLatencies);
  console.log(`  - Temporal Debouncer Mean        : ${debouncerStats.mean.toFixed(2)} µs`);
  console.log(`  - Temporal Debouncer Median      : ${debouncerStats.median.toFixed(2)} µs\n`);

  // 6. Compute Combined Pipeline Cadence & Latencies
  const combinedComputeMeanMs = (featureStats.mean + classifierStats.mean + debouncerStats.mean) / 1000;
  const combinedComputeP95Ms = (featureStats.p95 + classifierStats.p95 + debouncerStats.p95) / 1000;

  console.log('Combined Pipeline Computational Profile:');
  console.log(`  - Total Feature + Classifier + Debounce compute: ${combinedComputeMeanMs.toFixed(3)} ms (P95: ${combinedComputeP95Ms.toFixed(3)} ms)`);
  console.log(`  - Evaluation Interval at 30 FPS (every 15 frames): 500.0 ms`);
  console.log(`  - Evaluation Interval at 60 FPS (every 15 frames): 250.0 ms`);
  console.log(`  - Computational Load Margin : ${(combinedComputeMeanMs / 500.0 * 100).toFixed(2)}% of available frame budget\n`);

  // Write results to JSON
  const outputData = {
    benchmark_metadata: {
      timestamp: new Date().toISOString(),
      platform: process.platform,
      arch: process.arch,
      node_version: process.version,
      model_path: MODEL_PATH,
    },
    memory_allocation: {
      model_weights_transposed_bytes: weightsByteSize,
      model_bias_bytes: biasByteSize,
      scaler_mean_bytes: scalerMeanByteSize,
      scaler_scale_bytes: scalerScaleByteSize,
      total_parameter_bytes: totalTypedArrayMemoryBytes,
      total_parameter_kb: +(totalTypedArrayMemoryBytes / 1024).toFixed(2),
    },
    feature_builder_latency_us: featureStats,
    feature_builder_latency_ms: {
      mean: +(featureStats.mean / 1000).toFixed(4),
      median: +(featureStats.median / 1000).toFixed(4),
      p90: +(featureStats.p90 / 1000).toFixed(4),
      p95: +(featureStats.p95 / 1000).toFixed(4),
      max: +(featureStats.max / 1000).toFixed(4),
    },
    classifier_inference_latency_us: classifierStats,
    classifier_inference_latency_ms: {
      mean: +(classifierStats.mean / 1000).toFixed(4),
      median: +(classifierStats.median / 1000).toFixed(4),
      p90: +(classifierStats.p90 / 1000).toFixed(4),
      p95: +(classifierStats.p95 / 1000).toFixed(4),
      max: +(classifierStats.max / 1000).toFixed(4),
      throughput_ops_sec: Math.round(1000000 / classifierStats.mean),
    },
    debouncer_latency_us: debouncerStats,
    total_compute_latency_ms: {
      mean: +combinedComputeMeanMs.toFixed(4),
      p95: +combinedComputeP95Ms.toFixed(4),
    },
    cadence_timing_ms: {
      fps_standard: 30,
      buffer_window_duration_ms: 2000,
      stride_frames: 15,
      stride_cadence_ms: 500,
      debounce_required_streaks: 2,
      gesture_confirmation_cadence_ms: 1000,
      compute_margin_percent: +(combinedComputeMeanMs / 500.0 * 100).toFixed(2),
    },
  };

  fs.mkdirSync(path.dirname(OUTPUT_METRICS_PATH), { recursive: true });
  fs.writeFileSync(OUTPUT_METRICS_PATH, JSON.stringify(outputData, null, 2), 'utf-8');
  console.log(`Saved latency benchmark metrics to: ${OUTPUT_METRICS_PATH}`);
}

runBenchmark().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
