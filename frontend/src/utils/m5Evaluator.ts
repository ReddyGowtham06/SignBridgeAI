/**
 * SIGNBRIDGE AI — Isolated M5 MLP Inference Evaluator
 * ====================================================
 * Evaluator for the M5 small multi-layer perceptron (758 -> 128 -> 49)
 * with ReLU activations.
 *
 * Implements isolated forward pass for paired benchmarking against M1
 * without modifying production M1 inference.
 */

import { TOTAL_FEATURE_DIM } from './featureBuilder.ts';
import {
  NUM_CLASSES,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
  type PredictionCandidate,
  type PredictionResult,
  type InferenceOptions,
} from './signClassifier.ts';

export const M5_MODEL_URL = '/models/m5_evaluator.json';
export const M5_HIDDEN_DIM = 128;

export interface M5ModelData {
  model_type: string;
  version: string;
  feature_dim: number;
  hidden_dim: number;
  num_classes: number;
  classes: number[];
  labels: string[];
  id_to_label: Record<string, string>;
  scaler_mean: number[];
  scaler_scale: number[];
  W1: number[][]; // (758, 128)
  b1: number[];   // (128)
  W2: number[][]; // (128, 49)
  b2: number[];   // (49)
}

export interface LoadedM5Model {
  scalerMean: Float32Array;
  scalerScale: Float32Array;
  W1: Float32Array; // Flattened column-major or row-major: (758 * 128)
  b1: Float32Array; // (128)
  W2: Float32Array; // Flattened: (128 * 49)
  b2: Float32Array; // (49)
  classes: Int32Array;
  labels: string[];
  idToLabel: Record<string, string>;
}

let cachedM5Model: LoadedM5Model | null = null;

export async function loadM5Model(url: string = M5_MODEL_URL): Promise<LoadedM5Model> {
  if (cachedM5Model) return cachedM5Model;

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load M5 model from ${url}: HTTP ${res.status}`);
  }

  const raw: M5ModelData = await res.json();

  if (raw.feature_dim !== TOTAL_FEATURE_DIM) {
    throw new Error(`M5 feature dimension mismatch: expected ${TOTAL_FEATURE_DIM}, got ${raw.feature_dim}`);
  }
  if (raw.num_classes !== NUM_CLASSES) {
    throw new Error(`M5 num_classes mismatch: expected ${NUM_CLASSES}, got ${raw.num_classes}`);
  }

  const scalerMean = new Float32Array(raw.scaler_mean);
  const scalerScale = new Float32Array(raw.scaler_scale);

  // Flatten W1 (758 x 128)
  const W1 = new Float32Array(raw.feature_dim * raw.hidden_dim);
  for (let i = 0; i < raw.feature_dim; i++) {
    for (let j = 0; j < raw.hidden_dim; j++) {
      W1[i * raw.hidden_dim + j] = raw.W1[i][j];
    }
  }

  const b1 = new Float32Array(raw.b1);

  // Flatten W2 (128 x 49)
  const W2 = new Float32Array(raw.hidden_dim * raw.num_classes);
  for (let j = 0; j < raw.hidden_dim; j++) {
    for (let k = 0; k < raw.num_classes; k++) {
      W2[j * raw.num_classes + k] = raw.W2[j][k];
    }
  }

  const b2 = new Float32Array(raw.b2);
  const classes = new Int32Array(raw.classes);

  cachedM5Model = {
    scalerMean,
    scalerScale,
    W1,
    b1,
    W2,
    b2,
    classes,
    labels: raw.labels,
    idToLabel: raw.id_to_label,
  };

  return cachedM5Model;
}

/**
 * Perform M5 forward pass:
 * Input (758) -> Scale -> Dense1 (128) -> ReLU -> Dense2 (49) -> Softmax
 */
export function classifySignM5(
  features: Float32Array,
  model: LoadedM5Model,
  options: InferenceOptions = {}
): PredictionResult {
  const {
    presenceThreshold = DEFAULT_PRESENCE_THRESHOLD,
    motionThreshold = DEFAULT_MOTION_THRESHOLD,
    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
    applyRejectionGates = true,
  } = options;

  if (features.length !== TOTAL_FEATURE_DIM) {
    throw new Error(`Input feature length mismatch: expected ${TOTAL_FEATURE_DIM}, got ${features.length}`);
  }

  const D = TOTAL_FEATURE_DIM;
  const H = M5_HIDDEN_DIM;
  const K = NUM_CLASSES;

  // 1. Standardize features: x_norm = (x - mean) / scale
  const xNorm = new Float32Array(D);
  for (let i = 0; i < D; i++) {
    xNorm[i] = (features[i] - model.scalerMean[i]) / model.scalerScale[i];
  }

  // 2. Hidden Layer 1: h1 = ReLU(xNorm * W1 + b1)
  const h1 = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    let sum = model.b1[j];
    for (let i = 0; i < D; i++) {
      sum += xNorm[i] * model.W1[i * H + j];
    }
    h1[j] = sum > 0 ? sum : 0; // ReLU
  }

  // 3. Output Layer 2: logits = h1 * W2 + b2
  const logits = new Float32Array(K);
  let maxLogit = -Infinity;
  for (let k = 0; k < K; k++) {
    let sum = model.b2[k];
    for (let j = 0; j < H; j++) {
      sum += h1[j] * model.W2[j * K + k];
    }
    logits[k] = sum;
    if (sum > maxLogit) {
      maxLogit = sum;
    }
  }

  // 4. Stable Softmax
  const probabilities = new Float32Array(K);
  let sumExp = 0;
  for (let k = 0; k < K; k++) {
    const e = Math.exp(logits[k] - maxLogit);
    probabilities[k] = e;
    sumExp += e;
  }
  for (let k = 0; k < K; k++) {
    probabilities[k] /= sumExp;
  }

  // 5. Rank Top Candidates
  const indexed: PredictionCandidate[] = [];
  for (let k = 0; k < K; k++) {
    indexed.push({
      classIndex: k,
      label: model.labels[k],
      score: probabilities[k],
    });
  }
  indexed.sort((a, b) => b.score - a.score);

  const top1 = indexed[0];
  const maxConfidence = top1.score;

  // 6. Optional Rejection Gates (compatible with signClassifier)
  if (applyRejectionGates) {
    const leftPres = features[0];
    const rightPres = features[379];
    if (leftPres < presenceThreshold && rightPres < presenceThreshold) {
      return {
        status: 'rejected',
        predictedClass: null,
        predictedLabel: null,
        confidence: maxConfidence,
        topCandidates: indexed.slice(0, 5),
        logits,
        probabilities,
        rejectionReason: 'presence_too_low',
      };
    }

    let velSum = 0;
    let velCount = 0;
    if (leftPres >= presenceThreshold) {
      for (let i = 253; i < 316; i++) {
        velSum += Math.abs(features[i]);
        velCount++;
      }
    }
    if (rightPres >= presenceThreshold) {
      for (let i = 632; i < 695; i++) {
        velSum += Math.abs(features[i]);
        velCount++;
      }
    }
    const meanVel = velCount > 0 ? velSum / velCount : 0;
    if (meanVel < motionThreshold) {
      return {
        status: 'rejected',
        predictedClass: null,
        predictedLabel: null,
        confidence: maxConfidence,
        topCandidates: indexed.slice(0, 5),
        logits,
        probabilities,
        rejectionReason: 'motion_stationary',
      };
    }

    if (maxConfidence < confidenceThreshold) {
      return {
        status: 'rejected',
        predictedClass: null,
        predictedLabel: null,
        confidence: maxConfidence,
        topCandidates: indexed.slice(0, 5),
        logits,
        probabilities,
        rejectionReason: 'confidence_low',
      };
    }
  }

  return {
    status: 'accepted',
    predictedClass: top1.classIndex,
    predictedLabel: top1.label,
    confidence: maxConfidence,
    topCandidates: indexed.slice(0, 5),
    logits,
    probabilities,
    rejectionReason: null,
  };
}
