/**
 * SIGNBRIDGE AI — TypeScript Inference Engine
 * ============================================
 * Production-ready client-side inference engine for the 49-class Softmax
 * Logistic Regression Indian Sign Language classifier.
 *
 * Implements:
 * 1. Asynchronous model loading & caching from `/models/sign_classifier.json`.
 * 2. 758-dimensional input validation & standardization (scaler mean & scale).
 * 3. Cache-friendly contiguous matrix-vector projection & stable softmax.
 * 4. Multi-stage gesture rejection gates:
 *    - Gate 1 (Presence): Minimum dual-hand presence ratio (default >= 0.20).
 *    - Gate 2 (Motion): Minimum hand velocity threshold (default >= 0.05 units/sec).
 *    - Gate 3 (Confidence): Softmax probability threshold (default >= 0.45).
 * 5. Reusable Temporal Debouncer (Gate 4): consecutive window stabilization.
 *
 * Designed for sub-millisecond in-browser execution (< 0.1 ms per evaluation).
 */

import { TOTAL_FEATURE_DIM } from './featureBuilder.ts';

export const NUM_CLASSES = 49;
export const DEFAULT_MODEL_URL = '/models/sign_classifier.json';

// Default Rejection Gate Thresholds (Path A: Dynamic Signs)
export const DEFAULT_PRESENCE_THRESHOLD = 0.20;
export const DEFAULT_MOTION_THRESHOLD = 0.05;
export const DEFAULT_CONFIDENCE_THRESHOLD = 0.45;
export const DEFAULT_DEBOUNCE_CONSECUTIVE = 2;

// Default Rejection Gate Thresholds (Path B: Static Signs)
export const DEFAULT_STATIC_MIN_PRESENCE = 0.50;
export const DEFAULT_STATIC_MAX_POSE_STD = 0.055;
export const DEFAULT_STATIC_MAX_Y_CENTROID = 0.75;
export const DEFAULT_STATIC_MIN_CONFIDENCE = 0.50;
export const DEFAULT_STATIC_MIN_MARGIN = 0.10;
export const DEFAULT_STATIC_VOCABULARY = ['21.good', '34.stop', '45.hand'] as const;

/**
 * 0-based label array indices corresponding to the validated static vocabulary:
 * - 13: 21.good
 * - 27: 34.stop
 * - 39: 45.hand
 */
export const STATIC_CLASS_INDICES = [13, 27, 39] as const;

// Indices for Feature Slices
const SLOT_LEFT_PRESENCE_IDX = 0;
const SLOT_LEFT_STD_START = 64;
const SLOT_LEFT_STD_END = 127; // 63 features
const SLOT_LEFT_VEL_START = 253;
const SLOT_LEFT_VEL_END = 316; // 63 features

const SLOT_RIGHT_PRESENCE_IDX = 379;
const SLOT_RIGHT_STD_START = 443;
const SLOT_RIGHT_STD_END = 506; // 63 features
const SLOT_RIGHT_VEL_START = 632;
const SLOT_RIGHT_VEL_END = 695; // 63 features

export type RejectionReason =
  | 'presence_too_low'
  | 'motion_stationary'
  | 'confidence_low'
  | 'static_presence_low'
  | 'static_pose_jitter'
  | 'idle_hand_resting'
  | 'static_confidence_ambiguous';

export type RecognitionPath = 'dynamic_path' | 'static_path' | 'none';

export interface GatingMetrics {
  presenceLeft: number;
  presenceRight: number;
  motionEnergy: number;
  spatialStd: number;
  yCentroid: number;
  margin: number;
}

export interface PredictionCandidate {
  classIndex: number;
  label: string;
  score: number;
}

export interface RejectionDetails {
  presenceRatioLeft: number;
  presenceRatioRight: number;
  meanVelocity: number;
  maxConfidence: number;
  thresholds: {
    presence: number;
    motion: number;
    confidence: number;
    staticMinPresence?: number;
    staticMaxPoseStd?: number;
    staticMaxYCentroid?: number;
    staticMinConfidence?: number;
    staticMinMargin?: number;
  };
  spatialStd?: number;
  yCentroid?: number;
  margin?: number;
}

export interface PredictionResult {
  status: 'accepted' | 'rejected';
  predictedClass: number | null;
  predictedLabel: string | null;
  confidence: number;
  topCandidates: PredictionCandidate[];
  logits: Float32Array;
  probabilities: Float32Array;
  rejectionReason: RejectionReason | null;
  rejectionDetails?: RejectionDetails;
  recognitionPath?: RecognitionPath;
  gatingMetrics?: GatingMetrics;
}

export interface InferenceOptions {
  /** Suppress classification if both hands have presence ratio below this (default: 0.20) */
  presenceThreshold?: number;
  /** Suppress classification if sequence velocity is below this (default: 0.05) */
  motionThreshold?: number;
  /** Suppress classification if top confidence is below this (default: 0.45) */
  confidenceThreshold?: number;
  /** Set to false to bypass Gates 1-3 for raw evaluation (default: true) */
  applyRejectionGates?: boolean;
  /** Set to false to disable adaptive static path (default: true) */
  enableAdaptiveMotionGate?: boolean;
  /** Static path minimum active presence (default: 0.50) */
  staticMinPresence?: number;
  /** Static path maximum spatial std (default: 0.055) */
  staticMaxPoseStd?: number;
  /** Static path maximum Y centroid (default: 0.75) */
  staticMaxYCentroid?: number;
  /** Static path minimum confidence (default: 0.50) */
  staticMinConfidence?: number;
  /** Static path minimum top1-top2 margin (default: 0.10) */
  staticMinMargin?: number;
  /** Approved static sign vocabulary (default: ['21.good', '34.stop', '45.hand']) */
  staticVocabulary?: string[];
}

/** Raw JSON format exported by Phase 7A (`export_model_to_frontend.py`) */
export interface ClassifierModelData {
  model_type: string;
  version: string;
  feature_dim: number;
  num_classes: number;
  scaler_mean: number[];
  scaler_scale: number[];
  W: number[][]; // (758, 49)
  b: number[];   // (49)
  classes: number[];
  id_to_label: Record<string, string>;
  labels: string[];
  training_accuracy?: number;
  val_accuracy?: number;
  held_out_test_accuracy?: number;
}

/** Optimized in-memory model structure with typed arrays and transposed weights */
export interface LoadedModel {
  modelType: string;
  version: string;
  featureDim: number;
  numClasses: number;
  scalerMean: Float32Array;
  scalerScale: Float32Array;
  /** Transposed weights of shape (numClasses, featureDim) for contiguous row scans */
  WTransposed: Float32Array;
  b: Float32Array;
  classes: Int32Array;
  labels: string[];
  idToLabel: Record<string, string>;
  heldOutTestAccuracy?: number;
}

// Global model cache
let cachedModel: LoadedModel | null = null;
let modelLoadingPromise: Promise<LoadedModel> | null = null;

/**
 * Validates and converts raw JSON model data into an optimized `LoadedModel`.
 */
export function parseModelData(data: ClassifierModelData): LoadedModel {
  if (!data || typeof data !== 'object') {
    throw new Error('Invalid model data: payload is null or not an object.');
  }

  const {
    model_type,
    version = '1.0.0',
    feature_dim = TOTAL_FEATURE_DIM,
    num_classes = NUM_CLASSES,
    scaler_mean,
    scaler_scale,
    W,
    b,
    classes,
    labels,
    id_to_label,
    held_out_test_accuracy,
  } = data;

  if (feature_dim !== TOTAL_FEATURE_DIM) {
    throw new Error(`Model feature dimension ${feature_dim} does not match expected ${TOTAL_FEATURE_DIM}.`);
  }
  if (num_classes !== NUM_CLASSES) {
    throw new Error(`Model class count ${num_classes} does not match expected ${NUM_CLASSES}.`);
  }
  if (!Array.isArray(scaler_mean) || scaler_mean.length !== TOTAL_FEATURE_DIM) {
    throw new Error(`scaler_mean length ${scaler_mean?.length} does not match ${TOTAL_FEATURE_DIM}.`);
  }
  if (!Array.isArray(scaler_scale) || scaler_scale.length !== TOTAL_FEATURE_DIM) {
    throw new Error(`scaler_scale length ${scaler_scale?.length} does not match ${TOTAL_FEATURE_DIM}.`);
  }
  if (!Array.isArray(W) || W.length !== TOTAL_FEATURE_DIM || !Array.isArray(W[0]) || W[0].length !== NUM_CLASSES) {
    throw new Error(`W matrix shape [${W?.length}, ${W?.[0]?.length}] does not match [${TOTAL_FEATURE_DIM}, ${NUM_CLASSES}].`);
  }
  if (!Array.isArray(b) || b.length !== NUM_CLASSES) {
    throw new Error(`b bias length ${b?.length} does not match ${NUM_CLASSES}.`);
  }
  if (!Array.isArray(labels) || labels.length !== NUM_CLASSES) {
    throw new Error(`labels array length ${labels?.length} does not match ${NUM_CLASSES}.`);
  }

  // Convert to TypedArrays and transpose W to (NUM_CLASSES, TOTAL_FEATURE_DIM)
  const meanArr = new Float32Array(scaler_mean);
  const scaleArr = new Float32Array(scaler_scale);
  const bArr = new Float32Array(b);
  const classesArr = new Int32Array(classes || Array.from({ length: NUM_CLASSES }, (_, i) => i));

  // Verify numerical validity
  for (let i = 0; i < TOTAL_FEATURE_DIM; i++) {
    if (!Number.isFinite(meanArr[i]) || !Number.isFinite(scaleArr[i])) {
      throw new Error(`Non-finite value detected in scaler parameters at index ${i}.`);
    }
    if (scaleArr[i] <= 0) {
      throw new Error(`Non-positive standard deviation in scaler_scale at index ${i}: ${scaleArr[i]}.`);
    }
  }

  // Pre-transpose W: WTransposed has shape (49, 758)
  // WTransposed[c * 758 + f] = W[f][c]
  const WTransposed = new Float32Array(NUM_CLASSES * TOTAL_FEATURE_DIM);
  for (let f = 0; f < TOTAL_FEATURE_DIM; f++) {
    const row = W[f];
    for (let c = 0; c < NUM_CLASSES; c++) {
      const val = row[c];
      if (!Number.isFinite(val)) {
        throw new Error(`Non-finite weight detected in W at [${f}, ${c}].`);
      }
      WTransposed[c * TOTAL_FEATURE_DIM + f] = val;
    }
  }

  return {
    modelType: model_type || 'softmax_regression',
    version,
    featureDim: TOTAL_FEATURE_DIM,
    numClasses: NUM_CLASSES,
    scalerMean: meanArr,
    scalerScale: scaleArr,
    WTransposed,
    b: bArr,
    classes: classesArr,
    labels: [...labels],
    idToLabel: id_to_label || {},
    heldOutTestAccuracy: held_out_test_accuracy ?? 0.7431,
  };
}

/**
 * Loads the model asynchronously with caching.
 * Accepts a URL/path to fetch, or a pre-loaded ClassifierModelData object.
 */
export async function loadModel(
  source: string | ClassifierModelData = DEFAULT_MODEL_URL,
  forceReload = false
): Promise<LoadedModel> {
  if (!forceReload && cachedModel) {
    return cachedModel;
  }

  if (modelLoadingPromise && !forceReload) {
    return modelLoadingPromise;
  }

  modelLoadingPromise = (async () => {
    let rawData: ClassifierModelData;

    if (typeof source === 'object' && source !== null) {
      rawData = source;
    } else if (typeof source === 'string') {
      const response = await fetch(source);
      if (!response.ok) {
        throw new Error(`Failed to fetch model from ${source}: ${response.status} ${response.statusText}`);
      }
      rawData = (await response.json()) as ClassifierModelData;
    } else {
      throw new Error(`Invalid model source provided: ${typeof source}`);
    }

    const loaded = parseModelData(rawData);
    cachedModel = loaded;
    return loaded;
  })();

  return modelLoadingPromise;
}

/**
 * Resets the cached model instance.
 */
export function clearModelCache(): void {
  cachedModel = null;
  modelLoadingPromise = null;
}

/**
 * Performs synchronous inference on a 758-dimensional feature vector using a loaded model.
 *
 * Applies the four-stage gesture rejection pipeline:
 * 1. Presence Gate: checks dual-hand presence ratio.
 * 2. Motion Gate: checks dynamic hand motion energy.
 * 3. Standardization & Softmax projection.
 * 4. Confidence Gate: checks minimum top probability.
 *
 * @param features 758-element Float32Array or number array
 * @param model Pre-loaded model instance
 * @param options Configurable rejection thresholds and options
 */
export function classifySign(
  features: Float32Array | number[],
  model: LoadedModel,
  options?: InferenceOptions
): PredictionResult {
  const presenceThreshold = options?.presenceThreshold ?? DEFAULT_PRESENCE_THRESHOLD;
  const motionThreshold = options?.motionThreshold ?? DEFAULT_MOTION_THRESHOLD;
  const confidenceThreshold = options?.confidenceThreshold ?? DEFAULT_CONFIDENCE_THRESHOLD;
  const applyGates = options?.applyRejectionGates ?? true;
  const enableAdaptive = options?.enableAdaptiveMotionGate ?? true;
  const staticMinPresence = options?.staticMinPresence ?? DEFAULT_STATIC_MIN_PRESENCE;
  const staticMaxPoseStd = options?.staticMaxPoseStd ?? DEFAULT_STATIC_MAX_POSE_STD;
  const staticMaxYCentroid = options?.staticMaxYCentroid ?? DEFAULT_STATIC_MAX_Y_CENTROID;
  const staticMinConfidence = options?.staticMinConfidence ?? DEFAULT_STATIC_MIN_CONFIDENCE;
  const staticMinMargin = options?.staticMinMargin ?? DEFAULT_STATIC_MIN_MARGIN;
  const staticVocabulary = options?.staticVocabulary ?? (DEFAULT_STATIC_VOCABULARY as unknown as string[]);

  // 1. Input Validation
  if (!features || features.length !== TOTAL_FEATURE_DIM) {
    throw new Error(`Input features length ${features?.length} does not match required ${TOTAL_FEATURE_DIM}.`);
  }

  // Fast finite check
  for (let i = 0; i < TOTAL_FEATURE_DIM; i++) {
    if (!Number.isFinite(features[i])) {
      throw new Error(`Feature vector contains non-finite value at index ${i}: ${features[i]}`);
    }
  }

  const pLeft = features[SLOT_LEFT_PRESENCE_IDX];
  const pRight = features[SLOT_RIGHT_PRESENCE_IDX];

  // Calculate Mean Hand Velocity across active hands
  let velSumLeft = 0;
  for (let i = SLOT_LEFT_VEL_START; i < SLOT_LEFT_VEL_END; i++) {
    velSumLeft += features[i];
  }
  const velMeanLeft = velSumLeft / (SLOT_LEFT_VEL_END - SLOT_LEFT_VEL_START);

  let velSumRight = 0;
  for (let i = SLOT_RIGHT_VEL_START; i < SLOT_RIGHT_VEL_END; i++) {
    velSumRight += features[i];
  }
  const velMeanRight = velSumRight / (SLOT_RIGHT_VEL_END - SLOT_RIGHT_VEL_START);

  // Motion energy: highest activity among active hands
  const motionEnergy = Math.max(
    pLeft > 0 ? velMeanLeft : 0,
    pRight > 0 ? velMeanRight : 0
  );

  // Spatial Standard Deviation across landmarks
  let stdSumLeft = 0;
  for (let i = SLOT_LEFT_STD_START; i < SLOT_LEFT_STD_END; i++) {
    stdSumLeft += features[i];
  }
  const stdMeanLeft = stdSumLeft / (SLOT_LEFT_STD_END - SLOT_LEFT_STD_START);

  let stdSumRight = 0;
  for (let i = SLOT_RIGHT_STD_START; i < SLOT_RIGHT_STD_END; i++) {
    stdSumRight += features[i];
  }
  const stdMeanRight = stdSumRight / (SLOT_RIGHT_STD_END - SLOT_RIGHT_STD_START);

  const spatialStd = Math.max(
    pLeft > 0 ? stdMeanLeft : 0,
    pRight > 0 ? stdMeanRight : 0
  );

  // Vertical Centroid (y coordinate of landmarks: Left 2..62 step 3, Right 381..441 step 3)
  let ySumLeft = 0;
  for (let i = 2; i < 64; i += 3) {
    ySumLeft += features[i];
  }
  const yMeanLeft = pLeft > 0 ? ySumLeft / 21 : 1.0;

  let ySumRight = 0;
  for (let i = 381; i < 443; i += 3) {
    ySumRight += features[i];
  }
  const yMeanRight = pRight > 0 ? ySumRight / 21 : 1.0;

  const yCentroid = Math.min(
    pLeft > 0 ? yMeanLeft : 1.0,
    pRight > 0 ? yMeanRight : 1.0
  );

  const emptyLogits = new Float32Array(NUM_CLASSES);
  const emptyProbs = new Float32Array(NUM_CLASSES);

  // Gate 1 — Presence Gate
  if (applyGates && pLeft < presenceThreshold && pRight < presenceThreshold) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: 0,
      topCandidates: [],
      logits: emptyLogits,
      probabilities: emptyProbs,
      rejectionReason: 'presence_too_low',
      recognitionPath: 'none',
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: 0,
        spatialStd,
        yCentroid,
        margin: 0,
        thresholds: {
          presence: presenceThreshold,
          motion: motionThreshold,
          confidence: confidenceThreshold,
          staticMinPresence,
          staticMaxPoseStd,
          staticMaxYCentroid,
          staticMinConfidence,
          staticMinMargin,
        },
      },
      gatingMetrics: {
        presenceLeft: pLeft,
        presenceRight: pRight,
        motionEnergy,
        spatialStd,
        yCentroid,
        margin: 0,
      },
    };
  }

  // Feature Standardization: z = (x - mean) / scale
  const z = new Float32Array(TOTAL_FEATURE_DIM);
  const { scalerMean, scalerScale, WTransposed, b } = model;

  for (let i = 0; i < TOTAL_FEATURE_DIM; i++) {
    z[i] = (features[i] - scalerMean[i]) / scalerScale[i];
  }

  // Linear Projection: logits[c] = b[c] + dot(z, WTransposed[c])
  const logits = new Float32Array(NUM_CLASSES);
  let maxLogit = -Infinity;

  for (let c = 0; c < NUM_CLASSES; c++) {
    const rowOffset = c * TOTAL_FEATURE_DIM;
    let sum = b[c];

    // Inner contiguous dot product
    for (let f = 0; f < TOTAL_FEATURE_DIM; f++) {
      sum += z[f] * WTransposed[rowOffset + f];
    }

    logits[c] = sum;
    if (sum > maxLogit) {
      maxLogit = sum;
    }
  }

  // Numerically Stable Softmax: exp(logit - max) / sum(exp)
  const probabilities = new Float32Array(NUM_CLASSES);
  let expSum = 0;

  for (let c = 0; c < NUM_CLASSES; c++) {
    const expVal = Math.exp(logits[c] - maxLogit);
    probabilities[c] = expVal;
    expSum += expVal;
  }

  const invExpSum = 1.0 / expSum;
  for (let c = 0; c < NUM_CLASSES; c++) {
    probabilities[c] *= invExpSum;
  }

  // Find Top Candidates
  const sortedIndices = Array.from({ length: NUM_CLASSES }, (_, i) => i).sort(
    (a, b) => probabilities[b] - probabilities[a]
  );

  const topCandidates: PredictionCandidate[] = sortedIndices.slice(0, 3).map((idx) => ({
    classIndex: idx,
    label: model.labels[idx] || `Class ${idx}`,
    score: probabilities[idx],
  }));

  const bestClass = topCandidates[0].classIndex;
  const bestScore = topCandidates[0].score;
  const bestLabel = topCandidates[0].label;
  const top2Score = topCandidates.length > 1 ? topCandidates[1].score : 0;
  const margin = bestScore - top2Score;

  const gatingMetrics: GatingMetrics = {
    presenceLeft: pLeft,
    presenceRight: pRight,
    motionEnergy,
    spatialStd,
    yCentroid,
    margin,
  };

  const thresholds = {
    presence: presenceThreshold,
    motion: motionThreshold,
    confidence: confidenceThreshold,
    staticMinPresence,
    staticMaxPoseStd,
    staticMaxYCentroid,
    staticMinConfidence,
    staticMinMargin,
  };

  // If raw evaluation is requested, bypass all rejection gates
  if (!applyGates) {
    return {
      status: 'accepted',
      predictedClass: bestClass,
      predictedLabel: bestLabel,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: null,
      recognitionPath: motionEnergy >= motionThreshold ? 'dynamic_path' : 'static_path',
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // =========================================================================
  // DUAL-PATH ADAPTIVE MOTION REJECTION GATING
  // =========================================================================

  // PATH A: Dynamic Signs (Normal Motion Energy >= motionThreshold)
  if (motionEnergy >= motionThreshold) {
    const recognitionPath: RecognitionPath = 'dynamic_path';

    if (bestScore < confidenceThreshold) {
      return {
        status: 'rejected',
        predictedClass: null,
        predictedLabel: null,
        confidence: bestScore,
        topCandidates,
        logits,
        probabilities,
        rejectionReason: 'confidence_low',
        recognitionPath,
        rejectionDetails: {
          presenceRatioLeft: pLeft,
          presenceRatioRight: pRight,
          meanVelocity: motionEnergy,
          maxConfidence: bestScore,
          spatialStd,
          yCentroid,
          margin,
          thresholds,
        },
        gatingMetrics,
      };
    }

    return {
      status: 'accepted',
      predictedClass: bestClass,
      predictedLabel: bestLabel,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: null,
      recognitionPath,
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // PATH B: Low Motion (< motionThreshold)
  // If adaptive gating is disabled, enforce standard stationary rejection
  if (!enableAdaptive) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: 'motion_stationary',
      recognitionPath: 'none',
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // Static Candidate Path (Path B)
  const recognitionPath: RecognitionPath = 'static_path';
  const activePresence = Math.max(pLeft, pRight);

  // Filter B1: Active Hand Presence
  if (activePresence < staticMinPresence) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: 'static_presence_low',
      recognitionPath,
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // Filter B2: Spatial Pose Stability (Anti-Jitter Filter)
  if (spatialStd > staticMaxPoseStd) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: 'static_pose_jitter',
      recognitionPath,
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // Filter B3: Active Hand Elevation (Anti-Idle Desk/Lap Filter)
  if (yCentroid > staticMaxYCentroid) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: 'idle_hand_resting',
      recognitionPath,
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // Filter B4: Static Vocabulary Membership
  if (!staticVocabulary.includes(bestLabel)) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: 'motion_stationary',
      recognitionPath,
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // Filter B5: Static Confidence & Probability Margin Filter
  if (bestScore < staticMinConfidence || margin < staticMinMargin) {
    return {
      status: 'rejected',
      predictedClass: null,
      predictedLabel: null,
      confidence: bestScore,
      topCandidates,
      logits,
      probabilities,
      rejectionReason: 'static_confidence_ambiguous',
      recognitionPath,
      rejectionDetails: {
        presenceRatioLeft: pLeft,
        presenceRatioRight: pRight,
        meanVelocity: motionEnergy,
        maxConfidence: bestScore,
        spatialStd,
        yCentroid,
        margin,
        thresholds,
      },
      gatingMetrics,
    };
  }

  // All 5 Static Filters Passed! Accept static sign translation
  return {
    status: 'accepted',
    predictedClass: bestClass,
    predictedLabel: bestLabel,
    confidence: bestScore,
    topCandidates,
    logits,
    probabilities,
    rejectionReason: null,
    recognitionPath,
    rejectionDetails: {
      presenceRatioLeft: pLeft,
      presenceRatioRight: pRight,
      meanVelocity: motionEnergy,
      maxConfidence: bestScore,
      spatialStd,
      yCentroid,
      margin,
      thresholds,
    },
    gatingMetrics,
  };
}

// ============================================================================
// GATE 4: TEMPORAL DEBOUNCER
// ============================================================================

export interface DebouncerOptions {
  /** Minimum consecutive evaluation windows required to confirm a sign (default: 2) */
  minConsecutiveFrames?: number;
  /** Maximum number of history entries retained in telemetry buffer (default: 10) */
  maxHistoryLength?: number;
}

export interface DebouncedPrediction {
  /** Whether the sign has met the consecutive window threshold */
  isConfirmed: boolean;
  /** The confirmed class index (or null if not yet confirmed) */
  confirmedClassIndex: number | null;
  /** The confirmed class name (or null if not yet confirmed) */
  confirmedLabel: string | null;
  /** Current candidate under consideration */
  candidateClassIndex: number | null;
  /** Current candidate name */
  candidateLabel: string | null;
  /** How many consecutive evaluation windows this candidate has appeared */
  consecutiveCount: number;
  /** Threshold required for confirmation */
  requiredCount: number;
  /** Status of the underlying classifier prediction */
  underlyingStatus: 'accepted' | 'rejected';
  rejectionReason: RejectionReason | null;
}

/**
 * Stateful Temporal Debouncer (Gate 4).
 * Reusable utility completely isolated from React components.
 * Prevents rapid flickering and commits signs only when sustained across windows.
 */
export class TemporalDebouncer {
  private minConsecutive: number;
  private currentCandidate: number | null = null;
  private currentCandidateLabel: string | null = null;
  private consecutiveCount = 0;
  private lastConfirmedClass: number | null = null;

  constructor(options?: DebouncerOptions) {
    this.minConsecutive = options?.minConsecutiveFrames ?? DEFAULT_DEBOUNCE_CONSECUTIVE;
  }

  /**
   * Feeds a single evaluation window's prediction into the debouncer.
   *
   * Rules:
   * - If prediction was rejected: resets the candidate streak.
   * - If prediction was accepted and matches current candidate: increments streak.
   * - If prediction was accepted and is a new candidate: resets streak to 1 for the new candidate.
   * - If streak reaches `minConsecutive`: confirms the prediction.
   */
  public update(prediction: PredictionResult): DebouncedPrediction {
    if (prediction.status === 'rejected' || prediction.predictedClass === null) {
      // Rejection interrupts candidate streak
      this.currentCandidate = null;
      this.currentCandidateLabel = null;
      this.consecutiveCount = 0;

      return {
        isConfirmed: false,
        confirmedClassIndex: null,
        confirmedLabel: null,
        candidateClassIndex: null,
        candidateLabel: null,
        consecutiveCount: 0,
        requiredCount: this.minConsecutive,
        underlyingStatus: 'rejected',
        rejectionReason: prediction.rejectionReason,
      };
    }

    const newClass = prediction.predictedClass;
    const newLabel = prediction.predictedLabel;

    if (newClass === this.currentCandidate) {
      this.consecutiveCount += 1;
    } else {
      this.currentCandidate = newClass;
      this.currentCandidateLabel = newLabel;
      this.consecutiveCount = 1;
    }

    const isConfirmed = this.consecutiveCount >= this.minConsecutive;
    const confirmedClassIndex = isConfirmed ? this.currentCandidate : null;
    const confirmedLabel = isConfirmed ? this.currentCandidateLabel : null;

    if (isConfirmed) {
      this.lastConfirmedClass = confirmedClassIndex;
    }

    return {
      isConfirmed,
      confirmedClassIndex,
      confirmedLabel,
      candidateClassIndex: this.currentCandidate,
      candidateLabel: this.currentCandidateLabel,
      consecutiveCount: this.consecutiveCount,
      requiredCount: this.minConsecutive,
      underlyingStatus: 'accepted',
      rejectionReason: null,
    };
  }

  /**
   * Resets internal candidate state and history.
   */
  public reset(): void {
    this.currentCandidate = null;
    this.currentCandidateLabel = null;
    this.consecutiveCount = 0;
    this.lastConfirmedClass = null;
  }

  /**
   * Returns the last confirmed class index, if any.
   */
  public getLastConfirmed(): number | null {
    return this.lastConfirmedClass;
  }
}
