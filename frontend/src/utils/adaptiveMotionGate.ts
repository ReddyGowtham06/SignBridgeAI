/**
 * SIGNBRIDGE AI — Isolated Adaptive Motion Gate Module (Development-Only)
 * =======================================================================
 * Phase 9A.2A: Evaluates dual-path gesture rejection gating:
 * - Path A (Dynamic Path): standard motion threshold (> 0.05).
 * - Path B (Static Candidate Path): enables intentional static signs
 *   (21.good, 34.stop, 45.hand) under low motion, gated by:
 *     1. Active hand presence (>= 0.50)
 *     2. Pose stability / low spatial landmark standard deviation (<= 0.055)
 *     3. Signing space centroid (y <= 0.75, anti-idle filter)
 *     4. Static vocabulary membership (only 21.good, 34.stop, 45.hand)
 *     5. High confidence (>= 0.50) & margin (>= 0.10)
 *
 * STRICT PRESERVATION:
 * - Does NOT replace `signClassifier.ts` or production models.
 * - Records both original M1 and adaptive gate decisions for paired analysis.
 */

import { TOTAL_FEATURE_DIM } from './featureBuilder.ts';
import {
  NUM_CLASSES,
  DEFAULT_PRESENCE_THRESHOLD,
  DEFAULT_MOTION_THRESHOLD,
  DEFAULT_CONFIDENCE_THRESHOLD,
  type LoadedModel,
} from './signClassifier.ts';

export const STATIC_VOCABULARY = ['21.good', '34.stop', '45.hand'];

export const STATIC_MIN_PRESENCE = 0.50;
export const STATIC_MAX_POSE_STD = 0.055;
export const STATIC_MIN_CONFIDENCE = 0.50;
export const STATIC_MIN_MARGIN = 0.10;
export const STATIC_MAX_Y_CENTROID = 0.75;

export interface AdaptiveGateResult {
  metrics: {
    presenceLeft: number;
    presenceRight: number;
    motionEnergy: number;
    spatialStd: number;
    yCentroid: number;
    top1Label: string;
    top1Confidence: number;
    margin: number;
  };
  originalM1: {
    status: 'accepted' | 'rejected';
    predictedLabel: string | null;
    rejectionReason: string | null;
  };
  adaptiveGate: {
    status: 'accepted' | 'rejected';
    predictedLabel: string | null;
    rejectionReason: string | null;
    pathTaken: 'dynamic_path' | 'static_path' | 'none';
  };
}

export function evaluateAdaptiveMotionGate(
  features: Float32Array | number[],
  model: LoadedModel
): AdaptiveGateResult {
  const pLeft = features[0];
  const pRight = features[379];

  // Mean Velocity
  let velSumLeft = 0;
  for (let i = 253; i < 316; i++) velSumLeft += features[i];
  const velMeanLeft = velSumLeft / 63;

  let velSumRight = 0;
  for (let i = 632; i < 695; i++) velSumRight += features[i];
  const velMeanRight = velSumRight / 63;

  const motionEnergy = Math.max(
    pLeft > 0 ? velMeanLeft : 0,
    pRight > 0 ? velMeanRight : 0
  );

  // Spatial Standard Deviation (ddof=0)
  let stdSumLeft = 0;
  for (let i = 64; i < 127; i++) stdSumLeft += features[i];
  const stdMeanLeft = stdSumLeft / 63;

  let stdSumRight = 0;
  for (let i = 443; i < 506; i++) stdSumRight += features[i];
  const stdMeanRight = stdSumRight / 63;

  const spatialStd = Math.max(
    pLeft > 0 ? stdMeanLeft : 0,
    pRight > 0 ? stdMeanRight : 0
  );

  // Active Hand Vertical Centroid (y coordinate)
  let ySumLeft = 0;
  for (let i = 2; i < 64; i += 3) ySumLeft += features[i];
  const yMeanLeft = pLeft > 0 ? ySumLeft / 21 : 1.0;

  let ySumRight = 0;
  for (let i = 381; i < 443; i += 3) ySumRight += features[i];
  const yMeanRight = pRight > 0 ? ySumRight / 21 : 1.0;

  const yCentroid = Math.min(
    pLeft > 0 ? yMeanLeft : 1.0,
    pRight > 0 ? yMeanRight : 1.0
  );

  // M1 Forward Pass
  const standardized = new Float32Array(TOTAL_FEATURE_DIM);
  for (let i = 0; i < TOTAL_FEATURE_DIM; i++) {
    standardized[i] = (features[i] - model.scalerMean[i]) / model.scalerScale[i];
  }

  const logits = new Float32Array(NUM_CLASSES);
  let maxLogit = -Infinity;
  for (let c = 0; c < NUM_CLASSES; c++) {
    let dot = model.b[c];
    const wOffset = c * TOTAL_FEATURE_DIM;
    for (let i = 0; i < TOTAL_FEATURE_DIM; i++) {
      dot += standardized[i] * model.WTransposed[wOffset + i];
    }
    logits[c] = dot;
    if (dot > maxLogit) maxLogit = dot;
  }

  const probs = new Float32Array(NUM_CLASSES);
  let sumExp = 0;
  for (let c = 0; c < NUM_CLASSES; c++) {
    const exp = Math.exp(logits[c] - maxLogit);
    probs[c] = exp;
    sumExp += exp;
  }
  for (let c = 0; c < NUM_CLASSES; c++) {
    probs[c] /= sumExp;
  }

  // Find Top-1 & Top-2
  let top1Idx = 0;
  let top1Score = -1;
  let top2Score = -1;

  for (let c = 0; c < NUM_CLASSES; c++) {
    const s = probs[c];
    if (s > top1Score) {
      top2Score = top1Score;
      top1Score = s;
      top1Idx = c;
    } else if (s > top2Score) {
      top2Score = s;
    }
  }

  const top1Label = model.labels[top1Idx];
  const margin = top1Score - top2Score;

  // --- ORIGINAL M1 GATE ---
  let origStatus: 'accepted' | 'rejected' = 'accepted';
  let origReason: string | null = null;
  let origPred: string | null = top1Label;

  if (pLeft < DEFAULT_PRESENCE_THRESHOLD && pRight < DEFAULT_PRESENCE_THRESHOLD) {
    origStatus = 'rejected';
    origReason = 'presence_too_low';
    origPred = null;
  } else if (motionEnergy < DEFAULT_MOTION_THRESHOLD) {
    origStatus = 'rejected';
    origReason = 'motion_stationary';
    origPred = null;
  } else if (top1Score < DEFAULT_CONFIDENCE_THRESHOLD) {
    origStatus = 'rejected';
    origReason = 'confidence_low';
    origPred = null;
  }

  // --- ADAPTIVE MOTION GATE ---
  let adaptStatus: 'accepted' | 'rejected' = 'accepted';
  let adaptReason: string | null = null;
  let adaptPred: string | null = top1Label;
  let adaptPath: 'dynamic_path' | 'static_path' | 'none' = 'none';

  if (pLeft < DEFAULT_PRESENCE_THRESHOLD && pRight < DEFAULT_PRESENCE_THRESHOLD) {
    adaptStatus = 'rejected';
    adaptReason = 'presence_too_low';
    adaptPred = null;
    adaptPath = 'none';
  } else if (motionEnergy >= DEFAULT_MOTION_THRESHOLD) {
    adaptPath = 'dynamic_path';
    if (top1Score < DEFAULT_CONFIDENCE_THRESHOLD) {
      adaptStatus = 'rejected';
      adaptReason = 'confidence_low';
      adaptPred = null;
    }
  } else {
    // Static Candidate Path (Path B)
    adaptPath = 'static_path';
    const activePresence = Math.max(pLeft, pRight);

    if (activePresence < STATIC_MIN_PRESENCE) {
      adaptStatus = 'rejected';
      adaptReason = 'static_presence_low';
      adaptPred = null;
    } else if (spatialStd > STATIC_MAX_POSE_STD) {
      adaptStatus = 'rejected';
      adaptReason = 'static_pose_jitter';
      adaptPred = null;
    } else if (yCentroid > STATIC_MAX_Y_CENTROID) {
      adaptStatus = 'rejected';
      adaptReason = 'idle_hand_resting';
      adaptPred = null;
    } else if (!STATIC_VOCABULARY.includes(top1Label)) {
      adaptStatus = 'rejected';
      adaptReason = 'motion_stationary';
      adaptPred = null;
    } else if (top1Score < STATIC_MIN_CONFIDENCE || margin < STATIC_MIN_MARGIN) {
      adaptStatus = 'rejected';
      adaptReason = 'static_confidence_ambiguous';
      adaptPred = null;
    }
  }

  return {
    metrics: {
      presenceLeft: pLeft,
      presenceRight: pRight,
      motionEnergy,
      spatialStd,
      yCentroid,
      top1Label,
      top1Confidence: top1Score,
      margin,
    },
    originalM1: {
      status: origStatus,
      predictedLabel: origPred,
      rejectionReason: origReason,
    },
    adaptiveGate: {
      status: adaptStatus,
      predictedLabel: adaptPred,
      rejectionReason: adaptReason,
      pathTaken: adaptPath,
    },
  };
}
