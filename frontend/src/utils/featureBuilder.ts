/**
 * SIGNBRIDGE AI — TypeScript Feature Builder Module
 * ==================================================
 * Replicates the Python training feature extraction pipeline (`dataset_builder.py`).
 * Transforms a chronological sequence of MediaPipe hand landmark observations
 * into an exact 758-dimensional feature vector for sign language classification.
 *
 * Feature Layout (758 total Float32 elements):
 * -------------------------------------------------------------
 * Hand Slot 0: Left Hand (Indices 0 - 378: 379 features)
 *   [0]       Presence ratio (valid_frames / total_frames)
 *   [1-63]    Spatial Mean (21 landmarks x 3 coords: x, y, z)
 *   [64-126]  Spatial Standard Deviation (population, ddof=0)
 *   [127-189] Spatial Minimum Bound
 *   [190-252] Spatial Maximum Bound
 *   [253-315] Mean Absolute Temporal Velocity (units / second)
 *   [316-378] Temporal Velocity Standard Deviation (ddof=0)
 *
 * Hand Slot 1: Right Hand (Indices 379 - 757: 379 features)
 *   [379]     Presence ratio
 *   [380-442] Spatial Mean
 *   [443-505] Spatial Standard Deviation
 *   [506-568] Spatial Minimum Bound
 *   [569-631] Spatial Maximum Bound
 *   [632-694] Mean Absolute Temporal Velocity
 *   [695-757] Temporal Velocity Standard Deviation
 */

export const NUM_HANDS = 2;
export const LANDMARKS_PER_HAND = 21;
export const COORDS_PER_LANDMARK = 3;
export const COORDS_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK; // 63
export const FEATURES_PER_HAND = 1 + COORDS_PER_HAND * 6; // 379
export const TOTAL_FEATURE_DIM = NUM_HANDS * FEATURES_PER_HAND; // 758

export const SLOT_LEFT = 0;
export const SLOT_RIGHT = 1;

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface HandObservation {
  landmarks: LandmarkPoint[];
  categoryName?: string;
  displayName?: string;
  score?: number;
}

export interface SlottedFrame {
  timestampMs: number;
  /** Slot 0 = Left Hand, Slot 1 = Right Hand */
  landmarks: [LandmarkPoint[] | null, LandmarkPoint[] | null];
}

export interface RawMediaPipeFrame {
  timestampMs: number;
  landmarks?: Array<Array<{ x: number; y: number; z: number }>> | null;
  handedness?: Array<Array<{ categoryName?: string; displayName?: string; score?: number }>> | null;
}

/**
 * Assigns raw MediaPipe hand detections into canonical slots:
 * Slot 0 = Left Hand, Slot 1 = Right Hand.
 * Replicates the conflict resolution logic from Python `landmark_extractor.py`.
 */
export function slotMediaPipeResult(
  landmarks: Array<Array<{ x: number; y: number; z: number }>> | undefined | null,
  handedness: Array<Array<{ categoryName?: string; displayName?: string; score?: number }>> | undefined | null,
  timestampMs: number
): SlottedFrame {
  const slotted: [LandmarkPoint[] | null, LandmarkPoint[] | null] = [null, null];
  const presence: [boolean, boolean] = [false, false];

  if (!landmarks || landmarks.length === 0) {
    return { timestampMs, landmarks: slotted };
  }

  for (let handIdx = 0; handIdx < landmarks.length; handIdx++) {
    const handPts = landmarks[handIdx];
    if (!handPts || handPts.length !== LANDMARKS_PER_HAND) continue;

    let label = 'Unknown';
    if (handedness && handIdx < handedness.length && handedness[handIdx] && handedness[handIdx].length > 0) {
      label = handedness[handIdx][0].categoryName || handedness[handIdx][0].displayName || 'Unknown';
    }

    // Canonical slot assignment: Left -> Slot 0, Right -> Slot 1
    let targetSlot = label.toLowerCase() === 'left' ? SLOT_LEFT : SLOT_RIGHT;

    // Conflict resolution: if slot already taken, use the alternative slot if available
    if (presence[targetSlot]) {
      const otherSlot = 1 - targetSlot;
      if (!presence[otherSlot]) {
        targetSlot = otherSlot;
      }
    }

    slotted[targetSlot] = handPts.map((p) => ({ x: p.x, y: p.y, z: p.z }));
    presence[targetSlot] = true;
  }

  return { timestampMs, landmarks: slotted };
}

/**
 * Computes the 758-dimensional feature vector from a sequence of slotted frames.
 * Exactly matches `compute_video_feature_vector` in Python `dataset_builder.py`.
 *
 * @param frames Array of chronologically ordered slotted frames.
 * @returns Float32Array of shape (758,)
 */
export function computeVideoFeatureVector(frames: SlottedFrame[]): Float32Array {
  const totalFrames = frames.length;
  const features = new Float32Array(TOTAL_FEATURE_DIM);

  if (totalFrames === 0) {
    return features;
  }

  // Process Hand Slot 0 (Left) and Hand Slot 1 (Right) independently
  for (let slotIdx = 0; slotIdx < NUM_HANDS; slotIdx++) {
    const slotOffset = slotIdx * FEATURES_PER_HAND;

    // 1. Identify active frame indices for this slot
    const activeIndices: number[] = [];
    const presenceMask = new Uint8Array(totalFrames);

    for (let i = 0; i < totalFrames; i++) {
      const lm = frames[i].landmarks[slotIdx];
      if (lm && lm.length === LANDMARKS_PER_HAND) {
        presenceMask[i] = 1;
        activeIndices.push(i);
      }
    }

    const validFramesCount = activeIndices.length;
    const presenceRatio = validFramesCount / totalFrames;
    features[slotOffset] = presenceRatio;

    // If hand was never detected, its 378 spatial & velocity stats remain 0.0
    if (validFramesCount === 0) {
      continue;
    }

    // 2. Flatten active coordinates: (K, 63)
    const K = validFramesCount;
    const activeCoords = new Float32Array(K * COORDS_PER_HAND);

    for (let k = 0; k < K; k++) {
      const frameIdx = activeIndices[k];
      const lm = frames[frameIdx].landmarks[slotIdx]!;
      const rowOffset = k * COORDS_PER_HAND;
      for (let p = 0; p < LANDMARKS_PER_HAND; p++) {
        activeCoords[rowOffset + p * 3 + 0] = lm[p].x;
        activeCoords[rowOffset + p * 3 + 1] = lm[p].y;
        activeCoords[rowOffset + p * 3 + 2] = lm[p].z;
      }
    }

    // 3. Compute Spatial Statistics: Mean, Std (ddof=0), Min, Max
    const meanVals = new Float32Array(COORDS_PER_HAND);
    const stdVals = new Float32Array(COORDS_PER_HAND);
    const minVals = new Float32Array(COORDS_PER_HAND);
    const maxVals = new Float32Array(COORDS_PER_HAND);

    for (let c = 0; c < COORDS_PER_HAND; c++) {
      let sum = 0;
      let min = Infinity;
      let max = -Infinity;

      for (let k = 0; k < K; k++) {
        const val = activeCoords[k * COORDS_PER_HAND + c];
        sum += val;
        if (val < min) min = val;
        if (val > max) max = val;
      }

      const mean = sum / K;
      meanVals[c] = mean;
      minVals[c] = min;
      maxVals[c] = max;

      let varSum = 0;
      for (let k = 0; k < K; k++) {
        const diff = activeCoords[k * COORDS_PER_HAND + c] - mean;
        varSum += diff * diff;
      }
      stdVals[c] = Math.sqrt(varSum / K);
    }

    // Write spatial features into slot slice
    // Mean: [1..63]
    for (let c = 0; c < COORDS_PER_HAND; c++) {
      features[slotOffset + 1 + c] = meanVals[c];
    }
    // Std: [64..126]
    for (let c = 0; c < COORDS_PER_HAND; c++) {
      features[slotOffset + 1 + 63 + c] = stdVals[c];
    }
    // Min: [127..189]
    for (let c = 0; c < COORDS_PER_HAND; c++) {
      features[slotOffset + 1 + 126 + c] = minVals[c];
    }
    // Max: [190..252]
    for (let c = 0; c < COORDS_PER_HAND; c++) {
      features[slotOffset + 1 + 189 + c] = maxVals[c];
    }

    // 4. Compute Temporal Velocity across strictly adjacent valid pairs
    // consecMask is True at index i if hand is detected at both frame i and frame i+1
    const consecPairs: number[] = [];
    for (let i = 0; i < totalFrames - 1; i++) {
      if (presenceMask[i] === 1 && presenceMask[i + 1] === 1) {
        consecPairs.push(i);
      }
    }

    const numConsecutive = consecPairs.length;
    const velMean = new Float32Array(COORDS_PER_HAND);
    const velStd = new Float32Array(COORDS_PER_HAND);

    if (numConsecutive > 0) {
      const M = numConsecutive;
      // Array to store calculated velocity deltas: shape (M, 63)
      const velDeltas = new Float32Array(M * COORDS_PER_HAND);

      for (let m = 0; m < M; m++) {
        const i = consecPairs[m];
        const tCurr = frames[i].timestampMs;
        const tNext = frames[i + 1].timestampMs;
        const dtRaw = tNext - tCurr;
        // Guard against non-positive intervals; clamp minimum dt to 1 ms (0.001 s)
        const dtSec = Math.max(dtRaw / 1000.0, 0.001);

        const lmCurr = frames[i].landmarks[slotIdx]!;
        const lmNext = frames[i + 1].landmarks[slotIdx]!;

        const mOffset = m * COORDS_PER_HAND;
        for (let p = 0; p < LANDMARKS_PER_HAND; p++) {
          const c0 = p * 3 + 0;
          const c1 = p * 3 + 1;
          const c2 = p * 3 + 2;

          velDeltas[mOffset + c0] = (lmNext[p].x - lmCurr[p].x) / dtSec;
          velDeltas[mOffset + c1] = (lmNext[p].y - lmCurr[p].y) / dtSec;
          velDeltas[mOffset + c2] = (lmNext[p].z - lmCurr[p].z) / dtSec;
        }
      }

      // vel_mean = mean(abs(vel_deltas), axis=0)
      // vel_std  = std(vel_deltas, axis=0)  (ddof=0 on signed velocity)
      for (let c = 0; c < COORDS_PER_HAND; c++) {
        let absSum = 0;
        let signedSum = 0;

        for (let m = 0; m < M; m++) {
          const v = velDeltas[m * COORDS_PER_HAND + c];
          absSum += Math.abs(v);
          signedSum += v;
        }

        const meanAbs = absSum / M;
        const meanSigned = signedSum / M;
        velMean[c] = meanAbs;

        let varSum = 0;
        for (let m = 0; m < M; m++) {
          const diff = velDeltas[m * COORDS_PER_HAND + c] - meanSigned;
          varSum += diff * diff;
        }
        velStd[c] = Math.sqrt(varSum / M);
      }
    }

    // Mean Abs Velocity: [253..315]
    for (let c = 0; c < COORDS_PER_HAND; c++) {
      features[slotOffset + 1 + 252 + c] = velMean[c];
    }
    // Velocity Std: [316..378]
    for (let c = 0; c < COORDS_PER_HAND; c++) {
      features[slotOffset + 1 + 315 + c] = velStd[c];
    }
  }

  return features;
}

/**
 * Convenience wrapper converting raw MediaPipe frame results directly into a 758-dim feature vector.
 */
export function computeVideoFeatureVectorFromRaw(rawFrames: RawMediaPipeFrame[]): Float32Array {
  const slotted = rawFrames.map((f) => slotMediaPipeResult(f.landmarks, f.handedness, f.timestampMs));
  return computeVideoFeatureVector(slotted);
}

