/**
 * SIGNBRIDGE AI — Test Parity Runner for Feature Builder
 * =======================================================
 * Standalone Node CLI script invoked by parity test harness.
 * Reads a JSON payload of slotted video frames, computes the 758-dim feature vector
 * using `computeVideoFeatureVector` from `frontend/src/utils/featureBuilder.ts`,
 * and writes the resulting float array as JSON.
 */

import * as fs from 'node:fs';
import { computeVideoFeatureVector } from '../frontend/src/utils/featureBuilder.ts';

function main() {
  const args = process.argv.slice(2);
  let inputJson;

  if (args.length > 0 && args[0] !== '-') {
    inputJson = fs.readFileSync(args[0], 'utf-8');
  } else {
    inputJson = fs.readFileSync(0, 'utf-8');
  }

  const data = JSON.parse(inputJson);
  const frames = data.frames;

  const features = computeVideoFeatureVector(frames);

  const output = {
    length: features.length,
    features: Array.from(features),
  };

  const outputJson = JSON.stringify(output);
  if (args.length > 1) {
    fs.writeFileSync(args[1], outputJson, 'utf-8');
  } else {
    process.stdout.write(outputJson);
  }
}

main();

