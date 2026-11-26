/**
 * SIGNBRIDGE AI - Inference API Service
 */
import { apiClient } from './client';
import type { InferenceRequestPayload, InferenceResponsePayload } from './types';

export const inferenceApi = {
  async classify(payload: InferenceRequestPayload): Promise<InferenceResponsePayload> {
    return apiClient.request<InferenceResponsePayload>('inference/classify', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async getModelInfo(): Promise<Record<string, unknown>> {
    return apiClient.request<Record<string, unknown>>('inference/model-info', {
      method: 'GET',
    });
  },
};
