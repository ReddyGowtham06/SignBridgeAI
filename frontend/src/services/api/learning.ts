/**
 * SIGNBRIDGE AI - Learning Studio API Service
 */
import { apiClient } from './client';
import type { LessonItem, ProgressItem, ProgressRecordPayload, ProgressSummary } from './types';

export const learningApi = {
  async getLessons(): Promise<LessonItem[]> {
    return apiClient.request<LessonItem[]>('learning/lessons', {
      method: 'GET',
    });
  },

  async getProgress(): Promise<ProgressSummary> {
    return apiClient.request<ProgressSummary>('learning/progress', {
      method: 'GET',
    });
  },

  async recordProgress(payload: ProgressRecordPayload): Promise<ProgressItem> {
    return apiClient.request<ProgressItem>('learning/progress', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },
};
