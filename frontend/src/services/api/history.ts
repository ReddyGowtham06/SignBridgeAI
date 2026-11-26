/**
 * SIGNBRIDGE AI - Translation History API Service
 */
import { apiClient } from './client';
import type { HistoryCreatePayload, HistoryListResponse, TranslationHistoryItem } from './types';

export const historyApi = {
  async list(skip = 0, limit = 50): Promise<HistoryListResponse> {
    return apiClient.request<HistoryListResponse>(`history?skip=${skip}&limit=${limit}`, {
      method: 'GET',
    });
  },

  async create(payload: HistoryCreatePayload): Promise<TranslationHistoryItem> {
    return apiClient.request<TranslationHistoryItem>('history', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  async delete(historyId: string): Promise<void> {
    return apiClient.request<void>(`history/${historyId}`, {
      method: 'DELETE',
    });
  },
};
