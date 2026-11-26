/**
 * SIGNBRIDGE AI - Authentication API Service
 */
import { apiClient } from './client';
import type { AuthTokens, LoginPayload, RegisterPayload, User } from './types';

export const authApi = {
  async register(payload: RegisterPayload): Promise<AuthTokens> {
    const res = await apiClient.request<AuthTokens>('auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    apiClient.setTokens(res);
    return res;
  },

  async login(payload: LoginPayload): Promise<AuthTokens> {
    const res = await apiClient.request<AuthTokens>('auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    apiClient.setTokens(res);
    return res;
  },

  async getMe(): Promise<User> {
    return apiClient.request<User>('auth/me', {
      method: 'GET',
    });
  },

  async logout(): Promise<void> {
    try {
      await apiClient.request('auth/logout', { method: 'POST' });
    } finally {
      apiClient.clearTokens();
    }
  },
};
