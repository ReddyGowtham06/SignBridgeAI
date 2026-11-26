/**
 * SIGNBRIDGE AI - Base HTTP API Client
 * Manages authentication tokens, headers, concurrent token refresh, and normalized error responses.
 */
import type { AuthTokens } from './types';

const DEFAULT_API_BASE_URL = import.meta.env.PROD
  ? 'https://signbridge-api.vercel.app/api'
  : 'http://127.0.0.1:8000/api';

const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  (import.meta.env.VITE_API_URL as string | undefined) ||
  DEFAULT_API_BASE_URL;

export class ApiError extends Error {
  status: number;
  data: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

class ApiClient {
  private baseUrl: string;
  private refreshPromise: Promise<boolean> | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  getAccessToken(): string | null {
    return localStorage.getItem('signbridge_access_token');
  }

  getRefreshToken(): string | null {
    return localStorage.getItem('signbridge_refresh_token');
  }

  setTokens(tokens: AuthTokens): void {
    localStorage.setItem('signbridge_access_token', tokens.access_token);
    localStorage.setItem('signbridge_refresh_token', tokens.refresh_token);
    localStorage.setItem('signbridge_user', JSON.stringify(tokens.user));
  }

  clearTokens(): void {
    localStorage.removeItem('signbridge_access_token');
    localStorage.removeItem('signbridge_refresh_token');
    localStorage.removeItem('signbridge_user');
  }

  async request<T>(endpoint: string, options: RequestInit = {}, _isRetry = false): Promise<T> {
    const url = `${this.baseUrl}/${endpoint.replace(/^\/+/, '')}`;
    const headers = new Headers(options.headers || {});

    if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
    }

    const token = this.getAccessToken();
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }

    let response: Response;
    try {
      response = await fetch(url, { ...options, headers });
    } catch (err: unknown) {
      throw new ApiError(
        `Unable to reach SignBridge backend at ${this.baseUrl}. Please ensure the API server is running.`,
        0,
        err
      );
    }

    // Handle 401 unauthorized & attempt single refresh if token expired
    if (response.status === 401 && !_isRetry && this.getRefreshToken() && !endpoint.includes('auth/login') && !endpoint.includes('auth/register') && !endpoint.includes('auth/refresh')) {
      const refreshed = await this.tryRefreshToken();
      if (refreshed) {
        const nextHeaders = new Headers(options.headers || {});
        if (!nextHeaders.has('Content-Type') && !(options.body instanceof FormData)) {
          nextHeaders.set('Content-Type', 'application/json');
        }
        const newToken = this.getAccessToken();
        if (newToken) {
          nextHeaders.set('Authorization', `Bearer ${newToken}`);
        }
        return this.request<T>(endpoint, { ...options, headers: nextHeaders }, true);
      }
    }

    if (response.status === 204) {
      return null as unknown as T;
    }

    let data: unknown;
    try {
      data = await response.json();
    } catch {
      data = null;
    }

    if (!response.ok) {
      let errorMsg = `HTTP ${response.status}: ${response.statusText}`;
      if (typeof data === 'object' && data !== null) {
        if ('errors' in data && Array.isArray((data as { errors: Array<{ message?: string }> }).errors) && (data as { errors: Array<{ message?: string }> }).errors.length > 0) {
          errorMsg = (data as { errors: Array<{ message?: string }> }).errors.map((e) => e.message).filter(Boolean).join('; ');
        } else if ('detail' in data && typeof (data as { detail: unknown }).detail === 'string') {
          errorMsg = (data as { detail: string }).detail;
        }
      }
      throw new ApiError(errorMsg, response.status, data);
    }

    return data as T;
  }

  private async tryRefreshToken(): Promise<boolean> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      const refreshToken = this.getRefreshToken();
      if (!refreshToken) {
        this.clearTokens();
        return false;
      }

      try {
        const res = await fetch(`${this.baseUrl}/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });

        if (res.ok) {
          const tokens = (await res.json()) as AuthTokens;
          this.setTokens(tokens);
          return true;
        }
      } catch {
        // Refresh network/parse failure
      }

      this.clearTokens();
      return false;
    })();

    try {
      return await this.refreshPromise;
    } finally {
      this.refreshPromise = null;
    }
  }
}

export const apiClient = new ApiClient(API_BASE_URL);
