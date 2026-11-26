/**
 * SIGNBRIDGE AI - API Client Types
 * Aligned with Phase 10A FastAPI Pydantic schemas.
 */

export interface User {
  id: string;
  name: string;
  email: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

export interface RegisterPayload {
  name: string;
  email: string;
  password: string;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface TranslationHistoryItem {
  id: string;
  user_id: string;
  source_language: string;
  translation_direction: string;
  input_text: string | null;
  recognized_output: string;
  confidence: number | null;
  recognition_path: string | null;
  created_at: string;
}

export interface HistoryCreatePayload {
  source_language?: string;
  translation_direction?: string;
  input_text?: string | null;
  recognized_output: string;
  confidence?: number | null;
  recognition_path?: string | null;
}

export interface HistoryListResponse {
  items: TranslationHistoryItem[];
  total: number;
}

export interface LessonItem {
  id: string;
  title: string;
  description: string;
  difficulty: string;
  category: string;
  content_metadata: Record<string, unknown>;
  created_at: string;
}

export interface ProgressItem {
  id: string;
  user_id: string;
  lesson_id: string;
  completion_status: 'not_started' | 'in_progress' | 'completed';
  score: number;
  last_activity: string;
  lesson?: LessonItem;
}

export interface ProgressSummary {
  total_lessons: number;
  completed_lessons: number;
  in_progress_lessons: number;
  average_score: number;
  progress_records: ProgressItem[];
}

export interface ProgressRecordPayload {
  lesson_id: string;
  completion_status: 'not_started' | 'in_progress' | 'completed';
  score: number;
}

export interface InferenceRequestPayload {
  features: number[];
  motion_energy?: number | null;
  centroid_y?: number | null;
  spatial_stability?: number | null;
  hand_presence?: number | null;
}

export interface TopPrediction {
  label: string;
  class_index: number;
  probability: number;
}

export interface InferenceResponsePayload {
  status: 'accepted' | 'rejected';
  predicted_label: string | null;
  class_index: number | null;
  confidence: number;
  margin: number;
  recognition_path: 'dynamic_path' | 'static_path' | 'none';
  rejection_reason: string | null;
  top_predictions: TopPrediction[];
  gating_metrics: Record<string, unknown>;
}

export interface ApiErrorDetail {
  field?: string;
  message: string;
  type?: string;
}

export interface ApiErrorResponse {
  error: string;
  detail: string | ApiErrorDetail[];
  errors?: ApiErrorDetail[];
}
