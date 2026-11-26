import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  History,
  Trash2,
  RefreshCw,
  AlertCircle,
  Video,
  Clock,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  Loader2,
  ArrowRight,
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import { historyApi, ApiError } from '../../services/api';
import type { TranslationHistoryItem } from '../../services/api';

const PAGE_SIZE = 10;

export const HistoryPage = () => {
  const [items, setItems] = useState<TranslationHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchHistory = useCallback(async (pageIndex: number) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await historyApi.list(pageIndex * PAGE_SIZE, PAGE_SIZE);
      setItems(data.items);
      setTotal(data.total);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to load translation history.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchHistory(page);
  }, [fetchHistory, page]);

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    setError(null);
    try {
      await historyApi.delete(id);
      const nextTotal = Math.max(0, total - 1);
      const maxPage = Math.max(0, Math.ceil(nextTotal / PAGE_SIZE) - 1);
      const targetPage = Math.min(page, maxPage);
      if (targetPage !== page) {
        setPage(targetPage);
      } else {
        await fetchHistory(targetPage);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to delete translation record.');
      }
    } finally {
      setDeletingId(null);
    }
  };

  const formatTimestamp = (iso: string) => {
    try {
      return new Intl.DateTimeFormat('en-IN', {
        dateStyle: 'medium',
        timeStyle: 'short',
      }).format(new Date(iso));
    } catch {
      return iso;
    }
  };

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Translation History
            </h1>
            <Badge variant="slate" size="sm" dot>
              {total} {total === 1 ? 'Record' : 'Records'}
            </Badge>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Review your saved sign translations, recognition confidence scores, and gating paths.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void fetchHistory(page)}
          disabled={isLoading}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 min-h-[44px] sm:min-h-[38px] rounded-xl bg-slate-900/80 border border-slate-700/60 text-xs font-semibold text-slate-300 hover:text-white hover:border-cyan-500/40 transition-all disabled:opacity-50 cursor-pointer w-full sm:w-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-cyan-400' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-950/40 border border-rose-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-xs text-rose-200">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => void fetchHistory(page)}
            className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-xs font-semibold text-rose-200 transition-colors cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4].map((skeleton) => (
            <div
              key={skeleton}
              className="p-5 rounded-2xl bg-[#090f23]/60 border border-white/5 animate-pulse flex items-center justify-between gap-4"
            >
              <div className="space-y-2.5 flex-1">
                <div className="h-4 w-48 bg-slate-800 rounded" />
                <div className="h-3 w-72 bg-slate-800/70 rounded" />
              </div>
              <div className="h-8 w-8 bg-slate-800 rounded-lg" />
            </div>
          ))}
        </div>
      ) : items.length === 0 ? (
        /* Empty State */
        <div className="rounded-2xl p-10 bg-[#090f23]/60 border border-white/5 text-center space-y-4 max-w-xl mx-auto my-8">
          <div className="w-16 h-16 rounded-2xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-slate-300 mx-auto">
            <History className="w-8 h-8" />
          </div>
          <h2 className="text-xl font-bold text-white">No Saved Translations Yet</h2>
          <p className="text-sm text-slate-400 leading-relaxed">
            Perform Indian Sign Language signs in the Sign-to-Text studio and click{' '}
            <strong className="text-cyan-300">Save Translation</strong> to archive your transcripts here.
          </p>
          <Link
            to="/app/sign-to-text"
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-colors"
          >
            <Video className="w-4 h-4" />
            <span>Open Sign-to-Text Studio</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      ) : (
        /* History List */
        <div className="space-y-3">
          {items.map((item) => (
            <div
              key={item.id}
              className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 hover:border-cyan-500/20 transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-4"
            >
              <div className="space-y-2 min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 text-base font-bold text-white break-words">
                    <Sparkles className="w-4 h-4 text-cyan-400 shrink-0" />
                    {item.recognized_output}
                  </span>
                </div>

                <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-slate-400">
                  <span className="px-2 py-0.5 rounded bg-slate-800/80 border border-white/5 text-slate-300">
                    {item.source_language} • {item.translation_direction.replace(/_/g, ' → ')}
                  </span>

                  {item.recognition_path && (
                    <span
                      className={`px-2 py-0.5 rounded border ${
                        item.recognition_path === 'static_path'
                          ? 'bg-violet-500/15 border-violet-500/30 text-violet-300'
                          : 'bg-cyan-500/15 border-cyan-500/30 text-cyan-300'
                      }`}
                    >
                      {item.recognition_path === 'static_path' ? 'Path B (Static)' : 'Path A (Dynamic)'}
                    </span>
                  )}

                  {typeof item.confidence === 'number' && (
                    <span className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
                      {(item.confidence * 100).toFixed(1)}% conf
                    </span>
                  )}

                  <span className="inline-flex items-center gap-1 text-slate-400">
                    <Clock className="w-3 h-3" />
                    {formatTimestamp(item.created_at)}
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-end shrink-0">
                <button
                  type="button"
                  onClick={() => void handleDelete(item.id)}
                  disabled={deletingId === item.id}
                  title="Delete history entry"
                  className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 min-h-[40px] rounded-xl bg-slate-900/80 hover:bg-rose-950/50 border border-white/10 hover:border-rose-500/40 text-xs text-slate-400 hover:text-rose-300 transition-colors disabled:opacity-50 cursor-pointer w-full sm:w-auto"
                >
                  {deletingId === item.id ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="w-3.5 h-3.5" />
                  )}
                  <span>Delete</span>
                </button>
              </div>
            </div>
          ))}

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="pt-4 flex items-center justify-between text-xs text-slate-400 font-mono">
              <span>
                Page {page + 1} of {totalPages} ({total} total)
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0 || isLoading}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-300 hover:text-white disabled:opacity-40 cursor-pointer"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  <span>Prev</span>
                </button>
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1 || isLoading}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-300 hover:text-white disabled:opacity-40 cursor-pointer"
                >
                  <span>Next</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
