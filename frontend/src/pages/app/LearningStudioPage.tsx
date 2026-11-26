import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  GraduationCap,
  Award,
  CheckCircle2,
  Clock,
  BookOpen,
  RefreshCw,
  AlertCircle,
  Loader2,
  Video,
  Sparkles,
} from 'lucide-react';
import { Badge } from '../../components/common/Badge';
import { learningApi, ApiError } from '../../services/api';
import type { LessonItem, ProgressSummary, ProgressItem } from '../../services/api';

export const LearningStudioPage = () => {
  const [lessons, setLessons] = useState<LessonItem[]>([]);
  const [progress, setProgress] = useState<ProgressSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [savingLessonId, setSavingLessonId] = useState<string | null>(null);
  const [activeLesson, setActiveLesson] = useState<LessonItem | null>(null);
  const [practiceScore, setPracticeScore] = useState<number>(95);
  const [feedbackMsg, setFeedbackMsg] = useState<string | null>(null);

  const loadLearningData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [lessonsData, progressData] = await Promise.all([
        learningApi.getLessons(),
        learningApi.getProgress(),
      ]);
      setLessons(lessonsData);
      setProgress(progressData);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to load Learning Studio curriculum.');
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadLearningData();
  }, [loadLearningData]);

  const getLessonProgress = (lessonId: string): ProgressItem | undefined => {
    return progress?.progress_records.find((p) => p.lesson_id === lessonId);
  };

  const handleSaveProgress = async (
    lessonId: string,
    status: 'in_progress' | 'completed',
    score: number
  ) => {
    setSavingLessonId(lessonId);
    setError(null);
    setFeedbackMsg(null);
    try {
      await learningApi.recordProgress({
        lesson_id: lessonId,
        completion_status: status,
        score,
      });
      await loadLearningData();
      setActiveLesson(null);
      setFeedbackMsg(
        status === 'completed'
          ? `Lesson completed with ${score}% score!`
          : 'Lesson marked as in progress.'
      );
      setTimeout(() => setFeedbackMsg(null), 4000);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Failed to save learning progress.');
      }
    } finally {
      setSavingLessonId(null);
    }
  };

  const completionPct =
    progress && progress.total_lessons > 0
      ? Math.round((progress.completed_lessons / progress.total_lessons) * 100)
      : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              ISL Learning Studio
            </h1>
            <Badge variant="violet" size="sm" dot>
              Interactive Academy
            </Badge>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Master Indian Sign Language with structured curriculum modules and persistent progress tracking.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void loadLearningData()}
          disabled={isLoading}
          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-900/80 border border-slate-700/60 text-xs font-semibold text-slate-300 hover:text-white hover:border-violet-500/40 transition-all disabled:opacity-50 cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-violet-400' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Feedback Banner */}
      {feedbackMsg && (
        <div className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-500/30 flex items-center gap-2.5 text-xs text-emerald-200">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{feedbackMsg}</span>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-950/40 border border-rose-500/30 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5 text-xs text-rose-200">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => void loadLearningData()}
            className="px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 border border-rose-500/40 text-xs font-semibold text-rose-200 cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Progress Summary Cards */}
      {progress && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 space-y-1">
            <div className="text-xs font-mono text-slate-400 uppercase">Curriculum Completion</div>
            <div className="text-2xl font-extrabold text-white flex items-baseline gap-2">
              <span>{progress.completed_lessons}</span>
              <span className="text-sm font-normal text-slate-400">/ {progress.total_lessons} lessons</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden mt-2">
              <div
                className="h-full bg-gradient-to-r from-violet-500 to-cyan-400 transition-all duration-500"
                style={{ width: `${completionPct}%` }}
              />
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 space-y-1">
            <div className="text-xs font-mono text-slate-400 uppercase">Average Score</div>
            <div className="text-2xl font-extrabold text-cyan-400">{progress.average_score}%</div>
            <div className="text-[11px] text-slate-400">Across attempted modules</div>
          </div>

          <div className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 space-y-1">
            <div className="text-xs font-mono text-slate-400 uppercase">In Progress</div>
            <div className="text-2xl font-extrabold text-violet-400">{progress.in_progress_lessons}</div>
            <div className="text-[11px] text-slate-400">Active study modules</div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {[1, 2, 3, 4].map((s) => (
            <div
              key={s}
              className="p-6 rounded-2xl bg-[#090f23]/60 border border-white/5 animate-pulse space-y-4"
            >
              <div className="h-5 w-44 bg-slate-800 rounded" />
              <div className="h-3 w-full bg-slate-800/70 rounded" />
              <div className="h-3 w-3/4 bg-slate-800/70 rounded" />
            </div>
          ))}
        </div>
      ) : lessons.length === 0 ? (
        <div className="rounded-2xl p-10 bg-[#090f23]/60 border border-white/5 text-center space-y-4 max-w-xl mx-auto">
          <GraduationCap className="w-10 h-10 text-violet-400 mx-auto" />
          <h2 className="text-lg font-bold text-white">No Curriculum Lessons Available</h2>
          <p className="text-sm text-slate-400">Check backend connection or refresh to load lessons.</p>
        </div>
      ) : (
        /* Lessons Grid */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {lessons.map((lesson) => {
            const prog = getLessonProgress(lesson.id);
            const status = prog?.completion_status || 'not_started';
            const metadata = lesson.content_metadata as {
              vocabulary?: string[];
              target_motion?: string;
              estimated_minutes?: number;
            };

            return (
              <div
                key={lesson.id}
                className="p-6 rounded-2xl bg-[#090f23]/70 border border-white/5 hover:border-violet-500/30 transition-all flex flex-col justify-between space-y-5"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="violet" size="sm">
                        {lesson.category}
                      </Badge>
                      <Badge variant="slate" size="sm">
                        {lesson.difficulty}
                      </Badge>
                    </div>

                    {status === 'completed' ? (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-[11px] font-mono text-emerald-300">
                        <CheckCircle2 className="w-3 h-3" />
                        Completed ({prog?.score ?? 0}%)
                      </span>
                    ) : status === 'in_progress' ? (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-amber-500/15 border border-amber-500/30 text-[11px] font-mono text-amber-300">
                        <Clock className="w-3 h-3" />
                        In Progress
                      </span>
                    ) : (
                      <span className="text-[11px] font-mono text-slate-500">Not Started</span>
                    )}
                  </div>

                  <div>
                    <h2 className="text-lg font-bold text-white">{lesson.title}</h2>
                    <p className="text-xs text-slate-400 leading-relaxed mt-1">{lesson.description}</p>
                  </div>

                  {Array.isArray(metadata?.vocabulary) && metadata.vocabulary.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 pt-1">
                      {metadata.vocabulary.map((word) => (
                        <span
                          key={word}
                          className="px-2 py-0.5 rounded-md bg-slate-900 border border-white/10 text-[11px] font-mono text-cyan-300"
                        >
                          {word.replace(/^\d+\./, '')}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <div className="pt-4 border-t border-white/5 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
                    {metadata?.estimated_minutes && (
                      <span className="inline-flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {metadata.estimated_minutes} min
                      </span>
                    )}
                    {metadata?.target_motion && (
                      <span className="inline-flex items-center gap-1">
                        <BookOpen className="w-3 h-3" />
                        {metadata.target_motion}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {status === 'not_started' && (
                      <button
                        type="button"
                        onClick={() => void handleSaveProgress(lesson.id, 'in_progress', 50)}
                        disabled={savingLessonId === lesson.id}
                        className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 border border-white/10 text-xs font-semibold text-slate-200 transition-colors cursor-pointer disabled:opacity-50"
                      >
                        Start
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => {
                        setActiveLesson(lesson);
                        setPracticeScore(prog?.score && prog.score > 0 ? Math.round(prog.score) : 95);
                      }}
                      className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-violet-500/20 hover:bg-violet-500/30 border border-violet-500/40 text-xs font-semibold text-violet-200 transition-colors cursor-pointer"
                    >
                      <Award className="w-3.5 h-3.5" />
                      <span>{status === 'completed' ? 'Update Score' : 'Complete Lesson'}</span>
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Practice & Complete Modal */}
      {activeLesson && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-md max-h-[90vh] overflow-y-auto rounded-2xl bg-[#070c1d] border border-violet-500/30 p-5 sm:p-6 space-y-5 shadow-2xl">
            <div className="flex items-center gap-2.5">
              <div className="w-10 h-10 rounded-xl bg-violet-500/15 border border-violet-500/30 flex items-center justify-center text-violet-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">{activeLesson.title}</h3>
                <p className="text-xs text-slate-400">Record lesson completion & practice score</p>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/70 border border-white/5 text-xs text-slate-300 space-y-2">
              <p>Practice the signs for this module in the Sign-to-Text studio, then record your score below:</p>
              <Link
                to="/app/sign-to-text"
                className="inline-flex items-center gap-1.5 text-cyan-400 hover:text-cyan-300 font-semibold"
              >
                <Video className="w-3.5 h-3.5" />
                <span>Launch Sign-to-Text Camera Practice</span>
              </Link>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="practice-score" className="font-medium text-slate-300">
                  Practice Accuracy Score
                </label>
                <span className="font-mono font-bold text-cyan-400">{practiceScore}%</span>
              </div>
              <input
                id="practice-score"
                type="range"
                min={0}
                max={100}
                step={5}
                value={practiceScore}
                onChange={(e) => setPracticeScore(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
            </div>

            <div className="flex flex-col-reverse sm:flex-row sm:items-center justify-end gap-2.5 pt-2">
              <button
                type="button"
                onClick={() => setActiveLesson(null)}
                className="w-full sm:w-auto px-4 py-2.5 min-h-[44px] rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300 cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={savingLessonId === activeLesson.id}
                onClick={() => void handleSaveProgress(activeLesson.id, 'completed', practiceScore)}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-4 py-2.5 min-h-[44px] rounded-xl bg-violet-500 hover:bg-violet-400 text-slate-950 font-bold text-xs cursor-pointer disabled:opacity-50"
              >
                {savingLessonId === activeLesson.id ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    <span>Saving...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Save Completion</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
