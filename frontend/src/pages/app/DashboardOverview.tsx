import { useState, useEffect } from 'react'
import { LayoutDashboard, Video, Volume2, Bot, GraduationCap, History, ArrowRight, User as UserIcon, Award, AlertCircle } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Badge } from '../../components/common/Badge'
import { useAuth } from '../../context/AuthContext'
import { historyApi, learningApi, ApiError } from '../../services/api'
import type { ProgressSummary } from '../../services/api'

export const DashboardOverview = () => {
  const { user } = useAuth()
  const [historyCount, setHistoryCount] = useState<number | null>(null)
  const [progressSummary, setProgressSummary] = useState<ProgressSummary | null>(null)
  const [statsLoading, setStatsLoading] = useState(true)
  const [statsError, setStatsError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const fetchDashboardStats = async () => {
      setStatsLoading(true)
      setStatsError(null)
      try {
        const [historyRes, progressRes] = await Promise.all([
          historyApi.list(0, 1),
          learningApi.getProgress(),
        ])
        if (!active) return
        setHistoryCount(historyRes.total)
        setProgressSummary(progressRes)
      } catch (err) {
        if (!active) return
        setStatsError(err instanceof ApiError ? err.message : 'Unable to load account metrics.')
      } finally {
        if (active) setStatsLoading(false)
      }
    }
    void fetchDashboardStats()
    return () => {
      active = false
    }
  }, [])

  const quickActions = [
    {
      title: 'Sign to Text',
      description: 'Real-time ISL hand gesture recognition with 3D landmark tracking.',
      icon: Video,
      href: '/app/sign-to-text',
      color: 'cyan',
    },
    {
      title: 'Text to Speech',
      description: 'Synthesize spoken regional Indian languages from ISL transcriptions.',
      icon: Volume2,
      href: '/app/text-to-speech',
      color: 'violet',
    },
    {
      title: 'Text to Sign',
      description: 'Generate 3D avatar animations from typed or spoken sentences.',
      icon: Bot,
      href: '/app/text-to-sign',
      color: 'cyan',
    },
    {
      title: 'ISL Learning Studio',
      description: 'Master ISL signs and practice with real-time AI posture feedback.',
      icon: GraduationCap,
      href: '/app/learn',
      color: 'violet',
    },
    {
      title: 'Translation History',
      description: 'Review saved conversations, gloss transcripts, and audio recordings.',
      icon: History,
      href: '/app/history',
      color: 'slate',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              {user ? `Welcome back, ${user.name}` : 'Platform Overview'}
            </h1>
            <Badge variant="cyan" size="sm" dot>
              System Ready
            </Badge>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            {user ? `Signed in as ${user.email}` : 'Select a translation tool or study module to begin.'}
          </p>
        </div>
      </div>

      {/* Live Account Metrics */}
      {statsError ? (
        <div className="p-4 rounded-2xl bg-rose-950/30 border border-rose-500/20 flex items-center gap-2.5 text-xs text-rose-200">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>Backend metrics unavailable: {statsError}</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 flex items-center gap-4">
            <div className="w-11 h-11 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
              <History className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-mono text-slate-400 uppercase">Saved Translations</div>
              <div className="text-2xl font-extrabold text-white">
                {statsLoading ? '...' : historyCount ?? 'Unavailable'}
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 flex items-center gap-4">
            <div className="w-11 h-11 rounded-xl bg-violet-500/10 border border-violet-500/30 flex items-center justify-center text-violet-400 shrink-0">
              <GraduationCap className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-mono text-slate-400 uppercase">Completed Lessons</div>
              <div className="text-2xl font-extrabold text-white">
                {statsLoading
                  ? '...'
                  : progressSummary
                    ? `${progressSummary.completed_lessons} / ${progressSummary.total_lessons}`
                    : 'Unavailable'}
              </div>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-[#090f23]/70 border border-white/5 flex items-center gap-4">
            <div className="w-11 h-11 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <div className="text-xs font-mono text-slate-400 uppercase">Learning Score</div>
              <div className="text-2xl font-extrabold text-white">
                {statsLoading
                  ? '...'
                  : progressSummary
                    ? `${progressSummary.average_score}%`
                    : 'Unavailable'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Launch Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {quickActions.map((action) => {
          const Icon = action.icon
          return (
            <Link
              key={action.title}
              to={action.href}
              className="group p-6 rounded-2xl bg-[#090f23]/70 border border-white/5 hover:border-cyan-500/30 hover:bg-[#0c142e]/90 transition-all duration-300 flex flex-col justify-between"
            >
              <div className="space-y-4">
                <div className="w-12 h-12 rounded-xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center text-cyan-400 group-hover:scale-105 group-hover:border-cyan-400/40 transition-all">
                  <Icon className="w-6 h-6" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-white group-hover:text-cyan-300 transition-colors">
                    {action.title}
                  </h2>
                  <p className="text-xs text-slate-400 leading-relaxed mt-1.5">
                    {action.description}
                  </p>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-slate-400 group-hover:text-cyan-400 font-medium">
                <span>Launch Tool</span>
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </div>
            </Link>
          )
        })}
      </div>

      {/* System Status Banner */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-cyan-950/30 via-slate-900/40 to-violet-950/30 border border-cyan-500/20 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            {user ? <UserIcon className="w-5 h-5" /> : <LayoutDashboard className="w-5 h-5" />}
          </div>
          <div>
            <div className="text-sm font-semibold text-white">Neural Engine & Session Status</div>
            <div className="text-xs text-slate-400">
              Edge vision pipeline initialized. 21-point hand landmark extraction online.
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="cyan" size="sm" className="font-mono">
            60 FPS Ready
          </Badge>
          <Badge variant="violet" size="sm" className="font-mono">
            ISL v1.0
          </Badge>
        </div>
      </div>
    </div>
  )
}
