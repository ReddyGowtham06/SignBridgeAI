import { useState, useEffect } from 'react'
import { NavLink, Outlet, Link, useLocation, useNavigate } from 'react-router-dom'
import {
  HandMetal,
  LayoutDashboard,
  Video,
  Volume2,
  Bot,
  GraduationCap,
  History,
  ArrowLeft,
  Menu,
  X,
  Activity,
  ShieldCheck,
  ChevronRight,
  LogOut,
  User as UserIcon,
} from 'lucide-react'
import { useAuth } from '../context/AuthContext'


export const AppLayout = () => {
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  // Prevent background page scrolling when mobile drawer is open & close on Escape
  useEffect(() => {
    if (!mobileSidebarOpen) return
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileSidebarOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = prevOverflow
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [mobileSidebarOpen])

  const navItems = [
    { label: 'Overview', to: '/app', icon: LayoutDashboard, end: true },
    { label: 'Sign to Text', to: '/app/sign-to-text', icon: Video, end: false },
    { label: 'Text to Speech', to: '/app/text-to-speech', icon: Volume2, end: false },
    { label: 'Text to Sign', to: '/app/text-to-sign', icon: Bot, end: false },
    { label: 'ISL Learning Studio', to: '/app/learn', icon: GraduationCap, end: false },
    { label: 'Translation History', to: '/app/history', icon: History, end: false },
    { label: 'Evaluation Studio (Dev)', to: '/app/evaluate', icon: Activity, end: false },
  ]

  // Determine current page title for topbar breadcrumb
  const currentItem = navItems.find((item) =>
    item.end ? location.pathname === item.to : location.pathname.startsWith(item.to)
  )

  return (
    <div className="min-h-screen bg-[#050814] text-slate-100 flex flex-col md:flex-row">
      {/* Mobile Header Bar */}
      <div className="md:hidden flex items-center justify-between px-4 py-2.5 pt-safe bg-[#070c1d]/95 backdrop-blur-md border-b border-cyan-500/10 sticky top-0 z-40">
        <Link to="/app" className="flex items-center gap-2.5 min-h-[44px]">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
            <HandMetal className="w-4 h-4" />
          </div>
          <span className="font-extrabold text-sm tracking-wider text-white">
            SIGNBRIDGE<span className="text-cyan-400">AI</span>
          </span>
        </Link>
        <button
          type="button"
          onClick={() => setMobileSidebarOpen(!mobileSidebarOpen)}
          className="min-w-[44px] min-h-[44px] flex items-center justify-center rounded-xl bg-slate-900 border border-slate-700/60 text-slate-300 hover:text-white"
          aria-label="Toggle navigation menu"
          aria-expanded={mobileSidebarOpen}
        >
          {mobileSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>
      </div>

      {/* Backdrop for mobile drawer */}
      {mobileSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 md:hidden"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      {/* Sidebar Navigation */}
      <aside
        className={`fixed md:sticky top-0 left-0 bottom-0 z-50 w-64 bg-[#070c1d] border-r border-cyan-500/10 flex flex-col justify-between transition-transform duration-300 ease-in-out md:translate-x-0 ${
          mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        } h-screen`}
      >
        <div className="p-5 flex flex-col gap-6 overflow-y-auto">
          {/* Logo & Platform Info */}
          <div className="flex items-center justify-between">
            <Link to="/app" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 via-slate-900 to-violet-600/30 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:border-cyan-400 transition-colors shadow-sm">
                <HandMetal className="w-5 h-5 group-hover:scale-105 transition-transform" />
              </div>
              <div className="flex flex-col">
                <span className="font-extrabold text-base tracking-wider text-white">
                  SIGNBRIDGE<span className="text-cyan-400">AI</span>
                </span>
                <span className="text-[10px] tracking-widest text-slate-400 font-mono -mt-0.5 uppercase">
                  Studio Workspace
                </span>
              </div>
            </Link>
            <button
              type="button"
              className="md:hidden min-w-[44px] min-h-[44px] flex items-center justify-center rounded-xl text-slate-400 hover:text-white"
              onClick={() => setMobileSidebarOpen(false)}
              aria-label="Close navigation menu"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1.5 pt-2">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 px-3 mb-2 font-semibold">
              Application Modules
            </div>
            {navItems.map((item) => {
              const Icon = item.icon
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.end}
                  onClick={() => setMobileSidebarOpen(false)}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3.5 py-2.5 min-h-[44px] rounded-xl text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-gradient-to-r from-cyan-950/70 to-slate-900 border border-cyan-500/40 text-cyan-300 shadow-sm shadow-cyan-950'
                        : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                </NavLink>
              )
            })}
          </nav>
        </div>

        {/* Sidebar Footer: User Account, Logout & Back to Landing Page */}
        <div className="p-4 border-t border-white/5 bg-[#050917] space-y-3">
          {user && (
            <div className="flex items-center justify-between gap-2 px-2.5 py-2 rounded-xl bg-slate-900/70 border border-white/5">
              <div className="flex items-center gap-2 min-w-0">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/15 border border-cyan-500/30 flex items-center justify-center text-cyan-400 shrink-0">
                  <UserIcon className="w-3.5 h-3.5" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-white truncate">{user.name}</div>
                  <div className="text-[10px] text-slate-400 truncate">{user.email}</div>
                </div>
              </div>
              <button
                type="button"
                onClick={handleLogout}
                title="Sign Out"
                aria-label="Sign Out"
                className="min-w-[44px] min-h-[44px] flex items-center justify-center rounded-xl text-slate-400 hover:text-rose-400 hover:bg-rose-950/30 transition-colors shrink-0 cursor-pointer"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          )}

          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 px-1">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Engine Online
            </span>
            <span className="text-cyan-400">ISL v1.0</span>
          </div>

          <Link
            to="/"
            className="flex items-center justify-center gap-2 w-full px-3.5 py-2 rounded-xl text-xs font-semibold text-slate-300 bg-slate-900/80 border border-slate-700/60 hover:text-white hover:border-cyan-500/40 hover:bg-slate-800 transition-all"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Public Website</span>
          </Link>
        </div>
      </aside>

      {/* Main App Content Area with TopBar */}
      <div className="flex-1 flex flex-col min-w-0 min-h-screen">
        {/* Desktop Topbar Header */}
        <header className="hidden md:flex items-center justify-between px-8 py-4 bg-[#060a18]/80 backdrop-blur-md border-b border-white/5 sticky top-0 z-30">
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <Link to="/app" className="hover:text-cyan-400 transition-colors">
              SIGNBRIDGE
            </Link>
            <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
            <span className="text-slate-200 font-semibold">{currentItem?.label || 'Dashboard'}</span>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900/80 border border-cyan-500/20 text-xs font-mono text-slate-300">
              <Activity className="w-3.5 h-3.5 text-cyan-400" />
              <span>Edge Vision Mode</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
              <span>Privacy Active</span>
            </div>
            <Link
              to="/"
              className="text-xs text-slate-400 hover:text-cyan-400 transition-colors flex items-center gap-1"
            >
              <span>Landing Page</span>
              <ChevronRight className="w-3 h-3" />
            </Link>
          </div>
        </header>

        {/* Dynamic Outlet Area */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default AppLayout
