import React from 'react'
import { HandMetal, ShieldCheck, Globe, Code2, MessageSquare } from 'lucide-react'
import { Badge } from '../common/Badge'
import { Button } from '../common/Button'

export const Footer: React.FC = () => {
  return (
    <footer className="relative bg-[#03060f] border-t border-cyan-500/15 pt-20 pb-12 overflow-hidden text-slate-400">
      {/* Background glow */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-64 bg-cyan-600/5 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-12 pb-14 border-b border-white/5">
          {/* Column 1: Brand & Description (5 cols) */}
          <div className="lg:col-span-5 space-y-5">
            <a href="#home" className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 via-slate-900 to-violet-600/30 border border-cyan-500/40 text-cyan-400">
                <HandMetal className="w-5 h-5" />
              </div>
              <div className="flex flex-col">
                <span className="font-extrabold text-lg tracking-wider text-white flex items-center gap-1">
                  SIGNBRIDGE<span className="text-cyan-400">AI</span>
                </span>
                <span className="text-[10px] tracking-widest text-slate-400 font-mono -mt-1 uppercase">
                  Indian Sign Language Platform
                </span>
              </div>
            </a>

            <p className="text-sm text-slate-300/80 leading-relaxed max-w-sm">
              Breaking barriers and connecting worlds. SIGNBRIDGE AI is a next-generation accessibility
              platform dedicated to empowering India's Deaf community with real-time bidirectional ISL
              translation, computer vision landmarks, and expressive speech synthesis.
            </p>

            <div className="flex items-center gap-3">
              <Badge variant="cyan" size="sm" dot>
                Platform v1.0 Ready
              </Badge>
              <Badge variant="violet" size="sm">
                ISLRTC Aligned
              </Badge>
            </div>

            {/* Social / Community Links */}
            <div className="flex items-center gap-3 pt-2">
              <a
                href="#home"
                className="w-9 h-9 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-cyan-400 hover:border-cyan-500/40 transition-colors"
                aria-label="Code Repository"
              >
                <Code2 className="w-4 h-4" />
              </a>
              <a
                href="#home"
                className="w-9 h-9 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-cyan-400 hover:border-cyan-500/40 transition-colors"
                aria-label="Community Portal"
              >
                <Globe className="w-4 h-4" />
              </a>
              <a
                href="#home"
                className="w-9 h-9 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-center text-slate-400 hover:text-cyan-400 hover:border-cyan-500/40 transition-colors"
                aria-label="Discussions"
              >
                <MessageSquare className="w-4 h-4" />
              </a>
            </div>
          </div>

          {/* Column 2: Navigation Links (2 cols) */}
          <div className="lg:col-span-2 space-y-4">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-bold">
              Navigation
            </h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <a href="#home" className="hover:text-cyan-400 transition-colors">
                  Home
                </a>
              </li>
              <li>
                <a href="#how-it-works" className="hover:text-cyan-400 transition-colors">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#features" className="hover:text-cyan-400 transition-colors">
                  Platform Features
                </a>
              </li>
              <li>
                <a href="#learning" className="hover:text-cyan-400 transition-colors flex items-center gap-1.5">
                  <span>Learn ISL</span>
                  <Badge variant="cyan" size="sm" className="text-[9px] py-0 px-1">
                    Studio
                  </Badge>
                </a>
              </li>
              <li>
                <a href="#why-us" className="hover:text-cyan-400 transition-colors">
                  Why SignBridge
                </a>
              </li>
            </ul>
          </div>

          {/* Column 3: Features & Tech (2 cols) */}
          <div className="lg:col-span-2 space-y-4">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-bold">
              Core Tech
            </h4>
            <ul className="space-y-2.5 text-sm">
              <li>
                <a href="#features" className="hover:text-cyan-400 transition-colors">
                  Sign to Text
                </a>
              </li>
              <li>
                <a href="#features" className="hover:text-cyan-400 transition-colors">
                  Text to Speech
                </a>
              </li>
              <li>
                <a href="#features" className="hover:text-cyan-400 transition-colors">
                  Text to Sign (3D)
                </a>
              </li>
              <li>
                <a href="#features" className="hover:text-cyan-400 transition-colors">
                  Landmark Extraction
                </a>
              </li>
              <li>
                <a href="#features" className="hover:text-cyan-400 transition-colors">
                  Edge Privacy
                </a>
              </li>
            </ul>
          </div>

          {/* Column 4: Newsletter / Early Access (3 cols) */}
          <div className="lg:col-span-3 space-y-4">
            <h4 className="text-xs font-mono uppercase tracking-wider text-white font-bold">
              Stay Connected
            </h4>
            <p className="text-xs text-slate-300/80 leading-relaxed">
              Subscribe to updates on ISL dataset expansions, mobile app releases, and community workshops.
            </p>

            <form
              onSubmit={(e) => {
                e.preventDefault()
                alert('Thank you for subscribing to SIGNBRIDGE AI updates!')
              }}
              className="space-y-2"
            >
              <div className="relative">
                <input
                  type="email"
                  required
                  placeholder="Enter your email"
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900/90 border border-slate-700/80 text-white placeholder-slate-500 text-xs focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400 transition-all"
                />
              </div>
              <Button type="submit" variant="glow" size="sm" className="w-full justify-center">
                Get Updates
              </Button>
            </form>

            <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
              <span>Zero spam. Respecting your inbox.</span>
            </div>
          </div>
        </div>

        {/* Bottom Bar: Copyright & Status */}
        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span>© 2026 SIGNBRIDGE AI. All rights reserved.</span>
          </div>

          <div className="flex items-center gap-6 text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Neural Gateway: Online
            </span>
            <a href="#home" className="hover:text-white transition-colors">
              Privacy
            </a>
            <a href="#home" className="hover:text-white transition-colors">
              Terms
            </a>
            <a href="#home" className="hover:text-white transition-colors">
              Accessibility
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
