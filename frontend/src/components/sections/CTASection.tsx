import React from 'react'
import { Sparkles, ArrowRight, ShieldCheck, Heart } from 'lucide-react'
import { Button } from '../common/Button'
import { Badge } from '../common/Badge'

export const CTASection: React.FC = () => {
  return (
    <section className="relative py-20 bg-[#050814] overflow-hidden">
      {/* Background radial gradients */}
      <div className="absolute inset-0 bg-grid-pattern opacity-20 pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-5xl h-96 bg-gradient-to-r from-cyan-600/15 via-violet-600/15 to-transparent rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="rounded-3xl p-1 bg-gradient-to-r from-cyan-500/40 via-violet-500/30 to-cyan-500/40 shadow-2xl shadow-cyan-950/50">
          <div className="rounded-[22px] bg-gradient-to-b from-[#09122c] to-[#060a1a] p-8 sm:p-14 text-center space-y-6">
            <div className="inline-flex items-center gap-2">
              <Badge variant="cyan" size="md" dot>
                Early Access Open
              </Badge>
            </div>

            <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight max-w-2xl mx-auto">
              Ready to Bridge Conversations with{' '}
              <span className="bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
                SIGNBRIDGE AI?
              </span>
            </h2>

            <p className="text-sm sm:text-base text-slate-300 max-w-xl mx-auto leading-relaxed">
              Experience the power of real-time Indian Sign Language bidirectional translation.
              Join schools, hospitals, and community leaders building an inclusive India.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
              <Button
                variant="glow"
                size="lg"
                icon={<ArrowRight className="w-5 h-5" />}
                iconPosition="right"
                onClick={() => {
                  const target = document.getElementById('translator-preview')
                  target?.scrollIntoView({ behavior: 'smooth' })
                }}
              >
                Try Interactive Demo
              </Button>
              <Button
                variant="secondary"
                size="lg"
                icon={<Sparkles className="w-5 h-5 text-cyan-400" />}
                onClick={() => {
                  const target = document.getElementById('how-it-works')
                  target?.scrollIntoView({ behavior: 'smooth' })
                }}
              >
                Explore Architecture
              </Button>
            </div>

            <div className="pt-6 border-t border-white/5 flex flex-wrap items-center justify-center gap-6 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-cyan-400" />
                Browser Native • No App Required
              </span>
              <span className="text-slate-700">•</span>
              <span className="flex items-center gap-1.5">
                <Heart className="w-4 h-4 text-rose-400" />
                Free for Community & Students
              </span>
              <span className="text-slate-700">•</span>
              <span className="flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-violet-400" />
                21-Point Landmark Tracking
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

