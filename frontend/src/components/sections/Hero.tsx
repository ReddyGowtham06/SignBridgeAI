import React from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight, Sparkles, Compass, ShieldCheck, Globe } from 'lucide-react'
import { Button } from '../common/Button'
import { Badge } from '../common/Badge'
import { GestureMockPanel } from './GestureMockPanel'

export const Hero: React.FC = () => {
  const navigate = useNavigate()
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <section id="home" className="relative pt-24 pb-16 sm:pt-32 sm:pb-20 md:pt-40 md:pb-32 overflow-hidden">
      {/* Background Radial Glow Accents */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-gradient-to-tr from-cyan-600/15 via-violet-600/15 to-transparent rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-1/3 -left-48 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute top-1/2 -right-48 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl pointer-events-none -z-10" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center">
          {/* Left Column: Heading and CTAs */}
          <div className="lg:col-span-6 space-y-7 text-center lg:text-left">
            {/* Top Pill */}
            <div className="inline-flex items-center gap-2">
              <Badge variant="cyan" size="md" dot className="px-3.5 py-1.5 text-xs font-semibold">
                <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                <span>Next-Gen ISL Artificial Intelligence</span>
              </Badge>
            </div>

            {/* Main Headline */}
            <h1 className="text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-white leading-[1.12]">
              Breaking Barriers.{' '}
              <span className="block mt-1 bg-gradient-to-r from-cyan-400 via-sky-300 to-violet-400 bg-clip-text text-transparent">
                Connecting Worlds.
              </span>
            </h1>

            {/* Supporting Text */}
            <p className="text-base sm:text-lg text-slate-300/90 leading-relaxed max-w-xl mx-auto lg:mx-0">
              SIGNBRIDGE AI empowers millions in the Deaf and Hard-of-Hearing community across India
              with instant, bidirectional Indian Sign Language (ISL) translation, high-accuracy 3D hand
              landmark recognition, and natural regional voice synthesis.
            </p>

            {/* Call to Actions */}
            <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 pt-2">
              <Button
                variant="glow"
                size="lg"
                icon={<ArrowRight className="w-5 h-5" />}
                iconPosition="right"
                onClick={() => navigate('/app/sign-to-text')}
                className="w-full sm:w-auto"
              >
                Start Translating
              </Button>
              <Button
                variant="secondary"
                size="lg"
                icon={<Compass className="w-5 h-5 text-cyan-400" />}
                onClick={() => scrollToSection('how-it-works')}
                className="w-full sm:w-auto"
              >
                Explore Features
              </Button>
            </div>

            {/* Core Trust / Metric Highlights */}
            <div className="pt-6 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 text-left">
              <div className="space-y-1">
                <div className="text-xl sm:text-2xl font-bold text-cyan-300 font-mono">99.4%</div>
                <div className="text-xs text-slate-400 font-medium">Model Precision</div>
              </div>
              <div className="space-y-1">
                <div className="text-xl sm:text-2xl font-bold text-violet-300 font-mono">&lt;20ms</div>
                <div className="text-xs text-slate-400 font-medium">Neural Latency</div>
              </div>
              <div className="space-y-1">
                <div className="text-xl sm:text-2xl font-bold text-cyan-300 font-mono">2,500+</div>
                <div className="text-xs text-slate-400 font-medium">Curated Signs</div>
              </div>
              <div className="space-y-1">
                <div className="text-xl sm:text-2xl font-bold text-violet-300 font-mono">Pan-India</div>
                <div className="text-xs text-slate-400 font-medium">Regional Dialects</div>
              </div>
            </div>

            {/* Security & Accessibility badges */}
            <div className="flex flex-wrap items-center justify-center lg:justify-start gap-3 sm:gap-4 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4 text-cyan-400" />
                Edge Privacy Preserved
              </span>
              <span className="text-slate-600">•</span>
              <span className="flex items-center gap-1.5">
                <Globe className="w-4 h-4 text-violet-400" />
                Zero Special Hardware
              </span>
            </div>
          </div>

          {/* Right Column: Designed Mock Visual Interface */}
          <div className="lg:col-span-6 flex justify-center" id="translator-preview">
            <GestureMockPanel />
          </div>
        </div>
      </div>
    </section>
  )
}
