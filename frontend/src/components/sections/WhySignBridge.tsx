import React from 'react'
import {
  HeartHandshake,
  MessageCircle,
  Zap,
  Globe2,
  CheckCircle,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { Badge } from '../common/Badge'

export const WhySignBridge: React.FC = () => {
  const pillars = [
    {
      title: 'Uncompromised Accessibility',
      subtitle: 'Zero specialized hardware needed',
      description:
        'Engineered from the ground up alongside Indian Deaf educators and associations. Works universally on any standard smartphone, tablet, or laptop browser without expensive sensory gloves.',
      icon: HeartHandshake,
      color: 'cyan',
      statNumber: '18M+',
      statLabel: 'Deaf & Hard-of-Hearing Indians empowered',
      highlights: [
        'Browser-accessible with zero app installation barriers',
        'Works offline in low-connectivity rural schools and clinics',
        'High contrast mode and full screen reader compatibility',
      ],
    },
    {
      title: 'Frictionless Communication',
      subtitle: 'Natural conversations anywhere',
      description:
        'Empowers both signers and non-signers to interact with zero awkward pauses. Bridges critical everyday dialogues at hospital appointments, bank counters, classrooms, and family gatherings.',
      icon: MessageCircle,
      color: 'violet',
      statNumber: '2-Way',
      statLabel: 'Bidirectional signing and voice synthesis',
      highlights: [
        'Intuitive interface requiring zero technical training',
        'Simultaneous transcription with speech output',
        'Saved conversation history and quick common phrases',
      ],
    },
    {
      title: 'True Real-Time Interaction',
      subtitle: 'Sub-20ms edge latency',
      description:
        'Powered by lightweight, edge-quantized vision neural networks that evaluate 21 3D landmarks at 60 FPS, ensuring dialogue flows naturally as fast as human thought.',
      icon: Zap,
      color: 'cyan',
      statNumber: '<20ms',
      statLabel: 'Neural landmark inference latency',
      highlights: [
        'Continuous gesture streaming without sentence halting',
        'No roundtrip lag to remote cloud server farms',
        'Minimal battery and processing footprint on phones',
      ],
    },
    {
      title: 'Inclusive Indian Technology',
      subtitle: 'Honoring regional ISL dialects',
      description:
        'Unlike generic Western ASL models, SIGNBRIDGE AI is purposefully trained on Indian Sign Language grammar, regional nuances across states, and culturally resonant everyday gestures.',
      icon: Globe2,
      color: 'violet',
      statNumber: 'Pan-India',
      statLabel: 'Regional vocabulary & dialect awareness',
      highlights: [
        'ISLRTC standardized grammar and vocabulary alignment',
        'Support for Hindi, English, and regional dialect glosses',
        'Community-driven annotation pipeline for new slang and terms',
      ],
    },
  ]

  return (
    <section id="why-us" className="relative py-16 sm:py-24 lg:py-32 bg-[#060a17] overflow-hidden">
      {/* Background glow and subtle borders */}
      <div className="absolute inset-0 bg-grid-pattern opacity-25 pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-gradient-to-tr from-cyan-600/10 via-violet-600/10 to-transparent rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <Badge variant="violet" size="md" dot>
            Our Mission & Impact
          </Badge>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight">
            Why SIGNBRIDGE AI
          </h2>
          <p className="text-base sm:text-lg text-slate-300">
            More than just an AI model—a dedicated mission to guarantee equal conversational dignity,
            independence, and educational opportunities for India's Deaf community.
          </p>
        </div>

        {/* 4 Pillars Grid */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-8">
          {pillars.map((pillar, idx) => {
            const Icon = pillar.icon
            const isCyan = pillar.color === 'cyan'

            return (
              <div
                key={idx}
                className="group relative rounded-3xl p-5 sm:p-8 bg-gradient-to-b from-[#0b132b]/80 to-[#070c1e]/90 border border-white/10 hover:border-cyan-500/40 transition-all duration-300 shadow-xl shadow-black/40 backdrop-blur-xl flex flex-col justify-between"
              >
                <div>
                  {/* Top Bar with Icon and Stats */}
                  <div className="flex items-start justify-between mb-6">
                    <div
                      className={`w-14 h-14 rounded-2xl flex items-center justify-center border shadow-md ${isCyan
                          ? 'bg-cyan-500/15 border-cyan-400/30 text-cyan-400 group-hover:scale-105 transition-transform'
                          : 'bg-violet-500/15 border-violet-400/30 text-violet-400 group-hover:scale-105 transition-transform'
                        }`}
                    >
                      <Icon className="w-7 h-7" />
                    </div>

                    <div className="text-right">
                      <div
                        className={`text-2xl sm:text-3xl font-extrabold font-mono ${isCyan ? 'text-cyan-300' : 'text-violet-300'
                          }`}
                      >
                        {pillar.statNumber}
                      </div>
                      <div className="text-[11px] text-slate-400 font-medium">
                        {pillar.statLabel}
                      </div>
                    </div>
                  </div>

                  {/* Title and subtitle */}
                  <div className="space-y-1">
                    <h3 className="text-xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                      {pillar.title}
                    </h3>
                    <div className="text-xs font-mono text-cyan-400/80 uppercase tracking-wide">
                      {pillar.subtitle}
                    </div>
                  </div>

                  {/* Description */}
                  <p className="mt-4 text-sm text-slate-300/90 leading-relaxed">
                    {pillar.description}
                  </p>

                  {/* Highlights list */}
                  <div className="mt-6 pt-5 border-t border-white/5 space-y-2.5">
                    {pillar.highlights.map((item, hIdx) => (
                      <div key={hIdx} className="flex items-center gap-2 text-xs text-slate-300">
                        <CheckCircle
                          className={`w-4 h-4 shrink-0 ${isCyan ? 'text-cyan-400' : 'text-violet-400'
                            }`}
                        />
                        <span>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Bottom trust footer */}
                <div className="mt-8 pt-4 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span className="flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                    Community Validated
                  </span>
                  <span className="text-slate-400 font-medium">ISL Standardized</span>
                </div>
              </div>
            )
          })}
        </div>

        {/* Inclusion Banner Quote */}
        <div className="mt-14 rounded-2xl bg-gradient-to-r from-cyan-950/30 via-slate-900/60 to-violet-950/30 border border-cyan-500/20 p-8 text-center max-w-4xl mx-auto backdrop-blur-xl space-y-3">
          <Sparkles className="w-6 h-6 text-cyan-400 mx-auto" />
          <blockquote className="text-lg sm:text-xl font-medium text-slate-100 italic">
            "Communication is not a privilege; it is a fundamental human right. SIGNBRIDGE AI
            transforms visual Indian Sign Language into an open bridge where everyone is heard and
            understood."
          </blockquote>
          <div className="text-xs font-mono text-cyan-400 pt-2 uppercase tracking-wider">
            SignBridge AI Accessibility & Research Initiative
          </div>
        </div>
      </div>
    </section>
  )
}
