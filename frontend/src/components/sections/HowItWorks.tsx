import React, { useState } from 'react'
import {
  Camera,
  Cpu,
  Volume2,
  CheckCircle2,
  Zap,
  Activity,
} from 'lucide-react'
import { Badge } from '../common/Badge'

export const HowItWorks: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0)

  const steps = [
    {
      number: '01',
      title: 'Capture Sign',
      badge: 'Vision Stream',
      icon: Camera,
      color: 'cyan',
      description:
        'Using any standard smartphone camera or laptop webcam, the system captures continuous dynamic ISL gestures, hand shapes, and facial articulation at 60 FPS.',
      points: [
        'Real-time 21-point 3D hand landmark coordinate extraction',
        'Dual-hand coordination and spatial orientation tracking',
        'Facial expression and non-manual marker detection for grammar',
      ],
      techNote: 'MediaPipe Vision + Custom High-Frequency Landmark Extractor',
    },
    {
      number: '02',
      title: 'AI Recognition',
      badge: 'Neural Inference',
      icon: Cpu,
      color: 'violet',
      description:
        'Our state-of-the-art Spatial-Temporal Graph Convolutional Network (ST-GCN) and transformer language models map gesture trajectories against rich ISL corpora.',
      points: [
        'Temporal sequence modeling for continuous continuous sign flow',
        'Context-aware grammar mapping tuned specifically for Indian Sign Language',
        'Sub-20ms edge-optimized inference without cloud latency',
      ],
      techNote: 'Edge-Quantized ST-GCN + Indian Linguistic Attention Models',
    },
    {
      number: '03',
      title: 'Text & Speech Output',
      badge: 'Synthesis & Audio',
      icon: Volume2,
      color: 'cyan',
      description:
        'Recognized signs are instantly structured into grammatically coherent sentences, displayed as readable text, and voiced through natural speech synthesis.',
      points: [
        'Instant multi-lingual translation into English, Hindi & regional languages',
        'Expressive neural speech synthesis with customizable accents and cadence',
        'Bidirectional bridge: Spoken answers are translated back into 3D ISL signs',
      ],
      techNote: 'Neural Waveform Synthesizer + Multi-Dialect Indian Voice Engine',
    },
  ]

  return (
    <section id="how-it-works" className="relative py-16 sm:py-24 lg:py-32 bg-[#060a17]/90 border-t border-b border-cyan-500/10 overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 bg-grid-pattern opacity-20 pointer-events-none" />
      <div className="absolute top-1/2 left-0 w-80 h-80 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <Badge variant="cyan" size="md" dot>
            Seamless Pipeline
          </Badge>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight">
            How SignBridge AI Works
          </h2>
          <p className="text-base sm:text-lg text-slate-300">
            A frictionless, three-stage neural architecture transforming visual sign gestures into
            crystal-clear speech and text in real-time.
          </p>
        </div>

        {/* Step Indicator Tabs for Mobile & Desktop */}
        <div className="mt-14 grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {steps.map((item, index) => {
            const Icon = item.icon
            const isSelected = activeStep === index
            const isCyan = item.color === 'cyan'

            return (
              <div
                key={item.number}
                onClick={() => setActiveStep(index)}
                className={`group relative rounded-2xl p-5 sm:p-7 transition-all duration-300 cursor-pointer border ${
                  isSelected
                    ? isCyan
                      ? 'bg-gradient-to-b from-[#0c1630] to-[#070d20] border-cyan-400/60 shadow-xl shadow-cyan-950/70 md:scale-[1.02]'
                      : 'bg-gradient-to-b from-[#15102b] to-[#0a071c] border-violet-400/60 shadow-xl shadow-violet-950/70 md:scale-[1.02]'
                    : 'bg-[#090f23]/60 border-white/5 hover:border-slate-700 hover:bg-[#0c142e]/80'
                }`}
              >
                {/* Step numbering & Icon banner */}
                <div className="flex items-center justify-between mb-6">
                  <div
                    className={`flex items-center justify-center w-12 h-12 rounded-xl border ${
                      isSelected
                        ? isCyan
                          ? 'bg-cyan-500/20 border-cyan-400/40 text-cyan-300'
                          : 'bg-violet-500/20 border-violet-400/40 text-violet-300'
                        : 'bg-slate-800/60 border-slate-700/60 text-slate-400 group-hover:text-slate-200'
                    }`}
                  >
                    <Icon className="w-6 h-6" />
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-2xl font-black font-mono ${
                        isSelected
                          ? isCyan
                            ? 'text-cyan-400'
                            : 'text-violet-400'
                          : 'text-slate-600'
                      }`}
                    >
                      {item.number}
                    </span>
                  </div>
                </div>

                {/* Step Title & Badge */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <h3 className="text-xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                      {item.title}
                    </h3>
                  </div>
                  <p className="text-sm text-slate-300/90 leading-relaxed min-h-[4rem]">
                    {item.description}
                  </p>
                </div>

                {/* Sub points */}
                <ul className="mt-5 space-y-2.5 pt-4 border-t border-white/5">
                  {item.points.map((pt, pIdx) => (
                    <li key={pIdx} className="flex items-start gap-2 text-xs text-slate-300">
                      <CheckCircle2
                        className={`w-3.5 h-3.5 mt-0.5 shrink-0 ${
                          isSelected
                            ? isCyan
                              ? 'text-cyan-400'
                              : 'text-violet-400'
                            : 'text-slate-500'
                        }`}
                      />
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>

                {/* Technical Architecture Footer */}
                <div className="mt-6 pt-3 border-t border-white/5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span className="flex items-center gap-1">
                    <Activity className="w-3 h-3 text-cyan-400" />
                    Pipeline Component
                  </span>
                  <span className="text-slate-300 font-medium">Stage {item.number}</span>
                </div>
              </div>
            )
          })}
        </div>

        {/* Live Simulation Banner across the 3 stages */}
        <div className="mt-12 rounded-2xl bg-gradient-to-r from-cyan-950/40 via-slate-900/60 to-violet-950/40 border border-cyan-500/20 p-5 sm:p-6 backdrop-blur-xl flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-400/40 flex items-center justify-center shrink-0">
              <Zap className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white">
                Ultra-Low Latency Continuous Translation Loop
              </div>
              <div className="text-xs text-slate-400">
                Sign language video in &rarr; 3D landmark mesh &rarr; Phonetic syntax &rarr; Natural regional speech out in under 20 milliseconds.
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <Badge variant="cyan" size="sm" className="font-mono">
              60 FPS SYNC
            </Badge>
            <Badge variant="violet" size="sm" className="font-mono">
              OFFLINE READY
            </Badge>
          </div>
        </div>
      </div>
    </section>
  )
}
