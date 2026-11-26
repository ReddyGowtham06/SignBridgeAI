import React, { useState } from 'react'
import {
  Video,
  Volume2,
  GraduationCap,
  ArrowRight,
  Check,
  Bot,
  Activity,
} from 'lucide-react'
import { Badge } from '../common/Badge'
import { Button } from '../common/Button'

export const Features: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'sign-to-text' | 'text-to-speech' | 'text-to-sign' | 'learning'>('sign-to-text')

  const featuresList = [
    {
      id: 'sign-to-text',
      title: 'Sign to Text',
      category: 'Vision Intelligence',
      badge: 'Core Engine',
      icon: Video,
      color: 'cyan',
      headline: 'Instant, Continuous ISL Video Translation',
      description:
        'Transform continuous Indian Sign Language gestures captured from your webcam into structured, readable sentences. Our model accurately decodes dynamic hand movements, fingerspelling, and facial grammar.',
      highlights: [
        'Recognizes dynamic multi-word phrases and rapid fingerspelling',
        'Decodes non-manual facial cues like raised eyebrows and head tilts',
        'Maintains high accuracy even under challenging room lighting',
        'Edge-native processing ensures full conversational privacy',
      ],
      interactiveDemo: {
        title: 'Sign-to-Text Architecture',
        previewBadge: 'Live Landmark Stream',
        inputLabel: 'Continuous ISL Stream',
        inputVal: '[HANDS: DUAL CHEST ELEVATION] + [FACIAL: AFFIRMATIVE]',
        outputLabel: 'Translated English & Hindi Output',
        outputVal: '"I am glad to meet you today." • "मुझे आज आपसे मिलकर खुशी हुई।"',
        confidence: '99.4%',
      },
    },
    {
      id: 'text-to-speech',
      title: 'Text to Speech',
      category: 'Audio Synthesis',
      badge: 'Multi-Lingual',
      icon: Volume2,
      color: 'violet',
      headline: 'Expressive Neural Voice in Indian Languages',
      description:
        'Instantly speak the translated ISL sentences aloud with natural inflection and regional pronunciation. Breaks communication walls in shops, clinics, offices, and universities.',
      highlights: [
        'Native Indian English, Hindi, Tamil, Telugu, Bengali & Marathi voices',
        'Emotionally expressive prosody matches the intensity of the sign',
        'Adjustable playback speeds for comfortable conversation cadence',
        'Seamless integration with screen readers and accessibility peripherals',
      ],
      interactiveDemo: {
        title: 'Neural Speech Synthesizer',
        previewBadge: 'Acoustic Model v4',
        inputLabel: 'Target Text Sentence',
        inputVal: '"Where is the nearest metro station?"',
        outputLabel: 'Synthesized Audio Waveform',
        outputVal: '🔊 [Playing clear Hindi & Indian-English voice with natural intonation]',
        confidence: '12ms Synthesized',
      },
    },
    {
      id: 'text-to-sign',
      title: 'Text to Sign',
      category: 'Bidirectional Engine',
      badge: '3D Avatar',
      icon: Bot,
      color: 'cyan',
      headline: 'Bidirectional Speech-to-ISL 3D Animation',
      description:
        'The hearing partner speaks or types, and SIGNBRIDGE AI instantly animates a photorealistic 3D avatar performing standardized ISL signs with authentic facial cues.',
      highlights: [
        'Continuous two-way conversation without needing human interpreters',
        'Photorealistic 3D avatar with biomechanically accurate joint limits',
        'Translates spoken words and text idioms directly into ISL syntax',
        'Customizable avatar speed, angle rotation, and signing emphasis',
      ],
      interactiveDemo: {
        title: '3D Avatar Sign Generator',
        previewBadge: 'Real-time Skeletal Rig',
        inputLabel: 'Spoken or Typed Sentence',
        inputVal: '"Welcome to the national science conference."',
        outputLabel: 'Generated ISL Avatar Sequence',
        outputVal: '🕺 [Rendering ISL avatar: WELCOME + SCIENCE + CONFERENCE [ANNUAL]]',
        confidence: '60 FPS Render',
      },
    },
    {
      id: 'learning',
      title: 'ISL Learning Studio',
      category: 'Education & Practice',
      badge: 'Interactive Hub',
      icon: GraduationCap,
      color: 'violet',
      headline: 'Gamified Indian Sign Language Learning',
      description:
        'Master Indian Sign Language with structured lessons, interactive drills, and instant computer vision feedback. Perfect for students, teachers, parents, and healthcare professionals.',
      highlights: [
        'Curated curriculum spanning 2,500+ standard signs and fingerspelling',
        'Real-time webcam posture verification with instant feedback on finger positioning',
        'Daily streaks, flashcards, quizzes, and skill progress tracking',
        'Official ISLRTC (Indian Sign Language Research & Training Centre) aligned',
      ],
      interactiveDemo: {
        title: 'Interactive Practice Mode',
        previewBadge: 'Posture Assessment',
        inputLabel: 'Learning Module Target',
        inputVal: 'Sign the letter "K" in Indian Sign Language',
        outputLabel: 'AI Vision Evaluation',
        outputVal: '✅ "Thumb position correct! Index finger angle 94% accurate."',
        confidence: 'Grade: A+ (98%)',
      },
    },
  ]

  const currentFeature = featuresList.find((f) => f.id === activeTab) || featuresList[0]

  return (
    <section id="features" className="relative py-24 sm:py-32 bg-[#050814]">
      {/* Background Glow */}
      <div className="absolute top-1/3 right-10 w-96 h-96 bg-cyan-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 left-10 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <Badge variant="cyan" size="md" dot>
            Next-Gen Capabilities
          </Badge>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight">
            Designed for Universal Connection
          </h2>
          <p className="text-base sm:text-lg text-slate-300">
            Four powerful pillars that transform how Indian Sign Language is translated, spoken,
            visualized, and learned across India.
          </p>
        </div>

        {/* Feature Navigation Tabs */}
        <div className="mt-12 flex flex-wrap justify-center gap-2 sm:gap-3 p-1.5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-xl max-w-3xl mx-auto">
          {featuresList.map((f) => {
            const Icon = f.icon
            const isActive = activeTab === f.id
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => setActiveTab(f.id as any)}
                className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 cursor-pointer ${isActive
                    ? 'bg-gradient-to-r from-cyan-500 to-cyan-400 text-slate-950 shadow-md shadow-cyan-500/20'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
                  }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-slate-950' : 'text-cyan-400'}`} />
                <span>{f.title}</span>
              </button>
            )
          })}
        </div>

        {/* Detailed Feature Showcase Container */}
        <div className="mt-10 rounded-3xl p-1 bg-gradient-to-b from-cyan-500/20 via-violet-500/10 to-transparent border border-cyan-500/20 backdrop-blur-2xl">
          <div className="rounded-[22px] bg-[#070d22] p-6 sm:p-10 lg:p-12">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
              {/* Feature Text details */}
              <div className="lg:col-span-7 space-y-6">
                <div className="flex items-center gap-2.5">
                  <Badge variant={currentFeature.color as any} size="md">
                    {currentFeature.category}
                  </Badge>
                  <span className="text-xs font-mono text-slate-400">{currentFeature.badge}</span>
                </div>

                <h3 className="text-2xl sm:text-3xl font-extrabold text-white leading-tight">
                  {currentFeature.headline}
                </h3>

                <p className="text-slate-300 text-base leading-relaxed">
                  {currentFeature.description}
                </p>

                {/* Highlights List */}
                <div className="space-y-3 pt-2">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-cyan-400">
                    Key Technological Advantages
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {currentFeature.highlights.map((h, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-900/60 border border-white/5"
                      >
                        <div className="p-1 rounded-full bg-cyan-500/20 text-cyan-300 shrink-0 mt-0.5">
                          <Check className="w-3.5 h-3.5" />
                        </div>
                        <span className="text-xs text-slate-200 leading-snug">{h}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    icon={<ArrowRight className="w-4 h-4" />}
                    iconPosition="right"
                    onClick={() => {
                      const el = document.getElementById('why-us')
                      el?.scrollIntoView({ behavior: 'smooth' })
                    }}
                  >
                    Learn More About {currentFeature.title}
                  </Button>
                </div>
              </div>

              {/* Feature Interactive Visualization Box */}
              <div className="lg:col-span-5">
                <div className="rounded-2xl bg-gradient-to-b from-[#0a122e] to-[#060a1a] border border-cyan-500/30 p-6 shadow-2xl relative overflow-hidden">
                  {/* Decorative background grid */}
                  <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

                  <div className="relative z-10 space-y-5">
                    {/* Header */}
                    <div className="flex items-center justify-between pb-3 border-b border-cyan-500/20">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                          {currentFeature.interactiveDemo.title}
                        </span>
                      </div>
                      <Badge variant="cyan" size="sm" className="font-mono text-[10px]">
                        {currentFeature.interactiveDemo.previewBadge}
                      </Badge>
                    </div>

                    {/* Input Block */}
                    <div className="space-y-1.5 bg-[#050814]/90 p-3.5 rounded-xl border border-white/5">
                      <span className="text-[11px] font-mono text-slate-400">
                        {currentFeature.interactiveDemo.inputLabel}
                      </span>
                      <div className="text-sm font-mono text-cyan-300 font-medium">
                        {currentFeature.interactiveDemo.inputVal}
                      </div>
                    </div>

                    {/* Flow arrow indicator */}
                    <div className="flex justify-center -my-2">
                      <div className="w-7 h-7 rounded-full bg-cyan-950 border border-cyan-500/40 flex items-center justify-center text-cyan-300 shadow-sm">
                        <ArrowRight className="w-3.5 h-3.5 rotate-90" />
                      </div>
                    </div>

                    {/* Output Block */}
                    <div className="space-y-1.5 bg-gradient-to-r from-cyan-950/40 to-violet-950/40 p-4 rounded-xl border border-cyan-500/30">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-mono text-slate-300">
                          {currentFeature.interactiveDemo.outputLabel}
                        </span>
                        <Badge variant="cyan" size="sm" className="text-[9px]">
                          {currentFeature.interactiveDemo.confidence}
                        </Badge>
                      </div>
                      <div className="text-sm font-semibold text-white pt-1">
                        {currentFeature.interactiveDemo.outputVal}
                      </div>
                    </div>

                    {/* Live metrics footer */}
                    <div className="pt-2 flex items-center justify-between text-[11px] font-mono text-slate-400 border-t border-white/5">
                      <span>SIGNBRIDGE Core Pipeline</span>
                      <span className="text-cyan-400">Active Architecture</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 4 Feature Cards Grid Summary */}
        <div className="mt-14 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {featuresList.map((item) => {
            const Icon = item.icon
            const isSelected = activeTab === item.id
            return (
              <div
                key={item.id}
                onClick={() => setActiveTab(item.id as any)}
                className={`p-6 rounded-2xl border transition-all duration-300 cursor-pointer ${isSelected
                    ? 'bg-gradient-to-b from-[#0c1630] to-[#070e22] border-cyan-400/50 shadow-lg shadow-cyan-950/40'
                    : 'bg-[#090f24]/50 border-white/5 hover:border-slate-700 hover:bg-[#0c142e]/70'
                  }`}
              >
                <div className="w-10 h-10 rounded-xl bg-slate-800/60 border border-slate-700/60 flex items-center justify-center mb-4 text-cyan-400">
                  <Icon className="w-5 h-5" />
                </div>
                <h4 className="text-base font-bold text-white mb-2">{item.title}</h4>
                <p className="text-xs text-slate-300 line-clamp-3 leading-relaxed">
                  {item.description}
                </p>
                <div className="mt-4 pt-3 border-t border-white/5 flex items-center gap-1.5 text-xs text-cyan-400 font-medium">
                  <span>View Details</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
