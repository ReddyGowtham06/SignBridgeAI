import React, { useState } from 'react'
import {
  Sparkles,
  Volume2,
  Cpu,
  Activity,
  Layers,
  ChevronRight,
  ShieldCheck,
  Zap,
} from 'lucide-react'
import { Badge } from '../common/Badge'

interface GestureSample {
  id: string
  name: string
  gloss: string
  english: string
  hindi: string
  confidence: number
  latency: number
  landmarks: { x: number; y: number }[]
  category: string
}

const SAMPLE_GESTURES: GestureSample[] = [
  {
    id: 'namaste',
    name: 'Namaste',
    gloss: 'NAMASTE [FORMAL]',
    english: 'Greetings / Welcome',
    hindi: 'नमस्ते (प्रणाम)',
    confidence: 99.4,
    latency: 18,
    category: 'Greeting',
    landmarks: [
      { x: 50, y: 82 }, // Wrist
      // Thumb
      { x: 42, y: 72 },
      { x: 38, y: 62 },
      { x: 36, y: 52 },
      { x: 38, y: 44 },
      // Index
      { x: 46, y: 52 },
      { x: 45, y: 40 },
      { x: 45, y: 30 },
      { x: 46, y: 22 },
      // Middle
      { x: 50, y: 50 },
      { x: 50, y: 36 },
      { x: 50, y: 24 },
      { x: 50, y: 16 },
      // Ring
      { x: 54, y: 52 },
      { x: 55, y: 40 },
      { x: 55, y: 30 },
      { x: 54, y: 22 },
      // Pinky
      { x: 58, y: 56 },
      { x: 62, y: 46 },
      { x: 64, y: 38 },
      { x: 64, y: 30 },
    ],
  },
  {
    id: 'thank-you',
    name: 'Thank You',
    gloss: 'THANK-YOU [CHIN-FORWARD]',
    english: 'Thank You / Gratitude',
    hindi: 'धन्यवाद (आभार)',
    confidence: 98.7,
    latency: 16,
    category: 'Courtesy',
    landmarks: [
      { x: 50, y: 78 },
      { x: 40, y: 68 },
      { x: 35, y: 58 },
      { x: 32, y: 50 },
      { x: 34, y: 42 },
      { x: 46, y: 48 },
      { x: 48, y: 36 },
      { x: 52, y: 28 },
      { x: 56, y: 22 },
      { x: 52, y: 48 },
      { x: 56, y: 36 },
      { x: 60, y: 28 },
      { x: 64, y: 22 },
      { x: 56, y: 52 },
      { x: 60, y: 42 },
      { x: 64, y: 34 },
      { x: 68, y: 28 },
      { x: 60, y: 58 },
      { x: 64, y: 50 },
      { x: 68, y: 42 },
      { x: 72, y: 36 },
    ],
  },
  {
    id: 'how-are-you',
    name: 'How Are You',
    gloss: 'YOU [INDEX] + HEALTH-HOW',
    english: 'How are you?',
    hindi: 'आप कैसे हैं?',
    confidence: 97.9,
    latency: 21,
    category: 'Query',
    landmarks: [
      { x: 48, y: 80 },
      { x: 38, y: 70 },
      { x: 32, y: 60 },
      { x: 28, y: 52 },
      { x: 26, y: 44 },
      { x: 44, y: 48 },
      { x: 42, y: 34 },
      { x: 40, y: 22 },
      { x: 38, y: 14 },
      { x: 50, y: 52 },
      { x: 52, y: 40 },
      { x: 52, y: 32 },
      { x: 52, y: 25 },
      { x: 56, y: 54 },
      { x: 58, y: 44 },
      { x: 60, y: 36 },
      { x: 60, y: 30 },
      { x: 62, y: 58 },
      { x: 66, y: 50 },
      { x: 68, y: 44 },
      { x: 70, y: 38 },
    ],
  },
]

export const GestureMockPanel: React.FC = () => {
  const [selectedSign, setSelectedSign] = useState<GestureSample>(SAMPLE_GESTURES[0])
  const [isPlayingAudio, setIsPlayingAudio] = useState(false)

  const handleSimulateAudio = () => {
    setIsPlayingAudio(true)
    setTimeout(() => setIsPlayingAudio(false), 1800)
  }

  // Skeletal connection pairs (bone indices)
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4], // thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // index
    [0, 9], [9, 10], [10, 11], [11, 12], // middle
    [0, 13], [13, 14], [14, 15], [15, 16], // ring
    [0, 17], [17, 18], [18, 19], [19, 20], // pinky
    [5, 9], [9, 13], [13, 17], // palm base
  ]

  return (
    <div className="relative w-full max-w-xl mx-auto rounded-3xl p-1 bg-gradient-to-b from-cyan-500/30 via-violet-500/20 to-transparent shadow-2xl shadow-cyan-950/60">
      {/* Outer ambient glow */}
      <div className="absolute -inset-1 rounded-3xl bg-gradient-to-r from-cyan-500/20 to-violet-600/20 blur-xl opacity-75 -z-10" />

      <div className="rounded-[22px] bg-[#070c1d] border border-cyan-500/20 overflow-hidden backdrop-blur-2xl">
        {/* Top Camera HUD Control Bar */}
        <div className="flex items-center justify-between px-4 py-3 bg-[#0a122c]/90 border-b border-cyan-500/15">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-950/80 border border-emerald-500/30 text-emerald-300 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>ISL-CAM 1080P // 60FPS</span>
            </div>
            <span className="hidden sm:inline-block text-[11px] font-mono text-slate-400">
              POSE-NET 21.0
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="cyan" size="sm" className="font-mono text-[10px]">
              <Cpu className="w-3 h-3 text-cyan-400" />
              {selectedSign.latency}ms LATENCY
            </Badge>
            <div className="hidden sm:flex items-center gap-1 px-2 py-0.5 rounded bg-slate-800/60 text-slate-300 text-[10px] font-mono">
              <ShieldCheck className="w-3 h-3 text-cyan-400" />
              ON-DEVICE
            </div>
          </div>
        </div>

        {/* Viewport Frame with Simulated Hand Landmarks */}
        <div className="relative aspect-[4/3] sm:aspect-[16/11] bg-gradient-to-b from-[#060b1e] via-[#091129] to-[#040817] overflow-hidden flex items-center justify-center p-6">
          {/* Subtle grid backdrop */}
          <div className="absolute inset-0 bg-grid-pattern opacity-40 pointer-events-none" />

          {/* Corner Crosshair HUD Elements */}
          <div className="absolute top-4 left-4 w-5 h-5 border-t-2 border-l-2 border-cyan-400/70" />
          <div className="absolute top-4 right-4 w-5 h-5 border-t-2 border-r-2 border-cyan-400/70" />
          <div className="absolute bottom-4 left-4 w-5 h-5 border-b-2 border-l-2 border-cyan-400/70" />
          <div className="absolute bottom-4 right-4 w-5 h-5 border-b-2 border-r-2 border-cyan-400/70" />

          {/* Center Target Box */}
          <div className="absolute inset-10 border border-dashed border-cyan-500/20 rounded-2xl pointer-events-none flex items-start justify-end p-2">
            <span className="text-[10px] font-mono text-cyan-500/60 uppercase tracking-widest">
              ISL REGION OF INTEREST [ROI]
            </span>
          </div>

          {/* Animated Horizontal Scan Beam */}
          <div className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400/70 to-transparent shadow-[0_0_15px_#06b6d4] animate-scanline pointer-events-none" />

          {/* Simulated 21-Point Hand Skeletal Mesh */}
          <svg className="w-full h-full max-w-[320px] max-h-[320px] relative z-10 transition-all duration-500 filter drop-shadow-[0_0_12px_rgba(6,182,212,0.6)]">
            <defs>
              <linearGradient id="bone-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#06B6D4" stop-opacity="0.9" />
                <stop offset="100%" stop-color="#A855F7" stop-opacity="0.8" />
              </linearGradient>
            </defs>

            {/* Bone vectors */}
            {connections.map(([start, end], idx) => {
              const p1 = selectedSign.landmarks[start]
              const p2 = selectedSign.landmarks[end]
              if (!p1 || !p2) return null
              return (
                <line
                  key={`bone-${idx}`}
                  x1={`${p1.x}%`}
                  y1={`${p1.y}%`}
                  x2={`${p2.x}%`}
                  y2={`${p2.y}%`}
                  stroke="url(#bone-grad)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  className="transition-all duration-500"
                />
              )
            })}

            {/* Landmark Joint Nodes */}
            {selectedSign.landmarks.map((point, index) => {
              const isFingertip = [4, 8, 12, 16, 20].includes(index)
              const isWrist = index === 0

              return (
                <g key={`point-${index}`} className="transition-all duration-500">
                  {/* Outer pulse for key nodes */}
                  {(isFingertip || isWrist) && (
                    <circle
                      cx={`${point.x}%`}
                      cy={`${point.y}%`}
                      r={isWrist ? '7' : '6'}
                      fill="none"
                      stroke={isFingertip ? '#00f0ff' : '#c084fc'}
                      strokeWidth="1.5"
                      className="animate-ping opacity-60"
                    />
                  )}
                  {/* Core Joint Circle */}
                  <circle
                    cx={`${point.x}%`}
                    cy={`${point.y}%`}
                    r={isWrist ? '5' : isFingertip ? '4.5' : '3'}
                    fill={isFingertip ? '#00f0ff' : isWrist ? '#a855f7' : '#38bdf8'}
                    stroke="#050814"
                    strokeWidth="1.5"
                  />
                </g>
              )
            })}
          </svg>

          {/* Floating Confidence Float Tag inside viewport */}
          <div className="absolute bottom-5 left-5 z-20 flex items-center gap-2 bg-[#05091c]/90 border border-cyan-500/40 rounded-xl px-3 py-2 shadow-lg backdrop-blur-md">
            <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-400 font-mono leading-none">
                Confidence Match
              </span>
              <span className="text-sm font-bold text-cyan-300 font-mono">
                {selectedSign.confidence}%
              </span>
            </div>
          </div>

          <div className="absolute top-5 right-5 z-20 flex items-center gap-1.5 bg-[#05091c]/80 border border-violet-500/30 rounded-lg px-2.5 py-1 text-[11px] font-mono text-violet-300 backdrop-blur-md">
            <Layers className="w-3 h-3 text-violet-400" />
            21 Landmark Nodes
          </div>
        </div>

        {/* Live Translation Output Card */}
        <div className="p-4 sm:p-5 bg-[#091026] border-t border-cyan-500/15 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#060a19] p-3.5 rounded-xl border border-white/5">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Badge variant="cyan" size="sm">
                  {selectedSign.category}
                </Badge>
                <span className="text-[11px] font-mono text-slate-400">
                  {selectedSign.gloss}
                </span>
              </div>
              <div className="flex items-baseline gap-2.5 pt-0.5">
                <span className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                  "{selectedSign.english}"
                </span>
                <span className="text-base text-cyan-300 font-medium">
                  • {selectedSign.hindi}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={handleSimulateAudio}
              className={`flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-xl text-xs font-semibold transition-all border shrink-0 ${
                isPlayingAudio
                  ? 'bg-cyan-400 text-slate-950 border-cyan-300 shadow-md shadow-cyan-400/40'
                  : 'bg-slate-800/80 text-cyan-300 border-cyan-500/30 hover:bg-slate-800 hover:border-cyan-400'
              }`}
            >
              <Volume2 className={`w-4 h-4 ${isPlayingAudio ? 'animate-bounce' : ''}`} />
              <span>{isPlayingAudio ? 'Speaking...' : 'Play Audio'}</span>
            </button>
          </div>

          {/* Interactive Sign Test Switcher */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400 flex items-center gap-1.5 font-medium">
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                Select simulated ISL sign to preview recognition:
              </span>
              <span className="text-[11px] text-cyan-400/80 font-mono">Interactive Mock</span>
            </div>

            <div className="grid grid-cols-3 gap-2">
              {SAMPLE_GESTURES.map((item) => {
                const isActive = item.id === selectedSign.id
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setSelectedSign(item)}
                    className={`flex items-center justify-between p-2.5 rounded-xl text-left border transition-all ${
                      isActive
                        ? 'bg-gradient-to-r from-cyan-950/70 to-slate-900 border-cyan-400/60 shadow-md shadow-cyan-500/20 text-white'
                        : 'bg-slate-900/40 border-white/5 hover:border-slate-700 hover:bg-slate-800/40 text-slate-300'
                    }`}
                  >
                    <div className="truncate">
                      <div className="text-xs font-semibold truncate">{item.name}</div>
                      <div className="text-[10px] text-slate-400 font-mono truncate">
                        {item.hindi}
                      </div>
                    </div>
                    {isActive && <ChevronRight className="w-3.5 h-3.5 text-cyan-400 shrink-0 ml-1" />}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Bottom Status Banner */}
        <div className="px-4 py-2 bg-[#050917] border-t border-white/5 flex items-center justify-between text-[11px] text-slate-400 font-mono">
          <span className="flex items-center gap-1.5 text-slate-400">
            <Sparkles className="w-3 h-3 text-cyan-400" />
            Designed Landmark Visualizer • Ready for Live Neural Stream
          </span>
          <span className="text-cyan-400">v1.0-ISL</span>
        </div>
      </div>
    </div>
  )
}

