import React, { useState } from 'react'
import {
  GraduationCap,
  Sparkles,
  Award,
  BookOpen,
  CheckCircle2,
  Play,
  RotateCcw,
  Video,
} from 'lucide-react'
import { Badge } from '../common/Badge'
import { Button } from '../common/Button'

interface Flashcard {
  letter: string
  hindiChar: string
  hint: string
  difficulty: 'Beginner' | 'Intermediate'
}

const ALPHABET_CARDS: Flashcard[] = [
  { letter: 'A', hindiChar: 'अ', hint: 'Fist with thumb resting beside index finger', difficulty: 'Beginner' },
  { letter: 'B', hindiChar: 'ब', hint: 'Flat open palm with thumb tucked across palm', difficulty: 'Beginner' },
  { letter: 'C', hindiChar: 'स', hint: 'Curved hand forming a C shape', difficulty: 'Beginner' },
  { letter: 'D', hindiChar: 'ड', hint: 'Index finger pointing upward, others touching thumb', difficulty: 'Intermediate' },
  { letter: 'E', hindiChar: 'इ', hint: 'Fingers curled inward resting on thumb', difficulty: 'Intermediate' },
  { letter: 'L', hindiChar: 'ल', hint: 'Index finger and thumb forming right angle', difficulty: 'Beginner' },
]

export const LearningStudio: React.FC = () => {
  const [selectedCard, setSelectedCard] = useState<Flashcard>(ALPHABET_CARDS[0])
  const [practiceActive, setPracticeActive] = useState(false)
  const [score, setScore] = useState<number | null>(null)

  const handleStartPractice = () => {
    setPracticeActive(true)
    setScore(null)
    setTimeout(() => {
      setScore(96)
      setPracticeActive(false)
    }, 1500)
  }

  return (
    <section id="learning" className="relative py-16 sm:py-24 lg:py-32 bg-[#050814] border-t border-cyan-500/10 overflow-hidden">
      <div className="absolute top-1/2 right-1/4 w-80 h-80 bg-violet-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <Badge variant="violet" size="md" dot>
            Interactive Academy
          </Badge>
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight">
            ISL Learning Studio
          </h2>
          <p className="text-base sm:text-lg text-slate-300">
            Interactive, gamified Indian Sign Language courses equipped with real-time AI computer
            vision posture validation and instant feedback.
          </p>
        </div>

        {/* Studio Content Grid */}
        <div className="mt-14 grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left: Card Selection */}
          <div className="lg:col-span-5 space-y-5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                Select an ISL Character to Practice:
              </span>
              <Badge variant="cyan" size="sm">
                ISL Manual Alphabet
              </Badge>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {ALPHABET_CARDS.map((card) => {
                const isSelected = selectedCard.letter === card.letter
                return (
                  <button
                    key={card.letter}
                    type="button"
                    onClick={() => {
                      setSelectedCard(card)
                      setScore(null)
                    }}
                    className={`p-3 sm:p-4 min-h-[44px] rounded-2xl border text-center transition-all cursor-pointer ${isSelected
                        ? 'bg-gradient-to-b from-cyan-950/80 to-[#070f26] border-cyan-400/70 shadow-lg shadow-cyan-500/20 text-white md:scale-[1.02]'
                        : 'bg-slate-900/50 border-white/5 hover:border-slate-700 text-slate-300'
                      }`}
                  >
                    <div className="text-3xl font-black font-mono tracking-tight text-white">
                      {card.letter}
                    </div>
                    <div className="text-xs font-semibold text-cyan-300 mt-1">
                      {card.hindiChar}
                    </div>
                    <div className="text-[10px] text-slate-400 font-mono mt-1">
                      {card.difficulty}
                    </div>
                  </button>
                )
              })}
            </div>

            <div className="p-4 rounded-xl bg-slate-900/40 border border-white/5 space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
                <BookOpen className="w-4 h-4 text-cyan-400" />
                <span>Sign Technique Guide</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">{selectedCard.hint}</p>
            </div>
          </div>

          {/* Right: AI Posture Verification Simulator */}
          <div className="lg:col-span-7">
            <div className="rounded-3xl p-1 bg-gradient-to-b from-violet-500/30 via-cyan-500/20 to-transparent border border-cyan-500/20 shadow-2xl">
              <div className="rounded-[22px] bg-[#070c20] p-6 sm:p-8 space-y-6">
                <div className="flex items-center justify-between pb-4 border-b border-cyan-500/15">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-violet-500/20 border border-violet-400/40 flex items-center justify-center text-violet-300">
                      <GraduationCap className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-base font-bold text-white">
                        AI Practice Validator: Letter "{selectedCard.letter}"
                      </div>
                      <div className="text-xs text-slate-400 font-mono">
                        Visual Landmark Alignment Engine
                      </div>
                    </div>
                  </div>
                  <Badge variant="violet" size="sm" className="font-mono">
                    STUDIO MODE
                  </Badge>
                </div>

                {/* Simulated Practice Canvas */}
                <div className="relative rounded-2xl aspect-[16/9] bg-gradient-to-b from-[#0a122e] to-[#040714] border border-cyan-500/20 overflow-hidden flex flex-col items-center justify-center p-6 text-center">
                  <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

                  {practiceActive ? (
                    <div className="space-y-3 z-10">
                      <div className="w-12 h-12 rounded-full border-4 border-cyan-400 border-t-transparent animate-spin mx-auto" />
                      <div className="text-sm font-mono text-cyan-300 font-semibold animate-pulse">
                        Analyzing 3D joint vectors for "{selectedCard.letter}"...
                      </div>
                      <div className="text-xs text-slate-400 font-mono">
                        Evaluating thumb angle & finger curve alignment
                      </div>
                    </div>
                  ) : score !== null ? (
                    <div className="space-y-3 z-10">
                      <div className="w-14 h-14 rounded-2xl bg-emerald-500/20 border border-emerald-400/50 flex items-center justify-center mx-auto text-emerald-300 shadow-lg shadow-emerald-500/20">
                        <CheckCircle2 className="w-8 h-8" />
                      </div>
                      <div className="text-2xl font-black text-white font-mono">
                        {score}% Accuracy Score!
                      </div>
                      <p className="text-xs text-emerald-300 font-medium max-w-sm">
                        Excellent form! Finger joints align cleanly with standard ISL manual
                        alphabet reference models.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-4 z-10 max-w-md">
                      <div className="w-12 h-12 rounded-2xl bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center mx-auto text-cyan-300">
                        <Video className="w-6 h-6" />
                      </div>
                      <div>
                        <div className="text-lg font-bold text-white">
                          Ready to practice Letter "{selectedCard.letter}"?
                        </div>
                        <p className="text-xs text-slate-300 mt-1">
                          Test how accurately the computer vision model evaluates hand posture and
                          provides guidance.
                        </p>
                      </div>
                      <Button
                        variant="glow"
                        size="sm"
                        onClick={handleStartPractice}
                        icon={<Play className="w-4 h-4 fill-current" />}
                      >
                        Simulate AI Evaluation
                      </Button>
                    </div>
                  )}

                  {score !== null && (
                    <div className="mt-4 z-10">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={handleStartPractice}
                        icon={<RotateCcw className="w-3.5 h-3.5" />}
                      >
                        Try Again
                      </Button>
                    </div>
                  )}
                </div>

                {/* Studio Perks Summary */}
                <div className="grid grid-cols-3 gap-3 pt-2 text-center text-xs">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5">
                    <Award className="w-4 h-4 text-cyan-400 mx-auto mb-1" />
                    <div className="font-semibold text-white">Badges</div>
                    <div className="text-[10px] text-slate-400">Earn certificates</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5">
                    <Sparkles className="w-4 h-4 text-violet-400 mx-auto mb-1" />
                    <div className="font-semibold text-white">2,500+ Signs</div>
                    <div className="text-[10px] text-slate-400">ISL dictionary</div>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 mx-auto mb-1" />
                    <div className="font-semibold text-white">Instant Feedback</div>
                    <div className="text-[10px] text-slate-400">Joint angles</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
