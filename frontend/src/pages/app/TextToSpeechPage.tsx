import { Volume2, Sparkles } from 'lucide-react'
import { Badge } from '../../components/common/Badge'

export const TextToSpeechPage = () => {
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-white/5">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
              Text to Speech
            </h1>
            <Badge variant="violet" size="sm" dot>
              Audio Synthesis
            </Badge>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Expressive multi-lingual Indian speech synthesis for translated ISL text.
          </p>
        </div>
      </div>

      <div className="rounded-2xl p-6 sm:p-10 bg-[#090f23]/60 border border-white/5 text-center space-y-4 max-w-xl mx-auto my-8 sm:my-12">
        <div className="w-16 h-16 rounded-2xl bg-violet-500/10 border border-violet-500/30 flex items-center justify-center text-violet-400 mx-auto">
          <Volume2 className="w-8 h-8" />
        </div>
        <h2 className="text-xl font-bold text-white">Text to Speech Module</h2>
        <p className="text-sm text-slate-400 leading-relaxed">
          The multi-lingual voice synthesizer and dialect selector will be constructed here.
        </p>
        <div className="inline-flex items-center gap-2 pt-2 text-xs font-mono text-violet-400">
          <Sparkles className="w-4 h-4" />
          <span>Stage: Ready for Feature Implementation</span>
        </div>
      </div>
    </div>
  )
}

