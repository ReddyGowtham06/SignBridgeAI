import React from 'react'
import { cn } from '../../utils/cn'

export interface BadgeProps {
  children: React.ReactNode
  variant?: 'cyan' | 'violet' | 'emerald' | 'amber' | 'slate'
  size?: 'sm' | 'md'
  className?: string
  dot?: boolean
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'cyan',
  size = 'md',
  className,
  dot = false,
}) => {
  const variantStyles = {
    cyan: 'bg-cyan-950/60 text-cyan-300 border-cyan-500/30',
    violet: 'bg-violet-950/60 text-violet-300 border-violet-500/30',
    emerald: 'bg-emerald-950/60 text-emerald-300 border-emerald-500/30',
    amber: 'bg-amber-950/60 text-amber-300 border-amber-500/30',
    slate: 'bg-slate-800/60 text-slate-300 border-slate-700/50',
  }

  const dotColors = {
    cyan: 'bg-cyan-400',
    violet: 'bg-violet-400',
    emerald: 'bg-emerald-400',
    amber: 'bg-amber-400',
    slate: 'bg-slate-400',
  }

  const sizeStyles = {
    sm: 'text-[11px] px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
  }

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 font-medium rounded-full border backdrop-blur-md',
        sizeStyles[size],
        variantStyles[variant],
        className
      )}
    >
      {dot && (
        <span className={cn('w-1.5 h-1.5 rounded-full animate-pulse', dotColors[variant])} />
      )}
      {children}
    </span>
  )
}

