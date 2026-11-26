import React, { type ButtonHTMLAttributes } from 'react'
import { cn } from '../../utils/cn'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'glow'
  size?: 'sm' | 'md' | 'lg'
  icon?: React.ReactNode
  iconPosition?: 'left' | 'right'
}

export const Button: React.FC<ButtonProps> = ({
  children,
  className,
  variant = 'primary',
  size = 'md',
  icon,
  iconPosition = 'left',
  ...props
}) => {
  const baseStyles =
    'inline-flex items-center justify-center font-medium transition-all duration-300 rounded-xl cursor-pointer select-none active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none'

  const sizeStyles = {
    sm: 'text-xs px-3.5 py-1.5 gap-1.5',
    md: 'text-sm px-5 py-2.5 gap-2',
    lg: 'text-base px-6 py-3.5 gap-2.5 font-semibold',
  }

  const variantStyles = {
    primary:
      'bg-gradient-to-r from-cyan-500 to-cyan-400 text-slate-950 font-semibold shadow-lg shadow-cyan-500/20 hover:shadow-cyan-400/40 hover:brightness-110 border border-cyan-300/30',
    secondary:
      'bg-slate-900/80 text-white border border-slate-700/70 hover:border-slate-500 hover:bg-slate-800/90 shadow-sm backdrop-blur-md',
    outline:
      'bg-transparent text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/10 hover:border-cyan-400',
    ghost:
      'bg-transparent text-slate-300 hover:text-white hover:bg-white/5',
    glow:
      'bg-gradient-to-r from-cyan-500 via-indigo-500 to-violet-600 text-white font-semibold shadow-lg shadow-indigo-500/25 hover:shadow-cyan-500/30 hover:scale-[1.02] border border-cyan-400/30',
  }

  return (
    <button
      className={cn(baseStyles, sizeStyles[size], variantStyles[variant], className)}
      {...props}
    >
      {icon && iconPosition === 'left' && <span className="shrink-0">{icon}</span>}
      <span>{children}</span>
      {icon && iconPosition === 'right' && <span className="shrink-0">{icon}</span>}
    </button>
  )
}

