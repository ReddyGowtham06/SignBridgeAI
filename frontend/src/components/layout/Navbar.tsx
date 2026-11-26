import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Sparkles, Menu, X, ArrowUpRight, HandMetal } from 'lucide-react'
import { Button } from '../common/Button'
import { Badge } from '../common/Badge'

export const Navbar: React.FC = () => {
  const navigate = useNavigate()
  const [isScrolled, setIsScrolled] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    if (!mobileMenuOpen) return
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileMenuOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = prevOverflow
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [mobileMenuOpen])

  const navLinks = [
    { label: 'Home', href: '#home' },
    { label: 'How It Works', href: '#how-it-works' },
    { label: 'Features', href: '#features' },
    { label: 'Learn ISL', href: '#learning', badge: 'Studio' },
    { label: 'Why SignBridge', href: '#why-us' },
  ]

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-[#050814]/85 backdrop-blur-xl border-b border-cyan-500/10 shadow-lg shadow-black/40 py-3.5'
          : 'bg-transparent py-5'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          {/* Brand Logo */}
          <a href="#home" className="flex items-center gap-3 group">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 via-slate-900 to-violet-600/30 border border-cyan-500/30 shadow-md shadow-cyan-500/10 group-hover:border-cyan-400 group-hover:shadow-cyan-500/30 transition-all">
              <HandMetal className="w-5 h-5 text-cyan-400 group-hover:scale-110 transition-transform" />
              <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-cyan-400 animate-ping opacity-75" />
            </div>
            <div className="flex flex-col">
              <span className="font-extrabold text-lg tracking-wider text-white flex items-center gap-1.5">
                SIGNBRIDGE<span className="text-cyan-400">AI</span>
              </span>
              <span className="text-[10px] tracking-widest text-slate-400 font-mono -mt-1 uppercase">
                ISL Neural Engine
              </span>
            </div>
          </a>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1 bg-slate-900/40 p-1.5 rounded-full border border-white/5 backdrop-blur-md">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                className="relative px-4 py-2 text-sm font-medium text-slate-300 hover:text-white rounded-full transition-all duration-200 hover:bg-white/5 flex items-center gap-1.5"
              >
                {link.label}
                {link.badge && (
                  <Badge variant="cyan" size="sm" className="py-0 px-1.5 text-[9px]">
                    {link.badge}
                  </Badge>
                )}
              </a>
            ))}
          </nav>

          {/* Desktop CTA actions */}
          <div className="hidden md:flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate('/login')}
            >
              Sign In
            </Button>
            <Button
              variant="glow"
              size="sm"
              icon={<ArrowUpRight className="w-4 h-4" />}
              iconPosition="right"
              onClick={() => navigate('/app')}
            >
              Get Started
            </Button>
          </div>

          {/* Mobile Menu Trigger */}
          <button
            type="button"
            className="md:hidden min-w-[44px] min-h-[44px] flex items-center justify-center rounded-xl bg-slate-900/60 border border-slate-700/60 text-slate-300 hover:text-white"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle Navigation Menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[#070b1b]/95 backdrop-blur-2xl border-b border-cyan-500/20 px-4 pt-4 pb-6 space-y-3">
          <nav className="flex flex-col space-y-1">
            {navLinks.map((link) => (
              <a
                key={link.label}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className="flex items-center justify-between px-3 py-2.5 text-base font-medium text-slate-200 hover:text-cyan-300 hover:bg-slate-800/50 rounded-lg transition-colors"
              >
                <span>{link.label}</span>
                {link.badge && (
                  <Badge variant="cyan" size="sm">
                    {link.badge}
                  </Badge>
                )}
              </a>
            ))}
          </nav>
          <div className="pt-3 border-t border-slate-800/80 flex flex-col gap-2">
            <Button
              variant="secondary"
              className="w-full justify-center"
              onClick={() => {
                setMobileMenuOpen(false)
                navigate('/login')
              }}
            >
              Sign In
            </Button>
            <Button
              variant="glow"
              className="w-full justify-center"
              icon={<Sparkles className="w-4 h-4" />}
              onClick={() => {
                setMobileMenuOpen(false)
                navigate('/app')
              }}
            >
              Get Started
            </Button>
          </div>
        </div>
      )}
    </header>
  )
}

