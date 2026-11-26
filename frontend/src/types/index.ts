export interface NavItem {
  label: string
  href: string
  badge?: string
}

export interface FeatureItem {
  id: string
  title: string
  subtitle: string
  description: string
  icon: string
  badge?: string
  highlights: string[]
  color: 'cyan' | 'violet' | 'emerald' | 'amber'
}

export interface StepItem {
  step: string
  number: string
  title: string
  description: string
  details: string[]
  icon: string
}

export interface ValuePillar {
  title: string
  subtitle: string
  description: string
  icon: string
  statsLabel: string
  statsValue: string
}

export interface GestureMockData {
  id: string
  signName: string
  englishMeaning: string
  hindiTranslation: string
  confidence: number
  landmarksCount: number
  processingTimeMs: number
  category: string
}
