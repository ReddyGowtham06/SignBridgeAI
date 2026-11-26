import { Navbar } from '../components/layout/Navbar'
import { Hero } from '../components/sections/Hero'
import { HowItWorks } from '../components/sections/HowItWorks'
import { Features } from '../components/sections/Features'
import { LearningStudio } from '../components/sections/LearningStudio'
import { WhySignBridge } from '../components/sections/WhySignBridge'
import { CTASection } from '../components/sections/CTASection'
import { Footer } from '../components/layout/Footer'

export const LandingPage = () => {
  return (
    <div className="min-h-screen bg-[#050814] text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950 flex flex-col">
      {/* Fixed Navigation Header */}
      <Navbar />

      {/* Main Landing Page Flow */}
      <main className="flex-1">
        {/* Hero Section with Live Landmark Visualizer Mockup */}
        <Hero />

        {/* How It Works (3 Steps: Capture Sign, AI Recognition, Text & Speech Output) */}
        <HowItWorks />

        {/* Features (Sign to Text, Text to Speech, Text to Sign, ISL Learning Studio) */}
        <Features />

        {/* Dedicated ISL Learning Studio Interactive Hub */}
        <LearningStudio />

        {/* Why SignBridge (Accessibility, Simple Communication, Real-Time Interaction, Inclusive Tech) */}
        <WhySignBridge />

        {/* Community Call to Action */}
        <CTASection />
      </main>

      {/* Comprehensive SaaS Footer */}
      <Footer />
    </div>
  )
}

export default LandingPage

