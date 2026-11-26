import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import { ProtectedRoute } from './components/auth/ProtectedRoute'
import { PublicAuthRoute } from './components/auth/PublicAuthRoute'
import { LandingPage } from './pages/LandingPage'
import { LoginPage } from './pages/auth/LoginPage'
import { RegisterPage } from './pages/auth/RegisterPage'
import { AppLayout } from './layouts/AppLayout'
import { DashboardOverview } from './pages/app/DashboardOverview'
import { SignToTextPage } from './pages/app/SignToTextPage'
import { TextToSpeechPage } from './pages/app/TextToSpeechPage'
import { TextToSignPage } from './pages/app/TextToSignPage'
import { LearningStudioPage } from './pages/app/LearningStudioPage'
import { HistoryPage } from './pages/app/HistoryPage'
import { EvaluationStudioPage } from './pages/app/EvaluationStudioPage'

export function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Landing Page */}
          <Route path="/" element={<LandingPage />} />

          {/* Public Auth Routes (Redirect authenticated users to /app) */}
          <Route element={<PublicAuthRoute />}>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
          </Route>

          {/* Protected Application Shell & Dashboard Routes */}
          <Route element={<ProtectedRoute />}>
            <Route path="/app" element={<AppLayout />}>
              <Route index element={<DashboardOverview />} />
              <Route path="sign-to-text" element={<SignToTextPage />} />
              <Route path="text-to-speech" element={<TextToSpeechPage />} />
              <Route path="text-to-sign" element={<TextToSignPage />} />
              <Route path="learn" element={<LearningStudioPage />} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="evaluate" element={<EvaluationStudioPage />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
