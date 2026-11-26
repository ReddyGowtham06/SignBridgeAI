import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { HandMetal, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

export const ProtectedRoute: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#050814] flex flex-col items-center justify-center gap-4 text-slate-300">
        <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
          <HandMetal className="w-6 h-6" />
        </div>
        <div className="flex items-center gap-2 text-sm font-mono">
          <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
          <span>Verifying session...</span>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
};
