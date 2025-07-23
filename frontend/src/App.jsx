import React, { useEffect, useState } from "react";
import { GoogleOAuthProvider } from "@react-oauth/google";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate
} from "react-router-dom";
import { Provider as JotaiProvider } from "jotai";
import { useAtom } from "jotai";

import Home from "./pages/Home";
import Preferences from "./pages/Preferences"; // You'll need to create this
import Chat from "./pages/Chat"; // You'll need to create this
import Loading from "./components/Loading"; // You'll need to create this

// Import your stores
import {
  userAtom,
  isLoadingAtom,
  isAuthenticatedAtom
} from "./store/AuthStore";
import { AuthApi } from "./api/api";

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;

// Protected Route Component
function ProtectedRoute({ children }) {
  const [user, setUser] = useAtom(userAtom);
  const [isLoading, setIsLoading] = useAtom(isLoadingAtom);
  const [isAuthenticated, setIsAuthenticated] = useAtom(isAuthenticatedAtom);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("anya_token");

      if (!token) {
        setIsAuthenticated(false);
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const userData = await AuthApi.getCurrentUser();
        setUser(userData);
        setIsAuthenticated(true);
      } catch (error) {
        console.error("Auth check failed:", error);
        // Token might be expired, remove it
        localStorage.removeItem("anya_token");
        localStorage.removeItem("anya_user_info");
        setIsAuthenticated(false);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
  }, [setUser, setIsAuthenticated, setIsLoading]);

  if (isLoading) {
    return <Loading />; // Show loading spinner while checking auth
  }

  if (!isAuthenticated) {
    return <Navigate to="/" replace />;
  }

  return children;
}

// App Router Component
function AppRouter() {
  const [user] = useAtom(userAtom);
  const [isAuthenticated] = useAtom(isAuthenticatedAtom);

  return (
    <Router>
      <Routes>
        {/* Public Route - Home/Login page */}
        <Route
          path="/"
          element={
            isAuthenticated ? (
              // If authenticated, redirect based on onboarding status
              user?.onboarding_completed ? (
                <Navigate to="/chat" replace />
              ) : (
                <Navigate to="/preferences" replace />
              )
            ) : (
              // Not authenticated, show home page
              <Home />
            )
          }
        />

        {/* Protected Route - Preferences (Onboarding) */}
        <Route
          path="/preferences"
          element={
            <ProtectedRoute>
              {user?.onboarding_completed ? (
                <Navigate to="/chat" replace />
              ) : (
                <Preferences />
              )}
            </ProtectedRoute>
          }
        />

        {/* Protected Route - Chat Interface */}
        <Route
          path="/chat"
          element={
            <ProtectedRoute>
              {!user?.onboarding_completed ? (
                <Navigate to="/preferences" replace />
              ) : (
                <Chat />
              )}
            </ProtectedRoute>
          }
        />

        {/* Catch all route - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

// Main App Component
function App() {
  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <JotaiProvider>
        <div className="App">
          <AppRouter />
        </div>
      </JotaiProvider>
    </GoogleOAuthProvider>
  );
}

export default App;
