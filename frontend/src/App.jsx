import React, { useState, useEffect } from 'react';
import { onAuthStateChanged } from 'firebase/auth';
// Import the auth instance directly from your config file
import { auth } from './firebaseConfig.js'; 

// Import all page components
import AuthPage from './components/AuthPage.jsx';
import LandingPage from './components/LandingPage.jsx';
import MoodTrackingPage from './components/MoodTrackingPage.jsx';
import SleepPage from './components/SleepPage.jsx';
import JournalingPage from './components/JournalingPage.jsx';
import ExercisePage from './components/ExercisePage.jsx';

const App = () => {
  const [currentPage, setCurrentPage] = useState('landing');
  const [user, setUser] = useState(null);
  const [isAuthReady, setIsAuthReady] = useState(false);

  // This useEffect now correctly uses the imported auth instance
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      if (!currentUser) {
        // If the user logs out, always return to the landing page view
        setCurrentPage('landing');
      }
      setIsAuthReady(true);
    });

    return () => unsubscribe();
  }, []);

  const onNavigate = (page) => {
    setCurrentPage(page);
  };

  const renderContent = () => {
    // Show a loading message until Firebase has checked the auth state
    if (!isAuthReady) {
      return (
        <div className="flex items-center justify-center min-h-screen text-slate-500 font-sans">
          <p>Initializing Dashboard...</p>
        </div>
      );
    }

    // This is the core logic from your working file.
    // If a user is logged in, show the app content. Otherwise, show the AuthPage.
    if (!user) {
        return <AuthPage auth={auth} onAuthSuccess={(authedUser) => setUser(authedUser)} />;
    }

    // If a user IS logged in, show the correct page based on navigation
    switch (currentPage) {
      case 'landing':
        return <LandingPage auth={auth} user={user} onNavigate={onNavigate} />;
      case 'journaling':
        return <JournalingPage auth={auth} user={user} onNavigate={onNavigate} />;
      case 'moodtracking':
        return <MoodTrackingPage auth={auth} user={user} onNavigate={onNavigate} />;
      case 'sleep':
        return <SleepPage auth={auth} user={user} onNavigate={onNavigate} />;
      case 'exercise':
        return <ExercisePage auth={auth} user={user} onNavigate={onNavigate} />;
      default:
        return <LandingPage auth={auth} user={user} onNavigate={onNavigate} />;
    }
  };

  return <div className="app-container">{renderContent()}</div>;
};

export default App;