import React, { useState } from 'react';
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from 'firebase/auth';

/**
 * A component that provides a UI for both user login and sign-up.
 * It uses Firebase Authentication to handle the auth logic.
 *
 * @param {object} props - The component's props.
 * @param {import('firebase/auth').Auth} props.auth - The Firebase Auth instance from your firebaseConfig.js.
 * @param {function} props.onAuthSuccess - A callback function to be called on successful login/signup.
 */
const AuthPage = ({ auth, onAuthSuccess }) => {
    // State to toggle between Login and Sign Up forms
    const [isLogin, setIsLogin] = useState(true);

    // State for form inputs
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    
    // State for loading indicators and error messages
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleFormSubmit = async (e) => {
        e.preventDefault();
        if (!email || !password) {
            setError('Please enter both email and password.');
            return;
        }
        if (!auth) {
            setError('Firebase is not initialized. Please check the main App file.');
            console.error("Auth object is missing from props.");
            return;
        }
        setLoading(true);
        setError('');

        try {
            if (isLogin) {
                // Handle User Login
                const userCredential = await signInWithEmailAndPassword(auth, email, password);
                console.log('User logged in:', userCredential.user);
                onAuthSuccess(userCredential.user); // Notify parent component of success
            } else {
                // Handle New User Sign Up
                const userCredential = await createUserWithEmailAndPassword(auth, email, password);
                console.log('User signed up:', userCredential.user);
                onAuthSuccess(userCredential.user); // Notify parent component of success
            }
        } catch (err) {
            // Provide user-friendly error messages
            let friendlyMessage = 'An error occurred. Please try again.';
            switch (err.code) {
                case 'auth/user-not-found':
                case 'auth/wrong-password':
                    friendlyMessage = 'Invalid email or password.';
                    break;
                case 'auth/email-already-in-use':
                    friendlyMessage = 'An account with this email already exists.';
                    break;
                case 'auth/weak-password':
                    friendlyMessage = 'Password should be at least 6 characters long.';
                    break;
                default:
                    console.error('Authentication Error:', err.code, err.message);
            }
            setError(friendlyMessage);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="min-h-screen w-full flex items-center justify-center p-4 bg-gradient-to-br from-purple-800 to-blue-600 font-['Inter',_sans-serif]">
            <div className="w-full max-w-md bg-white/10 backdrop-blur-lg border border-white/15 rounded-2xl p-8 shadow-2xl text-white">
                <h2 className="text-3xl font-bold text-center mb-2">
                    {isLogin ? 'Welcome Back' : 'Create Your Account'}
                </h2>
                <p className="text-center text-white/70 mb-8">
                    {isLogin ? 'Log in to continue your journey.' : 'Start tracking your well-being.'}
                </p>
                
                <form onSubmit={handleFormSubmit} className="space-y-6">
                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-white/80">Email Address</label>
                        <input
                            id="email"
                            name="email"
                            type="email"
                            autoComplete="email"
                            required
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            className="mt-1 block w-full bg-white/10 rounded-lg p-3 border border-white/20 focus:ring-2 focus:ring-blue-400 focus:outline-none placeholder-white/40"
                            placeholder="you@example.com"
                        />
                    </div>
                    
                    <div>
                        <label htmlFor="password"className="block text-sm font-medium text-white/80">Password</label>
                        <input
                            id="password"
                            name="password"
                            type="password"
                            autoComplete={isLogin ? "current-password" : "new-password"}
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="mt-1 block w-full bg-white/10 rounded-lg p-3 border border-white/20 focus:ring-2 focus:ring-blue-400 focus:outline-none placeholder-white/40"
                            placeholder="••••••••"
                        />
                    </div>

                    {error && (
                        <p className="text-sm text-red-400 text-center">{error}</p>
                    )}

                    <div>
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-blue-500 transition-colors disabled:opacity-50"
                        >
                            {loading ? <div className="w-5 h-5 border-2 border-white/50 border-t-white rounded-full animate-spin"></div> : (isLogin ? 'Log In' : 'Sign Up')}
                        </button>
                    </div>
                </form>

                <div className="mt-6 text-center">
                    <button 
                        onClick={() => {
                            setIsLogin(!isLogin);
                            setError(''); // Clear errors when switching forms
                        }}
                        className="text-sm text-blue-300 hover:text-blue-200 transition-colors"
                    >
                        {isLogin ? "Don't have an account? Sign Up" : 'Already have an account? Log In'}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default AuthPage;
