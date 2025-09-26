import React from 'react';
import { motion } from 'framer-motion';

const JournalingPage = ({ auth, user, onNavigate }) => {
    return (
        <div className="min-h-screen w-full bg-gray-900 text-white font-sans p-8">
            {/* Header with a "Back to Home" button */}
            <header className="flex justify-between items-center mb-10">
                <h1 className="text-3xl font-bold">Daily Journal</h1>
                <button
                    onClick={() => onNavigate('landing')}
                    className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg transition-colors text-sm"
                >
                    Back to Home
                </button>
            </header>

            {/* Main content area */}
            <motion.main
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="max-w-4xl mx-auto"
            >
                <div className="bg-gray-800 p-6 rounded-xl shadow-lg">
                    <h2 className="text-xl font-semibold mb-4">What's on your mind today?</h2>
                    {/* The form for journal entry and the analytics will go here */}
                    <p className="text-gray-400">Journaling feature and sentiment analysis charts will be built here.</p>
                </div>
            </motion.main>
        </div>
    );
};

export default JournalingPage;
