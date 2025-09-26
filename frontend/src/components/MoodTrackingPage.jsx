import React, { useState, useEffect, useCallback } from 'react';
import { collection, addDoc, query, where, getDocs, orderBy } from 'firebase/firestore';
import { getFirestore } from 'firebase/firestore';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// Register all necessary Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, ArcElement, Title, Tooltip, Legend, Filler);

// Custom Modal Component instead of alert()
const Modal = ({ message, onClose }) => {
    if (!message) return null;
    return (
      <motion.div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          className="bg-white p-6 rounded-2xl shadow-xl w-full max-w-sm text-center font-sans"
          initial={{ scale: 0.9, y: 50 }}
          animate={{ scale: 1, y: 0 }}
          exit={{ scale: 0.9, y: 50 }}
        >
          <p className="text-lg font-medium text-gray-800 mb-4">{message}</p>
          <button
            onClick={onClose}
            className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-6 rounded-full transition-colors duration-300"
          >
            Close
          </button>
        </motion.div>
      </motion.div>
    );
};

// --- Reusable Chart Components ---
const MoodTrendChart = ({ data }) => {
    if (!data || !data.dates || data.dates.length === 0) {
        return <p className="text-center text-slate-500 h-full flex items-center justify-center">Log your mood to see the trend.</p>;
    }
    const chartData = {
        labels: data.dates,
        datasets: [{
            label: 'Mood Score',
            data: data.mood_scores,
            fill: true,
            borderColor: 'rgba(79, 70, 229, 0.8)',
            backgroundColor: 'rgba(79, 70, 229, 0.1)',
            tension: 0.4,
        }]
    };
    return <div className="h-full"><Line data={chartData} options={{ responsive: true, maintainAspectRatio: false }} /></div>;
};

const MoodDistributionChart = ({ data }) => {
    if (!data || !data.mood_distribution) {
        return <p className="text-center text-slate-500 h-full flex items-center justify-center">No mood distribution data.</p>;
    }
    const chartData = {
        labels: Object.keys(data.mood_distribution),
        datasets: [{
            data: Object.values(data.mood_distribution),
            backgroundColor: ['#10B981', '#F59E0B', '#EF4444', '#3B82F6', '#8B5CF6', '#EC4899', '#6EE7B7'],
            borderColor: 'transparent',
        }]
    };
    return <div className="h-48 w-48 mx-auto"><Doughnut data={chartData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#475569' } } } }} /></div>;
};


// --- Main Mood Tracking Page Component ---
export default function MoodTrackingPage({ auth, user, onNavigate }) {
    // Input states
    const [moodScore, setMoodScore] = useState(5);
    const [moodLabel, setMoodLabel] = useState('Neutral');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [modalMessage, setModalMessage] = useState('');
    const [showModal, setShowModal] = useState(false);

    // Data states
    const [moodAnalysis, setMoodAnalysis] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    const showCustomModal = (message) => {
      setModalMessage(message);
      setShowModal(true);
    };

    const closeModal = () => {
      setShowModal(false);
      setModalMessage('');
    };

    const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';

    const analyzeData = useCallback(async () => {
        if (!user || !user.uid || !auth || !auth.app) return;
        setIsLoading(true);
        try {
            const db = getFirestore(auth.app);
            const path = `/artifacts/${appId}/users/${user.uid}/entries`;
            const q = query(
                collection(db, path),
                orderBy("date", "asc")
            );

            const querySnapshot = await getDocs(q);

            const allMoodEntries = querySnapshot.docs
                .map(doc => doc.data())
                .filter(entry => entry.mood_score != null);

            if (allMoodEntries.length > 0) {
                const response = await fetch('http://localhost:8000/analyze_mood', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ logs: allMoodEntries })
                });

                if (!response.ok) {
                    throw new Error(`Network response was not ok (status: ${response.status})`);
                }

                const data = await response.json();
                setMoodAnalysis(data);
            } else {
                setMoodAnalysis(null);
            }
        } catch (error) {
            console.error("Error fetching or analyzing mood data:", error);
            showCustomModal("Could not analyze mood data. Please ensure your Python backend server is running and try again.");
        } finally {
            setIsLoading(false);
        }
    }, [user, auth, appId]);

    useEffect(() => {
        analyzeData();
    }, [analyzeData]);

    const handleMoodSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);
        if (!user || !user.uid || !auth || !auth.app) {
            showCustomModal("Authentication is not ready. Please wait a moment.");
            setIsSubmitting(false);
            return;
        }
        
        const newEntry = {
            userId: user.uid,
            date: new Date().toISOString().split('T')[0],
            mood_score: moodScore,
            mood_label: moodLabel,
            journal_text: null
        };
        try {
            const db = getFirestore(auth.app);
            const path = `/artifacts/${appId}/users/${user.uid}/entries`;
            await addDoc(collection(db, path), newEntry);
            showCustomModal("Mood logged successfully!");
            setMoodScore(5);
            setMoodLabel('Neutral');
            await analyzeData();
        } catch (error) {
            console.error("Error saving mood entry:", error);
            showCustomModal("Failed to save your mood. Please try again.");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen w-full bg-gradient-to-br from-cyan-100 via-purple-100 to-pink-200 text-slate-900 font-sans p-8">
            <Modal message={modalMessage} onClose={closeModal} />
            <header className="flex justify-between items-center mb-10 max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold text-slate-800">Mood Tracking</h1>
                <button
                    onClick={() => onNavigate('landing')}
                    className="bg-white/50 hover:bg-white/80 px-4 py-2 rounded-lg transition-colors text-sm text-slate-700 backdrop-blur-sm border border-slate-200"
                >
                    Back to Home
                </button>
            </header>

            <motion.main
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8"
            >
                {/* Left Column: Input Form */}
                <div className="lg:col-span-1 bg-white/60 p-6 rounded-2xl shadow-lg backdrop-blur-lg border border-white/50">
                    <h2 className="text-xl font-semibold mb-4 text-slate-800">How are you feeling today?</h2>
                    <form onSubmit={handleMoodSubmit} className="space-y-6">
                        <div>
                            <label className="block mb-2 text-slate-600">Your Mood: <span className="font-bold text-slate-800">{moodScore}/10</span></label>
                            <input type="range" min="1" max="10" value={moodScore} onChange={(e) => setMoodScore(parseInt(e.target.value))} className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600" />
                        </div>
                        <div>
                            <label className="block mb-2 text-slate-600">Mood Label</label>
                            <select value={moodLabel} onChange={(e) => setMoodLabel(e.target.value)} className="w-full bg-white/80 p-2 rounded-md border border-slate-300 focus:ring-2 focus:ring-indigo-500 focus:outline-none">
                                <option>Neutral</option>
                                <option>Happy</option>
                                <option>Sad</option>
                                <option>Angry</option>
                                <option>Anxious</option>
                                <option>Excited</option>
                                <option>Calm</option>
                            </select>
                        </div>
                        <button type="submit" disabled={isSubmitting} className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-2.5 rounded-lg font-semibold transition-colors disabled:bg-indigo-400 disabled:cursor-not-allowed">
                            {isSubmitting ? 'Logging...' : 'Log Mood'}
                        </button>
                    </form>
                </div>

                {/* Right Column: Analytics */}
                <div className="lg:col-span-2 bg-white/60 p-6 rounded-2xl shadow-lg backdrop-blur-lg border border-white/50 space-y-8">
                    {isLoading ? (
                        <div className="flex items-center justify-center h-full"><p className="text-slate-600">Analyzing your mood data...</p></div>
                    ) : (
                        <>
                            <div>
                                <h3 className="text-lg font-semibold text-slate-800 mb-4">Your Mood Trend</h3>
                                <div className="h-64"><MoodTrendChart data={moodAnalysis} /></div>
                            </div>
                            <div className="text-center">
                                <h3 className="text-lg font-semibold text-slate-800 mb-2">Mood Distribution</h3>
                                <MoodDistributionChart data={moodAnalysis} />
                            </div>
                        </>
                    )}
                </div>
            </motion.main>
        </div>
    );
}
