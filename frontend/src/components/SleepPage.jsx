import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { collection, addDoc, onSnapshot, query, where, getDocs, orderBy, serverTimestamp } from 'firebase/firestore';
import { getFirestore } from 'firebase/firestore';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// Helper function to process sleep data for the chart
const processSleepData = (logs) => {
    const dates = logs.map(log => {
        const date = log.timestamp?.toDate ? log.timestamp.toDate() : new Date(log.timestamp);
        return date.toLocaleDateString();
    }).reverse();
    const durations = logs.map(log => log.duration_minutes).reverse();
    return { dates, durations };
};

// Sleep Chart Component
const SleepTrendChart = ({ data }) => {
    if (!data || !data.dates || data.dates.length === 0) {
        return <p className="text-center text-slate-500">Log your sleep to see the trend.</p>;
    }
    const chartData = {
        labels: data.dates,
        datasets: [{
            label: 'Sleep Duration (minutes)',
            data: data.durations,
            borderColor: 'rgba(59, 130, 246, 0.8)',
            backgroundColor: 'rgba(59, 130, 246, 0.2)',
            tension: 0.4,
            fill: true,
        }]
    };
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Sleep Trend Over Time',
            },
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Duration (minutes)',
                },
            },
        },
    };
    return <div className="h-64"><Line data={chartData} options={options} /></div>;
};

// Custom Modal Component
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


// Main Sleep Page Component
export default function SleepPage({ auth, user, onNavigate }) {
  const [duration, setDuration] = useState('');
  const [quality, setQuality] = useState('Good');
  const [sleepLogs, setSleepLogs] = useState([]);
  const [chartData, setChartData] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [modalMessage, setModalMessage] = useState('');
  const [showModal, setShowModal] = useState(false);

  const showCustomModal = (message) => {
    setModalMessage(message);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setModalMessage('');
  };


  const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';

  // Fetch sleep logs from Firestore
  useEffect(() => {
    if (!user || !user.uid || !auth || !auth.app) return;
    const db = getFirestore(auth.app);
    const path = `/artifacts/${appId}/users/${user.uid}/sleepLogs`;
    const q = query(collection(db, path), orderBy('timestamp', 'desc'));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const fetchedLogs = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data(),
      }));
      setSleepLogs(fetchedLogs);
      setChartData(processSleepData(fetchedLogs));
    }, (error) => {
      console.error("Failed to fetch sleep logs:", error);
      showCustomModal("Failed to fetch sleep logs. Please check your connection.");
    });

    return () => unsubscribe();
  }, [user, auth, appId]);

  const handleSaveEntry = async (e) => {
    e.preventDefault();
    if (!duration) {
      showCustomModal("Please enter a duration.");
      return;
    }
    if (!user || !user.uid || !auth || !auth.app) {
        showCustomModal("Authentication is not ready. Please wait a moment.");
      return;
    }

    setIsSubmitting(true);
    const db = getFirestore(auth.app);
    try {
      const newEntry = {
        duration_minutes: parseInt(duration),
        quality: quality,
        timestamp: serverTimestamp(),
      };
      const path = `/artifacts/${appId}/users/${user.uid}/sleepLogs`;
      await addDoc(collection(db, path), newEntry);
      showCustomModal("Sleep log saved successfully!");
      setDuration('');
    } catch (error) {
      console.error('Error saving sleep log:', error);
      showCustomModal("Failed to save log. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-indigo-50 text-indigo-900 font-sans flex flex-col items-center p-4 sm:p-8">
      <Modal message={modalMessage} onClose={closeModal} />
      <div className="w-full max-w-4xl p-6 bg-white rounded-3xl shadow-lg sm:p-8">
        <header className="flex justify-between items-center mb-10">
          <motion.h1
            className="text-4xl font-extrabold text-indigo-600 drop-shadow-md"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            Sleep Tracker
          </motion.h1>
          <button
            onClick={() => onNavigate('landing')}
            className="bg-indigo-200 hover:bg-indigo-300 text-indigo-800 font-bold py-2 px-6 rounded-full transition-colors duration-300"
          >
            Back to Home
          </button>
        </header>

        <main className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Sleep Log Form */}
          <div className="flex flex-col space-y-4">
            <h2 className="text-2xl font-semibold text-indigo-700">Log Your Sleep</h2>
            <form onSubmit={handleSaveEntry} className="flex flex-col space-y-4">
              <div className="flex-1">
                <label htmlFor="duration" className="block text-lg text-indigo-600 font-medium mb-1">Duration (minutes)</label>
                <input
                  id="duration"
                  type="number"
                  className="w-full p-3 border border-gray-300 rounded-xl shadow-inner focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all duration-300"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                />
              </div>
              <div className="flex-1">
                <label htmlFor="quality" className="block text-lg text-indigo-600 font-medium mb-1">Sleep Quality</label>
                <select
                  id="quality"
                  className="w-full p-3 border border-gray-300 rounded-xl shadow-inner focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all duration-300"
                  value={quality}
                  onChange={(e) => setQuality(e.target.value)}
                >
                  <option>Poor</option>
                  <option>Fair</option>
                  <option>Good</option>
                  <option>Excellent</option>
                </select>
              </div>
              <motion.button
                type="submit"
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-xl shadow-lg transition-all duration-300 transform active:scale-95 flex items-center justify-center gap-2 mt-4"
                disabled={isSubmitting}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                {isSubmitting ? 'Saving...' : 'Save Log'}
              </motion.button>
            </form>
          </div>
          
          {/* Sleep Chart */}
          <div className="flex flex-col space-y-8">
            <h2 className="text-2xl font-semibold text-indigo-700">Sleep Trend</h2>
            <div className="bg-white rounded-xl shadow-md border border-gray-200 p-4">
              <SleepTrendChart data={chartData} />
            </div>
            
            {/* Sleep Logbook */}
            <h2 className="text-2xl font-semibold text-indigo-700 mt-8">Recent Logs</h2>
            <div className="bg-white rounded-xl shadow-md border border-gray-200 p-4 max-h-96 overflow-y-auto">
              <AnimatePresence>
                {sleepLogs.length === 0 ? (
                  <motion.p
                    className="text-center text-gray-400 italic py-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    No sleep logs yet.
                  </motion.p>
                ) : (
                  <ul className="space-y-3">
                    {sleepLogs.map((log) => (
                      <motion.li
                        key={log.id}
                        className="p-3 bg-gray-50 rounded-lg shadow-sm"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="flex justify-between items-center">
                          <span className="text-sm font-semibold text-indigo-700">
                            {log.timestamp ? new Date(log.timestamp.toDate()).toLocaleDateString() : 'N/A'}
                          </span>
                          <span className="text-xs text-gray-500">
                            {log.duration_minutes} min
                          </span>
                        </div>
                        <p className="mt-1 text-sm text-gray-600 italic">
                          Quality: {log.quality}
                        </p>
                      </motion.li>
                    ))}
                  </ul>
                )}
              </AnimatePresence>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
