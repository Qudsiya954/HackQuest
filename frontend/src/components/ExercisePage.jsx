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
import { collection, addDoc, onSnapshot, query, orderBy, serverTimestamp } from 'firebase/firestore';
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

// SVG Icons
const DumbbellIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-500"><path d="M14.4 14.4l2.4 2.4"/><path d="M14.4 14.4l-4.8 4.8"/><path d="M10.8 10.8L10 10l-4 4-2.4-2.4 4.8-4.8z"/><path d="M12 12l2.4 2.4-4.8-4.8z"/><path d="M14.4 14.4l4.8-4.8L20 8l-4.8 4.8z"/></svg>
);

const HistoryIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-500"><path d="M12 2v2"/><path d="M12 20v2"/><path d="M4.93 4.93l1.41 1.41"/><path d="M17.66 17.66l1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="M4.93 19.07l1.41-1.41"/><path d="M17.66 6.34l1.41-1.41"/></svg>
);

const UserCircleIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-400"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="10" r="3"/><path d="M7 20.662V19a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v1.662"/></svg>
);
const LoaderIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="animate-spin text-xl"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
);

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


export default function ExercisePage({ auth, user, onNavigate }) {
  const [activityType, setActivityType] = useState('Running');
  const [duration, setDuration] = useState('');
  const [quickNote, setQuickNote] = useState('');
  const [exerciseLogs, setExerciseLogs] = useState([]);
  const [comment, setComment] = useState('');
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalMessage, setModalMessage] = useState('');
  const [dailyReport, setDailyReport] = useState(null);

  const showCustomModal = (message) => {
    setModalMessage(message);
    setShowModal(true);
  };

  const closeModal = () => {
    setShowModal(false);
    setModalMessage('');
  };

  const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';

  // Fetch exercise logs from Firestore
  useEffect(() => {
    if (!user || !user.uid) return;

    const db = getFirestore(auth.app);
    const path = `/artifacts/${appId}/users/${user.uid}/exerciseLogs`;
    const q = query(collection(db, path), orderBy('timestamp', 'desc'));

    const unsubscribe = onSnapshot(q, (snapshot) => {
      const fetchedLogs = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data(),
      }));
      setExerciseLogs(fetchedLogs);
    }, (error) => {
      console.error("Failed to fetch exercise logs:", error);
      showCustomModal("Failed to fetch exercise logs. Please try again.");
    });

    return () => unsubscribe();
  }, [user, auth]);

  // Handle saving a new exercise entry
  const handleSaveEntry = async (e) => {
    e.preventDefault();
    if (!activityType || !duration || !quickNote) {
        showCustomModal("Please fill out all fields.");
        return;
    }

    if (!user || !user.uid) {
      showCustomModal("Authentication is not ready. Please wait a moment.");
      return;
    }

    try {
      setLoading(true);
      const newEntry = {
        activity_type: activityType,
        duration_minutes: parseInt(duration),
        quick_note: quickNote,
        timestamp: serverTimestamp(),
      };
      const db = getFirestore(auth.app);
      const path = `/artifacts/${appId}/users/${user.uid}/exerciseLogs`;
      await addDoc(collection(db, path), newEntry);
      
      const commentResult = await getExerciseComment(newEntry);
      
      setDailyReport({
        activityType: activityType,
        duration: duration,
        comment: commentResult,
      });

      setLoading(false);
      showCustomModal("Exercise log saved successfully!");
      setActivityType('Running');
      setDuration('');
      setQuickNote('');
    } catch (error) {
      console.error('Error saving exercise log:', error);
      setLoading(false);
      showCustomModal("Failed to save log. Please check your network connection.");
    }
  };

  // Function to get AI-generated exercise comment
  const getExerciseComment = async (entry) => {
    try {
      const payload = {
        date: new Date().toISOString(),
        activity_type: entry.activity_type,
        duration_minutes: entry.duration_minutes,
        quick_note: entry.quick_note,
      };

      const response = await fetch('http://127.0.0.1:8000/generate_exercise_comment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API call failed with status: ${response.status}`);
      }

      const result = await response.json();
      return result.exercise_comment;
    } catch (error) {
      console.error('Error getting exercise comment:', error);
      showCustomModal("Failed to generate exercise comment. Please ensure your backend is running.");
      return "Unable to generate a comment at this time.";
    }
  };
  
  const userId = user?.uid || 'anonymous';
  
  return (
    <div className="min-h-screen bg-emerald-50 text-emerald-900 font-sans flex flex-col items-center p-4 sm:p-8">
      <Modal message={modalMessage} onClose={closeModal} />
      <div className="w-full max-w-4xl p-6 bg-white rounded-3xl shadow-lg sm:p-8">
        <header className="text-center mb-10">
          <motion.h1
            className="text-4xl font-extrabold text-emerald-600 drop-shadow-md"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            Exercise Tracker
          </motion.h1>
          <motion.p
            className="text-emerald-500 mt-2"
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            Log your workouts and get a personalized insight.
          </motion.p>
          <AnimatePresence>
            {userId && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="mt-4 text-xs text-emerald-400 flex items-center justify-center space-x-2"
              >
                <UserCircleIcon />
                <span>User ID: {userId}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </header>

        <main className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Exercise Log Form */}
          <div className="flex flex-col space-y-4">
            <motion.h2
              className="text-2xl font-semibold text-emerald-700 flex items-center gap-2"
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <DumbbellIcon /> Log a workout
            </motion.h2>
            <form onSubmit={handleSaveEntry} className="flex flex-col space-y-4">
              <div className="flex-1">
                <label htmlFor="activityType" className="block text-lg text-emerald-600 font-medium mb-1">Activity Type</label>
                <select
                  id="activityType"
                  className="w-full p-3 border border-gray-300 rounded-xl shadow-inner focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-300"
                  value={activityType}
                  onChange={(e) => setActivityType(e.target.value)}
                >
                  <option>Running</option>
                  <option>Strength</option>
                  <option>Yoga</option>
                  <option>Walking</option>
                  <option>Other</option>
                </select>
              </div>
              <div className="flex-1">
                <label htmlFor="duration" className="block text-lg text-emerald-600 font-medium mb-1">Duration (minutes)</label>
                <input
                  id="duration"
                  type="number"
                  className="w-full p-3 border border-gray-300 rounded-xl shadow-inner focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-300"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                />
              </div>
              <div className="flex-1">
                <label htmlFor="quickNote" className="block text-lg text-emerald-600 font-medium mb-1">Quick Note</label>
                <textarea
                  id="quickNote"
                  rows="3"
                  className="w-full p-3 border border-gray-300 rounded-xl shadow-inner focus:outline-none focus:ring-2 focus:ring-emerald-500 transition-all duration-300"
                  value={quickNote}
                  onChange={(e) => setQuickNote(e.target.value)}
                />
              </div>
              <motion.button
                type="submit"
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-6 rounded-xl shadow-lg transition-all duration-300 transform active:scale-95 flex items-center justify-center gap-2 mt-4"
                disabled={loading}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                {loading ? (
                  <LoaderIcon />
                ) : (
                  <>
                    <DumbbellIcon /> Save Log
                  </>
                )}
              </motion.button>
            </form>
            <AnimatePresence>
              {dailyReport && (
                <motion.div
                  className="bg-emerald-100 p-4 rounded-xl text-emerald-800 font-medium shadow-md mt-6"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                >
                  <h3 className="text-lg font-semibold flex items-center gap-2">
                    Daily Report
                  </h3>
                  <p className="mt-2">{dailyReport.comment}</p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          {/* Exercise Logbook */}
          <div className="flex flex-col space-y-8">
            <motion.h2
              className="text-2xl font-semibold text-gray-700 flex items-center gap-2"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <HistoryIcon /> Exercise Logbook
            </motion.h2>
            <div className="bg-white rounded-xl shadow-md border border-gray-200 p-4 max-h-96 overflow-y-auto">
              <AnimatePresence>
                {exerciseLogs.length === 0 ? (
                  <motion.p
                    className="text-center text-gray-400 italic py-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    No exercise logs yet.
                  </motion.p>
                ) : (
                  <ul className="space-y-3">
                    {exerciseLogs.map((log) => (
                      <motion.li
                        key={log.id}
                        className="p-3 bg-gray-50 rounded-lg flex flex-col justify-between shadow-sm"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <div className="flex items-center justify-between">
                            <span className="text-sm font-semibold text-emerald-700">
                                {log.timestamp ? new Date(log.timestamp.toDate()).toLocaleDateString() : 'N/A'}
                            </span>
                            <span className="text-xs text-gray-500">
                                {log.duration_minutes} min
                            </span>
                        </div>
                        <div className="flex items-center gap-3 mt-2">
                            <span className="text-sm font-bold text-emerald-600">
                                {log.activity_type}
                            </span>
                        </div>
                        <p className="mt-2 text-sm text-gray-600 italic">"{log.quick_note}"</p>
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
