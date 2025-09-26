import React, { useState, useEffect, useMemo } from 'react';
import { getFirestore, collection, query, getDocs, orderBy, limit } from 'firebase/firestore';
import { motion, AnimatePresence } from 'framer-motion';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

// --- SVG Icons ---
const MoodIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>;
const SleepIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>;
const ExerciseIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>;
const TrophyIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a1 1 0 0 1 1 1v1a1 1 0 0 1-2 0V3a1 1 0 0 1 1-1zM5 8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V8z"/><path d="M12 12v5m0-13v2m-4-1h8"/></svg>;
const SparkleIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L9 9l-7 3 7 3 3 7 3-7 7-3-7-3-3-7z"/></svg>;

// --- Helper Functions & Animation Variants ---
const getMoodColor = (score) => {
    if (score >= 8) return 'bg-teal-400/80'; if (score >= 6) return 'bg-cyan-400/80';
    if (score >= 5) return 'bg-sky-400/80'; if (score >= 3) return 'bg-amber-400/80';
    if (score > 0) return 'bg-rose-400/80'; return 'bg-slate-300/50';
};
const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
};
const containerVariants = { visible: { transition: { staggerChildren: 0.1 } } };

// --- Re-styled Sub-Components for Dashboard ---
const KpiCard = ({ title, value, unit, icon, gradient }) => (
    <motion.div variants={cardVariants} className={`relative overflow-hidden p-4 rounded-xl shadow-lg flex items-center gap-4 text-white ${gradient}`}>
        <div className="relative z-10">{icon}</div>
        <div className="relative z-10"><p className="text-sm font-medium opacity-80">{title}</p><p className="text-2xl font-bold">{value}<span className="text-base font-medium opacity-80 ml-1">{unit}</span></p></div>
    </motion.div>
);

const ChartCard = ({ children, className = "" }) => (
    <motion.div variants={cardVariants} className={`bg-white/50 p-4 rounded-2xl shadow-lg backdrop-blur-lg border border-white/40 flex flex-col ${className}`}>
        {children}
    </motion.div>
);

const MoodHeatmap = ({ data }) => {
    const today = new Date();
    const startDate = new Date(today.getFullYear(), today.getMonth(), 1);
    const daysInMonth = new Date(today.getFullYear(), today.getMonth() + 1, 0).getDate();
    const startDayOfWeek = startDate.getDay();
    const moodByDate = useMemo(() => data.reduce((acc, entry) => { acc[new Date(entry.date).toDateString()] = entry.mood_score; return acc; }, {}), [data]);
    const allCells = [...Array(startDayOfWeek).fill(null), ...Array.from({ length: daysInMonth }, (_, i) => { const date = new Date(today.getFullYear(), today.getMonth(), i + 1); return { date, moodScore: moodByDate[date.toDateString()] }; })];
    
    return (
        <div className="h-full">
            <h3 className="text-base font-semibold text-slate-700 mb-2 text-center">{today.toLocaleString('default', { month: 'long' })} Mood Map</h3>
            <div className="grid grid-cols-7 gap-1.5">
                {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map(day => <div key={day} className="text-xs text-slate-500 font-bold text-center">{day}</div>)}
                {allCells.map((day, index) => (
                    <motion.div key={index} className="w-full aspect-square rounded-md flex items-center justify-center tooltip-container" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: index * 0.02 }}>
                        {day && (<><div className={`w-full h-full rounded-sm ${getMoodColor(day.moodScore)}`}></div><div className="tooltip-text">{day.date.toLocaleDateString()}: {day.moodScore ? `${day.moodScore}/10` : 'N/A'}</div></>)}
                    </motion.div>
                ))}
            </div>
        </div>
    );
};

// --- Main Dashboard Component ---
export default function Dashboard({ auth, user }) {
    const [allData, setAllData] = useState({ mood: [], sleep: [], exercise: [] });
    const [timeRange, setTimeRange] = useState(30);
    const [isLoading, setIsLoading] = useState(true);
    const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';

    useEffect(() => {
        const fetchAllData = async () => {
            if (!user || !user.uid || !auth || !auth.app) return;
            const db = getFirestore(auth.app);
            try {
                const queries = ['entries', 'sleepLogs', 'exerciseLogs'].map(c => query(collection(db, `/artifacts/${appId}/users/${user.uid}/${c}`), orderBy(c === 'entries' ? 'date' : 'timestamp', 'desc'), limit(90)));
                const [moodSnap, sleepSnap, exerciseSnap] = await Promise.all(queries.map(q => getDocs(q)));
                setAllData({
                    mood: moodSnap.docs.map(d => d.data()).filter(l => l?.mood_score != null && l.date),
                    sleep: sleepSnap.docs.map(d => d.data()).filter(l => l?.timestamp && typeof l.duration_minutes === 'number'),
                    exercise: exerciseSnap.docs.map(d => d.data()).filter(l => l),
                });
            } catch (error) { console.error("Error fetching dashboard data:", error); } 
            finally { setIsLoading(false); }
        };
        fetchAllData();
    }, [user, auth, appId]);

    const filteredData = useMemo(() => {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - timeRange);
        cutoffDate.setHours(0, 0, 0, 0);
        return {
            mood: allData.mood.filter(log => new Date(log.date) >= cutoffDate),
            sleep: allData.sleep.filter(log => log.timestamp.toDate() >= cutoffDate),
            exercise: allData.exercise.filter(log => log.timestamp.toDate() >= cutoffDate),
        };
    }, [allData, timeRange]);

    const kpiData = useMemo(() => {
        const { mood, sleep, exercise } = filteredData;
        const avgMood = mood.length ? (mood.reduce((s, l) => s + l.mood_score, 0) / mood.length).toFixed(1) : 'N/A';
        const avgSleep = sleep.length ? (sleep.reduce((s, l) => s + l.duration_minutes, 0) / sleep.length) : 0;
        const totalExercise = exercise.reduce((s, l) => s + Number(l.duration_minutes || 0), 0);
        const topActivity = exercise.length ? Object.keys(exercise.reduce((acc, l) => { acc[l.activity_type] = (acc[l.activity_type] || 0) + 1; return acc; }, {})).reduce((a, b, _, arr) => (arr[a] > arr[b] ? a : b), 'N/A') : 'N/A';
        return { avgMood, avgSleepHours: (avgSleep / 60).toFixed(1), totalExerciseHours: (totalExercise / 60).toFixed(1), topActivity };
    }, [filteredData]);

    const weeklyInsight = useMemo(() => {
        const weekCutoff = new Date();
        weekCutoff.setDate(weekCutoff.getDate() - 7);
        const weeklyMood = allData.mood.filter(log => new Date(log.date) >= weekCutoff);
        if (weeklyMood.length < 2) return "Log your mood a couple more times this week for a personalized insight!";
        const avgMood = (weeklyMood.reduce((s, l) => s + l.mood_score, 0) / weeklyMood.length);
        if (avgMood >= 7.5) return `Your average mood this week was ${avgMood.toFixed(1)}/10. It looks like you've had a fantastic week!`;
        return `Your average mood this week was ${avgMood.toFixed(1)}/10. Keep checking in with your feelings.`;
    }, [allData.mood]);

    const chartOptions = {
        responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { ticks: { font: { size: 10 } }, grid: { display: false } }, y: { ticks: { font: { size: 10 } }, grid: { color: 'rgba(200, 200, 200, 0.1)' } } }
    };

    const activityChartData = useMemo(() => {
        const aggregated = filteredData.exercise.reduce((acc, log) => { acc[log.activity_type] = (acc[log.activity_type] || 0) + Number(log.duration_minutes || 0); return acc; }, {});
        return {
            labels: Object.keys(aggregated),
            datasets: [{ label: 'Duration (minutes)', data: Object.values(aggregated), backgroundColor: ['rgba(52, 211, 153, 0.6)', 'rgba(96, 165, 250, 0.6)', 'rgba(251, 146, 60, 0.6)', 'rgba(168, 85, 247, 0.6)'] }]
        };
    }, [filteredData.exercise]);

    if (isLoading) return <div className="w-full max-w-6xl mx-auto mb-16 h-96 bg-white/50 rounded-2xl animate-pulse"></div>;

    return (
        <motion.div variants={containerVariants} initial="hidden" animate="visible" className="w-full max-w-6xl mx-auto mb-12">
            <style>{`.tooltip-container{position:relative}.tooltip-text{visibility:hidden;width:140px;background-color:#334155;color:#fff;text-align:center;border-radius:6px;padding:5px 0;position:absolute;z-index:1;bottom:125%;left:50%;margin-left:-70px;opacity:0;transition:opacity .3s}.tooltip-container:hover .tooltip-text{visibility:visible;opacity:1}`}</style>
            
            <motion.div variants={cardVariants} className="flex justify-center items-center gap-2 mb-6">
                {[7, 30, 90].map(days => ( <button key={days} onClick={() => setTimeRange(days)} className={`px-4 py-1.5 text-sm font-semibold rounded-full transition-all duration-300 ${timeRange === days ? 'bg-indigo-600 text-white shadow-lg' : 'bg-white/60 text-slate-700 hover:bg-white/90 hover:shadow-md'}`}>Last {days} Days</button>))}
            </motion.div>

            <motion.div variants={containerVariants} className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                <KpiCard title={`Avg. Mood`} value={kpiData.avgMood} unit="/ 10" icon={<MoodIcon />} gradient="bg-gradient-to-br from-violet-500 to-purple-600" />
                <KpiCard title={`Avg. Sleep`} value={kpiData.avgSleepHours} unit="hrs" icon={<SleepIcon />} gradient="bg-gradient-to-br from-sky-500 to-blue-600" />
                <KpiCard title={`Total Activity`} value={kpiData.totalExerciseHours} unit="hrs" icon={<ExerciseIcon />} gradient="bg-gradient-to-br from-emerald-500 to-teal-600" />
                <KpiCard title={`Top Activity`} value={kpiData.topActivity} unit="" icon={<TrophyIcon />} gradient="bg-gradient-to-br from-amber-500 to-orange-600" />
            </motion.div>
            
            <motion.div variants={containerVariants} className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <ChartCard className="lg:col-span-2 h-72">
                    <h3 className="text-base font-semibold text-slate-700 mb-2">Mood Trend</h3>
                    <Line options={chartOptions} data={{
                        labels: filteredData.mood.map(l => new Date(l.date).toLocaleDateString()),
                        datasets: [{ label: 'Mood Score', data: filteredData.mood.map(l => l.mood_score), borderColor: '#818cf8', backgroundColor: 'rgba(129, 140, 248, 0.1)', tension: 0.4, fill: true }]
                    }} />
                </ChartCard>

                <ChartCard className="h-72">
                    <h3 className="text-base font-semibold text-slate-700 mb-2">Sleep Trend</h3>
                    <Bar options={chartOptions} data={{
                        labels: filteredData.sleep.map(l => l.timestamp.toDate().toLocaleDateString()),
                        datasets: [{ label: 'Sleep (hrs)', data: filteredData.sleep.map(l => (l.duration_minutes / 60).toFixed(1)), backgroundColor: 'rgba(96, 165, 250, 0.5)' }]
                    }} />
                </ChartCard>

                <ChartCard className="h-64">
                    <MoodHeatmap data={allData.mood} />
                </ChartCard>

                <ChartCard className="h-64">
                    <h3 className="text-base font-semibold text-slate-700 mb-2">Activity Summary</h3>
                    <Bar options={{...chartOptions, indexAxis: 'y'}} data={activityChartData} />
                </ChartCard>

                <motion.div variants={cardVariants} className="h-64 bg-gradient-to-br from-indigo-500 to-purple-600 p-6 rounded-2xl shadow-lg flex flex-col justify-center items-center text-center text-white">
                    <div className="text-amber-300 mb-3"><SparkleIcon /></div>
                    <h4 className="font-semibold text-white mb-2 text-lg">Your Weekly Insight</h4>
                    <AnimatePresence mode="wait">
                        <motion.p key={timeRange} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.3 }} className="text-sm text-indigo-100 max-w-md">
                            {weeklyInsight}
                        </motion.p>
                    </AnimatePresence>
                </motion.div>
            </motion.div>
        </motion.div>
    );
}