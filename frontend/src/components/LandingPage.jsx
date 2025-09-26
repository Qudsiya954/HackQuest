import React, { useState, useEffect } from 'react';
import { signOut } from 'firebase/auth';
import { motion, AnimatePresence } from 'framer-motion';
import Dashboard from './Dashboard.jsx'; // <-- IMPORT THE NEW DASHBOARD COMPONENT

// --- SVG Icons (Styled for Wave Cards) ---
const JournalIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v18z"></path><path d="M14 2v6h6"></path></svg>
);
const MoodIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>
);
const SleepIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0V3a1 1 0 0 0-1-1zm0 20a1 1 0 0 0 1-1v-1a1 1 0 0 0-2 0v1a1 1 0 0 0 1 1zm0-18a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm0 18a1 1 0 0 0 1-1v-1a1 1 0 0 0-2 0v1a1 1 0 0 0 1 1zm0-16a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm0 16a1 1 0 0 0 1-1v-1a1 1 0 0 0-2 0v1a1 1 0 0 0 1 1zm0-14a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm0 14a1 1 0 0 0 1-1v-1a1 1 0 0 0-2 0v1a1 1 0 0 0 1 1zm-8 4h8l-2 3H4l-1 3h8l-2 3H2l-1 3zm11-14h1a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm0 14a1 1 0 0 0 1-1v-1a1 1 0 0 0-2 0v1a1 1 0 0 0 1 1zm-2 11h1a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1zm-1-1a1 1 0 0 0-1 1v1a1 1 0 0 0 2 0v-1a1 1 0 0 0-1-1z"></path></svg>
);
const DumbbellIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M14.4 14.4l2.4 2.4"/><path d="M14.4 14.4l-4.8 4.8"/><path d="M10.8 10.8L10 10l-4 4-2.4-2.4 4.8-4.8z"/><path d="M12 12l2.4 2.4-4.8-4.8z"/><path d="M14.4 14.4l4.8-4.8L20 8l-4.8 4.8z"/></svg>
);
const UserIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-slate-500"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
);
const SignOutIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
);

// --- Looping Typewriter Hook ---
const phrases = [
    "Your personal sanctuary for reflection and growth.",
    "Track your mood, sleep, and well-being with ease.",
    "Journal your thoughts and gain deeper insights.",
    "A happier, healthier you starts here.",
];

const useLoopingTypewriter = (interval = 3000, typeSpeed = 50) => {
    const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0);
    const [displayText, setDisplayText] = useState('');
    const [isTyping, setIsTyping] = useState(true);
    const currentText = phrases[currentPhraseIndex];
    useEffect(() => {
        let typingTimeout;
        const handleTyping = () => {
            if (isTyping) {
                if (displayText.length < currentText.length) {
                    typingTimeout = setTimeout(() => {
                        setDisplayText(currentText.substring(0, displayText.length + 1));
                    }, typeSpeed);
                } else {
                    typingTimeout = setTimeout(() => setIsTyping(false), interval);
                }
            } else {
                if (displayText.length > 0) {
                    typingTimeout = setTimeout(() => {
                        setDisplayText(currentText.substring(0, displayText.length - 1));
                    }, typeSpeed / 2);
                } else {
                    setIsTyping(true);
                    setCurrentPhraseIndex((prevIndex) => (prevIndex + 1) % phrases.length);
                }
            }
        };
        handleTyping();
        return () => clearTimeout(typingTimeout);
    }, [displayText, isTyping, currentText, currentPhraseIndex, interval, typeSpeed]);
    return displayText;
};

// --- Testimonial Carousel ---
const testimonials = [
    { quote: "Mindful has become my daily ritual. It's helped me understand my emotional patterns like never before.", author: "Jessica L." },
    { quote: "The journaling analysis is a game-changer. Seeing the key topics from my entries has been incredibly insightful.", author: "Mark C." },
    { quote: "A beautiful and calming space to reflect. I finally feel in tune with my mental well-being.", author: "Samantha R." },
];

const TestimonialCarousel = () => {
    const [index, setIndex] = useState(0);
    useEffect(() => {
        const interval = setInterval(() => {
            setIndex(prevIndex => (prevIndex + 1) % testimonials.length);
        }, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="relative h-24 w-full max-w-2xl mt-20 overflow-hidden">
            <AnimatePresence mode="wait">
                <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.8, ease: 'easeInOut' }}
                    className="absolute inset-0 flex flex-col items-center justify-center text-center"
                >
                    <p className="text-lg italic text-slate-700">"{testimonials[index].quote}"</p>
                    <p className="mt-2 text-sm font-semibold text-slate-500">- {testimonials[index].author}</p>
                </motion.div>
            </AnimatePresence>
        </div>
    );
};

// --- NEW FEATURE CARD WITH WAVE ANIMATION ---
const FeatureCard = ({ icon, title, description, onClick }) => {
    return (
        <motion.div
            onClick={onClick}
            className="e-card playing"
            whileHover={{ y: -10, boxShadow: '0px 12px 40px -12px rgba(0,0,0,0.55)' }}
            transition={{ type: "spring", stiffness: 300 }}
        >
            <div className="wave"></div>
            <div className="wave"></div>
            <div className="wave"></div>
            <div className="infotop">
                <div className="icon">{icon}</div>
                {title}
                <div className="name">{description}</div>
            </div>
        </motion.div>
    );
};


// --- Main Landing Page Component ---
export default function LandingPage({ auth, user, onNavigate }) {
    const animatedSubheading = useLoopingTypewriter();

    return (
        <div className="min-h-screen w-full text-slate-900 font-sans overflow-y-auto relative animated-gradient">
            {/* Styles for the page and NEW card styles */}
            <style>{`
                .animated-gradient {
                    background: linear-gradient(-45deg, #f0f9ff, #e0f2fe, #dbeafe, #e0e7ff);
                    background-size: 400% 400%;
                    animation: gradient 15s ease infinite;
                }
                @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
                
                /* NEW CARD STYLES from your CSS */
                .e-card {
                  margin: 1rem;
                  background: transparent;
                  box-shadow: 0px 8px 28px -9px rgba(0,0,0,0.45);
                  position: relative;
                  width: 240px;
                  height: 330px;
                  border-radius: 16px;
                  overflow: hidden;
                  cursor: pointer;
                }
                .wave {
                  position: absolute;
                  width: 540px;
                  height: 700px;
                  opacity: 0.6;
                  left: 0;
                  top: 0;
                  margin-left: -50%;
                  margin-top: -70%;
                  background: linear-gradient(744deg,#af40ff,#5b42f3 60%,#00ddeb);
                }
                .icon {
                  width: 3em;
                  margin-top: -1em;
                  padding-bottom: 1em;
                }
                .infotop {
                  display: flex;
                  flex-direction: column;
                  align-items: center;
                  justify-content: center;
                  text-align: center;
                  font-size: 20px;
                  position: absolute;
                  top: 0;
                  left: 0;
                  right: 0;
                  bottom: 0;
                  color: rgb(255, 255, 255);
                  font-weight: 600;
                  padding: 20px;
                }
                .name {
                  font-size: 14px;
                  font-weight: 100;
                  position: relative;
                  top: 1em;
                  text-transform: lowercase;
                }
                .wave:nth-child(2),
                .wave:nth-child(3) {
                  top: 210px;
                }
                .playing .wave {
                  border-radius: 40%;
                  animation: wave 3000ms infinite linear;
                }
                .playing .wave:nth-child(2) {
                  animation-duration: 4000ms;
                }
                .playing .wave:nth-child(3) {
                  animation-duration: 5000ms;
                }
                @keyframes wave {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
            `}</style>
            
            <div className="relative z-10">
                <motion.nav 
                    initial={{ y: -100, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.8, ease: "easeOut" }}
                    className="flex justify-between items-center p-6 max-w-7xl mx-auto"
                >
                     <h1 className="text-3xl font-bold tracking-wider text-indigo-600">Mindful</h1>
                    <div className="hidden md:flex items-center space-x-8">
                        <button onClick={() => onNavigate('journaling')} className="text-slate-600 hover:text-indigo-600 transition-colors font-medium">Journaling</button>
                        <button onClick={() => onNavigate('moodtracking')} className="text-slate-600 hover:text-indigo-600 transition-colors font-medium">Mood Tracking</button>
                        <button onClick={() => onNavigate('sleep')} className="text-slate-600 hover:text-indigo-600 transition-colors font-medium">Sleep</button>
                        <button onClick={() => onNavigate('exercise')} className="text-slate-600 hover:text-indigo-600 transition-colors font-medium">Exercise</button>
                    </div>
                    <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-2 bg-white/60 px-3 py-2 rounded-full border border-slate-200 shadow-sm">
                             <UserIcon />
                             <span className="text-sm font-medium text-slate-600">
                                {user?.uid ? `User: ${user.uid.substring(0, 6)}...` : 'Not Signed In'}
                            </span>
                        </div>
                        {user && (
                             <motion.button 
                                onClick={() => signOut(auth)} 
                                className="bg-white/80 text-red-600 px-4 py-2 rounded-full shadow-lg hover:bg-white transition-all duration-300 flex items-center space-x-2 font-semibold text-sm"
                                whileHover={{ scale: 1.05, boxShadow: "0 8px 20px rgba(0, 0, 0, 0.1)" }}
                                whileTap={{ scale: 0.95 }}
                                title="Sign Out"
                            >
                                <SignOutIcon />
                                <span>Sign Out</span>
                            </motion.button>
                        )}
                    </div>
                </motion.nav>

                <main className="flex flex-col items-center text-center px-4 pt-24 pb-16">
                    <motion.h2 
                        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1, delay: 0.2 }}
                        className="text-5xl md:text-7xl font-extrabold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-indigo-600 leading-tight"
                    >
                        Find Your Clarity
                    </motion.h2>
                    <p className="text-lg md:text-xl text-slate-600 mb-12 h-8">
                        {animatedSubheading}
                        <motion.span 
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} 
                            transition={{ repeat: Infinity, duration: 0.8, ease: "easeInOut" }}
                            className="ml-1"
                        >_</motion.span>
                    </p>

                    {/* --- NEWLY ADDED DASHBOARD --- */}
                    {user && <Dashboard auth={auth} user={user} />}

                    <motion.div 
                        className="flex flex-wrap justify-center gap-8"
                    >
                        <FeatureCard 
                            icon={<JournalIcon/>}
                            title="Journaling"
                            description="Uncover insights from your daily thoughts."
                            onClick={() => onNavigate('journaling')}
                        />
                        <FeatureCard 
                            icon={<MoodIcon/>}
                            title="Mood Tracking"
                            description="Visualize your emotional patterns over time."
                            onClick={() => onNavigate('moodtracking')}
                        />
                         <FeatureCard 
                            icon={<SleepIcon/>}
                            title="Sleep"
                            description="Log your sleep to improve your rest."
                            onClick={() => onNavigate('sleep')}
                        />
                        <FeatureCard 
                            icon={<DumbbellIcon/>}
                            title="Exercise"
                            description="Track your workouts and stay motivated."
                            onClick={() => onNavigate('exercise')}
                        />
                    </motion.div>
                    
                    <TestimonialCarousel />
                </main>
            </div>
        </div>
    );
};