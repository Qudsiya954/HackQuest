from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set, Tuple
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
from transformers import pipeline
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware
import nltk
from nltk.tokenize import sent_tokenize
import re
from collections import defaultdict
from datetime import datetime, timedelta
import statistics
from functools import lru_cache
import numpy as np

load_dotenv()

app = FastAPI(title="Mental Well-being API", version="2.0")  

# --- CORS Configuration (CRUCIAL FIX) ---
# This tells your backend to accept requests from your frontend
origins = [
    "http://localhost:5173", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (POST, GET, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- The rest of your Python code remains the same ---


# Models
sentiment_analyzer = SentimentIntensityAnalyzer()
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)


GEMINI_API_KEY =os.getenv("GEMINI_API_Key")

class MoodEntry(BaseModel):
    date: str
    physical_score: Optional[int] = None
    mood_label: Optional[str] = None
    journal_text: Optional[str] = None
    
class MoodLogRequest(BaseModel):
    logs: List[MoodEntry]

class SleepEntry(BaseModel):
    date: str
    sleep_hours: float
    
class ExerciseEntry(BaseModel):
    date: str
    activity_type: str          
    duration_minutes: int
    quick_note: str            

class WeeklySummaryRequest(BaseModel):
    mood_data: Dict[str, Any]  # {"dates": [...], "mood_scores": [...], "mood_distribution": {...}}
    sleep_data: List[SleepEntry] 
    exercise_data: List[ExerciseEntry]
    journal_data: Dict[str, Any] 
    
class ProcessedMoodEntry(BaseModel):
    date: str
    physical_score: Optional[int] = None 
   
    joy: Optional[float] = None
    sadness: Optional[float] = None
    anger: Optional[float] = None
    fear: Optional[float] = None
    
    sentiment_positive: Optional[float] = None  
    sentiment_neutral: Optional[float] = None   # 0 to 100
    sentiment_negative: Optional[float] = None 
    
class CorrelationDataPoint(BaseModel):
    date: str
    sleep_hours: Optional[float] = None
    physical_energy: Optional[int] = None
    avg_negative_emotion_score: Optional[float] = None
    avg_positive_emotion_score: Optional[float] = None
    emotional_volatility: Optional[float] = None
    sentiment_score: Optional[float] = None  
    exercise_impact: Optional[float] = None

class PatternInsight(BaseModel):
    pattern_type: str
    strength: float  # 0-1 correlation strength
    description: str
    confidence_level: str

class PredictiveInsights(BaseModel):
    burnout_risk_score: int  # 0-100
    risk_level: str  # LOW, MODERATE, HIGH
    predicted_emotional_state: str
    activity_recommendation: str
    insights_summary: str
    pattern_insights: List[PatternInsight]
    trend_direction: str  # IMPROVING, STABLE, DECLINING
    confidence_score: float  # 0-1 based on data quality and patterns

class PredictiveAnalyticsResponse(BaseModel):
    correlation_data: List[CorrelationDataPoint]
    predictive_insights: PredictiveInsights
    emotional_trends: Dict[str, float]  # Average scores for each emotion
    sentiment_trends: Dict[str, float]  # Average sentiment scores
    correlations: Dict[str, float]  # Key correlation coefficients

class AnalyticsRequest(BaseModel):
    mood_logs: List[ProcessedMoodEntry]
    sleep_logs: List[SleepEntry]
    exercise_logs: List[ExerciseEntry]
    analysis_days: Optional[int] = 7
# 0 to 100

def extract_meaningful_topics(text: str, max_topics: int = 7):
    """
    Extract meaningful, complete topics that are LLM-friendly for aggregation
    Focus on complete thoughts and emotional contexts
    """
    sentences = sent_tokenize(text)
    meaningful_topics = []
    important_patterns = [
        r'i feel\s+\w+',  # "I feel happy/sad/etc"
        r'i felt\s+\w+',  # "I felt frustrated/proud/etc"
        r'made me\s+\w+', # "made me happy/angry/etc"
        r'i was\s+\w+',   # "I was stressed/excited/etc"
        r'i had\s+\w+',   # "I had difficulty/success/etc"
        r'i received\s+\w+', # "I received good news/etc"
        r'i managed\s+to\s+\w+', # "I managed to accomplish/etc"
        r'helped me\s+\w+',  # "helped me stay calm/etc"
        r'which brought\s+\w+', # "which brought joy/etc"
        r'i am grateful\s+for\s+\w+', # gratitude expressions
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 4: 
            continue
            
        sentence_lower = sentence.lower()
        for pattern in important_patterns:
            if re.search(pattern, sentence_lower):
                # Extract the complete thought around the pattern
                clean_sentence = re.sub(r'^(today|yesterday|later|then|also|despite that)\s*,?\s*', '', sentence, flags=re.IGNORECASE)
                clean_sentence = clean_sentence.strip(' .,!?')
                if len(clean_sentence.split()) >= 4:
                    meaningful_topics.append(clean_sentence)
                break
        
        key_indicators = [
            'stressful', 'frustrated', 'overwhelmed', 'anxious', 'worried', 'sad', 'angry',
            'happy', 'joy', 'excited', 'proud', 'satisfied', 'content', 'grateful', 'accomplished',
            'workout', 'exercise', 'work', 'meeting', 'project', 'friend', 'family', 'sleep',
            'progress', 'achievement', 'success', 'difficulty', 'challenge', 'good news', 'bad news'
        ]
        
        if any(indicator in sentence_lower for indicator in key_indicators):
            if len(sentence.split()) >= 6 and len(sentence.split()) <= 15:  # Good length for topics
                clean_sentence = re.sub(r'^(today|yesterday|later|then|also|despite that)\s*,?\s*', '', sentence, flags=re.IGNORECASE)
                clean_sentence = clean_sentence.strip(' .,!?')
                if len(clean_sentence.split()) >= 4:
                    meaningful_topics.append(clean_sentence)
    
    unique_topics = []
    for topic in meaningful_topics:
        is_duplicate = False
        topic_words = set(topic.lower().split())
        
        for existing in unique_topics:
            existing_words = set(existing.lower().split())
            overlap = len(topic_words & existing_words) / len(topic_words.union(existing_words))
            if overlap >= 0.6:
                is_duplicate = True
                break
        
        if not is_duplicate and len(topic) > 10:  # Minimum meaningful length
            unique_topics.append(topic)
    
    return unique_topics[:max_topics]

def get_top_emotions(text: str):
    """
    Get top 4 emotions with intensity for bar charts - always returns 4 emotions
    """
    emotion_scores = emotion_analyzer(text)[0]
    
    emotion_mapping = {
        'joy': 'Joy',
        'sadness': 'Sadness',
        'anger': 'Anger',
        'fear': 'Anxiety',
        'surprise': 'Surprise',
        'disgust': 'Frustration',
        'love': 'Love'
    }
    
    all_emotions = []
    for emotion_data in emotion_scores:
        emotion_label = emotion_data['label'].lower()
        score = emotion_data['score']
        
        emotion_display = emotion_mapping.get(emotion_label, emotion_label.title())
        
        intensity = round(score * 10, 1)
        
        all_emotions.append({
            'emotion': emotion_display,
            'intensity': intensity
        })
    
    all_emotions.sort(key=lambda x: x['intensity'], reverse=True)
    return all_emotions[:4]  
def get_sentiment_distribution(all_texts: list):
    """
    Simple sentiment distribution
    """
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

    for text in all_texts:
        score = sentiment_analyzer.polarity_scores(text)
        if score["compound"] > 0.1:
            sentiment_counts["positive"] += 1
        elif score["compound"] < -0.1:
            sentiment_counts["negative"] += 1
        else:
            sentiment_counts["neutral"] += 1

    total = sum(sentiment_counts.values()) or 1
    return {k: round((v / total) * 100, 1) for k, v in sentiment_counts.items()}

def generate_llm_comment(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    try:
        comment = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        comment = "Unable to generate comment at the moment."
    
    return comment



# --- API Route ---
@app.post("/analyze_mood")
def analyze_mood(request: MoodLogRequest):
    logs = [log for log in request.logs if log.mood_score is not None and log.mood_label is not None]
    if not logs:
        return {"message": "No mood logs provided."}

    dates = [log.date for log in logs]
    scores = [log.mood_score for log in logs]

    mood_distribution: Dict[str, int] = {}
    for log in logs:
        mood_distribution[log.mood_label] = mood_distribution.get(log.mood_label, 0) + 1

    return {
        "dates": dates,
        "physical_scores": scores,
        "mood_distribution": mood_distribution
    }


@app.post("/generate_sleep_comment")
def generate_sleep_comment(entry: SleepEntry):
    prompt = f"""
    You are an empathetic wellness assistant. Provide a short (1-2 sentences) comment based ONLY on today's sleep hours.
    - Sleep <6 hours → express concern and suggest more rest.
    - Sleep 6-8 hours → appreciation/encouragement.
    - Sleep >8 hours → praise balance.
    Tone: empathetic, concise, supportive.

    User's sleep hours: {entry.sleep_hours}
    """
    comment = generate_llm_comment(prompt)
    return {"sleep_comment": comment}


@app.post("/generate_exercise_comment")
def generate_exercise_comment(entry: ExerciseEntry):
    prompt = f"""
    You are an empathetic wellness assistant. Based on today's exercise entry, provide a short (1-2 sentences) supportive comment.
    - Activity type: {entry.activity_type}
    - Duration: {entry.duration_minutes} minutes
    - User's note: "{entry.quick_note}"

    Guidelines:
    - Appreciate effort and consistency.
    - Encourage and motivate if duration is low or note indicates fatigue.
    - Use empathy and positive reinforcement.
    """
    comment = generate_llm_comment(prompt)
    return {"exercise_comment": comment}

   
@app.post("/analyze_journal")
def analyze_journal(request: MoodLogRequest):
    logs = [log for log in request.logs if log.journal_text]
    if not logs:
        return {"message": "No journal entries provided."}

    all_texts = [log.journal_text for log in logs]
    joined_text = " ".join(all_texts)
    top_emotions = get_top_emotions(joined_text)
    sentiment_distribution = get_sentiment_distribution(all_texts)
    topics = extract_meaningful_topics(joined_text, max_topics=7)
    prompt = f"""
   You are a supportive mental health assistant. 💜  
   Here is a journal entry summary:  
   - Emotions: {top_emotions}  
   - Sentiment: {sentiment_distribution}  
   - Topics: {topics}  

    Write a short (1–2 sentence) empathetic and encouraging note.  
   ✨ Keep it warm, validating, and human-like.  
   🌱 Use 2–3 emojis to add care and positivity.  
   ❌ Do NOT give medical, therapeutic, or diagnostic advice.  
   ✅ Focus only on appreciation, gentle encouragement, or validation.  
  """

    ai_comment = generate_llm_comment(prompt)

    return {
        "sentiment_distribution": sentiment_distribution,
        "emotions": top_emotions,
        "topics": topics,
        "ai_comment": ai_comment
    }
    
    
    
    
    

@lru_cache(maxsize=1000)
def get_topic_score(topic_lower: str, important_keywords: tuple, emotion_words: tuple) -> int:
    """Cached scoring function to avoid repeated string operations"""
    score = 0
    
    # Score based on important keywords
    for keyword in important_keywords:
        if keyword in topic_lower:
            score += 2
    
    # Prefer topics with emotional indicators
    for word in emotion_words:
        if word in topic_lower:
            score += 1
    
    # Prefer topics that aren't too generic
    if len(topic_lower.split()) >= 3:
        score += 1
        
    return score

def extract_key_topics(topics: List[str], max_topics: int = 8) -> List[str]:
    """
    Extract most relevant topics from journal entries intelligently
    Focus on preserving important themes for LLM understanding
    """
    if not topics:
        return []
    
    # Use tuples for caching compatibility
    important_keywords = (
        'work', 'job', 'career', 'stress', 'anxiety', 'worried', 'concern',
        'family', 'relationship', 'friend', 'love', 'support',
        'health', 'sleep', 'tired', 'energy', 'exercise', 'sick',
        'happy', 'excited', 'proud', 'accomplished', 'grateful',
        'sad', 'frustrated', 'angry', 'disappointed', 'overwhelmed',
        'goal', 'plan', 'future', 'change', 'decision', 'progress'
    )
    
    emotion_words = ('feel', 'felt', 'emotion', 'mood', 'heart')
    
    # Remove duplicates and very short topics
    unique_topics = list(set(topic for topic in topics if len(topic.strip()) > 5))
    
    # Score topics using cached function
    scored_topics = []
    for topic in unique_topics:
        topic_lower = topic.lower()
        score = get_topic_score(topic_lower, important_keywords, emotion_words)
        scored_topics.append((topic, score))
    
    # Sort by score and take top topics
    scored_topics.sort(key=lambda x: x[1], reverse=True)
    selected_topics = [topic for topic, score in scored_topics[:max_topics]]
    
    # Ensure we don't lose context - truncate long topics smartly
    final_topics = []
    for topic in selected_topics:
        if len(topic) > 80:  # If topic is very long
            # Try to keep the most meaningful part
            sentences = topic.split('. ')
            if len(sentences) > 1:
                # Take first sentence if it's meaningful
                first_sentence = sentences[0]
                if len(first_sentence) > 20:
                    topic = first_sentence + "..."
                else:
                    topic = topic[:80] + "..."
            else:
                topic = topic[:80] + "..."
        final_topics.append(topic)
    
    return final_topics

def calculate_weekly_averages(mood_data: Dict, sleep_data: List[SleepEntry], exercise_data: List[ExerciseEntry], journal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate weekly averages and key metrics for 7-day period"""
    
    # Mood averages (from your mood analysis output)
    mood_scores = mood_data.get("mood_scores", [])
    dates = mood_data.get("dates", [])
    mood_distribution = mood_data.get("mood_distribution", {})
    
    avg_mood = sum(mood_scores) / len(mood_scores) if mood_scores else 0
    
    # Sleep average across 7 days
    total_sleep = sum(entry.sleep_hours for entry in sleep_data)
    avg_sleep = total_sleep / len(sleep_data) if sleep_data else 0
    
    # Exercise totals and averages across 7 days
    total_exercise_minutes = sum(entry.duration_minutes for entry in exercise_data)
    exercise_sessions = len(exercise_data)
    avg_exercise_duration = total_exercise_minutes / exercise_sessions if exercise_sessions > 0 else 0
    
    # Exercise activity breakdown
    activity_counts = defaultdict(int)
    activity_durations = defaultdict(int)
    for entry in exercise_data:
        activity_counts[entry.activity_type] += 1
        activity_durations[entry.activity_type] += entry.duration_minutes
    
    top_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    raw_topics = journal_data.get("topics", [])
    journal_topics = extract_key_topics(raw_topics, max_topics=6)  # Get 6 most relevant topics
    
    # Get emotions and sentiment from journal data
    top_emotions = journal_data.get("emotions", [])[:5]  # Top 5 emotions
    sentiment_distribution = journal_data.get("sentiment_distribution", {})
    
    return {
        "date_range": f"{dates[0]} to {dates[-1]}" if dates else "N/A",
        "avg_mood_score": round(avg_mood, 1),
        "mood_distribution": mood_distribution,
        "avg_sleep_hours": round(avg_sleep, 1),
        "total_exercise_minutes": total_exercise_minutes,
        "exercise_sessions": exercise_sessions,
        "avg_exercise_duration": round(avg_exercise_duration, 1),
        "top_activities": [{"activity": activity, "sessions": count} for activity, count in top_activities],
        "journal_topics": journal_topics,
        "total_topics_available": len(journal_data.get("topics", [])),
        "top_emotions": top_emotions,
        "sentiment_distribution": sentiment_distribution,
        "total_days": len(dates) if dates else 7 
    }

def create_concise_prompt(averages: Dict[str, Any]) -> str:
    """Create a concise prompt to minimize API usage while preserving important context"""
    
    activities_str = ", ".join([f"{a['activity']}({a['sessions']}x)" for a in averages['top_activities']])
    
    top_emotions = averages.get('top_emotions', [])
    if top_emotions and len(top_emotions) > 0:
        emotions_list = []
        for emotion in top_emotions[:3]:  
            if isinstance(emotion, dict):
                
                emotion_name = emotion.get('emotion', 'Unknown')
                intensity = emotion.get('intensity', 0)
                emotions_list.append(f"{emotion_name}({intensity:.1f})")
            elif isinstance(emotion, str):
                
                emotions_list.append(emotion)
            else:
                
                emotions_list.append(str(emotion))
        emotions_str = ", ".join(emotions_list)
    else:
        emotions_str = "None"
    topics = averages['journal_topics']
    if topics:
        topics_str = " | ".join(topics[:4])  
    else:
        topics_str = "No significant themes"
    
    sentiment_dist = averages.get('sentiment_distribution', {})
    positive_sentiment = sentiment_dist.get('positive', 0) if isinstance(sentiment_dist, dict) else 0
    negative_sentiment = sentiment_dist.get('negative', 0) if isinstance(sentiment_dist, dict) else 0
    prompt = f"""Weekly well-being analysis ({averages['date_range']}):

📊 WEEKLY DATA:
• Mood: {averages['avg_mood_score']}/10 average across {averages['total_days']} days
• Sleep: {averages['avg_sleep_hours']} hours/night average
• Exercise: {averages['total_exercise_minutes']}min total, {averages['exercise_sessions']} sessions
• Activities: {activities_str if activities_str else 'No activities recorded'}
• Top emotions: {emotions_str}
• Sentiment: {positive_sentiment:.2f} positive, {negative_sentiment:.2f} negative

🎯 KEY JOURNAL THEMES:
{topics_str}

Please provide a focused 4-part analysis:

1. **WEEK OVERVIEW**: How was the overall week based on mood, sleep, and activity patterns?

2. **EMOTIONAL PATTERNS**: What emotional trends and patterns emerged throughout the week?

3. **POSITIVE HIGHLIGHTS**: What went well this week? What achievements or positive moments stand out?

4. **ACTIONABLE INSIGHTS**: What are 2-3 key recommendations for improving well-being next week?

Keep each section to 1-2 sentences for clarity and focus."""
    
    return prompt

@app.post("/generate_weekly_summary")
async def generate_weekly_summary(request: WeeklySummaryRequest):
    """
    Generate weekly well-being summary with intelligent topic selection for 7-day period
    
    Input format:
    {
        "mood_data": {"dates": [...], "mood_scores": [...], "mood_distribution": {...}},
        "sleep_data": [{"date": "2024-01-01", "sleep_hours": 7.5}, ...],
        "exercise_data": [{"date": "2024-01-01", "activity_type": "Running", "duration_minutes": 30, "quick_note": "..."}, ...],
        "journal_data": {"sentiment_distribution": {...}, "emotions": [...], "topics": [...]}
    }
    """
    try:
        
        if not request.mood_data and not request.sleep_data and not request.exercise_data:
            return {
                "success": False,
                "error": "No data provided for analysis",
                "ai_summary": "Unable to generate summary without data."
            }
        
        
        averages = calculate_weekly_averages(
            request.mood_data, 
            request.sleep_data, 
            request.exercise_data, 
            request.journal_data
        )
        
        
        prompt = create_concise_prompt(averages)
        
        
        ai_summary = generate_llm_comment(prompt)
        
        
        return {
            "success": True,
            "weekly_metrics": {
                "date_range": averages["date_range"],
                "avg_mood_score": averages["avg_mood_score"],
                "mood_distribution": averages["mood_distribution"],
                "avg_sleep_hours": averages["avg_sleep_hours"],
                "total_exercise_minutes": averages["total_exercise_minutes"],
                "exercise_sessions": averages["exercise_sessions"],
                "avg_exercise_duration": averages["avg_exercise_duration"],
                "top_activities": averages["top_activities"],
                "journal_insights": {
                    "selected_topics": averages["journal_topics"],
                    "total_topics_available": averages["total_topics_available"],
                    "top_emotions": averages["top_emotions"][:3],  # Top 3 emotions for response
                    "sentiment": averages["sentiment_distribution"]
                }
            },
            "ai_summary": ai_summary,
            "data_summary": {
                "total_days_analyzed": averages["total_days"],
                "mood_entries": len(request.mood_data.get("mood_scores", [])),
                "sleep_entries": len(request.sleep_data),
                "exercise_entries": len(request.exercise_data),
                "topics_intelligence": f"Selected {len(averages['journal_topics'])} most relevant from {averages['total_topics_available']} total topics"
            },
            "prompt_used": prompt  
        }
        
    except Exception as e:
        return {
            "success": False, 
            "error": f"Error generating weekly summary: {str(e)}", 
            "ai_summary": "Unable to generate comment at the moment due to processing error."
        }



def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
    """Calculate Pearson correlation coefficient"""
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    
    try:
        return np.corrcoef(x_values, y_values)[0, 1] if not np.isnan(np.corrcoef(x_values, y_values)[0, 1]) else 0.0
    except:
        return 0.0

def calculate_emotional_volatility(emotions: Dict[str, float]) -> float:
    """Calculate emotional volatility as standard deviation of emotion scores"""
    if not emotions:
        return 0.0
    
    emotion_values = list(emotions.values())
    return float(np.std(emotion_values)) if len(emotion_values) > 1 else 0.0

def calculate_compound_sentiment(positive: Optional[float], neutral: Optional[float], negative: Optional[float]) -> Optional[float]:
    """Calculate compound sentiment score from positive, neutral, negative percentages"""
    if positive is None or negative is None:
        return None
    
    # Convert to decimal (0-1) and calculate compound score (-1 to 1)
    pos_decimal = positive / 100.0
    neg_decimal = negative / 100.0
    
    # Simple compound calculation: positive weight - negative weight
    compound = pos_decimal - neg_decimal
    
    # Scale to -100 to 100 for consistency
    return compound * 100.0

def detect_patterns(correlation_data: List[CorrelationDataPoint]) -> List[PatternInsight]:
    """Detect patterns in the data using correlation analysis"""
    patterns = []
    
    # Extract valid data pairs for correlation analysis
    sleep_energy_pairs = [(d.sleep_hours, d.physical_energy) for d in correlation_data 
                         if d.sleep_hours is not None and d.physical_energy is not None]
    
    sleep_sentiment_pairs = [(d.sleep_hours, d.sentiment_score) for d in correlation_data 
                           if d.sleep_hours is not None and d.sentiment_score is not None]
    
    exercise_energy_pairs = [(d.exercise_impact, d.physical_energy) for d in correlation_data 
                           if d.exercise_impact is not None and d.physical_energy is not None]
    
    exercise_mood_pairs = [(d.exercise_impact, d.sentiment_score) for d in correlation_data 
                         if d.exercise_impact is not None and d.sentiment_score is not None]
    
    negative_sleep_pairs = [(d.avg_negative_emotion_score, d.sleep_hours) for d in correlation_data 
                          if d.avg_negative_emotion_score is not None and d.sleep_hours is not None]
    
    volatility_sleep_pairs = [(d.emotional_volatility, d.sleep_hours) for d in correlation_data 
                            if d.emotional_volatility is not None and d.sleep_hours is not None]
    
    # Sleep-Energy Pattern
    if len(sleep_energy_pairs) >= 3:
        sleep_vals, energy_vals = zip(*sleep_energy_pairs)
        correlation = calculate_correlation(list(sleep_vals), list(energy_vals))
        if abs(correlation) > 0.3:
            strength = min(1.0, abs(correlation))
            confidence = "HIGH" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.5 else "LOW"
            direction = "positive" if correlation > 0 else "negative"
            patterns.append(PatternInsight(
                pattern_type="sleep_energy_correlation",
                strength=strength,
                description=f"Strong {direction} correlation between sleep and energy levels (r={correlation:.2f})",
                confidence_level=confidence
            ))
    
    # Sleep-Sentiment Pattern
    if len(sleep_sentiment_pairs) >= 3:
        sleep_vals, sentiment_vals = zip(*sleep_sentiment_pairs)
        correlation = calculate_correlation(list(sleep_vals), list(sentiment_vals))
        if abs(correlation) > 0.3:
            strength = min(1.0, abs(correlation))
            confidence = "HIGH" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.5 else "LOW"
            direction = "positive" if correlation > 0 else "negative"
            patterns.append(PatternInsight(
                pattern_type="sleep_mood_correlation",
                strength=strength,
                description=f"{direction.title()} relationship between sleep quality and mood (r={correlation:.2f})",
                confidence_level=confidence
            ))
    
    # Exercise-Energy Pattern
    if len(exercise_energy_pairs) >= 3:
        exercise_vals, energy_vals = zip(*exercise_energy_pairs)
        correlation = calculate_correlation(list(exercise_vals), list(energy_vals))
        if abs(correlation) > 0.3:
            strength = min(1.0, abs(correlation))
            confidence = "HIGH" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.5 else "LOW"
            direction = "positive" if correlation > 0 else "negative"
            patterns.append(PatternInsight(
                pattern_type="exercise_energy_correlation",
                strength=strength,
                description=f"Exercise shows {direction} impact on energy levels (r={correlation:.2f})",
                confidence_level=confidence
            ))
    
    # Exercise-Mood Pattern
    if len(exercise_mood_pairs) >= 3:
        exercise_vals, mood_vals = zip(*exercise_mood_pairs)
        correlation = calculate_correlation(list(exercise_vals), list(mood_vals))
        if abs(correlation) > 0.3:
            strength = min(1.0, abs(correlation))
            confidence = "HIGH" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.5 else "LOW"
            direction = "positive" if correlation > 0 else "negative"
            patterns.append(PatternInsight(
                pattern_type="exercise_mood_correlation",
                strength=strength,
                description=f"Exercise has {direction} correlation with mood (r={correlation:.2f})",
                confidence_level=confidence
            ))
    
    # Negative Emotions-Sleep Pattern
    if len(negative_sleep_pairs) >= 3:
        negative_vals, sleep_vals = zip(*negative_sleep_pairs)
        correlation = calculate_correlation(list(negative_vals), list(sleep_vals))
        if abs(correlation) > 0.3:
            strength = min(1.0, abs(correlation))
            confidence = "HIGH" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.5 else "LOW"
            if correlation < -0.3:
                patterns.append(PatternInsight(
                    pattern_type="negative_emotions_sleep_correlation",
                    strength=strength,
                    description=f"Higher negative emotions correlate with poor sleep (r={correlation:.2f})",
                    confidence_level=confidence
                ))
    
    # Emotional Volatility Pattern
    if len(volatility_sleep_pairs) >= 3:
        volatility_vals, sleep_vals = zip(*volatility_sleep_pairs)
        correlation = calculate_correlation(list(volatility_vals), list(sleep_vals))
        if abs(correlation) > 0.3:
            strength = min(1.0, abs(correlation))
            confidence = "HIGH" if abs(correlation) > 0.7 else "MODERATE" if abs(correlation) > 0.5 else "LOW"
            direction = "inversely" if correlation < 0 else "directly"
            patterns.append(PatternInsight(
                pattern_type="emotional_volatility_pattern",
                strength=strength,
                description=f"Emotional volatility is {direction} related to sleep quality (r={correlation:.2f})",
                confidence_level=confidence
            ))
    
    return patterns

def calculate_trend_direction(correlation_data: List[CorrelationDataPoint]) -> str:
    """Calculate overall wellness trend direction"""
    if len(correlation_data) < 3:
        return "INSUFFICIENT_DATA"
    
    # Create trend scores for recent vs earlier data
    mid_point = len(correlation_data) // 2
    earlier_data = correlation_data[:mid_point] if mid_point > 0 else []
    recent_data = correlation_data[mid_point:] if mid_point < len(correlation_data) else correlation_data
    
    def calculate_wellness_score(data_points: List[CorrelationDataPoint]) -> float:
        scores = []
        for d in data_points:
            if d.physical_energy is not None:
                scores.append(d.physical_energy / 5.0)  # Normalize to 0-1
            if d.sentiment_score is not None:
                scores.append((d.sentiment_score + 100) / 200.0)  # Convert -100,100 to 0-1
            if d.sleep_hours is not None:
                sleep_score = min(1.0, max(0.0, (d.sleep_hours - 4) / 5.0))  # 4-9 hours mapped to 0-1
                scores.append(sleep_score)
            if d.avg_negative_emotion_score is not None:
                scores.append(1.0 - d.avg_negative_emotion_score)  # Invert negative emotions
        
        return statistics.mean(scores) if scores else 0.5
    
    if not earlier_data or not recent_data:
        return "STABLE"
    
    earlier_score = calculate_wellness_score(earlier_data)
    recent_score = calculate_wellness_score(recent_data)
    
    difference = recent_score - earlier_score
    
    if difference > 0.1:
        return "IMPROVING"
    elif difference < -0.1:
        return "DECLINING"
    else:
        return "STABLE"

def calculate_enhanced_burnout_risk(sleep_avg: float, energy_avg: float, negative_emotion_avg: float, 
                                  sentiment_score: float, exercise_frequency: float, 
                                  emotional_volatility: float, patterns: List[PatternInsight]) -> Tuple[int, str]:
    """Enhanced burnout risk calculation incorporating pattern insights"""
    risk_score = 0
    
    
    sentiment_decimal = sentiment_score / 100.0 
    
    
    if sleep_avg < 5.5:
        risk_score += 25
    elif sleep_avg < 6.5:
        risk_score += 20
    elif sleep_avg < 7.0:
        risk_score += 12
    elif sleep_avg < 7.5:
        risk_score += 6
    elif sleep_avg < 8.0:
        risk_score += 2
    
   
    if energy_avg <= 1.5:
        risk_score += 20
    elif energy_avg <= 2.5:
        risk_score += 15
    elif energy_avg <= 3.0:
        risk_score += 8
    elif energy_avg <= 3.5:
        risk_score += 3
    
    
    risk_score += min(25, int(negative_emotion_avg * 100 * 0.25))
    
   
    if sentiment_decimal <= -0.5:
        risk_score += 15
    elif sentiment_decimal <= -0.3:
        risk_score += 10
    elif sentiment_decimal <= -0.1:
        risk_score += 6
    elif sentiment_decimal < 0:
        risk_score += 2
    
    
    volatility_risk = min(10, int(emotional_volatility * 50))
    risk_score += volatility_risk
    
   
    if exercise_frequency >= 0.8:
        risk_score = max(0, risk_score - 10)
    elif exercise_frequency >= 0.6:
        risk_score = max(0, risk_score - 7)
    elif exercise_frequency >= 0.4:
        risk_score = max(0, risk_score - 4)
    elif exercise_frequency >= 0.2:
        risk_score = max(0, risk_score - 2)
    
   
    pattern_adjustment = 0
    for pattern in patterns:
        if pattern.pattern_type == "sleep_energy_correlation" and pattern.strength > 0.6:
            if "negative" in pattern.description:
                pattern_adjustment += 5  # Concerning pattern
        elif pattern.pattern_type == "exercise_mood_correlation" and pattern.strength > 0.6:
            if "positive" in pattern.description:
                pattern_adjustment -= 3  # Protective pattern
        elif pattern.pattern_type == "negative_emotions_sleep_correlation" and pattern.strength > 0.6:
            pattern_adjustment += 4  # Concerning pattern
        elif pattern.pattern_type == "emotional_volatility_pattern" and pattern.strength > 0.6:
            pattern_adjustment += 3  # Volatility adds risk
    
    risk_score += pattern_adjustment
    risk_score = min(100, max(0, risk_score))
    
   
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"
    
    return risk_score, risk_level

def predict_emotional_state_enhanced(emotional_trends: Dict[str, float], sentiment_score: float, 
                                   patterns: List[PatternInsight], trend_direction: str) -> str:
    """Enhanced emotional state prediction using trends and patterns"""
    
    if not emotional_trends:
        return "Unable to determine emotional state due to insufficient data"
    
    
    sentiment_decimal = sentiment_score / 100.0  
    
    
    sorted_emotions = sorted(emotional_trends.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_score = sorted_emotions[0] if sorted_emotions else ("neutral", 0.0)
    
    
    prediction = ""
    
    
    trend_modifier = ""
    if trend_direction == "IMPROVING":
        trend_modifier = "improving "
    elif trend_direction == "DECLINING":
        trend_modifier = "concerning "
    
    
    if top_emotion == 'sadness' and top_score > 0.3:
        if sentiment_decimal <= -0.4:
            prediction = f"Likely experiencing {trend_modifier}significant depressive symptoms"
        elif sentiment_decimal <= -0.2:
            prediction = f"Likely experiencing {trend_modifier}mild to moderate sadness"
        else:
            prediction = f"Mixed emotional state with {trend_modifier}sadness component"
    
    elif top_emotion == 'fear' and top_score > 0.3:
        if sentiment_decimal <= -0.3:
            prediction = f"Likely experiencing {trend_modifier}significant anxiety or worry"
        else:
            prediction = f"Likely experiencing {trend_modifier}mild anxiety or concern"
    
    elif top_emotion == 'anger' and top_score > 0.3:
        if sentiment_decimal <= -0.2:
            prediction = f"Likely experiencing {trend_modifier}frustration or irritability"
        else:
            prediction = f"Likely experiencing {trend_modifier}mild frustration"
    
    elif top_emotion == 'joy' and top_score > 0.3:
        if sentiment_decimal >= 0.3:
            prediction = f"Very positive and {trend_modifier}joyful emotional state"
        elif sentiment_decimal >= 0.1:
            prediction = f"Generally positive and {trend_modifier}content state"
        else:
            prediction = f"Mixed emotional state with {trend_modifier}joy component"
    
    else:
        
        if sentiment_decimal >= 0.2:
            prediction = f"Generally positive {trend_modifier}emotional state"
        elif sentiment_decimal <= -0.2:
            prediction = f"Generally negative {trend_modifier}emotional state"
        else:
            prediction = f"Neutral to mixed {trend_modifier}emotional state"
    
    
    strong_patterns = [p for p in patterns if p.strength > 0.6]
    if strong_patterns:
        prediction
    
    return prediction

@app.post("/analytics/predictive-insights", response_model=PredictiveAnalyticsResponse)
async def get_predictive_insights(request: AnalyticsRequest):
    """
    Generate enhanced predictive insights with strong pattern learning from correlations
    """
    try:
        
        if request.analysis_days < 3 or request.analysis_days > 7:
            request.analysis_days = min(7, max(3, request.analysis_days))
        
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.analysis_days)
        
        
        mood_data = {}
        for mood_log in request.mood_logs:
            log_date = datetime.strptime(mood_log.date, "%Y-%m-%d")
            if start_date <= log_date <= end_date:
                mood_data[mood_log.date] = mood_log
        
        sleep_data = {}
        for sleep_log in request.sleep_logs:
            log_date = datetime.strptime(sleep_log.date, "%Y-%m-%d")
            if start_date <= log_date <= end_date:
                sleep_data[sleep_log.date] = sleep_log.sleep_hours
        
        
        exercise_impact = {}
        exercise_dates = set()
        for exercise_log in request.exercise_logs:
            log_date = datetime.strptime(exercise_log.date, "%Y-%m-%d")
            if start_date <= log_date <= end_date:
                exercise_dates.add(exercise_log.date)
               
                base_impact = min(1.0, exercise_log.duration_minutes / 60.0)  
                exercise_impact[exercise_log.date] = base_impact
        
       
        correlation_data = []
        all_dates = sorted(set(list(mood_data.keys()) + list(sleep_data.keys()) + list(exercise_impact.keys())))
        
        for date in all_dates:
            negative_emotions = None
            positive_emotions = None
            sentiment_score = None
            physical_energy = None
            volatility = None
            
            if date in mood_data:
                mood = mood_data[date]
                physical_energy = mood.physical_score
                
                
                sentiment_score = calculate_compound_sentiment(
                    mood.sentiment_positive, 
                    mood.sentiment_neutral, 
                    mood.sentiment_negative
                )
                
                
                emotions = {}
                if mood.sadness is not None: emotions['sadness'] = mood.sadness
                if mood.anger is not None: emotions['anger'] = mood.anger
                if mood.fear is not None: emotions['fear'] = mood.fear
                if mood.joy is not None: emotions['joy'] = mood.joy
                
                if emotions:
                    negative_scores = [emotions.get(e, 0) for e in ['sadness', 'anger', 'fear']]
                    positive_scores = [emotions.get(e, 0) for e in ['joy']]
                    
                    negative_emotions = statistics.mean([s for s in negative_scores if s is not None])
                    positive_emotions = statistics.mean([s for s in positive_scores if s is not None])
                    volatility = calculate_emotional_volatility(emotions)
            
            correlation_data.append(CorrelationDataPoint(
                date=date,
                sleep_hours=sleep_data.get(date),
                physical_energy=physical_energy,
                avg_negative_emotion_score=negative_emotions,
                avg_positive_emotion_score=positive_emotions,
                emotional_volatility=volatility,
                sentiment_score=sentiment_score,
                exercise_impact=exercise_impact.get(date, 0.0)
            ))
        
        
        valid_sleep = [d.sleep_hours for d in correlation_data if d.sleep_hours is not None]
        valid_energy = [d.physical_energy for d in correlation_data if d.physical_energy is not None]
        valid_negative_emotions = [d.avg_negative_emotion_score for d in correlation_data if d.avg_negative_emotion_score is not None]
        valid_sentiment = [d.sentiment_score for d in correlation_data if d.sentiment_score is not None]
        valid_volatility = [d.emotional_volatility for d in correlation_data if d.emotional_volatility is not None]
        
        sleep_avg = statistics.mean(valid_sleep) if valid_sleep else 7.5
        energy_avg = statistics.mean(valid_energy) if valid_energy else 3.0
        negative_emotion_avg = statistics.mean(valid_negative_emotions) if valid_negative_emotions else 0.0
        sentiment_avg = statistics.mean(valid_sentiment) if valid_sentiment else 0.0
        volatility_avg = statistics.mean(valid_volatility) if valid_volatility else 0.0
        exercise_frequency = len(exercise_dates) / request.analysis_days
        
        
        emotional_trends = {}
        sentiment_trends = {}
        valid_moods = [mood for mood in mood_data.values()]
        
        if valid_moods:
            emotions = ['joy', 'sadness', 'anger', 'fear']
            for emotion in emotions:
                scores = [getattr(mood, emotion) for mood in valid_moods if getattr(mood, emotion) is not None]
                if scores:
                    emotional_trends[emotion] = statistics.mean(scores)
            
            
            sentiment_fields = ['positive', 'neutral', 'negative']
            for field in sentiment_fields:
                attr_name = f'sentiment_{field}'
                scores = [getattr(mood, attr_name) for mood in valid_moods if getattr(mood, attr_name) is not None]
                if scores:
                    sentiment_trends[field] = statistics.mean(scores)
        
        
        patterns = detect_patterns(correlation_data)
        
        
        trend_direction = calculate_trend_direction(correlation_data)
        
        # Enhanced burnout risk calculation
        burnout_score, risk_level = calculate_enhanced_burnout_risk(
            sleep_avg, energy_avg, negative_emotion_avg, sentiment_avg, 
            exercise_frequency, volatility_avg, patterns
        )
        
        
        predicted_state = predict_emotional_state_enhanced(
            emotional_trends, sentiment_avg, patterns, trend_direction
        )
        
        
        dominant_negative = [emotion for emotion, score in emotional_trends.items() 
                           if emotion in ['sadness', 'anger', 'fear'] and score > 0.2]
        
        recommendation = generate_activity_recommendation(
            burnout_score, risk_level, energy_avg, sleep_avg, exercise_frequency, dominant_negative
        )
        
        
        if patterns:
            strong_patterns = [p for p in patterns if p.strength > 0.6]
            if strong_patterns:
                recommendation += f" Based on detected patterns: Focus on areas showing strong correlations."
        
        
        data_quality = len(correlation_data) / request.analysis_days
        pattern_strength = statistics.mean([p.strength for p in patterns]) if patterns else 0.0
        confidence_score = min(1.0, (data_quality + pattern_strength) / 2.0)
        
        
        insights_summary = f"Enhanced analysis of {request.analysis_days} days with {len(patterns)} significant patterns detected. "
        insights_summary += f"Trend: {trend_direction.lower()}. "
        
        if confidence_score > 0.7:
            insights_summary += "High confidence in predictions due to strong data patterns. "
        elif confidence_score > 0.4:
            insights_summary += "Moderate confidence in predictions. "
        else:
            insights_summary += "Lower confidence due to limited data patterns. "
        
        if burnout_score >= 70:
            insights_summary += f"HIGH burnout risk detected (score: {burnout_score}/100). "
        elif burnout_score >= 40:
            insights_summary += f"MODERATE burnout risk (score: {burnout_score}/100). "
        else:
            insights_summary += f"LOW burnout risk (score: {burnout_score}/100). "
        
        # Calculate key correlations for response
        correlations = {}
        if len(valid_sleep) >= 3 and len(valid_energy) >= 3:
            correlations['sleep_energy'] = calculate_correlation(valid_sleep, valid_energy)
        if len(valid_sleep) >= 3 and len(valid_sentiment) >= 3:
            correlations['sleep_sentiment'] = calculate_correlation(valid_sleep, valid_sentiment)
        if exercise_frequency > 0 and len(valid_energy) >= 3:
            exercise_vals = [d.exercise_impact for d in correlation_data if d.exercise_impact is not None]
            if len(exercise_vals) >= 3:
                correlations['exercise_energy'] = calculate_correlation(exercise_vals, valid_energy)
        
        predictive_insights = PredictiveInsights(
            burnout_risk_score=burnout_score,
            risk_level=risk_level,
            predicted_emotional_state=predicted_state,
            activity_recommendation=recommendation,
            insights_summary=insights_summary,
            pattern_insights=patterns,
            trend_direction=trend_direction,
            confidence_score=confidence_score
        )
        
        return PredictiveAnalyticsResponse(
            correlation_data=correlation_data,
            predictive_insights=predictive_insights,
            emotional_trends=emotional_trends,
            sentiment_trends=sentiment_trends,
            correlations=correlations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating enhanced predictive insights: {str(e)}")

def generate_activity_recommendation(risk_score: int, risk_level: str, energy_avg: float,
                                   sleep_avg: float, exercise_frequency: float, 
                                   dominant_negative_emotions: List[str]) -> str:
    """Enhanced pattern-aware activity recommendations"""
    
    base_recommendation = ""
    
   
    if risk_level == "HIGH":
        if energy_avg <= 2 and sleep_avg < 6:
            base_recommendation = "PRIORITY: Sleep optimization (7-8 hours) and energy conservation. Avoid demanding activities."
        elif energy_avg <= 2:
            base_recommendation = "Focus on gentle, restorative activities: light stretching, meditation, short nature walks."
        elif sleep_avg < 6:
            base_recommendation = "Prioritize sleep hygiene: consistent schedule, screen limits, relaxation techniques."
        else:
            base_recommendation = "Comprehensive stress management needed: professional support, mindfulness, reduced commitments."
    
    elif risk_level == "MODERATE":
        if energy_avg <= 2.5:
            base_recommendation = "Balanced approach: alternate rest days with light activity (walking, gentle yoga)."
        elif exercise_frequency < 0.3:
            base_recommendation = "Gradually increase activity: start with 20-minute daily walks, build consistency."
        else:
            base_recommendation = "Maintain current routines while adding stress-relief practices (breathing, journaling)."
    
    else:  
        if energy_avg >= 4 and exercise_frequency >= 0.7:
            base_recommendation = "Excellent foundation! Consider new challenges or activities to maintain engagement."
        else:
            base_recommendation = "Build on your solid base: slight increases in preferred activities."
    
     
    if 'sadness' in dominant_negative_emotions:
        base_recommendation += " Include mood-lifting activities: sunlight exposure, social connections."
    if 'fear' in dominant_negative_emotions:
        base_recommendation += " Add anxiety management: progressive relaxation, grounding techniques."
    if 'anger' in dominant_negative_emotions:
        base_recommendation += " Include tension release: vigorous walking, punching bag, or physical outlet."
    
    return base_recommendation