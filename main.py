from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
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
import asyncio
from functools import lru_cache

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
kw_model = KeyBERT("all-MiniLM-L6-v2")

GEMINI_API_KEY =os.getenv("GEMINI_API_Key")

class MoodEntry(BaseModel):
    date: str
    mood_score: Optional[int] = None
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
# Helper Fucntions
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
    return all_emotions[:3]  
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
        "mood_scores": scores,
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
