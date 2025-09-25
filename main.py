from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
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

GEMINI_API_KEY =os.getenv("GEMINI_API")

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
#     prompt = f"""
#    You are a supportive mental health assistant. 💜  
#    Here is a journal entry summary:  
#    - Emotions: {top_emotions}  
#    - Sentiment: {sentiment_distribution}  
#    - Topics: {topics}  

#     Write a short (1–2 sentence) empathetic and encouraging note.  
#    ✨ Keep it warm, validating, and human-like.  
#    🌱 Use 2–3 emojis to add care and positivity.  
#    ❌ Do NOT give medical, therapeutic, or diagnostic advice.  
#    ✅ Focus only on appreciation, gentle encouragement, or validation.  
#   """

    # ai_comment = generate_llm_comment(prompt)

    return {
        "sentiment_distribution": sentiment_distribution,
        "emotions": top_emotions,
        "topics": topics,
        # "ai_comment": ai_comment
    }