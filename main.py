from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
from transformers import pipeline

app = FastAPI(title="Mental Well-being API", version="2.0")

# Models
sentiment_analyzer = SentimentIntensityAnalyzer()
kw_model = KeyBERT()
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

class MoodEntry(BaseModel):
    date: str
    mood_score: Optional[int] = None
    mood_label: Optional[str] = None
    journal_text: Optional[str] = None
    sleep_hours: Optional[float] = None
    exercise_done: Optional[bool] = None

class MoodLogRequest(BaseModel):
    logs: List[MoodEntry]

# --- Mood API ---
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

# --- Journal API with Advanced Sentiment & Emotion Analysis ---
@app.post("/analyze_journal")
def analyze_journal(request: MoodLogRequest):
    logs = [log for log in request.logs if log.journal_text]
    if not logs:
        return {"message": "No journal entries provided."}

    all_texts = [log.journal_text for log in logs]

    # --- Sentiment Analysis (positive/neutral/negative) ---
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for text in all_texts:
        score = sentiment_analyzer.polarity_scores(text)
        if score["compound"] > 0.05:
            sentiment_counts["positive"] += 1
        elif score["compound"] < -0.05:
            sentiment_counts["negative"] += 1
        else:
            sentiment_counts["neutral"] += 1
    total = sum(sentiment_counts.values())
    sentiment_distribution = {k: round((v / total) * 100, 2) for k, v in sentiment_counts.items()}

    # --- Advanced Emotion Analysis ---
    emotion_summary = {}
    for text in all_texts:
        scores = emotion_analyzer(text)[0]  # list of emotions with scores
        for item in scores:
            label = item['label']
            score = item['score']
            emotion_summary[label] = max(score, emotion_summary.get(label, 0))  # keep max intensity

    # --- Keyword / Theme Extraction (important ones only) ---
    joined_text = " ".join(all_texts)
    extracted = kw_model.extract_keywords(
        joined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=15,
        use_maxsum=True,
        nr_candidates=30
    )

    keywords = []
    seen_tokens = set()
    for word, score in extracted:
        if score < 0.3:
            continue
        tokens = set(word.lower().split())
        if not tokens & seen_tokens:
            keywords.append(word)
            seen_tokens.update(tokens)
        if len(keywords) >= 7:
            break

    return {
        "sentiment_distribution": sentiment_distribution,
        "emotions": emotion_summary,
        "topics": keywords
    }

@app.post("/generate_insights")
def generate_insights(request: MoodLogRequest):
    logs = [log for log in request.logs if log.journal_text]
    if not logs:
        return {"message": "No journal entries for insights."}

    insights = []

    # 1️⃣ Emotion / sentiment hints from journal
    negative_keywords = ["stress", "anxiety", "pressure", "deadline"]
    for i, log in enumerate(logs):
        if any(nk in log.journal_text.lower() for nk in negative_keywords):
            insights.append(f"Entry {i+1}: Your journal mentions stress-related topics. Consider relaxing activities.")

    # 2️⃣ Sleep habit correlation
    sleep_logs = [log.sleep_hours for log in logs if log.sleep_hours is not None]
    if sleep_logs:
        avg_sleep = sum(sleep_logs) / len(sleep_logs)
        if avg_sleep < 6:
            insights.append(f"Your average sleep is {avg_sleep:.1f} hours. More sleep may improve your mood.")

    # 3️⃣ Exercise habit correlation
    exercise_logs = [log.exercise_done for log in logs if log.exercise_done is not None]
    if exercise_logs:
        exercise_ratio = sum(exercise_logs) / len(exercise_logs)
        if exercise_ratio < 0.5:
            insights.append("Consider regular exercise; it often improves mood and reduces stress.")

    # 4️⃣ Fallback if no major insights
    if not insights:
        insights.append("Keep up the journaling! Tracking your mood regularly helps improve self-awareness.")

    return {
        "insights": insights
    }
