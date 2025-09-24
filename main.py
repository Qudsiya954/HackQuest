from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT

app = FastAPI(title="Mental Well-being API", version="1.0")

# Models
sentiment_analyzer = SentimentIntensityAnalyzer()
kw_model = KeyBERT()

class MoodEntry(BaseModel):
    date: str
    mood_score: Optional[int] = None
    mood_label: Optional[str] = None
    journal_text: Optional[str] = None

class MoodLogRequest(BaseModel):
    logs: List[MoodEntry]

@app.post("/analyze_mood")
def analyze_mood(request: MoodLogRequest):
    logs = [log for log in request.logs if log.mood_score is not None and log.mood_label is not None]
    if not logs:
        return {"message": "No mood logs provided."}

    # Dates and scores
    dates = [log.date for log in logs]
    scores = [log.mood_score for log in logs]

    # Mood distribution
    mood_distribution: Dict[str, int] = {}
    for log in logs:
        mood_distribution[log.mood_label] = mood_distribution.get(log.mood_label, 0) + 1

    return {
        "dates": dates,
        "mood_scores": scores,
        "mood_distribution": mood_distribution
    }
@app.post("/analyze_journal")
def analyze_journal(request: MoodLogRequest):
    logs = [log for log in request.logs if log.journal_text]
    if not logs:
        return {"message": "No journal entries provided."}

    all_texts = [log.journal_text for log in logs]

    # Sentiment analysis
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

    # Keyword extraction (semantic-aware + token-level deduplication)
    # Keyword extraction (important ones only for dashboard)
    joined_text = " ".join(all_texts)
    extracted = kw_model.extract_keywords(
        joined_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=15,          # extract more candidates
        use_maxsum=True,
        nr_candidates=30
    )

    keywords = []
    seen_tokens = set()
    for word, score in extracted:
        if score < 0.3:  # keep only important keywords
            continue
        tokens = set(word.lower().split())
        if not tokens & seen_tokens:
            keywords.append(word)
            seen_tokens.update(tokens)
        if len(keywords) >= 7:  # enough for dashboard visualization
            break

    return {
        "sentiment_distribution": sentiment_distribution,
        "keywords": keywords
    }
