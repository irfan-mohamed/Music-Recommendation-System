import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from src.recommendation import recommend



app = FastAPI(title="Spotify Recommender API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global objects
df = None
pipeline = None


# Loading artifacts

@app.on_event("startup")
def load_artifacts():
    global df, pipeline

    model_path = os.path.join(BASE_DIR, "models", "kmeans_pipeline.pkl")
    data_path = os.path.join(BASE_DIR, "data", "processed", "tracks_with_clusters.csv")
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

@app.get("/")
def home():
    return {"message": "Spotify Recommender API is running"}


# Recommendation endpoint

@app.get("/recommend/{track_id}")
def get_recommendations(track_id: str, top_n: int = 10):
    global df, pipeline

    if df is None or pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    label, result = recommend(track_id, df, pipeline, top_n)

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return label, result.to_dict(orient="records")
