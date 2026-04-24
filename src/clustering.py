import sys
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.pipeline import build_pipeline
from src.preprocessing import clean_data

# Load data
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Final Dataset.csv")
df = pd.read_csv(DATA_PATH)

df = clean_data(df)

# Features
num_features = [
    'danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness',
    'valence','tempo','duration_ms',
    'energy_dance_ratio','acoustic_softness','vocal_presence'
]

cat_features = ['key','mode','time_signature']


# Build pipeline
pipeline = build_pipeline(num_features, cat_features)
pipeline.fit(df)
clusters = pipeline.predict(df)


# Evaluation
X_transformed = pipeline[:-1].transform(df)
score = silhouette_score(X_transformed, clusters)

print("Clusters:", set(clusters))
print("Silhouette Score:", score)

df["cluster"] = pipeline.predict(df)

cluster_labels = {
    0 : 'Energetic Happy',
    1 : 'Party',
    2 : 'Calm Acoustic',
    3 : 'Balanced'
}

df["cluster_label"] = df['cluster'].map(cluster_labels)
# Save dataset
df.to_csv(
    os.path.join(BASE_DIR, "data", "processed", "tracks_with_clusters.csv"),
    index=False
)

# Save model
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(pipeline, os.path.join(BASE_DIR, "models", "kmeans_pipeline.pkl"))
