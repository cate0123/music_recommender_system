# src/preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Base project directory (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Raw data path
DATA_PATH = os.path.join(BASE_DIR, "data", "spotify_tracks.csv")

# Processed data directory
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean_dataframe(df):
    # Ensure expected columns
    expected = {'id', 'name', 'genre', 'artists', 'album', 'duration_ms', 'explicit', 'popularity'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df.copy()
    # Basic cleaning
    df['genre'] = df['genre'].fillna('Unknown')
    df['album'] = df['album'].fillna('Unknown')
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
    # Convert explicit to 0/1
    df['explicit'] = df['explicit'].astype(int)
    # Filter unrealistic durations (ms) -> 30s to 15min
    df = df[(df['duration_ms'] >= 30_000) & (df['duration_ms'] <= 15 * 60 * 1000)].copy()
    # Clamp popularity
    df['popularity'] = df['popularity'].clip(0, 100)
    return df


def encode_and_save(df, val_ratio=0.15, test_ratio=0.15, random_state=42):
    df = df.copy()
    # Label encode categorical columns
    le_genre = LabelEncoder().fit(df['genre'])
    le_artist = LabelEncoder().fit(df['artists'])
    le_album = LabelEncoder().fit(df['album'])
    df['genre_id'] = le_genre.transform(df['genre'])
    df['artists_id'] = le_artist.transform(df['artists'])
    df['album_id'] = le_album.transform(df['album'])

    # Convert duration to seconds and normalize duration + popularity
    df['duration_s'] = df['duration_ms'] / 1000.0
    scaler = MinMaxScaler()
    df[['duration_s', 'popularity']] = scaler.fit_transform(df[['duration_s', 'popularity']])

    # --- Fixed train/val/test split ---
    # Ensure validation and test tracks exist in training
    track_ids = df['id'].tolist()
    train_val_ids, test_ids = train_test_split(track_ids, test_size=test_ratio, random_state=random_state)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_ratio / (1 - test_ratio), random_state=random_state)

    train = df[df['id'].isin(train_ids)].reset_index(drop=True)
    val = df[df['id'].isin(val_ids)].reset_index(drop=True)
    test = df[df['id'].isin(test_ids)].reset_index(drop=True)

    # Save CSVs
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    # Persist encoders and scaler
    joblib.dump(le_genre, os.path.join(PROCESSED_DIR, "le_genre.joblib"))
    joblib.dump(le_artist, os.path.join(PROCESSED_DIR, "le_artist.joblib"))
    joblib.dump(le_album, os.path.join(PROCESSED_DIR, "le_album.joblib"))
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))

    print(f"Saved processed splits to {PROCESSED_DIR}")
    return train, val, test


if __name__ == "__main__":
    df = load_raw()
    df = clean_dataframe(df)
    encode_and_save(df)
