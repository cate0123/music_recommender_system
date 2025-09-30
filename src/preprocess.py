# src/preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

 #Base project directory (two levels up from this file)S
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
    expected = {'id','name','genre','artists','album','duration_ms','explicit','popularity'}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df.copy()
    # Basic cleaning
    df['genre'] = df['genre'].fillna('Unknown')
    df['album'] = df['album'].fillna('Unknown')
    df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
    # convert explicit to 0/1 (handles strings like True/False or 0/1)
    df['explicit'] = df['explicit'].astype(int)
    # filter unrealistic durations (ms) -> 30s to 15min
    df = df[(df['duration_ms'] >= 30_000) & (df['duration_ms'] <= 15*60*1000)].copy()
    # clamp popularity
    df['popularity'] = df['popularity'].clip(0,100)
    return df

def encode_and_save(df, test_size=0.15, random_state=42):
    df = df.copy()
    # Label encode categorical columns
    le_genre = LabelEncoder().fit(df['genre'])
    le_artist = LabelEncoder().fit(df['artists'])
    le_album = LabelEncoder().fit(df['album'])
    df['genre_id'] = le_genre.transform(df['genre'])
    df['artists_id'] = le_artist.transform(df['artists'])
    df['album_id'] = le_album.transform(df['album'])

    # convert duration to seconds and normalize duration + popularity
    df['duration_s'] = df['duration_ms'] / 1000.0
    scaler = MinMaxScaler()
    df[['duration_s', 'popularity']] = scaler.fit_transform(df[['duration_s', 'popularity']])

    # train/val/test split (stratify by genre_id)
    train_val, test = train_test_split(df, test_size=test_size, stratify=df['genre_id'], random_state=random_state)
    train, val = train_test_split(train_val, test_size=test_size/(1-test_size), stratify=train_val['genre_id'], random_state=random_state)

    # Save CSVs
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    processed_dir = os.path.join(base, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    train.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    # persist encoders and scaler
    joblib.dump(le_genre, os.path.join(processed_dir, "le_genre.joblib"))
    joblib.dump(le_artist, os.path.join(processed_dir, "le_artist.joblib"))
    joblib.dump(le_album, os.path.join(processed_dir, "le_album.joblib"))
    joblib.dump(scaler, os.path.join(processed_dir, "scaler.joblib"))

    print(f"Saved processed splits to {processed_dir}")
    return train, val, test

if __name__ == "__main__":
    df = load_raw()
    df = clean_dataframe(df)
    encode_and_save(df)
