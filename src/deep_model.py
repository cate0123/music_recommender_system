# src/deep_model.py
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras import layers, Model, optimizers, losses, callbacks # type: ignore
from src.utils import load_processed

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_metadata_ncf(num_genres, num_artists, num_albums, emb_dim=32):
    # Inputs
    inp_genre = layers.Input(shape=(), dtype='int32', name='genre_id')
    inp_artist = layers.Input(shape=(), dtype='int32', name='artists_id')
    inp_album = layers.Input(shape=(), dtype='int32', name='album_id')
    inp_duration = layers.Input(shape=(1,), dtype='float32', name='duration_s')
    inp_explicit = layers.Input(shape=(1,), dtype='float32', name='explicit')

    # Embedding layers with explicit names
    emb_genre = layers.Embedding(input_dim=num_genres, output_dim=emb_dim, name='emb_genre')(inp_genre)
    emb_artists = layers.Embedding(input_dim=num_artists, output_dim=emb_dim, name='emb_artists')(inp_artist)
    emb_album = layers.Embedding(input_dim=num_albums, output_dim=emb_dim, name='emb_album')(inp_album)

    g = layers.Flatten()(emb_genre)
    a = layers.Flatten()(emb_artists)
    al = layers.Flatten()(emb_album)

    x = layers.Concatenate()([g, a, al, inp_duration, inp_explicit])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid', name='pred_pop')(x)  # predicts normalized popularity

    model = Model(inputs=[inp_genre, inp_artist, inp_album, inp_duration, inp_explicit], outputs=out)
    model.compile(optimizer=optimizers.Adam(1e-3), loss=losses.MeanSquaredError(), metrics=['RootMeanSquaredError'])
    return model

def df_to_inputs(df):
    X = {
        'genre_id': df['genre_id'].astype(int).values,
        'artists_id': df['artists_id'].astype(int).values,
        'album_id': df['album_id'].astype(int).values,
        'duration_s': df['duration_s'].astype(float).values,
        'explicit': df['explicit'].astype(float).values
    }
    y = df['popularity'].astype(float).values
    return X, y

def train_and_save(epochs=30, batch_size=256):
    train, val, test, full = load_processed()
    # vocab sizes
    num_genres = int(train['genre_id'].max()) + 1
    num_artists = int(train['artists_id'].max()) + 1
    num_albums = int(train['album_id'].max()) + 1
    model = build_metadata_ncf(num_genres, num_artists, num_albums, emb_dim=32)
    X_train, y_train = df_to_inputs(train)
    X_val, y_val = df_to_inputs(val)
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=cb)
    # Save model and embeddings
    model_path = os.path.join(MODEL_DIR, "ncf_metadata.h5")
    model.save(model_path)
    print("Saved model to", model_path)
    # Save embedding matrices separately
    emb_mats = {}
    for layer in model.layers:
        if isinstance(layer, layers.Embedding):
            emb_mats[layer.name] = layer.get_weights()[0]
    joblib.dump(emb_mats, os.path.join(MODEL_DIR, "embeddings.joblib"))
    print("Saved embeddings to models/embeddings.joblib")
    return model

if __name__ == "__main__":
    train_and_save()
