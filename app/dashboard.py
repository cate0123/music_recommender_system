# app/dashboard.py
import streamlit as st
import pandas as pd
import os
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import load_processed

from src.CF import ItemSimilarityRecommender
from src.deep_model import train_and_save
import joblib



st.set_page_config(layout="wide", page_title="Music Recommender Demo")

st.title("Music Recommender Demo (Metadata-based)")

@st.cache_data
def load_data():
    train, val, test, full = load_processed()
    return train, val, test, full

train, val, test, full = load_data()
st.sidebar.header("Options")
k = st.sidebar.slider("Top-K", 1, 20, 10)

st.subheader("Sample tracks")
sample = st.selectbox("Pick a seed track (from train)", train[['id','name']].sample(50, random_state=1).values.tolist(), format_func=lambda x: f"{x[1]} ({x[0]})")
seed_id = sample[0] if sample else train['id'].iloc[0]

st.markdown("### Popularity baseline")
top_pop = train.sort_values('popularity', ascending=False).head(k)[['name','id','popularity']]
st.dataframe(top_pop)

st.markdown("### Item-similarity (TF-IDF) recommendations")
rec = ItemSimilarityRecommender(train)
sim_ids = rec.recommend(seed_id, k=k)
sim_df = full[full['id'].isin(sim_ids)][['name','artists','album','genre','popularity']].reset_index(drop=True)
st.dataframe(sim_df)

st.markdown("### Embedding-based recommendations (if available)")
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
emb_path = os.path.join(models_dir, "embeddings.joblib")
if os.path.exists(emb_path):
    emb_mats = joblib.load(emb_path)
    g_emb = emb_mats.get('emb_genre'); a_emb = emb_mats.get('emb_artists'); al_emb = emb_mats.get('emb_album')
    if g_emb is not None and a_emb is not None and al_emb is not None:
        full_df = full.reset_index(drop=True)
        def build_vecs(df):
            import numpy as np
            vecs = []
            for _, r in df.iterrows():
                g = g_emb[int(r['genre_id'])]
                a = a_emb[int(r['artists_id'])]
                al = al_emb[int(r['album_id'])]
                vecs.append(np.concatenate([g,a,al,[r['duration_s'], r['explicit']]]))
            return np.vstack(vecs)
        vectors = build_vecs(full_df)
        id_to_idx = {rid: idx for idx, rid in enumerate(full_df['id'].tolist())}
        if seed_id in id_to_idx:
            from sklearn.metrics.pairwise import cosine_similarity
            q = vectors[id_to_idx[seed_id]].reshape(1,-1)
            sims = cosine_similarity(q, vectors)[0]
            top_idx = sims.argsort()[::-1][1:k+1]
            emb_ids = full_df.iloc[top_idx]['id'].tolist()
            emb_df = full[full['id'].isin(emb_ids)][['name','artists','album','genre','popularity']].reset_index(drop=True)
            st.dataframe(emb_df)
        else:
            st.info("Seed not in indexed set for embedding recommendations.")
    else:
        st.info("Embedding matrices not found with expected names; run training.")
else:
    st.info("No embeddings found. Run the deep model training script to generate embeddings.")
