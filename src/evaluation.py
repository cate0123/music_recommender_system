# src/evaluation.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from utils import load_processed, top_k_popular
from src.CF import ItemSimilarityRecommender  # type: ignore
from src.deep_model import train_and_save  # if needed to generate embeddings

def ndcg_at_k(recommended_ids, true_relevance_map, k=10):
    """Compute NDCG@k for a single recommendation list."""
    gains = []
    for i, rid in enumerate(recommended_ids[:k]):
        rel = true_relevance_map.get(rid, 0.0)
        gains.append((2**rel - 1) / np.log2(i + 2))
    dcg = np.sum(gains)
    ideal = sorted(true_relevance_map.values(), reverse=True)[:k]
    idcg = np.sum([(2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal)]) if ideal else 0.0
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_proxy_k(recommender_func, seeds, test_df, k=10):
    """
    Evaluate a recommender function using NDCG@k over a set of seeds.
    recommender_func must accept (seed, k) as arguments.
    """
    true_rel = {rid: pop for rid, pop in zip(test_df['id'], test_df['popularity'])}
    scores = []
    for seed in seeds:
        recs = recommender_func(seed, k)
        ndcg = ndcg_at_k(recs, true_rel, k)
        scores.append(ndcg)
    return np.mean(scores)

def evaluate_all(k=10, sample_n=200):
    """Evaluate all recommenders: popularity, CF (TF-IDF), embedding-based."""
    train, val, test, full = load_processed()
    seeds = test['id'].sample(min(sample_n, len(test)), random_state=42).tolist()

    # -------------------------
    # Popularity baseline
    # -------------------------
    pop_recs = lambda seed, k: top_k_popular(train, k=k)  # ignores seed
    pop_score = evaluate_proxy_k(pop_recs, seeds, test, k=k)
    print(f"Popularity baseline NDCG@{k}: {pop_score:.4f}")

    # -------------------------
    # TF-IDF Item-similarity baseline
    # -------------------------
    rec = ItemSimilarityRecommender(train)
    tfidf_score = evaluate_proxy_k(rec.recommend, seeds, test, k=k)
    print(f"TF-IDF item-sim NDCG@{k}: {tfidf_score:.4f}")

    # -------------------------
    # Embedding-based recommender
    # -------------------------
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    emb_path = os.path.join(model_dir, "embeddings.joblib")

    if os.path.exists(emb_path):
        emb_mats = joblib.load(emb_path)
        g_emb = emb_mats.get('emb_genre')
        a_emb = emb_mats.get('emb_artists')
        al_emb = emb_mats.get('emb_album')

        if g_emb is None or a_emb is None or al_emb is None:
            print("Expected 'emb_genre', 'emb_artists', 'emb_album' in embeddings.joblib")
        else:
            df = full.reset_index(drop=True)
            id_to_idx = {rid: idx for idx, rid in enumerate(df['id'].tolist())}

            # Precompute vector for each track
            vectors = np.vstack([
                np.concatenate([
                    g_emb[int(r['genre_id'])],
                    a_emb[int(r['artists_id'])],
                    al_emb[int(r['album_id'])],
                    [r['duration_s'], r['explicit']]
                ]) for _, r in df.iterrows()
            ])

            # Embedding-based recommender function
            def embed_recs(seed, k=10):
                if seed not in id_to_idx:
                    return top_k_popular(train, k=k)
                q = vectors[id_to_idx[seed]].reshape(1, -1)
                sims = cosine_similarity(q, vectors)[0]
                top_idx = np.argsort(sims)[::-1][1:k+1]  # exclude self
                return df.iloc[top_idx]['id'].tolist()

            emb_score = evaluate_proxy_k(embed_recs, seeds, test, k=k)
            print(f"Embedding-based NDCG@{k}: {emb_score:.4f}")
    else:
        print("No trained embeddings found. Run deep_model.py to generate embeddings.")

if __name__ == "__main__":
    evaluate_all()
