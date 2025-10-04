# src/evaluation.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_processed, top_k_popular
from src.CF import ItemSimilarityRecommender  
# src/evaluation.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import load_processed, top_k_popular
from src.CF import ItemSimilarityRecommender  


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


def precision_recall_at_k(recommended_ids, true_relevance_map, k=10, threshold=0.5):
    """Compute Precision@K and Recall@K for one seed.
       threshold: relevance cutoff for popularity (scaled 0–1)."""
    recommended_topk = recommended_ids[:k]
    relevant_items = {rid for rid, pop in true_relevance_map.items() if pop >= threshold}
    hits = sum([1 for rid in recommended_topk if rid in relevant_items])
    precision = hits / k
    recall = hits / len(relevant_items) if relevant_items else 0.0
    return precision, recall


def evaluate_proxy_k(recommender_func, seeds, test_df, k=10, threshold=0.5):
    """Evaluate NDCG, Precision@K, Recall@K over a set of seeds."""
    true_rel = {rid: pop for rid, pop in zip(test_df['id'], test_df['popularity'])}
    ndcg_scores, precisions, recalls = [], [], []

    for seed in seeds:
        recs = recommender_func(seed, k)
        ndcg_scores.append(ndcg_at_k(recs, true_rel, k))
        p, r = precision_recall_at_k(recs, true_rel, k, threshold)
        precisions.append(p)
        recalls.append(r)

    return np.mean(ndcg_scores), np.mean(precisions), np.mean(recalls)


def evaluate_all(k=10, sample_n=200):
    """Evaluate all recommenders: Popularity, CF, Embedding-based."""
    train, val, test, full = load_processed()
    seeds = test['id'].sample(min(sample_n, len(test)), random_state=42).tolist()

    results = []

    # -------------------------
    # Popularity baseline
    # -------------------------
    pop_recs = lambda seed, k: top_k_popular(train, k=k)
    ndcg, prec, rec = evaluate_proxy_k(pop_recs, seeds, test, k=k)
    results.append({'Model': 'Popularity', 'NDCG@10': ndcg, 'Precision@10': prec, 'Recall@10': rec})

    # -------------------------
    # Item-similarity CF baseline
    # -------------------------
    rec = ItemSimilarityRecommender(train)
    ndcg, prec, rec_val = evaluate_proxy_k(rec.recommend, seeds, test, k=k)
    results.append({'Model': 'Item-based CF', 'NDCG@10': ndcg, 'Precision@10': prec, 'Recall@10': rec_val})

    # -------------------------
    # Embedding-based recommender (manual feature concat)
    # -------------------------
    df = full.reset_index(drop=True)
    id_to_idx = {rid: idx for idx, rid in enumerate(df['id'].tolist())}

    vectors = np.vstack([
        np.array([
            df.loc[idx, 'genre_id'],
            df.loc[idx, 'artists_id'],
            df.loc[idx, 'album_id'],
            df.loc[idx, 'duration_s'],
            df.loc[idx, 'explicit'],
            df.loc[idx, 'popularity']
        ])
        for idx in range(len(df))
    ])

    def embed_recs(seed, k=10):
        if seed not in id_to_idx:
            return top_k_popular(train, k=k)
        q = vectors[id_to_idx[seed]].reshape(1, -1)
        sims = cosine_similarity(q, vectors)[0]
        top_idx = np.argsort(sims)[::-1][1:k+1]
        return df.iloc[top_idx]['id'].tolist()

    ndcg, prec, rec_val = evaluate_proxy_k(embed_recs, seeds, test, k=k)
    results.append({'Model': 'Embedding-based', 'NDCG@10': ndcg, 'Precision@10': prec, 'Recall@10': rec_val})

    # -------------------------
    # Show results in a table
    # -------------------------
    results_df = pd.DataFrame(results)
    print("\n===== Evaluation Results =====")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_all()


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


def precision_recall_at_k(recommended_ids, true_relevance_map, k=10, threshold=0.5):
    """Compute Precision@K and Recall@K for one seed.
       threshold: relevance cutoff for popularity (scaled 0–1)."""
    recommended_topk = recommended_ids[:k]
    relevant_items = {rid for rid, pop in true_relevance_map.items() if pop >= threshold}
    hits = sum([1 for rid in recommended_topk if rid in relevant_items])
    precision = hits / k
    recall = hits / len(relevant_items) if relevant_items else 0.0
    return precision, recall


def evaluate_proxy_k(recommender_func, seeds, test_df, k=10, threshold=0.2):
    """Evaluate NDCG, Precision@K, Recall@K over a set of seeds."""
    true_rel = {rid: pop for rid, pop in zip(test_df['id'], test_df['popularity'])}
    ndcg_scores, precisions, recalls = [], [], []

    for seed in seeds:
        recs = recommender_func(seed, k)
        ndcg_scores.append(ndcg_at_k(recs, true_rel, k))
        p, r = precision_recall_at_k(recs, true_rel, k, threshold)
        precisions.append(p)
        recalls.append(r)

    return np.mean(ndcg_scores), np.mean(precisions), np.mean(recalls)


def evaluate_all(k=10, sample_n=200):
    """Evaluate all recommenders: Popularity, CF, Embedding-based."""
    train, val, test, full = load_processed()
    seeds = test['id'].sample(min(sample_n, len(test)), random_state=42).tolist()

    results = []

    # -------------------------
    # Popularity baseline
    # -------------------------
    pop_recs = lambda seed, k: top_k_popular(train, k=k)
    ndcg, prec, rec = evaluate_proxy_k(pop_recs, seeds, test, k=k)
    results.append({'Model': 'Popularity', 'NDCG@10': ndcg, 'Precision@10': prec, 'Recall@10': rec})

    # -------------------------
    # Item-similarity CF baseline
    # -------------------------
    rec = ItemSimilarityRecommender(train)
    ndcg, prec, rec_val = evaluate_proxy_k(rec.recommend, seeds, test, k=k)
    results.append({'Model': 'Item-based CF', 'NDCG@10': ndcg, 'Precision@10': prec, 'Recall@10': rec_val})

    # -------------------------
    # Embedding-based recommender (manual feature concat)
    # -------------------------
    df = full.reset_index(drop=True)
    id_to_idx = {rid: idx for idx, rid in enumerate(df['id'].tolist())}

    vectors = np.vstack([
        np.array([
            df.loc[idx, 'genre_id'],
            df.loc[idx, 'artists_id'],
            df.loc[idx, 'album_id'],
            df.loc[idx, 'duration_s'],
            df.loc[idx, 'explicit'],
            df.loc[idx, 'popularity']
        ])
        for idx in range(len(df))
    ])

    def embed_recs(seed, k=10):
        if seed not in id_to_idx:
            return top_k_popular(train, k=k)
        q = vectors[id_to_idx[seed]].reshape(1, -1)
        sims = cosine_similarity(q, vectors)[0]
        top_idx = np.argsort(sims)[::-1][1:k+1]
        return df.iloc[top_idx]['id'].tolist()

    ndcg, prec, rec_val = evaluate_proxy_k(embed_recs, seeds, test, k=k)
    results.append({'Model': 'Embedding-based', 'NDCG@10': ndcg, 'Precision@10': prec, 'Recall@10': rec_val})

    # -------------------------
    # Show results in a table
    # -------------------------
    results_df = pd.DataFrame(results)
    print("\n===== Evaluation Results =====")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_all()
