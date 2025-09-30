
# src/collaborative_filtering.py
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.utils import load_processed

class ItemSimilarityRecommender:
    def __init__(self, train_df):
        self.train = train_df.copy().reset_index(drop=True)
        self._build_meta()
        self._build_tfidf()

    def _build_meta(self):
        # create a combined metadata text column
        self.items = self.train[['id','name','genre','artists','album']].copy()
        self.items['meta'] = (self.items['genre'].astype(str) + ' ' +
                              self.items['artists'].astype(str) + ' ' +
                              self.items['album'].astype(str) + ' ' +
                              self.items['name'].astype(str))
        self.id_to_index = {id_: idx for idx, id_ in enumerate(self.items['id'].tolist())}

    def _build_tfidf(self):
        self.tfidf = TfidfVectorizer(min_df=1, ngram_range=(1,2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.items['meta'])
        self.sim_matrix = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def recommend(self, track_id, k=10):
        if track_id not in self.id_to_index:
            # fallback to popularity
            return self.train.sort_values('popularity', ascending=False)['id'].head(k).tolist()
        idx = self.id_to_index[track_id]
        sims = self.sim_matrix[idx]
        top_idx = np.argsort(sims)[::-1][1:k+1]  # exclude self
        return self.items.iloc[top_idx]['id'].tolist()

if __name__ == "__main__":
    train, val, test, full = load_processed()
    rec = ItemSimilarityRecommender(train)
    seed = train['id'].iloc[0]
    print("Seed:", seed)
    print("Similar:", rec.recommend(seed, k=10))

