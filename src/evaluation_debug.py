# src/evaluation_debug.py
import pandas as pd
from src import utils

def debug_evaluation():
    # Load processed data
    train, val, test, full = utils.load_processed()
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}\n")

    # Show top 5 popular tracks
    top5 = utils.top_k_popular(train, k=5)
    print("Top 5 popular tracks (IDs):", top5, "\n")

    # Show ID to index mapping
    id_map = utils.id_to_index_map(train)
    print("First 5 ID mappings:", list(id_map.items())[:5], "\n")

    # Sample of 5 users/items from validation set
    sample_val = val.head(5)
    for i, row in sample_val.iterrows():
        track_id = row['id']
        print(f"Validation track {i}: ID={track_id}, popularity={row['popularity']}")

        # Popularity-based recommendation
        pop_recs = utils.top_k_popular(train, k=10)
        print("  Popularity-based recs:", pop_recs)

        # TODO: Add your Item-based CF or Embedding-based recommendations
        # For example:
        # item_recs = item_based_model.recommend(track_id, k=10)
        # embedding_recs = embedding_model.recommend(track_id, k=10)
        # print("  Item-based CF recs:", item_recs)
        # print("  Embedding-based recs:", embedding_recs)
        print()

    # Check overlap with validation set
    val_ids = set(val['id'])
    pop_hits = [tid for tid in pop_recs if tid in val_ids]
    print("Popularity-based hits in validation:", pop_hits)

if __name__ == "__main__":
    debug_evaluation()
