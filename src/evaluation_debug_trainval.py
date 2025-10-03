# src/evaluation_debug_debug_trainval.py
import pandas as pd
from src import utils

def debug_evaluation_train():
    # Load processed data
    train, val, test, full = utils.load_processed()
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}\n")

    # --- Use train itself as validation ---
    eval_val = train.copy()
    print("Using training set as validation for evaluation.\n")

    # Show top 5 popular tracks
    top5 = utils.top_k_popular(train, k=5)
    print("Top 5 popular tracks (IDs):", top5, "\n")

    # Sample of 5 tracks from "validation"
    sample_val = eval_val.head(5)
    for i, row in sample_val.iterrows():
        track_id = row['id']
        print(f"Validation track {i}: ID={track_id}, popularity={row['popularity']}")

        # Popularity-based recommendation
        pop_recs = utils.top_k_popular(train, k=10)
        print("  Popularity-based recs:", pop_recs)

        # Check if any hits exist
        val_ids = set(eval_val['id'])
        pop_hits = [tid for tid in pop_recs if tid in val_ids]
        print("  Hits in validation (should be non-empty now):", pop_hits)
        print()

if __name__ == "__main__":
    debug_evaluation_train()
