# src/utils.py
import os
import pandas as pd
import numpy as np

def load_processed():
    """
    Load train, validation, test, and full datasets.
    Returns:
        train, val, test, full (pd.DataFrame)
    """
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    
    # Check files exist
    for fname in ["train.csv", "val.csv", "test.csv"]:
        if not os.path.exists(os.path.join(base, fname)):
            raise FileNotFoundError(f"{fname} not found in {base}")

    train = pd.read_csv(os.path.join(base, "train.csv"))
    val = pd.read_csv(os.path.join(base, "val.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))

    # Ensure 'id' and 'popularity' columns exist
    for df_name, df in zip(['train', 'val', 'test'], [train, val, test]):
        for col in ['id', 'popularity']:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing in {df_name}.csv")

    full = pd.concat([train, val, test], ignore_index=True)
    return train, val, test, full


def top_k_popular(train_df, k=10, return_scores=False):
    """
    Return top-k popular track IDs from training data.
    Args:
        train_df: pd.DataFrame with 'id' and 'popularity'
        k: number of tracks
        return_scores: if True, also return popularity scores
    Returns:
        List of track IDs (or list of tuples if return_scores=True)
    """
    top = train_df.sort_values("popularity", ascending=False).head(k)
    if return_scores:
        return list(zip(top['id'], top['popularity']))
    else:
        return top['id'].tolist()


def id_to_index_map(df):
    """
    Map track ID to row index for fast lookup in embeddings or feature matrices.
    Args:
        df: pd.DataFrame with 'id' column
    Returns:
        dict: {track_id: row_index}
    """
    if 'id' not in df.columns:
        raise ValueError("'id' column missing in DataFrame")
    return {rid: idx for idx, rid in enumerate(df['id'].tolist())}
