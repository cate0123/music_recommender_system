# src/utils.py
import os
import pandas as pd
import numpy as np

def load_processed():
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    train = pd.read_csv(os.path.join(base, "train.csv"))
    val = pd.read_csv(os.path.join(base, "val.csv"))
    test = pd.read_csv(os.path.join(base, "test.csv"))
    full = pd.concat([train, val, test], ignore_index=True)
    return train, val, test, full

def top_k_popular(train_df, k=10):
    return train_df.sort_values("popularity", ascending=False)['id'].head(k).tolist()

def id_to_index_map(df):
    return {rid: idx for idx, rid in enumerate(df['id'].tolist())}
