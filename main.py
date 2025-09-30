# main.py
from src.preprocess import load_raw, clean_dataframe, encode_and_save
from src.utils import load_processed
import os

def run_preprocess():
    print("Loading raw CSV...")
    df = load_raw()
    print("Cleaning dataframe...")
    df = clean_dataframe(df)
    print("Encoding and saving splits...")
    train, val, test = encode_and_save(df)
    print("Preprocessing complete.")

def run_basic_pipeline():
    # Ensure processed exists
    train, val, test, full = load_processed()
    print("Train/Val/Test sizes:", len(train), len(val), len(test))

if __name__ == "__main__":
    # Run preprocessing then basic pipeline
    run_preprocess()
    run_basic_pipeline()
    print("Now you can run EDA, train models and evaluate.")
