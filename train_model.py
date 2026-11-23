#!/usr/bin/env python
"""
Standalone script to train and save the credit card approval model.
Run this before starting the Flask app, or the app will auto-train.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.ml_pipeline import train_and_save_model, generate_synthetic_data

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(os.path.dirname(Config.DATA_PATH), exist_ok=True)

    # Generate synthetic data if not present
    if not os.path.exists(Config.DATA_PATH):
        print("Generating synthetic dataset...")
        df = generate_synthetic_data()
        df.to_csv(Config.DATA_PATH, index=False)
        print(f"Data saved to {Config.DATA_PATH}")

    # Train and save model
    pipeline = train_and_save_model(Config.DATA_PATH, Config.MODEL_PATH, retrain=True)
    print("Training completed successfully.")
