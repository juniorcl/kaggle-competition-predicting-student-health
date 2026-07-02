import os
import pandas as pd
from src.layer_one.config import MODEL_REGISTRY


MODEL_DIR = "models/layer_1/"

os.makedirs(MODEL_DIR, exist_ok=True)


X_TRAIN = pd.read_parquet("data/processed/X_train_raw.parquet")
Y_TRAIN = pd.read_parquet("data/interim/y_train.parquet")

category_features = [
    'diet_type',
    'stress_level',
    'sleep_quality',
    'physical_activity_level',
    'smoking_alcohol',
    'gender'
]

X_TRAIN = X_TRAIN.astype({feature: 'category' for feature in category_features})
Y_TRAIN_ENC = Y_TRAIN.loc[:, 'health_condition']


if __name__ == "__main__":

    for model_name, model_instance in MODEL_REGISTRY.items():
        print(f"\n---------- Train {model_name} ----------")

        model_path = os.path.join(MODEL_DIR, f"model_{model_name}.pkl")

        if os.path.exists(model_path):
            print(f"Skipping {model_name} (already trained).")
            continue

        print(f"Training {model_name}...")
        model_instance(X_TRAIN, Y_TRAIN_ENC, model_path, n_trials=30, n_splits=3)