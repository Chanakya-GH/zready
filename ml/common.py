from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data" / "creditcard.csv"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # Time dropped

def load_split(seed=42):
    df = pd.read_csv(DATA)
    X = df[FEATURES].astype(np.float32).values
    y = df["Class"].values.astype(np.int64)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
