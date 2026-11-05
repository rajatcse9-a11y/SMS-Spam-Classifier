import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Spam.csv"
MODEL_DIR = BASE_DIR / "models"


def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.drop_duplicates().reset_index(drop=True)
    df["clean"] = df["text"].apply(clean)
    return df


def get_split(test_size=0.2, seed=42):
    df = load_data()
    return train_test_split(
        df["clean"], df["label"],
        test_size=test_size, stratify=df["label"], random_state=seed,
    )
