from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from ml_classifier import MLClassifier


def main() -> None:
    train_path = PROJECT_ROOT / "data" / "imdb_train.csv"
    test_path = PROJECT_ROOT / "data" / "imdb_test.csv"
    model_path = PROJECT_ROOT / "models" / "ml_classifier.joblib"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            "Run: python scripts/download_imdb.py"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_path}\n"
            "Run: python scripts/download_imdb.py"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError(
            f"Training data must contain columns: {required_columns}"
        )

    if not required_columns.issubset(test_df.columns):
        raise ValueError(
            f"Test data must contain columns: {required_columns}"
        )

    train_df = train_df.dropna(subset=["text", "label"])
    test_df = test_df.dropna(subset=["text", "label"])

    classifier = MLClassifier()

    print("Training ML classifier...")
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")

    classifier.train(
        texts=train_df["text"],
        labels=train_df["label"],
    )

    print("\nEvaluating model...")

    predictions = classifier.predict(test_df["text"])

    accuracy = accuracy_score(
        test_df["label"],
        predictions,
    )

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report")
    print(
        classification_report(
            test_df["label"],
            predictions,
            digits=4,
        )
    )

    classifier.save(model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()