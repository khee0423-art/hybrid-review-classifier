from pathlib import Path

import pandas as pd
from datasets import load_dataset

def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("stanfordnlp/imdb")
   
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    label_mapping = {
        0: "negative",
        1: "positive",
    }

    train_df["label"] = train_df["label"].map(label_mapping)
    test_df["label"] = test_df["label"].map(label_mapping)

    train_df = train_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    train_df.to_csv(
        output_dir / "imdb_train.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "imdb_test.csv",
        index=False,
    )
    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

    print(train_df.head())

if __name__ == "__main__":
    main()