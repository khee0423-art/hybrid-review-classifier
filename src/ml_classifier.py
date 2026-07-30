from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class MLClassifier:
    """
    Text classifier using TF-IDF and Logistic Regression.

    The classifier returns both predicted labels and confidence scores.
    """

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
    ) -> None:
        self.pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        lowercase=True,
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        self.is_fitted = False

    def train(
        self,
        texts: Iterable[str],
        labels: Iterable[str | int],
    ) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        texts:
            Review text data.
        labels:
            Target labels, such as positive/negative.
        """
        text_list = self._validate_texts(texts)
        label_list = list(labels)

        if len(text_list) != len(label_list):
            raise ValueError("The number of texts and labels must be equal.")

        if len(set(label_list)) < 2:
            raise ValueError("Training data must contain at least two classes.")

        self.pipeline.fit(text_list, label_list)
        self.is_fitted = True

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """
        Predict labels for review texts.
        """
        self._check_is_fitted()
        text_list = self._validate_texts(texts)

        return self.pipeline.predict(text_list)

    def predict_with_confidence(
        self,
        texts: Iterable[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return predicted labels and confidence scores.

        Confidence is the highest predicted class probability.
        """
        self._check_is_fitted()
        text_list = self._validate_texts(texts)

        probabilities = self.pipeline.predict_proba(text_list)
        predicted_indexes = np.argmax(probabilities, axis=1)

        classes = self.pipeline.classes_
        predictions = classes[predicted_indexes]
        confidence_scores = probabilities[
            np.arange(len(predicted_indexes)),
            predicted_indexes,
        ]

        return predictions, confidence_scores

    def save(self, file_path: str | Path) -> None:
        """
        Save the trained model.
        """
        self._check_is_fitted()

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, path)

    def load(self, file_path: str | Path) -> None:
        """
        Load a previously trained model.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self.pipeline = joblib.load(path)
        self.is_fitted = True

    @staticmethod
    def _validate_texts(texts: Iterable[str]) -> list[str]:
        text_list = list(texts)

        if not text_list:
            raise ValueError("Text data cannot be empty.")

        cleaned_texts = []

        for text in text_list:
            if text is None:
                cleaned_texts.append("")
            else:
                cleaned_texts.append(str(text).strip())

        if not any(cleaned_texts):
            raise ValueError("All review texts are empty.")

        return cleaned_texts

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "The model has not been trained or loaded."
            )
# test