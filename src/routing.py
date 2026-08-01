from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RoutingResult:
    """
    Result returned by the hybrid review classifier.
    """

    label: str
    route: str
    ml_label: str
    ml_confidence: float
    confidence_threshold: float
    llm_called: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the routing result to a dictionary."""
        return asdict(self)


class HybridRouter:
    """
    Route high-confidence predictions to the ML classifier and
    low-confidence predictions to the LLM classifier.

    Parameters
    ----------
    ml_classifier:
        A classifier whose predict() method returns:
        (label, confidence)

    llm_classifier:
        A classifier whose classify() method returns:
        "positive" or "negative"

    confidence_threshold:
        Minimum ML confidence required to accept the ML prediction.
    """

    VALID_LABELS = {"positive", "negative"}

    def __init__(
        self,
        ml_classifier: Any,
        llm_classifier: Any,
        confidence_threshold: float = 0.80,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self.ml_classifier = ml_classifier
        self.llm_classifier = llm_classifier
        self.confidence_threshold = confidence_threshold

        self.total_requests = 0
        self.ml_requests = 0
        self.llm_requests = 0

    def classify(self, review: str) -> RoutingResult:
        """
        Classify one review using the hybrid routing strategy.
        """
        cleaned_review = review.strip()

        if not cleaned_review:
            raise ValueError("Review text cannot be empty.")

        self.total_requests += 1

        ml_label, ml_confidence = self.ml_classifier.predict(
            cleaned_review
        )

        ml_label = str(ml_label).strip().lower()
        ml_confidence = float(ml_confidence)

        self._validate_ml_result(
            label=ml_label,
            confidence=ml_confidence,
        )

        if ml_confidence >= self.confidence_threshold:
            self.ml_requests += 1

            return RoutingResult(
                label=ml_label,
                route="ml",
                ml_label=ml_label,
                ml_confidence=ml_confidence,
                confidence_threshold=self.confidence_threshold,
                llm_called=False,
            )

        llm_label = self.llm_classifier.classify(cleaned_review)
        llm_label = str(llm_label).strip().lower()

        self._validate_label(llm_label)

        self.llm_requests += 1

        return RoutingResult(
            label=llm_label,
            route="llm",
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            confidence_threshold=self.confidence_threshold,
            llm_called=True,
        )

    def classify_batch(
        self,
        reviews: list[str],
    ) -> list[RoutingResult]:
        """
        Classify multiple reviews.
        """
        if not reviews:
            return []

        results: list[RoutingResult] = []

        for review in reviews:
            result = self.classify(review)
            results.append(result)

        return results

    def get_summary(self) -> dict[str, int | float]:
        """
        Return routing statistics for the current session.
        """
        if self.total_requests == 0:
            ml_rate = 0.0
            llm_rate = 0.0
        else:
            ml_rate = self.ml_requests / self.total_requests
            llm_rate = self.llm_requests / self.total_requests

        return {
            "total_requests": self.total_requests,
            "ml_requests": self.ml_requests,
            "llm_requests": self.llm_requests,
            "ml_routing_rate": ml_rate,
            "llm_routing_rate": llm_rate,
            "confidence_threshold": self.confidence_threshold,
        }

    def reset_summary(self) -> None:
        """
        Reset routing statistics.
        """
        self.total_requests = 0
        self.ml_requests = 0
        self.llm_requests = 0

    def _validate_ml_result(
        self,
        label: str,
        confidence: float,
    ) -> None:
        """Validate the ML classifier output."""
        self._validate_label(label)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "ML confidence must be between 0 and 1. "
                f"Received: {confidence}"
            )

    def _validate_label(self, label: str) -> None:
        """Validate a sentiment label."""
        if label not in self.VALID_LABELS:
            raise ValueError(
                "Classifier output must be positive or negative. "
                f"Received: {label}"
            )


if __name__ == "__main__":
    from llm_classifier import LLMClassifier
    from ml_classifier import MLClassifier

    ml_classifier = MLClassifier()
    llm_classifier = LLMClassifier()

    router = HybridRouter(
        ml_classifier=ml_classifier,
        llm_classifier=llm_classifier,
        confidence_threshold=0.80,
    )

    sample_reviews = [
        "This product is absolutely amazing.",
        "It was okay, but I am not sure whether I liked it.",
        "The quality was terrible and I want a refund.",
    ]

    for sample_review in sample_reviews:
        routing_result = router.classify(sample_review)

        print("\nReview:", sample_review)
        print("Final label:", routing_result.label)
        print("Route:", routing_result.route)
        print(
            "ML confidence:",
            f"{routing_result.ml_confidence:.4f}",
        )

    print("\nRouting summary")
    print(router.get_summary())
