from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from llm_classifier import LLMClassifier
from ml_classifier import MLClassifier


@dataclass
class RoutingResult:
    """
    Result of one hybrid classification request.
    """

    final_label: str
    route: str
    ml_label: str
    ml_confidence: float
    confidence_threshold: float
    llm_called: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridRouter:
    """
    Route reviews between the ML classifier and Gemini.

    High-confidence ML predictions are accepted directly.
    Low-confidence predictions are sent to the LLM classifier.
    """

    VALID_LABELS = {"positive", "negative"}

    def __init__(
        self,
        ml_classifier: MLClassifier,
        llm_classifier: LLMClassifier,
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
        Classify one review using ML or Gemini.
        """
        cleaned_review = review.strip()

        if not cleaned_review:
            raise ValueError("Review text cannot be empty.")

        ml_label, ml_confidence = self._get_ml_prediction(
            cleaned_review
        )

        ml_label = str(ml_label).strip().lower()
        ml_confidence = float(ml_confidence)

        self._validate_label(ml_label)
        self._validate_confidence(ml_confidence)

        self.total_requests += 1

        if ml_confidence >= self.confidence_threshold:
            self.ml_requests += 1

            return RoutingResult(
                final_label=ml_label,
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
            final_label=llm_label,
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
        return [
            self.classify(review)
            for review in reviews
        ]

    def _get_ml_prediction(
        self,
        review: str,
    ) -> tuple[str, float]:
        """
        Call the ML classifier.

        The ML classifier must return:
        (label, confidence)
        """
        if hasattr(self.ml_classifier, "predict"):
            result = self.ml_classifier.predict(review)

        elif hasattr(self.ml_classifier, "classify"):
            result = self.ml_classifier.classify(review)

        else:
            raise AttributeError(
                "MLClassifier must have either a predict() "
                "or classify() method."
            )

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "ML classifier must return a tuple: "
                "(label, confidence). "
                f"Received: {result!r}"
            )

        label, confidence = result

        return str(label), float(confidence)

    def get_summary(self) -> dict[str, int | float]:
        """
        Return routing statistics for the current session.
        """
        if self.total_requests == 0:
            ml_routing_rate = 0.0
            llm_routing_rate = 0.0
        else:
            ml_routing_rate = (
                self.ml_requests / self.total_requests
            )
            llm_routing_rate = (
                self.llm_requests / self.total_requests
            )

        summary: dict[str, int | float] = {
            "total_requests": self.total_requests,
            "ml_requests": self.ml_requests,
            "llm_requests": self.llm_requests,
            "ml_routing_rate": ml_routing_rate,
            "llm_routing_rate": llm_routing_rate,
            "confidence_threshold": self.confidence_threshold,
        }

        cost_tracker = getattr(
            self.llm_classifier,
            "cost_tracker",
            None,
        )

        if cost_tracker is not None:
            cost_summary = cost_tracker.get_summary()

            summary["llm_total_cost_usd"] = float(
                cost_summary["total_cost_usd"]
            )
            summary["llm_total_tokens"] = int(
                cost_summary["total_tokens"]
            )

        return summary

    def reset_summary(self) -> None:
        """
        Reset routing counts and LLM cost-session totals.
        """
        self.total_requests = 0
        self.ml_requests = 0
        self.llm_requests = 0

        cost_tracker = getattr(
            self.llm_classifier,
            "cost_tracker",
            None,
        )

        if cost_tracker is not None:
            cost_tracker.reset_session()

    def _validate_label(self, label: str) -> None:
        if label not in self.VALID_LABELS:
            raise ValueError(
                "Label must be positive or negative. "
                f"Received: {label}"
            )

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "ML confidence must be between 0 and 1. "
                f"Received: {confidence}"
            )


if __name__ == "__main__":
    ml_classifier = MLClassifier()
    llm_classifier = LLMClassifier()

    router = HybridRouter(
        ml_classifier=ml_classifier,
        llm_classifier=llm_classifier,
        confidence_threshold=0.80,
    )

    sample_reviews = [
        "This product is absolutely amazing.",
        "It is not bad, but I am not sure whether I like it.",
        "The quality is terrible and I want a refund.",
    ]

    for sample_review in sample_reviews:
        result = router.classify(sample_review)

        print("\nReview:", sample_review)
        print("Final label:", result.final_label)
        print("Route:", result.route)
        print("ML label:", result.ml_label)
        print(
            "ML confidence:",
            f"{result.ml_confidence:.4f}",
        )
        print("LLM called:", result.llm_called)

    print("\nRouting summary")
    print(router.get_summary())