from __future__ import annotations

import os

from google import genai
from google.genai import types

from cost_tracker import (
    CostTracker,
    ModelPricing,
    extract_gemini_usage,
)


class LLMClassifier:
    """
    Review sentiment classifier using the Gemini API.
    """

    VALID_LABELS = {"positive", "negative"}

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your environment variables."
            )

        self.model = model
        self.client = genai.Client(api_key=resolved_api_key)

        self.cost_tracker = cost_tracker or CostTracker(
            pricing={
                self.model: ModelPricing(
                    input_per_million=0.10,
                    output_per_million=0.40,
                )
            },
            log_path="logs/llm_costs.csv",
        )

    def classify(self, review: str) -> str:
        """
        Classify one review as positive or negative.
        """
        cleaned_review = review.strip()

        if not cleaned_review:
            raise ValueError("Review text cannot be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=cleaned_review,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a sentiment classification system. "
                    "Classify the review as either positive or negative. "
                    "Return only one lowercase word: positive or negative."
                ),
                temperature=0,
                max_output_tokens=10,
            ),
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        label = response.text.strip().lower()

        if label not in self.VALID_LABELS:
            raise ValueError(
                f"Unexpected Gemini response: {label}"
            )

        input_tokens, output_tokens = extract_gemini_usage(response)

        cost_record = self.cost_tracker.record(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            category="review_classification",
        )

        print(
            f"Prediction: {label} | "
            f"Input tokens: {input_tokens} | "
            f"Output tokens: {output_tokens} | "
            f"Cost: ${cost_record.total_cost_usd:.8f}"
        )

        return label


if __name__ == "__main__":
    classifier = LLMClassifier()

    sample_reviews = [
        "This product is amazing and works perfectly.",
        "The quality is terrible and I want a refund.",
    ]

    for sample_review in sample_reviews:
        result = classifier.classify(sample_review)

        print("Review:", sample_review)
        print("Label:", result)

    print("\nCost summary")
    print(classifier.cost_tracker.get_summary())