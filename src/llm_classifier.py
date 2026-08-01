from __future__ import annotations

import os

from openai import OpenAI

from cost_tracker import (
    CostTracker,
    ModelPricing,
    extract_openai_usage,
)


class LLMClassifier:
    """
    Review classifier using an OpenAI language model.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        api_key: str | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your environment or .env file."
            )

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key)

        self.cost_tracker = cost_tracker or CostTracker(
            pricing={
                self.model: ModelPricing(
                    input_per_million=0.0,   # 실제 단가로 교체
                    output_per_million=0.0,  # 실제 단가로 교체
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

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are a sentiment classification system. "
                "Classify the review as either positive or negative. "
                "Return only one word: positive or negative."
            ),
            input=cleaned_review,
        )

        label = response.output_text.strip().lower()

        if label not in {"positive", "negative"}:
            raise ValueError(
                f"Unexpected model response: {label}"
            )

        input_tokens, output_tokens = extract_openai_usage(response)

        cost_record = self.cost_tracker.record(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_id=response.id,
            category="review_classification",
        )

        print(
            f"Prediction: {label} | "
            f"Input tokens: {input_tokens} | "
            f"Output tokens: {output_tokens} | "
            f"Cost: ${cost_record.total_cost_usd:.8f}"
        )

        return label