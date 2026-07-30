from __future__ import annotations

import os

from openai import OpenAI


class LLMClassifier:
    """
    Review classifier using an OpenAI language model.
    """

    def __init__(
        self,
        model: str = "gpt-5.5",
        api_key: str | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your environment or .env file."
            )

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key)

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

        return label
