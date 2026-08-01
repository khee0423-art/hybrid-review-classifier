"""
cost_tracker.py

LLM 호출별 토큰 사용량과 비용을 계산하고 CSV로 기록하는 모듈.

지원 기능
---------
1. 모델별 input/output token 단가 등록
2. 호출별 비용 계산
3. 누적 토큰 및 누적 비용 집계
4. CSV 로그 저장
5. OpenAI/Gemini 응답 객체에서 usage 정보 추출 보조
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelPricing:
    """
    모델의 토큰 단가.

    가격은 1백만 토큰당 USD 기준이다.
    예:
        input_per_million=0.10
        output_per_million=0.40
    """

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        if self.input_per_million < 0:
            raise ValueError("input_per_million must be non-negative.")

        if self.output_per_million < 0:
            raise ValueError("output_per_million must be non-negative.")


@dataclass
class CostRecord:
    """한 번의 LLM 호출 비용 기록."""

    timestamp_utc: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    request_id: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[str] = None


class CostTracker:
    """
    LLM 호출 비용을 추적하는 클래스.

    Parameters
    ----------
    pricing:
        모델 이름을 key로 하고 ModelPricing을 value로 갖는 딕셔너리.
    log_path:
        호출별 비용을 저장할 CSV 파일 경로.
    """

    CSV_FIELDS = [
        "timestamp_utc",
        "model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
        "request_id",
        "category",
        "metadata",
    ]

    def __init__(
        self,
        pricing: Dict[str, ModelPricing],
        log_path: str | Path = "logs/llm_costs.csv",
    ) -> None:
        if not pricing:
            raise ValueError("At least one model pricing configuration is required.")

        self.pricing = pricing
        self.log_path = Path(log_path)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0

        self._lock = Lock()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        """CSV 파일이 없으면 헤더를 생성한다."""

        if self.log_path.exists():
            return

        with self.log_path.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.CSV_FIELDS)
            writer.writeheader()

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> Dict[str, float | int]:
        """
        토큰 수를 기준으로 비용을 계산한다.

        Returns
        -------
        dict
            input/output/total token과 비용 정보.
        """

        if model not in self.pricing:
            available_models = ", ".join(sorted(self.pricing))
            raise KeyError(
                f"Pricing for model '{model}' is not configured. "
                f"Available models: {available_models}"
            )

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative.")

        model_pricing = self.pricing[model]

        input_cost = (
            input_tokens / 1_000_000
        ) * model_pricing.input_per_million

        output_cost = (
            output_tokens / 1_000_000
        ) * model_pricing.output_per_million

        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
        }

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_id: Optional[str] = None,
        category: Optional[str] = None,
        metadata: Optional[str] = None,
    ) -> CostRecord:
        """
        한 번의 LLM 호출 비용을 계산하고 CSV에 기록한다.

        category 예시
        -------------
        - review_classification
        - fallback
        - summarization
        """

        cost = self.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        record = CostRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(cost["total_tokens"]),
            input_cost_usd=float(cost["input_cost_usd"]),
            output_cost_usd=float(cost["output_cost_usd"]),
            total_cost_usd=float(cost["total_cost_usd"]),
            request_id=request_id,
            category=category,
            metadata=metadata,
        )

        with self._lock:
            self.total_input_tokens += record.input_tokens
            self.total_output_tokens += record.output_tokens
            self.total_cost_usd += record.total_cost_usd
            self.call_count += 1

            self._append_to_csv(record)

        return record

    def _append_to_csv(self, record: CostRecord) -> None:
        """CostRecord를 CSV 파일에 추가한다."""

        with self.log_path.open(
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.CSV_FIELDS)
            writer.writerow(asdict(record))

    def get_summary(self) -> Dict[str, float | int]:
        """현재 실행 세션의 누적 비용을 반환한다."""

        total_tokens = (
            self.total_input_tokens
            + self.total_output_tokens
        )

        average_cost = (
            self.total_cost_usd / self.call_count
            if self.call_count > 0
            else 0.0
        )

        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "average_cost_per_call_usd": average_cost,
        }

    def reset_session(self) -> None:
        """
        메모리에 저장된 세션 누적값만 초기화한다.

        기존 CSV 로그는 삭제하지 않는다.
        """

        with self._lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost_usd = 0.0
            self.call_count = 0

    def estimate_batch_cost(
        self,
        model: str,
        number_of_requests: int,
        average_input_tokens: int,
        average_output_tokens: int,
    ) -> Dict[str, float | int]:
        """
        여러 요청을 실행하기 전 예상 비용을 계산한다.
        """

        if number_of_requests < 0:
            raise ValueError("number_of_requests must be non-negative.")

        single_request = self.calculate_cost(
            model=model,
            input_tokens=average_input_tokens,
            output_tokens=average_output_tokens,
        )

        return {
            "number_of_requests": number_of_requests,
            "estimated_input_tokens": (
                average_input_tokens * number_of_requests
            ),
            "estimated_output_tokens": (
                average_output_tokens * number_of_requests
            ),
            "estimated_total_tokens": (
                int(single_request["total_tokens"])
                * number_of_requests
            ),
            "estimated_total_cost_usd": (
                float(single_request["total_cost_usd"])
                * number_of_requests
            ),
        }


def extract_openai_usage(response: Any) -> tuple[int, int]:
    """
    OpenAI 계열 응답에서 input/output token 수를 추출한다.

    Responses API와 Chat Completions 형태를 모두 일부 지원한다.
    """

    usage = getattr(response, "usage", None)

    if usage is None:
        raise ValueError("The response does not contain usage information.")

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)

    # Chat Completions 호환 필드
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)

    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)

    if input_tokens is None or output_tokens is None:
        raise ValueError(
            "Could not extract input/output token counts "
            "from the OpenAI response."
        )

    return int(input_tokens), int(output_tokens)


def extract_gemini_usage(response: Any) -> tuple[int, int]:
    """
    Gemini 응답에서 prompt/candidate token 수를 추출한다.
    """

    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        raise ValueError(
            "The Gemini response does not contain usage_metadata."
        )

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)

    if input_tokens is None or output_tokens is None:
        raise ValueError(
            "Could not extract token counts from Gemini usage_metadata."
        )

    return int(input_tokens), int(output_tokens)


if __name__ == "__main__":
    # 아래 가격은 예시다.
    # 실제 사용하는 모델의 최신 단가로 수정해야 한다.
    tracker = CostTracker(
        pricing={
            "gemini-model": ModelPricing(
                input_per_million=0.10,
                output_per_million=0.40,
            ),
            "openai-model": ModelPricing(
                input_per_million=0.50,
                output_per_million=1.50,
            ),
        },
        log_path="logs/llm_costs.csv",
    )

    record = tracker.record(
        model="gemini-model",
        input_tokens=1_200,
        output_tokens=150,
        request_id="review-0001",
        category="review_classification",
        metadata="ML confidence below threshold",
    )

    print("Current request")
    print(record)

    print("\nSession summary")
    print(tracker.get_summary())

    print("\nEstimated batch cost")
    print(
        tracker.estimate_batch_cost(
            model="gemini-model",
            number_of_requests=10_000,
            average_input_tokens=1_200,
            average_output_tokens=150,
        )
    )
