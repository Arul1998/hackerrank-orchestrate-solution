from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class UsageMetrics:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    images_processed: int = 0
    cache_hits: int = 0
    errors: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None

    def add_usage(self, input_tokens: int, output_tokens: int, images: int = 0) -> None:
        self.model_calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.images_processed += images

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or datetime.now(timezone.utc)
        return max((end - self.started_at).total_seconds(), 0.001)

    def summary(self) -> dict:
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "images_processed": self.images_processed,
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def estimated_cost_usd(
        self,
        input_price_per_million: float = 2.50,
        output_price_per_million: float = 10.00,
        image_price_each: float = 0.003,
    ) -> float:
        text_cost = (self.input_tokens / 1_000_000) * input_price_per_million
        text_cost += (self.output_tokens / 1_000_000) * output_price_per_million
        image_cost = self.images_processed * image_price_each
        return round(text_cost + image_cost, 4)
