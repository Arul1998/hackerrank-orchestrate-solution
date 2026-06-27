from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings
from .image_utils import normalize_image_bytes
from .metrics import UsageMetrics


class LLMClient:
    def __init__(self, settings: Settings, metrics: UsageMetrics):
        settings.validate()
        self.settings = settings
        self.metrics = metrics
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, namespace: str, key: str) -> Path:
        return self.settings.cache_dir / namespace / f"{key}.json"

    def _read_cache(self, namespace: str, key: str) -> dict | None:
        path = self._cache_path(namespace, key)
        if path.exists():
            self.metrics.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, namespace: str, key: str, payload: dict) -> None:
        path = self._cache_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _encode_image(path: Path) -> tuple[str, str]:
        raw, media_type = normalize_image_bytes(path)
        return base64.b64encode(raw).decode("utf-8"), media_type

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=1, min=25, max=60),
    )
    def _chat(self, model: str, messages: list[dict], image_count: int = 0) -> dict[str, Any]:
        time.sleep(self.settings.request_delay_seconds)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except RateLimitError as exc:
            if "insufficient_quota" in str(exc).lower():
                raise
            raise
        usage = response.usage
        self.metrics.add_usage(
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            images=image_count,
        )
        content = response.choices[0].message.content or "{}"
        return self._extract_json(content)

    def complete_text(self, system_prompt: str, user_prompt: str, cache_key: str | None = None) -> dict:
        if cache_key:
            cached = self._read_cache("text", cache_key)
            if cached is not None:
                return cached

        result = self._chat(
            model=self.settings.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if cache_key:
            self._write_cache("text", cache_key, result)
        return result

    def analyze_images(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        cache_key: str | None = None,
        model: str | None = None,
    ) -> dict:
        if cache_key:
            cached = self._read_cache("vision", cache_key)
            if cached is not None:
                return cached

        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for path in image_paths:
            encoded, media_type = self._encode_image(path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{media_type};base64,{encoded}",
                        "detail": "high",
                    },
                }
            )

        result = self._chat(
            model=model or self.settings.vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            image_count=len(image_paths),
        )
        if cache_key:
            self._write_cache("vision", cache_key, result)
        return result
