from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import httpx

from .config import AppConfig
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .utils import RateLimiter, ensure_directory, estimate_tokens, iso_now

_logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class ModelResponse:
    candidates: list[str]
    raw: str
    usage: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResponse":
        return cls(
            candidates=list(data.get("candidates", [])),
            raw=data.get("raw", ""),
            usage=dict(data.get("usage", {})),
        )


class DeepSeekClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("Переменная окружения DEEPSEEK_API_KEY не задана")
        self._client = httpx.AsyncClient(base_url=API_BASE_URL, timeout=60.0)
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._rate_limiter = RateLimiter(config.rate_limit)
        self._memory_cache: dict[str, ModelResponse] = {}
        self.cache_dir = ensure_directory(config.cache_dir)
        self.resume = config.resume
        self.total_requests = 0
        self.total_failures = 0
        self.total_retries = 0
        self.total_tokens = 0
        self._token_budget = config.token_budget
        self._usage_details: list[dict[str, Any]] = []

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DeepSeekClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _cache_key(
        self,
        parent_path: str,
        count: int,
        lang: str,
        temperature: float,
        top_p: float,
        attempt: int,
    ) -> str:
        payload = {
            "model": self.config.model,
            "parent": parent_path,
            "count": count,
            "lang": lang,
            "seed": self.config.seed,
            "temperature": temperature,
            "top_p": top_p,
            "attempt": attempt,
        }
        dumped = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, key: str) -> ModelResponse | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._cache_path(key)
        if self.resume and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                response = ModelResponse.from_dict(data.get("response", data))
                self._memory_cache[key] = response
                return response
            except Exception as exc:  # pragma: no cover - corrupted cache
                _logger.warning("Не удалось прочитать кэш %s: %s", path, exc)
        return None

    def _save_cache(self, key: str, response: ModelResponse, payload: Dict[str, Any]) -> None:
        data = {
            "timestamp": iso_now(),
            "request": payload,
            "response": {
                "candidates": response.candidates,
                "raw": response.raw,
                "usage": response.usage,
            },
        }
        path = self._cache_path(key)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._memory_cache[key] = response

    @staticmethod
    def _parse_candidates(content: str) -> list[str]:
        candidates: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = line.lstrip("-•*")
            line = line.strip()
            line = line.lstrip("0123456789.)- ")
            line = line.strip()
            if not line:
                continue
            candidates.append(line)
        return candidates

    def _enforce_budget(self, prompt_tokens: int, completion_tokens: int) -> None:
        total = prompt_tokens + completion_tokens
        if self._token_budget is None:
            return
        if self.total_tokens + total > self._token_budget:
            raise RuntimeError("Превышен лимит токенов для запуска")

    async def _post_with_retries(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with self._semaphore:
                    await self._rate_limiter.wait()
                    response = await self._client.post("/chat/completions", headers=self._headers(), json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {429, 500, 502, 503, 504}:
                    self.total_retries += 1
                    last_error = exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
            except httpx.HTTPError as exc:  # pragma: no cover - network errors
                last_error = exc
                self.total_retries += 1
                await asyncio.sleep(delay)
                delay *= 2
                continue
        self.total_failures += 1
        raise RuntimeError(f"DeepSeek API не отвечает: {last_error}")

    async def generate(
        self,
        parent_path: str,
        count: int,
        lang: str,
        existing: Sequence[str],
        final_level: bool,
        attempt: int,
    ) -> ModelResponse:
        temperature = min(1.2, self.config.temperature + attempt * 0.2)
        top_p = self.config.top_p
        key = self._cache_key(parent_path, count, lang, temperature, top_p, attempt)
        cached = self._load_cache(key)
        if cached:
            return cached

        user_prompt = build_user_prompt(parent_path, count, lang, existing, final_level, attempt)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.seed is not None:
            payload["seed"] = self.config.seed

        approx_prompt_tokens = estimate_tokens(user_prompt) + estimate_tokens(SYSTEM_PROMPT)
        approx_completion_tokens = count * 16
        self._enforce_budget(approx_prompt_tokens, approx_completion_tokens)

        response_json = await self._post_with_retries(payload)
        self.total_requests += 1
        choices = response_json.get("choices", [])
        raw_text = ""
        if choices:
            contents = [choice.get("message", {}).get("content", "") for choice in choices]
            raw_text = "\n".join(contents)
        candidates = self._parse_candidates(raw_text)
        usage = response_json.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", approx_prompt_tokens))
        completion_tokens = int(usage.get("completion_tokens", approx_completion_tokens))
        self._enforce_budget(prompt_tokens, completion_tokens)
        self.total_tokens += prompt_tokens + completion_tokens
        usage.setdefault("prompt_tokens", prompt_tokens)
        usage.setdefault("completion_tokens", completion_tokens)
        self._usage_details.append(usage)

        model_response = ModelResponse(candidates=candidates, raw=raw_text, usage=usage)
        self._save_cache(key, model_response, payload)
        return model_response

    def stats(self) -> Dict[str, Any]:
        return {
            "requests": self.total_requests,
            "retries": self.total_retries,
            "failures": self.total_failures,
            "tokens": self.total_tokens,
            "usage": self._usage_details,
        }


__all__ = ["DeepSeekClient", "ModelResponse"]
