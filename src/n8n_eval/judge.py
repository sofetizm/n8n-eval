"""LLM judge: does the answer satisfy a rubric written for this case? Answers pass/fail with a reason."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from .checks import CheckResult
from .config import Judge

PROMPT = """Ты проверяешь ответ ИИ-ассистента по критерию.

Сообщение пользователя:
{question}

Ответ ассистента:
{answer}

Критерий: {rubric}

Ответ проходит, только если критерий выполнен полностью. Не оценивай стиль и длину, если критерий о них не говорит.
Верни только JSON: {{"pass": true или false, "reason": "одно предложение"}}"""


class JudgeError(RuntimeError):
    pass


def parse(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise JudgeError(f"judge returned no JSON: {text[:200]}")
    data = json.loads(match.group(0))
    return {"pass": bool(data.get("pass")), "reason": str(data.get("reason", ""))}


class LLMJudge:
    def __init__(self, config: Judge, client: httpx.AsyncClient | None = None, retries: int = 8, max_wait: float = 600):
        if not config.api_key:
            raise JudgeError("judge.api_key is empty: set it in the suite or the environment")
        self.config = config
        self.retries = retries
        self.max_wait = max_wait
        self.client = client or httpx.AsyncClient(timeout=120)
        self.tokens = {"prompt": 0, "completion": 0}

    async def _complete(self, prompt: str) -> str:
        body = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1024,
            **self.config.extra,
        }
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(self.retries + 1):
            response = await self.client.post(url, json=body, headers=headers)
            if response.status_code in (429, 500, 502, 503) and attempt < self.retries:
                wait = float(response.headers.get("retry-after") or 2**attempt)
                if wait > self.max_wait:
                    raise JudgeError(f"judge rate limit: retry after {wait:.0f} s")
                await asyncio.sleep(wait)
                continue
            if response.status_code >= 400:
                raise JudgeError(f"judge API error {response.status_code}: {response.text[:300]}")
            data = response.json()
            usage = data.get("usage") or {}
            self.tokens["prompt"] += int(usage.get("prompt_tokens", 0))
            self.tokens["completion"] += int(usage.get("completion_tokens", 0))
            return data["choices"][0]["message"].get("content") or ""
        raise JudgeError("unreachable")

    async def check(self, question: str, answer: str, rubric: str) -> CheckResult:
        prompt = PROMPT.format(question=question, answer=answer, rubric=rubric)
        for attempt in range(3):
            try:
                verdict = parse(await self._complete(prompt))
                return CheckResult("judge", verdict["pass"], verdict["reason"])
            except (JudgeError, json.JSONDecodeError) as error:
                if "rate limit" in str(error) or "API error" in str(error) or attempt == 2:
                    raise
        raise JudgeError("unreachable")

    async def aclose(self) -> None:
        await self.client.aclose()
