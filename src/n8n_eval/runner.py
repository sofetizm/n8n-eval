from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from .checks import CheckResult, run_checks
from .config import Case, Suite
from .judge import LLMJudge
from .n8n import N8nClient, Trace

DEFAULT_PRICES = {  # USD per 1M tokens (input, output); override with `prices:` in the suite
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
}


def extract(data: Any, path: str) -> Any:
    """Dot path into a JSON response: 'output', 'data.0.text'; empty path returns the whole response."""
    for part in [p for p in path.split(".") if p]:
        if isinstance(data, list):
            data = data[int(part)] if part.isdigit() and int(part) < len(data) else None
        elif isinstance(data, dict):
            data = data.get(part)
        else:
            return None
    return data


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def cost(trace: Trace | None, prices: dict[str, Any]) -> float | None:
    if trace is None or not (trace.prompt_tokens or trace.completion_tokens):
        return None
    table = {**DEFAULT_PRICES, **{k: tuple(v) for k, v in prices.items()}}
    model = next((m for m in trace.models if m in table), None)
    if model is None:
        return None
    per_in, per_out = table[model]
    return (trace.prompt_tokens * per_in + trace.completion_tokens * per_out) / 1_000_000


_WAIT = re.compile(r"try again in\s*((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", re.I)


def is_rate_limit(error: str) -> bool:
    lowered = error.lower()
    return "rate limit" in lowered or "rate_limit" in lowered or "429" in lowered


def retry_after(error: str) -> float | None:
    """Seconds from messages like 'Please try again in 1m2.5s' (Groq, OpenAI); 20 s if no hint is given."""
    match = _WAIT.search(error)
    if not match or not match.group(1):
        return 20.0
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(h|ms|m|s)", match.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a results file; when a run was retried, only its last attempt is kept."""
    if not path.exists():
        return []
    latest: dict[tuple, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["label"], row["case"], row["repeat"])] = row
    return list(latest.values())


class TooManyErrors(RuntimeError):
    pass


class Runner:
    def __init__(
        self,
        suite: Suite,
        label: str,
        out_path: str | Path,
        judge: LLMJudge | None = None,
        n8n: N8nClient | None = None,
        http: httpx.AsyncClient | None = None,
        progress: Callable[[int, int], None] | None = None,
        max_consecutive_errors: int = 5,
    ):
        self.suite = suite
        self.label = label
        self.out_path = Path(out_path)
        self.judge = judge
        self.n8n = n8n
        self.http = http or httpx.AsyncClient(timeout=suite.target.timeout)
        self.progress = progress
        self._lock = asyncio.Lock()
        self._done = 0
        self._errors_in_row = 0
        self.max_consecutive_errors = max_consecutive_errors

    def _body(self, case: Case, marker: str) -> dict[str, Any]:
        target = self.suite.target
        body = dict(case.input) if isinstance(case.input, dict) else {target.input_field: case.input}
        body["_eval"] = marker
        if target.session_field:
            body.setdefault(target.session_field, marker)
        return body

    async def _post(self, body: dict[str, Any], wait_registration: float = 30.0) -> httpx.Response:
        """n8n registers webhooks of a just-activated workflow with a short delay, so a 404
        'not registered' is retried for a while before it counts as a failure."""
        deadline = time.monotonic() + wait_registration
        while True:
            response = await self.http.post(self.suite.target.webhook, json=body, headers=self.suite.target.headers)
            if response.status_code != 404 or "not registered" not in response.text or time.monotonic() > deadline:
                return response
            await asyncio.sleep(1)

    async def run_one(self, case: Case, repeat: int, max_wait: float = 120.0, attempts: int = 30) -> dict[str, Any]:
        """One run; when the LLM behind n8n hits a per-minute rate limit, waits as the provider asks and retries.
        A long wait (a daily limit) is returned as an error, which stops the run so it can be resumed later."""
        for _ in range(attempts):
            row = await self._attempt(case, repeat)
            error = row.get("error", "")
            if not is_rate_limit(error):
                return row
            wait = retry_after(error)
            if wait is None or wait > max_wait:
                row["error"] = "rate limit: " + error
                return row
            await asyncio.sleep(wait + 1)
        return row

    async def _attempt(self, case: Case, repeat: int) -> dict[str, Any]:
        marker = f"{self.label}:{case.id}:{repeat}:{uuid.uuid4().hex[:8]}"
        started_after = datetime.now(timezone.utc) - timedelta(seconds=2)
        row: dict[str, Any] = {
            "label": self.label,
            "case": case.id,
            "repeat": repeat,
            "tags": case.tags,
            "input": case.input,
            "marker": marker,
        }
        t0 = time.perf_counter()
        try:
            response = await self._post(self._body(case, marker))
            row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            row["status_code"] = response.status_code
            try:
                payload: Any = response.json()
            except ValueError:
                payload = response.text
            row["output"] = as_text(
                extract(payload, self.suite.target.output_path) if not isinstance(payload, str) else payload
            )
            if response.status_code >= 400:
                row["error"] = f"HTTP {response.status_code}: {response.text[:300]}"
        except httpx.HTTPError as error:
            row.update(latency_ms=round((time.perf_counter() - t0) * 1000, 1), output="", error=repr(error)[:300])

        reached_n8n = "error" not in row or row.get("status_code", 0) >= 500
        trace = await self.n8n.find_execution(marker, started_after) if self.n8n and reached_n8n else None
        row["trace"] = trace.to_dict() if trace else None
        if "error" in row and trace and trace.error and trace.execution_id:
            row["error"] += f" | n8n: {trace.error}"
        row["cost_usd"] = cost(trace, self.suite.prices)

        checks: list[CheckResult] = []
        if "error" in row:
            checks.append(CheckResult("request", False, row["error"]))
        else:
            checks = run_checks(case.expect, row["output"], row["latency_ms"], trace.tools if trace else None)
            if "judge" in case.expect:
                if self.judge is None:
                    checks.append(CheckResult("judge", False, "no judge configured"))
                else:
                    question = case.input if isinstance(case.input, str) else json.dumps(case.input, ensure_ascii=False)
                    checks.append(await self.judge.check(question, row["output"], case.expect["judge"]))
        row["checks"] = [c.to_dict() for c in checks]
        row["passed"] = bool(checks) and all(c.passed for c in checks)
        return row

    async def run(self, cases: list[Case], repeats: int, concurrency: int) -> list[dict[str, Any]]:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        done = {
            (r["case"], r["repeat"])
            for r in load_rows(self.out_path)
            if r.get("label") == self.label and not r.get("error")
        }
        todo = [(c, i) for i in range(repeats) for c in cases if (c.id, i) not in done]
        total = len(todo)
        semaphore = asyncio.Semaphore(concurrency)

        async def worker(case: Case, repeat: int) -> None:
            async with semaphore:
                if self._errors_in_row >= self.max_consecutive_errors:
                    return
                row = await self.run_one(case, repeat)
            async with self._lock:
                self._errors_in_row = self._errors_in_row + 1 if row.get("error") else 0
                with self.out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._done += 1
                if self.progress:
                    self.progress(self._done, total)

        await asyncio.gather(*(worker(c, i) for c, i in todo))
        if self._errors_in_row >= self.max_consecutive_errors:
            last = json.loads(self.out_path.read_text(encoding="utf-8").splitlines()[-1]).get("error", "")
            raise TooManyErrors(
                f"{self._errors_in_row} requests in a row failed, stopping (last: {last[:200]}). "
                "Runs that failed are retried next time with the same --out, finished ones are kept."
            )
        wanted = {c.id for c in cases}
        return [
            r
            for r in load_rows(self.out_path)
            if r.get("label") == self.label and r["case"] in wanted and r["repeat"] < repeats
        ]

    async def aclose(self) -> None:
        await self.http.aclose()
