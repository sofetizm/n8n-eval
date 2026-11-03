from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .checks import KNOWN

_ENV = re.compile(r"\$\{(\w+)(?::-([^${}]*))?\}")


def expand_env(value: Any) -> Any:
    """Replaces ${VAR} and ${VAR:-default} in strings."""
    if isinstance(value, str):
        while True:
            expanded = _ENV.sub(lambda m: os.environ.get(m.group(1)) or m.group(2) or "", value)
            if expanded == value:
                return expanded
            value = expanded
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


@dataclass
class Case:
    id: str
    input: str | dict[str, Any]
    expect: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = set(self.expect) - KNOWN
        if unknown:
            raise ValueError(f"case {self.id}: unknown checks {sorted(unknown)}; known: {sorted(KNOWN)}")
        if not self.expect:
            raise ValueError(f"case {self.id}: no checks in 'expect'")


@dataclass
class Target:
    webhook: str
    input_field: str = "message"
    output_path: str = "output"
    session_field: str | None = "sessionId"
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0


@dataclass
class N8nApi:
    url: str = ""
    api_key: str = ""
    workflow: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.api_key)


@dataclass
class Judge:
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "openai/gpt-oss-120b"
    api_key: str = ""
    extra: dict[str, Any] = field(default_factory=lambda: {"reasoning_effort": "low"})


@dataclass
class Suite:
    name: str
    target: Target
    cases: list[Case]
    n8n: N8nApi = field(default_factory=N8nApi)
    judge: Judge = field(default_factory=Judge)
    repeats: int = 3
    concurrency: int = 2
    prices: dict[str, list[float]] = field(default_factory=dict)
    gate: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def select(
        self, tags: list[str] | None = None, ids: list[str] | None = None, limit: int | None = None
    ) -> list[Case]:
        cases = [c for c in self.cases if (not tags or set(tags) & set(c.tags)) and (not ids or c.id in ids)]
        return cases[:limit] if limit else cases


def load_cases(path: Path) -> list[Case]:
    if path.suffix in (".yaml", ".yml"):
        rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [Case(**row) for row in rows]
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case ids: {sorted(dupes)}")
    return cases


def load_suite(path: str | Path, overrides: dict[str, Any] | None = None) -> Suite:
    path = Path(path)
    raw = expand_env(yaml.safe_load(path.read_text(encoding="utf-8")))
    for key, value in (overrides or {}).items():
        section, _, name = key.partition(".")
        if name:
            raw.setdefault(section, {})[name] = value
        else:
            raw[key] = value
    cases_ref = raw.pop("cases")
    cases = load_cases(path.parent / cases_ref) if isinstance(cases_ref, str) else [Case(**c) for c in cases_ref]
    return Suite(
        name=raw.get("name", path.stem),
        target=Target(**raw["target"]),
        cases=cases,
        n8n=N8nApi(**raw.get("n8n", {})),
        judge=Judge(**raw.get("judge", {})),
        repeats=int(raw.get("repeats", 3)),
        concurrency=int(raw.get("concurrency", 2)),
        prices=raw.get("prices", {}),
        gate=raw.get("gate", {}),
        path=path,
    )
