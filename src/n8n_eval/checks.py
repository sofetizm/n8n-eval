"""Deterministic checks of one workflow answer. The LLM judge lives in judge.py."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def check_contains(output: str, expected: Any) -> CheckResult:
    text = normalize(output)
    missing = [e for e in _as_list(expected) if normalize(str(e)) not in text]
    return CheckResult("contains", not missing, f"missing: {missing}" if missing else "")


def check_contains_any(output: str, expected: Any) -> CheckResult:
    text = normalize(output)
    ok = any(normalize(str(e)) in text for e in _as_list(expected))
    return CheckResult("contains_any", ok, "" if ok else f"none of: {_as_list(expected)}")


def check_not_contains(output: str, forbidden: Any) -> CheckResult:
    text = normalize(output)
    found = [f for f in _as_list(forbidden) if normalize(str(f)) in text]
    return CheckResult("not_contains", not found, f"found: {found}" if found else "")


def check_regex(output: str, pattern: str) -> CheckResult:
    ok = re.search(pattern, output, re.I | re.S) is not None
    return CheckResult("regex", ok, "" if ok else f"no match: {pattern}")


def check_json(output: str, required_keys: Any) -> CheckResult:
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return CheckResult("json", False, "not valid JSON")
    keys = [] if required_keys in (True, None) else _as_list(required_keys)
    missing = [k for k in keys if not isinstance(data, dict) or k not in data]
    return CheckResult("json", not missing, f"missing keys: {missing}" if missing else "")


def check_tools(called: list[str] | None, expected: Any) -> CheckResult:
    if called is None:
        return CheckResult("tools", False, "no execution data: set n8n.api_key to check tools")
    missing = [t for t in _as_list(expected) if t not in called]
    return CheckResult("tools", not missing, f"not called: {missing}; called: {called}" if missing else "")


def check_no_tools(called: list[str] | None, forbidden: Any) -> CheckResult:
    if called is None:
        return CheckResult("no_tools", False, "no execution data: set n8n.api_key to check tools")
    names = called if forbidden is True else [t for t in _as_list(forbidden) if t in called]
    return CheckResult("no_tools", not names, f"called: {names}" if names else "")


def check_max_latency(latency_ms: float, limit: float) -> CheckResult:
    ok = latency_ms <= float(limit)
    return CheckResult("max_latency_ms", ok, "" if ok else f"{latency_ms:.0f} ms > {limit}")


TEXT_CHECKS = {
    "contains": check_contains,
    "contains_any": check_contains_any,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "json": check_json,
}
KNOWN = set(TEXT_CHECKS) | {"tools", "no_tools", "max_latency_ms", "judge"}


def run_checks(expect: dict[str, Any], output: str, latency_ms: float, tools: list[str] | None) -> list[CheckResult]:
    results = []
    for name, value in expect.items():
        if name in TEXT_CHECKS:
            results.append(TEXT_CHECKS[name](output, value))
        elif name == "tools":
            results.append(check_tools(tools, value))
        elif name == "no_tools":
            results.append(check_no_tools(tools, value))
        elif name == "max_latency_ms":
            results.append(check_max_latency(latency_ms, value))
    return results
