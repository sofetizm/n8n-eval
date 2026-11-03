"""Quality gate: exit code 1 in CI when a workflow change makes the agent worse.

gate:
  min_pass_rate: 0.85            # share of passed runs (point estimate)
  min_checks: {judge: 0.8}       # per-check minimums
  max_p95_latency_ms: 20000
  max_cost_per_1000_usd: 2.0
  max_regression: 0.05           # vs --baseline: fail if the pass rate dropped by more than this
  fail_on_significant_drop: true # vs --baseline: fail if the whole 95% CI of the change is below zero
"""

from __future__ import annotations

from typing import Any


def evaluate_gate(gate: dict[str, Any], summary: dict[str, Any], comparison: dict[str, Any] | None = None) -> list[str]:
    failures = []
    rate = summary["pass"]["rate"]
    if "min_pass_rate" in gate and rate < gate["min_pass_rate"]:
        failures.append(f"pass rate {rate:.1%} < {gate['min_pass_rate']:.0%}")
    for name, minimum in (gate.get("min_checks") or {}).items():
        value = summary["checks"].get(name, {}).get("rate")
        if value is None:
            failures.append(f"check '{name}' was not run")
        elif value < minimum:
            failures.append(f"check '{name}' {value:.1%} < {minimum:.0%}")
    p95 = summary["latency_ms"].get("p95")
    if "max_p95_latency_ms" in gate and p95 is not None and p95 > gate["max_p95_latency_ms"]:
        failures.append(f"p95 latency {p95:.0f} ms > {gate['max_p95_latency_ms']} ms")
    cost = summary.get("cost_per_1000_usd")
    if "max_cost_per_1000_usd" in gate and cost is not None and cost > gate["max_cost_per_1000_usd"]:
        failures.append(f"cost ${cost:.3f} per 1000 > ${gate['max_cost_per_1000_usd']}")
    if comparison and comparison.get("cases"):
        if "max_regression" in gate and comparison["diff"] < -gate["max_regression"]:
            failures.append(
                f"pass rate dropped by {-comparison['diff']:.1%} vs baseline (max {gate['max_regression']:.0%})"
            )
        if gate.get("fail_on_significant_drop", True) and comparison["ci_high"] < 0:
            failures.append(
                f"significant drop vs baseline: 95% CI {comparison['ci_low']:+.1%} … {comparison['ci_high']:+.1%}"
            )
    return failures
