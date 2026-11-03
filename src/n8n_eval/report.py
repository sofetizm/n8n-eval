from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import stats


def matrix(rows: Sequence[dict[str, Any]], key: str | None = None) -> dict[str, list[bool]]:
    """case -> outcomes per repeat; with `key`, the outcome of one named check."""
    out: dict[str, list[bool]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["case"], r["repeat"])):
        if key is None:
            out[r["case"]].append(bool(r["passed"]))
        else:
            found = [c["passed"] for c in r["checks"] if c["name"] == key]
            if found:
                out[r["case"]].append(all(found))
    return dict(out)


def _pct(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    m = matrix(rows)
    check_names = sorted({c["name"] for r in rows for c in r["checks"]})
    case_tags = {r["case"]: set(r.get("tags", [])) for r in rows}
    tags = sorted(set().union(*case_tags.values())) if case_tags else []
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None and not r.get("error")]
    traces = [r["trace"] for r in rows if r.get("trace") and r["trace"].get("execution_id")]
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    tool_counts: dict[str, int] = defaultdict(int)
    for t in traces:
        for tool in t["tools"]:
            tool_counts[tool] += 1
    return {
        "label": rows[0]["label"] if rows else "",
        "pass": stats.pass_rate(m),
        "checks": {name: stats.pass_rate(matrix(rows, name)) for name in check_names},
        "tags": {tag: stats.pass_rate({k: v for k, v in m.items() if tag in case_tags[k]}) for tag in tags},
        "flaky": stats.flaky_cases(m),
        "single_run_spread": stats.single_run_spread(m),
        "errors": sum(1 for r in rows if r.get("error")),
        "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95)},
        "traced_runs": len(traces),
        "llm_calls_mean": float(np.mean([t["llm_calls"] for t in traces])) if traces else None,
        "tokens_mean": float(np.mean([t["prompt_tokens"] + t["completion_tokens"] for t in traces]))
        if traces
        else None,
        "cost_per_1000_usd": float(np.mean(costs) * 1000) if costs else None,
        "tool_calls": dict(tool_counts),
        "failures": [
            {
                "case": r["case"],
                "repeat": r["repeat"],
                "output": r.get("output", "")[:300],
                "failed": [f"{c['name']}: {c['detail']}" for c in r["checks"] if not c["passed"]],
            }
            for r in rows
            if not r["passed"]
        ],
    }


def _rate(x: dict[str, Any]) -> str:
    return f"{x['rate']:.0%} ({x['ci_low']:.0%}–{x['ci_high']:.0%})"


def _num(value: float | None, fmt: str) -> str:
    return "—" if value is None else format(value, fmt)


def markdown(summary: dict[str, Any], title: str | None = None) -> str:
    p = summary["pass"]
    lines = [f"### {title or summary['label']}", ""]
    lines.append(
        f"**Прошли проверки: {_rate(p)}** — {p['cases']} случаев × {p['runs'] // max(p['cases'], 1)} "
        f"повтора, 95% ДИ по случаям."
    )
    spread = summary.get("single_run_spread")
    if spread:
        lines.append(f"Один прогон без повторов дал бы от {spread['min']:.0%} до {spread['max']:.0%}.")
    if summary["flaky"]:
        lines.append(
            f"Нестабильные случаи (то проходят, то нет): {len(summary['flaky'])} — "
            + ", ".join(f"`{c}`" for c in summary["flaky"][:10])
        )
    lines += ["", "| Проверка | Прошли |", "|---|---|"]
    lines += [f"| {name} | {_rate(x)} |" for name, x in summary["checks"].items()]
    if summary["tags"]:
        lines += ["", "| Группа | Прошли |", "|---|---|"]
        lines += [f"| {tag} | {_rate(x)} |" for tag, x in summary["tags"].items()]
    lat = summary["latency_ms"]
    lines += [
        "",
        "| Задержка p50 / p95 | Вызовов LLM | Токенов на ответ | $ за 1000 ответов | Ошибок |",
        "|---|---|---|---|---|",
        f"| {_num(lat['p50'], '.0f')} / {_num(lat['p95'], '.0f')} мс | {_num(summary['llm_calls_mean'], '.1f')} | "
        f"{_num(summary['tokens_mean'], '.0f')} | {_num(summary['cost_per_1000_usd'], '.3f')} | "
        f"{summary['errors']} |",
    ]
    if summary["failures"]:
        lines += ["", "<details><summary>Провалы</summary>", ""]
        for f in summary["failures"][:30]:
            lines.append(f"- `{f['case']}` #{f['repeat']}: {'; '.join(f['failed'])[:300]}")
        lines += ["", "</details>"]
    return "\n".join(lines)


def compare_markdown(base_rows: Sequence[dict], cand_rows: Sequence[dict], delta: float = 0.05) -> tuple[str, dict]:
    result = stats.compare(matrix(base_rows), matrix(cand_rows))
    if not result.get("cases"):
        return "Нет общих случаев для сравнения.", result
    needed = stats.cases_needed(result["sd_diff"], delta)
    base_label, cand_label = base_rows[0]["label"], cand_rows[0]["label"]
    lines = [
        f"### {cand_label} против {base_label}",
        "",
        "| | " + base_label + " | " + cand_label + " | Разница (95% ДИ) |",
        "|---|---|---|---|",
        f"| Прошли проверки | {result['baseline']:.0%} | {result['candidate']:.0%} | "
        f"**{result['diff']:+.1%}** ({result['ci_low']:+.1%} … {result['ci_high']:+.1%}) |",
        "",
    ]
    if result["broken"]:
        lines.append("Сломались: " + ", ".join(f"`{c}`" for c in result["broken"]))
    if result["fixed"]:
        lines.append("Починились: " + ", ".join(f"`{c}`" for c in result["fixed"]))
    if needed:
        lines.append(
            f"\nЧтобы надёжно (мощность 80%) заметить падение на {delta:.0%}, нужно ≈{needed} случаев "
            f"при текущем разбросе; сейчас {result['cases']}."
        )
    result["cases_needed"] = needed
    return "\n".join(lines), result
