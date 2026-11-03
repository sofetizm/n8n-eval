"""Report for examples/bookstore/run_experiment.sh: summaries, paired comparisons, repeat analysis, charts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from n8n_eval import stats  # noqa: E402
from n8n_eval.report import compare_markdown, matrix, summarize  # noqa: E402
from n8n_eval.runner import load_rows  # noqa: E402

VARIANTS = [
    ("gpt-oss-120b", "gpt-oss-120b, полный промпт"),
    ("gpt-oss-20b", "gpt-oss-20b, полный промпт"),
    ("short-prompt", "gpt-oss-20b, короткий промпт"),
]
TAGS = [("orders", "заказы"), ("policy", "правила"), ("mixed", "заказ + правила"), ("safety", "границы и безопасность")]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def pct(x: float) -> str:
    return f"{x:.0%}"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_groups(summaries: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    groups = [("all", "все случаи")] + TAGS
    height = 0.26
    for i, (label, title) in enumerate(VARIANTS):
        s = summaries.get(label)
        if not s:
            continue
        values = [s["pass"]["rate"]] + [s["tags"].get(t, {}).get("rate", np.nan) for t, _ in TAGS]
        y = np.arange(len(groups)) + (i - 1) * height
        ax.barh(y, values, height=height * 0.9, color=COLORS[i], label=title)
        for yy, v in zip(y, values):
            if not np.isnan(v):
                ax.text(v + 0.01, yy, pct(v), va="center", fontsize=8, color=TEXT)
    ax.set_yticks(np.arange(len(groups)), [g[1] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=3, labelcolor=TEXT)
    ax.set_title("Доля прогонов, прошедших все проверки", loc="left", fontsize=11, color=TEXT)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def chart_repeats(rows_by_variant: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=SURFACE)
    for i, (label, _) in enumerate(VARIANTS):
        rows = rows_by_variant.get(label)
        if not rows:
            continue
        m = matrix(rows)
        repeats = min(len(v) for v in m.values())
        singles = [np.mean([runs[r] for runs in m.values()]) for r in range(repeats)]
        p = stats.pass_rate(m)
        ax.plot([p["ci_low"], p["ci_high"]], [i, i], color=COLORS[i], linewidth=6, alpha=0.25, solid_capstyle="round")
        ax.scatter(singles, [i] * len(singles), color=COLORS[i], s=40, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.scatter([p["rate"]], [i], marker="|", s=300, color=TEXT, zorder=4)
    ax.set_yticks(range(len(VARIANTS)), [t for _, t in VARIANTS])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Один прогон против повторов", loc="left", fontsize=11, color=TEXT)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main(results_dir: str) -> None:
    root = Path(results_dir)
    rows = {label: load_rows(root / f"{label}.jsonl") for label, _ in VARIANTS}
    rows = {k: v for k, v in rows.items() if v}
    summaries = {k: summarize(v) for k, v in rows.items()}
    lines = ["# Результаты эксперимента", ""]

    lines += [
        "| Версия | Прошли все проверки (95% ДИ) | Нестабильных случаев | Вызовов LLM | Токенов | p50 / p95, с | $ за 1000 ответов |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, title in VARIANTS:
        s = summaries.get(label)
        if not s:
            continue
        p, lat = s["pass"], s["latency_ms"]
        lines.append(
            f"| {title} | **{pct(p['rate'])}** ({pct(p['ci_low'])}–{pct(p['ci_high'])}) | "
            f"{len(s['flaky'])} из {p['cases']} | {s['llm_calls_mean'] or 0:.1f} | {s['tokens_mean'] or 0:.0f} | "
            f"{(lat['p50'] or 0) / 1000:.1f} / {(lat['p95'] or 0) / 1000:.1f} | {s['cost_per_1000_usd'] or 0:.3f} |"
        )
    lines.append("")

    check_names = sorted({c for s in summaries.values() for c in s["checks"]})
    lines += [
        "| Проверка | " + " | ".join(t for label, t in VARIANTS if label in summaries) + " |",
        "|---" * (1 + len(summaries)) + "|",
    ]
    for name in check_names:
        cells = [
            pct(summaries[label]["checks"][name]["rate"]) if name in summaries[label]["checks"] else "—"
            for label, _ in VARIANTS
            if label in summaries
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

    comparisons = {}
    for base, cand in (("gpt-oss-120b", "gpt-oss-20b"), ("gpt-oss-20b", "short-prompt")):
        if base in rows and cand in rows:
            text, result = compare_markdown(rows[base], rows[cand])
            comparisons[f"{cand} vs {base}"] = result
            lines += [text, ""]

    if "gpt-oss-120b" in rows:
        m = matrix(rows["gpt-oss-120b"])
        p = stats.pass_rate(m)
        spread = stats.single_run_spread(m)
        lines += [
            "## Повторы",
            "",
            f"- Отдельные прогоны gpt-oss-120b: от {pct(spread.get('min', 0))} до {pct(spread.get('max', 0))}.",
            f"- 95% ДИ, если считать прогоны независимыми: {pct(p['naive_ci_low'])}–{pct(p['naive_ci_high'])}; "
            f"по случаям (правильно): {pct(p['ci_low'])}–{pct(p['ci_high'])}.",
            f"- Нестабильные случаи: {', '.join(summaries['gpt-oss-120b']['flaky']) or 'нет'}.",
            "",
        ]

    for label, title in VARIANTS:
        s = summaries.get(label)
        if s and s["failures"]:
            lines += [f"<details><summary>Провалы: {title}</summary>", ""]
            lines += [f"- `{f['case']}` #{f['repeat']}: {'; '.join(f['failed'])[:250]}" for f in s["failures"]]
            lines += ["", "</details>", ""]

    chart_groups(summaries, root / "groups.png")
    chart_repeats(rows, root / "repeats.png")
    lines += ["![Группы](groups.png)", "", "![Повторы](repeats.png)", ""]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "experiment.json").write_text(
        json.dumps({"summaries": summaries, "comparisons": comparisons}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(root / "RESULTS.md")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/bookstore/results")
