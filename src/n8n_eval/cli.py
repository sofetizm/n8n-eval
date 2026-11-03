from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .config import load_suite
from .gate import evaluate_gate
from .judge import JudgeError, LLMJudge
from .n8n import N8nClient
from .report import compare_markdown, markdown, summarize
from .runner import Runner, TooManyErrors, load_rows


def _write(path: str | None, text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        print(f"saved {path}", file=sys.stderr)


def _step_summary(text: str) -> None:
    """Adds the report to the GitHub Actions job page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")


def _rows(path: str, label: str | None = None) -> list[dict]:
    rows = load_rows(Path(path))
    if label:
        rows = [r for r in rows if r["label"] == label]
    if not rows:
        sys.exit(f"no results in {path}")
    return rows


async def _run(args) -> list[dict]:
    overrides = {}
    if args.webhook:
        overrides["target.webhook"] = args.webhook
    if args.workflow:
        overrides["n8n.workflow"] = args.workflow
    suite = load_suite(args.suite, overrides)
    cases = suite.select(args.tag, args.case, args.limit)
    repeats = args.repeats or suite.repeats
    needs_judge = any("judge" in c.expect for c in cases)
    judge = LLMJudge(suite.judge) if needs_judge else None
    n8n = N8nClient(suite.n8n) if suite.n8n.enabled else None
    label = args.label or suite.n8n.workflow or suite.name
    out = args.out or f"results/{suite.name}/{label}.jsonl"
    print(
        f"{suite.name}: {len(cases)} cases × {repeats} repeats -> {out}"
        + ("" if n8n else " (no n8n API key: tools, tokens and cost are not traced)"),
        file=sys.stderr,
    )
    runner = Runner(
        suite,
        label,
        out,
        judge=judge,
        n8n=n8n,
        progress=lambda n, total: print(f"\r{n}/{total}", end="", file=sys.stderr, flush=True),
    )
    try:
        rows = await runner.run(cases, repeats, args.concurrency or suite.concurrency)
    finally:
        print(file=sys.stderr)
        await runner.aclose()
        for client in (judge, n8n):
            if client:
                await client.aclose()
    args.out = out
    return rows


def cmd_run(args) -> None:
    try:
        rows = asyncio.run(_run(args))
    except (TooManyErrors, JudgeError) as error:
        sys.exit(f"{error}\nFinished runs are saved; run the same command again to continue.")
    summary = summarize(rows)
    text = markdown(summary)
    print(text)
    _step_summary(text)
    _write(str(Path(args.out).with_suffix(".summary.json")), json.dumps(summary, ensure_ascii=False, indent=2))
    if args.gate or args.baseline:
        _gate(args, rows, summary)


def _gate(args, rows: list[dict], summary: dict) -> None:
    suite = load_suite(args.suite)
    comparison = None
    if args.baseline:
        text, comparison = compare_markdown(_rows(args.baseline), rows, suite.gate.get("max_regression", 0.05))
        print(text)
        _step_summary(text)
    failures = evaluate_gate(suite.gate, summary, comparison)
    if failures:
        message = "**Гейт не пройден:**\n" + "\n".join(f"- {f}" for f in failures)
        print(message)
        _step_summary(message)
        sys.exit(1)
    print("Гейт пройден.")
    _step_summary("Гейт пройден ✅")


def cmd_report(args) -> None:
    rows = _rows(args.results, args.label)
    text = markdown(summarize(rows))
    print(text)
    _write(args.md, text)


def cmd_compare(args) -> None:
    text, result = compare_markdown(_rows(args.baseline), _rows(args.candidate), args.delta)
    print(text)
    _step_summary(text)
    _write(args.md, text)
    _write(args.json, json.dumps(result, ensure_ascii=False, indent=2))


def cmd_gate(args) -> None:
    rows = _rows(args.results)
    _gate(args, rows, summarize(rows))


def cmd_up(args) -> None:
    from .bootstrap import bootstrap

    result = bootstrap(args.url, args.workflows, args.credentials, args.email, args.password)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::add-mask::{result['api_key']}")
    if args.env_file:
        with open(args.env_file, "a", encoding="utf-8") as f:
            f.write(f"N8N_API_KEY={result['api_key']}\n")
    print(json.dumps({k: v for k, v in result.items() if k != "api_key"}, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="n8n-eval", description="Evaluate n8n AI workflows: run, compare, gate")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="send every case to the workflow webhook and check the answers")
    p.add_argument("suite")
    p.add_argument("--label", help="name of this run, e.g. main or pr-42")
    p.add_argument("--out", help="results JSONL (runs already in it are skipped, so a stopped run resumes)")
    p.add_argument("--repeats", type=int)
    p.add_argument("--concurrency", type=int)
    p.add_argument("--tag", action="append", help="only cases with this tag")
    p.add_argument("--case", action="append", help="only this case id")
    p.add_argument("--limit", type=int)
    p.add_argument("--webhook", help="override target.webhook")
    p.add_argument("--workflow", help="override n8n.workflow")
    p.add_argument("--baseline", help="results JSONL to compare with; applies the regression gate")
    p.add_argument("--gate", action="store_true", help="apply the suite's gate and exit 1 if it fails")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="Markdown report for a results file")
    p.add_argument("results")
    p.add_argument("--label")
    p.add_argument("--md")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="paired comparison of two runs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--delta", type=float, default=0.05, help="drop to detect, for the sample size estimate")
    p.add_argument("--md")
    p.add_argument("--json")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("gate", help="apply the suite's gate to a results file")
    p.add_argument("suite")
    p.add_argument("results")
    p.add_argument("--baseline")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("up", help="prepare a fresh n8n for CI: owner, API key, credentials, workflows")
    p.add_argument("--url", default="http://localhost:5678")
    p.add_argument("--workflows", nargs="*", default=[])
    p.add_argument("--credentials", help="JSON file with credentials; ${VAR} is replaced from the environment")
    p.add_argument("--email", default="ci@example.com")
    p.add_argument("--password", default="Ci-password-123")
    p.add_argument("--env-file", default=os.environ.get("GITHUB_ENV"), help="append N8N_API_KEY=... here")
    p.set_defaults(func=cmd_up)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
