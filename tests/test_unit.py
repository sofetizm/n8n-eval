import asyncio
import json

import httpx
import pytest

from n8n_eval import stats
from n8n_eval.checks import run_checks
from n8n_eval.config import Case, Judge, expand_env, load_suite
from n8n_eval.gate import evaluate_gate
from n8n_eval.judge import LLMJudge, parse
from n8n_eval.report import compare_markdown, markdown, matrix, summarize
from n8n_eval.runner import Runner, extract


def names(results):
    return {r.name: r.passed for r in results}


def test_text_checks():
    out = "Ваш заказ 1042 отправлен, трек-номер RU123456789. Ожидается 6 ноября."
    res = run_checks(
        {
            "contains": ["ru123456789", "Отправлен"],
            "contains_any": ["доставлен", "отправлен"],
            "not_contains": "отменён",
            "regex": r"6\s+ноября",
        },
        out,
        100,
        None,
    )
    assert all(r.passed for r in res)
    res = run_checks({"contains": ["ё-мое", "трек"], "not_contains": ["заказ"]}, out, 100, None)
    assert names(res) == {"contains": False, "not_contains": False}
    assert run_checks({"contains": "ЕЛКА"}, "ёлка", 1, None)[0].passed


def test_json_tools_latency_checks():
    assert run_checks({"json": ["a"]}, '{"a": 1}', 1, None)[0].passed
    assert not run_checks({"json": True}, "not json", 1, None)[0].passed
    res = run_checks({"tools": ["get_order"], "no_tools": ["refund"], "max_latency_ms": 500}, "", 900, ["get_order"])
    assert names(res) == {"tools": True, "no_tools": True, "max_latency_ms": False}
    assert not run_checks({"no_tools": True}, "", 1, ["x"])[0].passed
    assert "api_key" in run_checks({"tools": ["x"]}, "", 1, None)[0].detail


def test_case_validation():
    with pytest.raises(ValueError, match="unknown checks"):
        Case("a", "q", {"contain": "x"})
    with pytest.raises(ValueError, match="no checks"):
        Case("a", "q", {})


def test_expand_env(monkeypatch):
    monkeypatch.setenv("TOKEN", "abc")
    assert expand_env({"a": "Bearer ${TOKEN}", "b": ["${MISSING:-def}", 3]}) == {"a": "Bearer abc", "b": ["def", 3]}
    assert expand_env("${MISSING:-${TOKEN}}") == "abc"


def test_load_suite(tmp_path):
    (tmp_path / "cases.jsonl").write_text(
        '{"id": "a", "input": "q1", "expect": {"contains": "x"}, "tags": ["t"]}\n'
        '{"id": "b", "input": "q2", "expect": {"contains": "y"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "suite.yaml").write_text(
        "name: s\ntarget: {webhook: http://x/webhook/a}\ncases: cases.jsonl\nrepeats: 2\n"
    )
    suite = load_suite(tmp_path / "suite.yaml", {"target.webhook": "http://y"})
    assert suite.target.webhook == "http://y" and suite.repeats == 2 and [c.id for c in suite.select(["t"])] == ["a"]
    (tmp_path / "cases.jsonl").write_text('{"id": "a", "input": "q", "expect": {"contains": "x"}}\n' * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(tmp_path / "suite.yaml")


def test_extract():
    data = {"output": "hi", "data": [{"text": "a"}]}
    assert extract(data, "output") == "hi" and extract(data, "data.0.text") == "a"
    assert extract(data, "data.5.text") is None and extract(data, "") == data


def test_cluster_ci_is_wider_than_naive():
    m = {f"c{i}": [i % 4 != 0] * 5 for i in range(40)}
    p = stats.pass_rate(m)
    assert p["rate"] == pytest.approx(0.75)
    assert (p["ci_high"] - p["ci_low"]) > (p["naive_ci_high"] - p["naive_ci_low"])


def test_flaky_and_spread():
    m = {"a": [True, False, True], "b": [True, True, True], "c": [False, False, True]}
    assert stats.flaky_cases(m) == ["a", "c"]
    assert stats.single_run_spread(m) == {"min": 1 / 3, "max": 1.0, "repeats": 3}


def test_compare_paired():
    base = {f"c{i}": [True, True] for i in range(30)}
    cand = {**base, **{f"c{i}": [False, False] for i in range(6)}}
    r = stats.compare(base, cand)
    assert r["diff"] == pytest.approx(-0.2) and r["ci_high"] < 0
    assert sorted(r["broken"]) == sorted(f"c{i}" for i in range(6)) and r["fixed"] == []
    assert stats.cases_needed(r["sd_diff"], 0.05) > 30


def rows(label, outcomes, tags=("t",)):
    out = []
    for case, runs in outcomes.items():
        for i, ok in enumerate(runs):
            out.append(
                {
                    "label": label,
                    "case": case,
                    "repeat": i,
                    "tags": list(tags),
                    "passed": ok,
                    "checks": [{"name": "contains", "passed": ok, "detail": "" if ok else "missing"}],
                    "latency_ms": 100 + i,
                    "output": "x",
                    "trace": {
                        "execution_id": "1",
                        "tools": ["get_order"],
                        "llm_calls": 2,
                        "prompt_tokens": 1000,
                        "completion_tokens": 100,
                    },
                    "cost_usd": 0.0002,
                }
            )
    return out


def test_summary_markdown_and_gate():
    r = rows("main", {"a": [True, True], "b": [True, False], "c": [False, False]})
    s = summarize(r)
    assert s["pass"]["rate"] == pytest.approx(0.5) and s["flaky"] == ["b"] and s["tool_calls"] == {"get_order": 6}
    assert s["cost_per_1000_usd"] == pytest.approx(0.2) and len(s["failures"]) == 3
    text = markdown(s)
    assert "Прошли проверки: 50%" in text and "`b`" in text
    assert evaluate_gate({"min_pass_rate": 0.4, "max_cost_per_1000_usd": 1}, s) == []
    fails = evaluate_gate({"min_pass_rate": 0.9, "min_checks": {"judge": 0.8}, "max_p95_latency_ms": 50}, s)
    assert len(fails) == 3


def test_compare_markdown_and_regression_gate():
    base = rows("main", {f"c{i}": [True, True] for i in range(20)})
    cand = rows("pr", {f"c{i}": [i >= 5, i >= 5] for i in range(20)})
    text, result = compare_markdown(base, cand)
    assert "Сломались" in text and result["diff"] == pytest.approx(-0.25)
    fails = evaluate_gate({"max_regression": 0.05}, summarize(cand), result)
    assert any("dropped" in f for f in fails) and any("significant" in f for f in fails)
    assert matrix(cand, "contains")["c0"] == [False, False]


def test_judge_parse_and_call():
    assert parse('```json\n{"pass": true, "reason": "ок"}\n```') == {"pass": True, "reason": "ок"}

    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0 and "Критерий: называет срок" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"pass": false, "reason": "нет"}'}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            },
        )

    judge = LLMJudge(Judge(api_key="k"), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = asyncio.run(judge.check("вопрос", "ответ", "называет срок"))
    assert not result.passed and result.detail == "нет" and judge.tokens == {"prompt": 50, "completion": 5}


def test_runner_resumes(tmp_path):
    (tmp_path / "cases.jsonl").write_text(
        '{"id": "a", "input": "заказ 1042", "expect": {"contains": "1042"}}\n'
        '{"id": "b", "input": "привет", "expect": {"contains": "1042"}}\n',
        encoding="utf-8",
    )
    (tmp_path / "s.yaml").write_text("name: s\ntarget: {webhook: http://n8n/webhook/x}\ncases: cases.jsonl\n")
    suite = load_suite(tmp_path / "s.yaml")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["_eval"].startswith("main:") and body["sessionId"] == body["_eval"]
        return httpx.Response(200, json={"output": f"вы спросили: {body['message']}"})

    def make():
        return Runner(
            suite, "main", tmp_path / "r.jsonl", http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        )

    result = asyncio.run(make().run(suite.cases, repeats=2, concurrency=2))
    assert len(result) == 4 and len(calls) == 4
    assert {r["case"]: r["passed"] for r in result} == {"a": True, "b": False}
    result = asyncio.run(make().run(suite.cases, repeats=3, concurrency=2))
    assert len(result) == 6 and len(calls) == 6


def test_runner_records_http_errors(tmp_path):
    (tmp_path / "s.yaml").write_text(
        "name: s\ntarget: {webhook: http://n8n/webhook/x}\ncases:\n  - {id: a, input: q, expect: {contains: x}}\n"
    )
    suite = load_suite(tmp_path / "s.yaml")
    runner = Runner(
        suite,
        "main",
        tmp_path / "r.jsonl",
        http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))),
    )
    (row,) = asyncio.run(runner.run(suite.cases, 1, 1))
    assert not row["passed"] and row["checks"][0]["name"] == "request" and "500" in row["error"]


def test_runner_retries_errors_and_stops_on_error_streak(tmp_path):
    (tmp_path / "s.yaml").write_text(
        "name: s\ntarget: {webhook: http://n8n/webhook/x}\ncases:\n"
        + "".join(f"  - {{id: c{i}, input: q, expect: {{contains: ok}}}}\n" for i in range(10))
    )
    suite = load_suite(tmp_path / "s.yaml")
    state = {"fail": True, "calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["fail"]:
            return httpx.Response(500, json={"message": "Service unavailable"})
        return httpx.Response(200, json={"output": "ok"})

    def make():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return Runner(suite, "main", tmp_path / "r.jsonl", http=client, max_consecutive_errors=3)

    from n8n_eval.runner import TooManyErrors

    with pytest.raises(TooManyErrors, match="Service unavailable"):
        asyncio.run(make().run(suite.cases, 1, 1))
    assert state["calls"] == 3
    state["fail"] = False
    result = asyncio.run(make().run(suite.cases, 1, 1))
    assert len(result) == 10 and all(r["passed"] for r in result) and state["calls"] == 13


def test_parse_real_execution():
    """tests/execution_sample.json is GET /api/v1/executions?includeData=true from n8n 2.46 (demo workflow)."""
    from pathlib import Path

    from n8n_eval.n8n import parse_execution
    from n8n_eval.runner import cost

    execution = json.loads((Path(__file__).parent / "execution_sample.json").read_text(encoding="utf-8"))
    trace = parse_execution(execution)
    assert trace.status == "success" and trace.tools == ["get_order"] and trace.llm_calls == 2
    assert trace.models == ["openai/gpt-oss-120b"] and trace.prompt_tokens > 0 and trace.completion_tokens > 0
    assert set(trace.node_ms) >= {"Webhook", "Support agent", "Chat model", "get_order", "Respond"}
    assert cost(trace, {}) > 0 and cost(trace, {"openai/gpt-oss-120b": [0, 0]}) == 0


def test_retry_after_parsing():
    from n8n_eval.runner import is_rate_limit, retry_after

    msg = "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): Limit 8000. Please try again in 7.66s."
    assert is_rate_limit(msg) and retry_after(msg) == pytest.approx(7.66)
    assert retry_after("Please try again in 1m2.5s") == pytest.approx(62.5)
    assert retry_after("tokens per day (TPD). Please try again in 2h3m") == pytest.approx(7380)
    assert retry_after("try again in 350ms") == pytest.approx(0.35)
    assert retry_after("Rate limit reached") == 20.0
    assert not is_rate_limit("HTTP 500: Error in workflow")


def test_runner_waits_on_minute_rate_limit(tmp_path, monkeypatch):
    (tmp_path / "s.yaml").write_text(
        "name: s\ntarget: {webhook: http://n8n/webhook/x}\ncases:\n  - {id: a, input: q, expect: {contains: ok}}\n"
    )
    suite = load_suite(tmp_path / "s.yaml")
    calls = {"n": 0}
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("n8n_eval.runner.asyncio.sleep", fake_sleep)

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(500, json={"message": "Rate limit reached ... Please try again in 3.5s"})
        return httpx.Response(200, json={"output": "ok"})

    runner = Runner(suite, "main", tmp_path / "r.jsonl", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    (row,) = asyncio.run(runner.run(suite.cases, 1, 1))
    assert row["passed"] and calls["n"] == 3 and sleeps == [4.5, 4.5]

    calls["n"] = -100
    row = asyncio.run(runner.run_one(suite.cases[0], 1, max_wait=1))
    assert row["error"].startswith("rate limit:")
