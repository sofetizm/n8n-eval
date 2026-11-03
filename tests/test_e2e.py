"""End to end against a real n8n with the mock LLM (tests/mock_llm.py):

    N8N_TEST_URL=http://localhost:5678 MOCK_LLM_URL=http://localhost:8100/v1 pytest tests/test_e2e.py

The n8n instance must be fresh or set up by a previous run of this test (same owner credentials).
"""

import asyncio
import os
from pathlib import Path

import pytest

from n8n_eval.bootstrap import bootstrap
from n8n_eval.config import load_suite
from n8n_eval.judge import LLMJudge
from n8n_eval.n8n import N8nClient
from n8n_eval.report import summarize
from n8n_eval.runner import Runner

URL = os.environ.get("N8N_TEST_URL")
MOCK = os.environ.get("MOCK_LLM_URL", "http://localhost:8100/v1")
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bookstore"
pytestmark = pytest.mark.skipif(not URL, reason="N8N_TEST_URL is not set")


@pytest.fixture(scope="module")
def api_key(tmp_path_factory, monkeypatch_module=None):
    os.environ["LLM_API_KEY"] = "mock"
    os.environ["LLM_BASE_URL"] = MOCK
    workflows = sorted(str(p) for p in (EXAMPLE / "workflows").glob("*.json"))
    result = bootstrap(URL, workflows, str(EXAMPLE / "credentials.json"), "ci@example.com", "Ci-password-123")
    assert {w["name"] for w in result["workflows"]} >= {"bookstore-support", "bookstore-support-20b"}
    return result["api_key"]


def test_suite_against_n8n(api_key, tmp_path):
    os.environ.update(N8N_URL=URL, N8N_API_KEY=api_key, JUDGE_BASE_URL=MOCK)
    suite = load_suite(EXAMPLE / "suite.yaml")
    cases = suite.select(tags=["orders"])
    runner = Runner(suite, "e2e", tmp_path / "r.jsonl", judge=LLMJudge(suite.judge), n8n=N8nClient(suite.n8n))
    rows = asyncio.run(runner.run(cases, repeats=1, concurrency=2))
    assert len(rows) == len(cases) and not any(r.get("error") for r in rows)
    by_case = {r["case"]: r for r in rows}
    shipped = by_case["order-shipped"]
    assert shipped["passed"], shipped["checks"]
    assert shipped["trace"]["tools"] == ["get_order"] and shipped["trace"]["llm_calls"] == 2
    assert shipped["trace"]["prompt_tokens"] > 0 and shipped["cost_usd"] > 0
    assert "openai/gpt-oss-120b" in shipped["trace"]["models"]
    assert by_case["order-no-number"]["trace"]["tools"] == []
    summary = summarize(rows)
    assert summary["traced_runs"] == len(cases) and summary["tool_calls"]["get_order"] >= 8


def test_variant_workflow_is_traced_separately(api_key, tmp_path):
    os.environ.update(N8N_URL=URL, N8N_API_KEY=api_key)
    suite = load_suite(
        EXAMPLE / "suite.yaml",
        {"target.webhook": f"{URL}/webhook/bookstore-support-20b", "n8n.workflow": "bookstore-support-20b"},
    )
    runner = Runner(suite, "e2e-20b", tmp_path / "r.jsonl", n8n=N8nClient(suite.n8n))
    (row,) = asyncio.run(runner.run(suite.select(ids=["order-track"]), repeats=1, concurrency=1))
    assert row["passed"] and row["trace"]["models"] == ["openai/gpt-oss-20b"]
