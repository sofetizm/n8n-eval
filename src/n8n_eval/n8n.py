"""n8n public API: find the execution behind a webhook call and read tools, tokens and node timings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from .config import N8nApi

TOOL_MARKERS = ("tool", "Tool")


@dataclass
class Trace:
    execution_id: str | None = None
    status: str | None = None
    tools: list[str] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: list[str] = field(default_factory=list)
    node_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _token_usage(run: dict[str, Any]) -> tuple[int, int]:
    for item in ((run.get("data") or {}).get("ai_languageModel") or [[]])[0] or []:
        usage = (item.get("json") or {}).get("tokenUsage") or (item.get("json") or {}).get("tokenUsageEstimate")
        if usage:
            return int(usage.get("promptTokens", 0)), int(usage.get("completionTokens", 0))
    return 0, 0


def is_tool(node_type: str) -> bool:
    short = node_type.rsplit(".", 1)[-1]
    return short.startswith("tool") or short.endswith("Tool")


def is_chat_model(node_type: str) -> bool:
    return node_type.rsplit(".", 1)[-1].startswith("lmChat")


def model_name(node: dict[str, Any]) -> str:
    model = (node.get("parameters") or {}).get("model", "")
    if isinstance(model, dict):
        model = model.get("value") or model.get("cachedResultName") or ""
    return str(model)


def parse_execution(execution: dict[str, Any]) -> Trace:
    """Execution JSON from GET /api/v1/executions/{id}?includeData=true -> Trace."""
    trace = Trace(execution_id=str(execution.get("id")), status=execution.get("status"))
    nodes = {n["name"]: n for n in (execution.get("workflowData") or {}).get("nodes", [])}
    result = (execution.get("data") or {}).get("resultData") or {}
    run_data: dict[str, list[dict]] = result.get("runData") or {}
    for name, runs in run_data.items():
        node = nodes.get(name, {})
        node_type = node.get("type", "")
        trace.node_ms[name] = float(sum(r.get("executionTime", 0) for r in runs))
        if is_tool(node_type):
            trace.tools.extend([name] * len(runs))
        if is_chat_model(node_type):
            trace.llm_calls += len(runs)
            model = model_name(node)
            if model and model not in trace.models:
                trace.models.append(model)
            for run in runs:
                p, c = _token_usage(run)
                trace.prompt_tokens += p
                trace.completion_tokens += c
    error = result.get("error")
    if error:
        node = (error.get("node") or {}).get("name") if isinstance(error, dict) else None
        message = error.get("description") or error.get("message") if isinstance(error, dict) else error
        trace.error = (f"{node}: " if node else "") + str(message)[:500]
    return trace


class N8nClient:
    def __init__(self, config: N8nApi, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient(
            base_url=config.url.rstrip("/") + "/api/v1", headers={"X-N8N-API-KEY": config.api_key}, timeout=60
        )
        self._workflow_id: str | None = None

    async def workflow_id(self) -> str:
        if self._workflow_id:
            return self._workflow_id
        ref = self.config.workflow
        if not ref:
            raise ValueError("n8n.workflow is empty: set the workflow name or id")
        cursor = None
        while True:
            params = {"limit": 250, **({"cursor": cursor} if cursor else {})}
            data = (await self.client.get("/workflows", params=params)).raise_for_status().json()
            for wf in data.get("data", []):
                if ref in (wf.get("id"), wf.get("name")):
                    self._workflow_id = str(wf["id"])
                    return self._workflow_id
            cursor = data.get("nextCursor")
            if not cursor:
                raise ValueError(f"workflow '{ref}' not found in n8n")

    async def find_execution(self, marker: str, started_after: datetime, attempts: int = 10) -> Trace:
        """The execution whose webhook body carries `marker`; executions are saved shortly after the response."""
        workflow = await self.workflow_id()
        for attempt in range(attempts):
            params = {"workflowId": workflow, "limit": 20, "includeData": "true"}
            data = (await self.client.get("/executions", params=params)).raise_for_status().json()
            for execution in data.get("data", []):
                started = _parse_time(execution.get("startedAt"))
                if started and started < started_after:
                    continue
                if marker in str((execution.get("data") or {}).get("resultData", {}).get("runData", {})):
                    return parse_execution(execution)
            await asyncio.sleep(0.5 * (attempt + 1))
        return Trace(error="execution not found in n8n (is saving successful executions enabled?)")

    async def aclose(self) -> None:
        await self.client.aclose()
