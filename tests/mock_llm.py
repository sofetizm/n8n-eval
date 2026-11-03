"""OpenAI-compatible mock with tool calling, for end-to-end tests of n8n workflows without an API key.

uvicorn tests.mock_llm:app --port 8100
"""

from __future__ import annotations

import json
import re
import time
import uuid

from fastapi import FastAPI, Request

app = FastAPI()
POLICY_WORDS = ("доставк", "возврат", "вернуть", "оплат", "сертификат", "бонус", "самовывоз", "рассрочк")


def _message(content: str | None = None, tool_calls: list | None = None) -> dict:
    return {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}


def _call(name: str, args: dict) -> dict:
    return {
        "id": "call_" + uuid.uuid4().hex[:8],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def respond(body: dict) -> dict:
    messages = body.get("messages", [])
    tools = {t["function"]["name"] for t in body.get("tools") or []}
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if isinstance(user, list):
        user = " ".join(part.get("text", "") for part in user if isinstance(part, dict))
    tool_results = [m["content"] for m in messages if m["role"] == "tool"]
    if "Ты проверяешь ответ" in str(user):
        verdict = "нет" not in str(user).split("Ответ ассистента:")[-1].split("Критерий:")[0]
        message = _message(json.dumps({"pass": verdict, "reason": "mock"}, ensure_ascii=False))
    elif tool_results:
        message = _message("По данным магазина: " + " ".join(map(str, tool_results)))
    else:
        calls = []
        order = re.search(r"\b(\d{4})\b", user)
        if order and "get_order" in tools:
            calls.append(_call("get_order", {"order_number": order.group(1)}))
        if any(w in user.lower() for w in POLICY_WORDS) and "search_policy" in tools:
            calls.append(_call("search_policy", {"query": user}))
        message = _message(tool_calls=calls) if calls else _message("Я помогаю только с вопросами о магазине.")
    prompt_tokens = sum(len(str(m.get("content") or "")) for m in messages) // 4
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20, "total_tokens": prompt_tokens + 20},
    }


@app.post("/v1/chat/completions")
async def chat(request: Request) -> dict:
    return respond(await request.json())


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}
