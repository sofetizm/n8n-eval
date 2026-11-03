"""Prepares a fresh n8n (a CI container) for evaluation.

1. creates the owner account and an API key (internal REST API of the n8n UI);
2. creates credentials from a JSON file, taking secrets from environment variables;
3. imports workflows, links their nodes to those credentials by name, and activates them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from .config import expand_env

API_KEY_SCOPES = [
    "workflow:create",
    "workflow:read",
    "workflow:update",
    "workflow:delete",
    "workflow:list",
    "workflow:activate",
    "workflow:deactivate",
    "execution:read",
    "execution:list",
    "credential:create",
    "credential:delete",
]
WORKFLOW_FIELDS = ("name", "nodes", "connections", "settings")


def wait_ready(client: httpx.Client, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            settings = client.get("/rest/settings")
            if settings.status_code == 200 and "userManagement" in settings.json().get("data", {}):
                return
        except (httpx.HTTPError, ValueError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("n8n did not start")
        time.sleep(1)


def setup_owner(client: httpx.Client, email: str, password: str) -> None:
    body = {"email": email, "firstName": "CI", "lastName": "Eval", "password": password}
    response = client.post("/rest/owner/setup", json=body)
    if response.status_code >= 400:
        login = client.post("/rest/login", json={"emailOrLdapLoginId": email, "email": email, "password": password})
        login.raise_for_status()


def create_api_key(client: httpx.Client) -> str:
    available = client.get("/rest/api-keys/scopes")
    scopes = available.json().get("data", API_KEY_SCOPES) if available.status_code == 200 else API_KEY_SCOPES
    body: dict[str, Any] = {"label": f"n8n-eval-{int(time.time())}", "expiresAt": None, "scopes": scopes}
    response = client.post("/rest/api-keys", json=body)
    if response.status_code == 400:
        body.pop("scopes")
        response = client.post("/rest/api-keys", json=body)
    response.raise_for_status()
    data = response.json().get("data", response.json())
    key = data.get("rawApiKey") or data.get("apiKey")
    if not key:
        raise RuntimeError(f"unexpected API key response: {list(data)}")
    return key


def create_credentials(api: httpx.Client, path: str | None) -> dict[str, dict[str, str]]:
    """Returns {name: {"id": ..., "type": ...}}."""
    if not path:
        return {}
    created = {}
    for cred in expand_env(json.loads(Path(path).read_text(encoding="utf-8"))):
        response = api.post("/credentials", json={k: cred[k] for k in ("name", "type", "data")})
        if response.status_code >= 400:
            raise RuntimeError(f"credential '{cred['name']}': {response.status_code} {response.text[:300]}")
        created[cred["name"]] = {"id": str(response.json()["id"]), "type": cred["type"]}
    return created


def link_credentials(workflow: dict[str, Any], credentials: dict[str, dict[str, str]]) -> dict[str, Any]:
    for node in workflow.get("nodes", []):
        for cred_type, ref in (node.get("credentials") or {}).items():
            found = credentials.get(ref.get("name", ""))
            if found:
                node["credentials"][cred_type] = {"id": found["id"], "name": ref["name"]}
    return workflow


def existing_workflows(api: httpx.Client) -> dict[str, str]:
    found, cursor = {}, None
    while True:
        params = {"limit": 250, **({"cursor": cursor} if cursor else {})}
        data = api.get("/workflows", params=params).raise_for_status().json()
        found.update({wf["name"]: str(wf["id"]) for wf in data.get("data", [])})
        cursor = data.get("nextCursor")
        if not cursor:
            return found


def import_workflow(
    api: httpx.Client, path: str, credentials: dict[str, dict[str, str]], existing: dict[str, str]
) -> dict[str, str]:
    """Creates the workflow, or replaces the one with the same name, and activates it."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    body = link_credentials({k: raw[k] for k in WORKFLOW_FIELDS if k in raw}, credentials)
    body.setdefault("settings", {})
    workflow_id = existing.get(body["name"])
    if workflow_id:
        response = api.put(f"/workflows/{workflow_id}", json=body)
    else:
        response = api.post("/workflows", json=body)
    if response.status_code >= 400:
        raise RuntimeError(f"workflow {path}: {response.status_code} {response.text[:300]}")
    workflow_id = str(response.json()["id"])
    activated = api.post(f"/workflows/{workflow_id}/activate")
    if activated.status_code >= 400:
        raise RuntimeError(f"activate {body['name']}: {activated.status_code} {activated.text[:300]}")
    return {"id": workflow_id, "name": body["name"]}


def bootstrap(
    url: str, workflows: list[str], credentials_file: str | None, email: str, password: str
) -> dict[str, Any]:
    url = url.rstrip("/")
    with httpx.Client(base_url=url, timeout=60) as ui:
        wait_ready(ui)
        setup_owner(ui, email, password)
        api_key = create_api_key(ui)
    with httpx.Client(base_url=url + "/api/v1", headers={"X-N8N-API-KEY": api_key}, timeout=60) as api:
        credentials = create_credentials(api, credentials_file)
        existing = existing_workflows(api)
        imported = [import_workflow(api, path, credentials, existing) for path in workflows]
    return {"api_key": api_key, "credentials": list(credentials), "workflows": imported}
