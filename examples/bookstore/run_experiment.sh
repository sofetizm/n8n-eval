#!/usr/bin/env bash
# Three versions of the support agent on the same 32 cases × 3 repeats.
# Needs a running n8n with the workflows imported (n8n-eval up) and LLM_API_KEY.
set -euo pipefail
cd "$(dirname "$0")"
N8N=${N8N_URL:-http://localhost:5678}

n8n-eval run suite.yaml --label gpt-oss-120b --out results/gpt-oss-120b.jsonl
n8n-eval run suite.yaml --label gpt-oss-20b --out results/gpt-oss-20b.jsonl \
  --workflow bookstore-support-20b --webhook "$N8N/webhook/bookstore-support-20b"
n8n-eval run suite.yaml --label short-prompt --out results/short-prompt.jsonl \
  --workflow bookstore-support-short-prompt --webhook "$N8N/webhook/bookstore-support-short-prompt"

python ../../scripts/experiment_report.py results
