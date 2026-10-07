#!/usr/bin/env bash
# Serve the small worker model with vLLM's OpenAI-compatible API.
# Needs a CUDA GPU. The harness finds it through VLLM_BASE_URL (default http://127.0.0.1:8000/v1).
#
#   VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507 serve/vllm.sh
#
# Tool calling must be on, or the worker cannot use read/write/edit/bash.
# `hermes` is the parser for Qwen-family models; other families need their own.
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

exec vllm serve "$MODEL" \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8000}" \
  --max-model-len "${VLLM_CONTEXT:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${VLLM_TOOL_PARSER:-hermes}" \
  --enable-prefix-caching
