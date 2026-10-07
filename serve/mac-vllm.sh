#!/usr/bin/env bash
# Serve the worker model on an Apple Silicon Mac with vllm-metal (vLLM on MLX).
# Install once:
#   brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
#   brew install vllm-project/vllm-metal/vllm-metal
#
# The model name the server reports is VLLM_MODEL, so the harness needs the same value.
# --gpu-memory-utilization caps the KV cache as a share of the Mac's unified memory.
# vllm-metal's default lets it grow to most of the machine; 0.3 is plenty for a 4B model.
set -euo pipefail

MODEL="${VLLM_MODEL:-mlx-community/Qwen3-4B-Instruct-2507-8bit}"

exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT:-8000}" \
  --max-model-len "${VLLM_CONTEXT:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.3}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${VLLM_TOOL_PARSER:-hermes}"
