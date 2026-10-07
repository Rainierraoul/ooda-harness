#!/usr/bin/env bash
# Serve the worker model on an Apple Silicon Mac with vllm-metal (vLLM on MLX).
# Install once:
#   brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
#   brew install vllm-project/vllm-metal/vllm-metal
#
# The model name the server reports is VLLM_MODEL, so the harness needs the same value.
# --gpu-memory-utilization caps the KV cache as a share of the Mac's unified memory.
# vllm-metal's default lets it grow to most of the machine; 0.3 is plenty for a 4B model.
#
# Optional:
#   VLLM_REVISION          pin the weights to a Hugging Face commit, for reproducible runs
#   VLLM_REASONING_PARSER  separate a thinking model's reasoning from its answer (e.g. qwen3)
set -euo pipefail

extra=()
if [ -n "${VLLM_REVISION:-}" ]; then extra+=(--revision "$VLLM_REVISION"); fi
if [ -n "${VLLM_REASONING_PARSER:-}" ]; then extra+=(--reasoning-parser "$VLLM_REASONING_PARSER"); fi

MODEL="${VLLM_MODEL:-mlx-community/Qwen3-4B-Instruct-2507-8bit}"

exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "${VLLM_PORT:-8000}" \
  --max-model-len "${VLLM_CONTEXT:-32768}" \
  --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.3}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${VLLM_TOOL_PARSER:-hermes}" \
  ${extra[@]+"${extra[@]}"}
