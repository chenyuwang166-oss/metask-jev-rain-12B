#!/usr/bin/env bash
# Start the service in the foreground; it prints one health JSON line when ready.
#   bash serve.sh --model /abs/model --state-dir /abs/state --run-id NAME [--port 8000] [--bind 127.0.0.1]
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
exec "$PY" -B "$PKG/launch.py" "$@"
