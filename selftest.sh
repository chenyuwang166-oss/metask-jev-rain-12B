#!/usr/bin/env bash
# Maintainer tool: public-data selftest (see MAINTAINER.md).
#   bash selftest.sh --model /abs/model --public /abs/jevbench/datasets/public --output-dir /abs/new/dir
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python3}"
exec "$PY" -B "$PKG/selftest.py" "$@"
