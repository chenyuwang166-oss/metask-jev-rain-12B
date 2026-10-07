#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# Serving is fully offline (evaluators run it without network); only prepare_model.sh downloads.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python -B launch.py "$@"
