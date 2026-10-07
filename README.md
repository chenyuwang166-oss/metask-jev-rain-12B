# metask-jev-rain-12B

## Quick start

Linux, Bash, Python 3.10 and an NVIDIA GPU with at least 24 GB (measured on RTX 4090 24 GB).
Copy the commands as they are. Steps 1–5 run in terminal 1; steps 6–7 in terminal 2.
The service serves `POST /v1/systemone` and `GET /health`, so the unchanged `typesafe`
adapter of the official JevBench harness reaches it. Serving runs fully offline.

1. Clone and check out the submitted commit, then enter the package.
   ```bash
   git clone https://github.com/chenyuwang166-oss/metask-jev-rain-12B && cd metask-jev-rain-12B && git checkout v1.0-frozen
   ```
2. Create and activate the model environment.
   ```bash
   python3.10 -m venv ../jev12b-venv && source ../jev12b-venv/bin/activate
   ```
3. Install the runtime.
   ```bash
   pip install -r requirements.txt
   ```
4. Download the pinned community INT4 weights.
   ```bash
   bash prepare_model.sh ../jev12b-model
   ```
5. Start the service (foreground; wait for the printed health result).
   ```bash
   bash serve.sh --model ../jev12b-model
   ```
6. Check readiness: `status: "ok"` and empty `warnings`.
   ```bash
   curl --fail http://localhost:8000/health
   ```
7. Run the official TypeSafe adapter; use a fresh `RUN_DIR` for each evaluation.
   ```bash
   "$JEVBENCH_PYTHON" -m jevbench.cli run --adapter typesafe --endpoint http://localhost:8000 --model metask-jev-rain-12B --key-env '' --price-in-per-m 0.05 --price-out-per-m 0.15 --tasks "$TASKS_JSONL" --results "${RUN_DIR:?Set a fresh results directory}/results.jsonl" --ledger "$RUN_DIR/ledger.jsonl" --raw-dir "$RUN_DIR/raw" --manifest "$RUN_DIR/manifest.json" --cap-usd 15
   ```

## Installation and model preparation

`requirements.txt` pins only the main runtime (vLLM 0.31.0 and torch
2.13.0+cu130) and includes Hugging Face Hub for preparation. It uses the PyTorch
cu130 index plus PyPI; pip resolves transitive dependencies. The supplied
201-entry `requirements.lock` remains an optional exact-version reproduction
path: `pip install -r requirements.lock`. It is not a wheel-hash lock.
`install.sh` is an optional venv helper using the minimal requirements.
The launcher checks only vLLM major 0 and torch major 2, not the full lock;
Python 3.10 remains required. Use the pinned requirements for the submitted run.

`prepare_model.sh` downloads only the checkpoint repository
`cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` at
`12deda07c573047b191b63f0768c0412120f3ed4`, verifies INT4 metadata and records
weight hashes in the external model directory. No separate processor command
is needed. At startup, missing processor configs are copied from
`google/gemma-4-12b-it` only when present there. Existing configs are preserved;
the resolved base revision, copied hashes and absent filenames are recorded in
`PREPARATION.json`. First startup may need network access and authorized model
access. Subsequent startup reuses this record; the model child runs offline.

## Runtime and health

`--state-dir` defaults to `./state` relative to the package directory;
`--run-id` defaults to a UTC timestamp with microseconds. Both remain optional
explicit overrides. Restart the foreground service for each independent run.
Ctrl-C stops the service and its engine process group.

`generation_reference.intended_overrides` declares the frozen greedy settings.
Only differences matching the declared values while greedy decoding is active
are informational: `/health.info.intended_overrides` records checkpoint values,
configured values and reasons. Unexpected differences and unreadable/missing
metadata still warn. An unknown actual attention backend also warns.

Actual backend names come first from loaded worker attention layers, then from
engine logs (including readable names such as Flash Attention). Requested backend
settings are reported separately. No request value is treated as proof of the
active implementation. Startup tries auto, then FLASH_ATTN, then TRITON_ATTN on
engine initialization failure. Attempt logs live under
`<state-dir>/<run-id>-startup/`; each attempt has isolated quota state.
The FlashInfer sampler and telemetry are disabled.

## Frozen behavior and evidence

Frozen `google/gemma-4-12b-it`, community INT4 weights, single-forward readout
with a short draft for the low-confidence 15% routing budget. Threshold
2.9999032097890423; quota 0.15; fuse 0.20; draft limit 300 words / 512 tokens;
long inputs compressed to 8192 and fast-only; actual engine usage accounting.
Public-data calibration is not held-out validation.

Supplied measured result: 231/231 tasks, accuracy 0.879, USD 0.047 per 1k,
p50 0.055 s, RTX 4090. Price uses the declared reference assumption of
USD 0.05 / 0.15 per million input/output tokens, not hardware running costs.
This packaging revision was checked offline; it has not rerun the model or
verified a fresh dependency installation. See DISCLOSURE.md for evidence fields,
harness.md for evaluator details, and environment.md for dependency provenance.

## Optional integrity check

```bash
sha256sum -c MANIFEST.sha256
```

The LF manifest covers package files except itself; clone metadata, Python caches
and default runtime state are excluded. Keep environments, weights and evaluation
outputs outside the package. Maintainers must fill the package commit and UTC
freeze time before submission, then regenerate the manifest. FORM_NOTES.md is
the concise form draft. PROVENANCE.json retains original hashes and records the
delivered runtime hashes.
