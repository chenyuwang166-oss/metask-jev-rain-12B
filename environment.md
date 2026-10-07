# Measured environment and installation contract

Measured on Ubuntu 22.04.5, Python 3.10.12, NVIDIA driver 580.105.08,
RTX 4090 24 GB. Runtime Python is restricted to 3.10 (including installation).
vLLM 0.31.0; torch 2.13.0+cu130; transformers 5.17.0;
huggingface_hub 1.33.0; tokenizers 0.23.2; safetensors 0.8.0;
numpy 2.2.6; fastapi 0.136.3; uvicorn 0.54.0; requests 2.34.2.

`requirements.measured.lock` is the supplied 201-entry freeze. `resolve_env.py`
reproduces `requirements.lock` and `environment.resolved.json` offline from it.
All 201 entries are retained; the only override is torch 2.13.0 to the explicitly
supplied measured build 2.13.0+cu130. No packages were guessed to be uninstallable.
`install.sh` installs the minimal requirements.txt and runs `pip check`.
The 201-entry lock is optional for exact-version reproduction. The public CUDA 13.0 wheel
index is included for the measured torch build.

Lock SHA256: `32f0414b901856effaa2a6ff466cc9363c0ae70a7b6d5c6260a6433e184c3cdc`.
This is a version lock, not a wheel-hash lock: the supplied freeze has no artifact
hashes. Do not claim `--require-hashes` verification or a successful clean install.
The full dependency set, wheel availability and clean installation on Linux/Python
3.10 remain unverified in this task. Installation or `pip check` failure is fatal.
OS, driver and the model/processor files are outside pip's scope. The torch CUDA
build does not establish a separately installed system CUDA toolkit version.

Serving and selftest keep VLLM_USE_FLASHINFER_SAMPLER=0. The supervisor tries
auto attention selection, then FLASH_ATTN, then TRITON_ATTN on initialization
failure. Supplied historical measurements used FLASH_ATTN; the revised cold run
must record the actual backend reported by /health. Both disable
usage telemetry and enable HF/transformers offline mode. Run preparation before
serving, with the installed environment's Python and network access.

Weights are fixed to cyankiwi/gemma-4-12B-it-qat-AWQ-INT4, compressed-tensors
INT4 auto-detected by vLLM. There is no mandatory four-file processor supplement.
At startup, model_setup.py copies only missing processor configs that exist
in google/gemma-4-12b-it and records its resolved commit and file hashes.
prepare_model.sh only downloads and verifies the pinned checkpoint repository.

The dependency lock is inherited unchanged from the frozen 4B template; this task
has not independently verified a clean 12B installation. Measured hardware is
RTX 4090 24 GB. Package vllm_options remain gpu_memory_utilization=0.85 and
max_num_seqs=1. Effective max_model_len is 18432, physical input guard 16384 and
soft input budget 8192. Long inputs use only the fast path. Cold-run measurements
and peak memory remain named fields in DISCLOSURE.md; do not infer them from the
4B template. launch.py checks only vLLM major 0 and torch major 2, not the full lock.
