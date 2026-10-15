# metask-jev-rain-12B

> **Status (2026-10-15):** This repository is our 2026-10-08 frozen-base service package (evaluated release: tag v1.1), kept for the record. Our current official JevBench result for 12B comes from the trained model, measured and ranked by the benchmark organizers (not self-reported):
>
> | Model | Trained | Leaderboard | Metric | Score | As of |
> |---|---|---|---|---|---|
> | metask-jev-rain-12B-trained | yes | JevBench open-weight leaderboard v1.6.18 | Capability Score | 77.9 (highest among ranked entries labeled 12B in the system name) | 2026-10-14 |
>
> Measured by the leaderboard operator (Benchmark Heaven) on 1,500 decision items per system (1,200 sealed + 300 public).
>
> Leaderboard: https://benchmarkheaven.com/jev-models · Model: https://huggingface.co/MeTask-Rain/metask-jev-rain-12B-trained.
>
> 说明（2026-10-15）：本仓库是 2026-10-08 的冻结底座服务旧包（评测版本为标签 v1.1），留作记录。现行 12B 官方成绩来自训练版模型（见上表，榜方实测，非自测）。榜方 Benchmark Heaven 测量，每个系统 1,500 道决策题（1,200 封存 + 300 公开）。

A self-hosted decision service for JevBench. It serves `POST /v1/systemone` for the unchanged
`typesafe` adapter of the official harness, and `GET /health`. It loads a frozen community INT4
checkpoint of Gemma 4 12B with its weights unchanged. Facts, use of public items and measurements
are in [DISCLOSURE.md](DISCLOSURE.md).

## Requirements

- One NVIDIA GPU used only by this service: at least 24 GB of memory, compute capability 8.0 or
  newer, NVIDIA driver 580 or newer. The commands use the host's GPU 0 (`--gpus '"device=0"'`).
- Docker with the NVIDIA Container Toolkit.
- Runtime image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
  (Python 3.12.3, vLLM 0.31.0, torch 2.13.0+cu130). It is the linux/amd64 image of vLLM release
  `v0.31.0` with index digest `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`,
  linux/amd64 manifest digest `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`
  (used in the commands) and config digest
  `sha256:c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3` (the classic Docker image
  store shows the config digest as the image ID; the containerd image store identifies images by
  manifest or index digest, depending on the reference used when pulling).
  Nothing is installed and the service has no network access.
- Model folder: the files of
  https://huggingface.co/cyankiwi/gemma-4-12B-it-qat-AWQ-INT4/tree/12deda07c573047b191b63f0768c0412120f3ed4
  as regular files, for example downloaded with
  `hf download cyankiwi/gemma-4-12B-it-qat-AWQ-INT4 --revision 12deda07c573047b191b63f0768c0412120f3ed4 --local-dir "$MODEL"`.
  A Hugging Face cache `snapshots/<revision>` directory holds links into `../../blobs` and cannot be
  mounted on its own. No preparation step: Gemma 4 keeps its processor settings in
  `processor_config.json`, so the folder has no `preprocessor_config.json` or
  `video_preprocessor_config.json`. At startup the service checks the SHA-256 of seven small files
  and the sizes of the three weight shards, and refuses other weight, tokenizer or processor files
  in the folder; it never writes to the folder.
- Code: this repository at the commit given in the submission email.
- Harness: https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc

## Run

Copy the commands as they are, in the same Bash terminal. First set four absolute host paths and
the task list:

```bash
PKG=/abs/path/to/metask-jev-rain-12B            # this repository, mounted read-only
MODEL=/abs/path/to/gemma-4-12B-it-qat-AWQ-INT4  # the model folder above, mounted read-only
WORK=/abs/path/to/new-work-dir                  # new for each evaluation; the only place anything is written
JEVBENCH=/abs/path/to/jevbench                  # the harness checkout above, mounted read-only
TASKS=/jevbench/datasets/public/easy.jsonl,/jevbench/datasets/public/original.jsonl,/jevbench/datasets/public/hard.jsonl  # container paths
```

`TASKS` is the comma-separated list of task files as seen inside the container: a file below
`$JEVBENCH` is below `/jevbench`, a file below `$WORK` is below `/work`. The value above is the
public set. Task paths cannot contain commas. Start with an empty `WORK`; the only files you may put
there are task files. `WORK` must be on a local filesystem that the container user can write to and
that is not mounted `noexec` (compiled kernels are loaded from `$WORK/state/cache`).

1. Start the service:

   ```bash
   docker run -d --name jev12b --init --gpus '"device=0"' --network none --read-only --shm-size 8g \
     -v "$PKG":/pkg:ro -v "$MODEL":/model:ro -v "$WORK":/work -v "$JEVBENCH":/jevbench:ro \
     --entrypoint bash vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b \
     /pkg/serve.sh --model /model --state-dir /work/state --run-id run1
   ```

2. Follow the log until the service is ready:

   ```bash
   docker logs -f jev12b
   ```

   Startup loads and compiles the model. When a line starting with `{"status": "ok"` appears, press
   Ctrl-C (this ends only the log display) and go to step 3. If the log ends before that line, the
   container has exited: the reason is in `docker logs jev12b` and in
   `$WORK/state/run1-startup/attempt-*.log`; go to step 4. After correcting the cause, use a new
   `WORK` and restart from step 1.

3. Run the official harness in the same container, then summarize:

   ```bash
   docker exec -e PYTHONPATH=/jevbench jev12b python3 -B -m jevbench.cli run --adapter typesafe \
     --endpoint http://127.0.0.1:8000 --model metask-jev-rain-12B --key-env '' --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl --raw-dir /work/run1/raw \
     --manifest /work/run1/manifest.json --cap-usd 15 --price-in-per-m 0.05 --price-out-per-m 0.15
   docker exec -e PYTHONPATH=/jevbench jev12b python3 -B -m jevbench.cli summarize --tasks "$TASKS" \
     --results /work/run1/results.jsonl --ledger /work/run1/ledger.jsonl
   ```

4. Stop and remove the container:

   ```bash
   docker stop -t 40 jev12b && docker rm jev12b
   ```

Each evaluation uses a new `WORK` and a new container (step 4 removes the previous one); the run
name stays `run1` because `WORK` is new. The slow-path quota is counted per run in arrival order, so
a run always starts from a freshly started service.

## What to expect

- Health line: `"status": "ok"`, `"warnings": []`, `"model_key": "gemma-4-12b-awq"`,
  `"mode": "routed"`, `"prefix_caching": true`, `"supports_slow_session": true`.
  `info.attention_backend` names the attention backend vLLM selected (DISCLOSURE, section 4);
  `info.runtime` lists the Python, vLLM, torch, transformers, tokenizers, triton and
  compressed-tensors versions, the CUDA version of torch and the GPU name.
- Startup: each startup attempt waits at most 1800 s for readiness. The first attempt lets vLLM
  select the attention backend; only if the engine fails to start does a second attempt request
  `TRITON_ATTN`. The log of attempt N is `$WORK/state/run1-startup/attempt-N.log`.
- Writes: the service writes files only below `$WORK/state` (plus shared memory in `/dev/shm`) and
  the harness only below `$WORK/run1`. The service creates `state/run1-startup/` (`attempts.jsonl`,
  `attempt-N.log`, quota state `state-N/`, `tmp/`, `home/`) and `state/cache/` (compilation caches).
  The package and model mounts are read-only. Cache and configuration path variables (`HF_HOME`,
  `XDG_*`, `TRITON_*`, `TORCH_HOME` and similar) must not be passed to the container: a value outside
  `/work/state` is refused at startup.
- Responses: an item the service cannot process returns HTTP 422 without answer and without usage;
  the harness scores it as wrong, and because that item has no cost, `summarize` reports the price
  per 1,000 decisions as null. After an engine failure, and from the third consecutive item failure
  on, the service answers every `POST /v1/systemone` with HTTP 503 and `/health` reports
  `"status": "error"`; the harness stops after three consecutive 503 responses. No substitute answer
  is ever returned (DISCLOSURE, section 4).
