# metask-jev-rain-12B: disclosure

Facts about the system, the weights, the use of public JevBench items and the measurements. How to
run the service: [README.md](README.md). Dates without a time of day are calendar dates in UTC+8;
times of runs are given in UTC.

## 1. Identity

| Item | Value |
|---|---|
| System name | `metask-jev-rain-12B` (team MeTask) |
| Code | this repository at the commit given in the submission email; MIT license ([LICENSE](LICENSE)) |
| Weights loaded | `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` at https://huggingface.co/cyankiwi/gemma-4-12B-it-qat-AWQ-INT4/tree/12deda07c573047b191b63f0768c0412120f3ed4 (compressed-tensors W4A16, detected by vLLM from the checkpoint) |
| Base chain | `google/gemma-4-12B-it` → `google/gemma-4-12B-it-qat-q4_0-unquantized` (Google's quantization-aware-training release, named as `base_model` in the cyankiwi model card) → `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` (the only weights loaded) |
| Weights license | Apache-2.0, as declared by the cyankiwi model card (`license: apache-2.0`, license link https://ai.google.dev/gemma/docs/gemma_4_license) |
| Training by us | none for this system: no fine-tuning, LoRA, adapter or other parameter update of the weights it loads; `model.adapter` is `null` |
| Internal model key | `gemma-4-12b-awq`: the key of the weights profile and the calibration in `config.json` and `calibration.json`; it is not the system name. The profile label `awq` selects the quantized profile; vLLM loads the checkpoint with its compressed-tensors kernel |

### Training data

This system loads the third-party weights above unchanged; we added no data of our own to them.
Google documents the training data of the Gemma models. The cyankiwi model card names the
calibration data of its 4-bit quantization "STEM and Agentic" (Hugging Face dataset
`cyankiwi/calibration`). We cannot verify whether any of these third-party data contain JevBench
items.

### Model files checked at startup

The service reads the model folder without writing to it and refuses to start if any of these
files is missing, is a symbolic link or differs (`launch.py`, `MODEL_SHA256` and `SHARD_BYTES`), or
if the folder holds any other file a loader may read (weights, tokenizer, processor or template
files). The model card, `.gitattributes` and the Hugging Face download cache directory are ignored.

| File | Bytes | SHA-256 |
|---|---|---|
| `model-00001-of-00003.safetensors` | 5,368,362,132 | `093872a6b9b1c316c66241f1d8fadf7b6d16f7368914e34fbec0e5343fc10a78` |
| `model-00002-of-00003.safetensors` | 5,357,673,316 | `c43491e1a10ca18f90c7ffd69338124cdbfa47315a1da3d4e49bd0180f73de11` |
| `model-00003-of-00003.safetensors` | 496,095,448 | `a93a9a3917fd4c2aad6108b8b0f02c0acd074b9f306039d09447b81fcc092254` |
| `config.json` | 5,496 | `53b52e5eeb829992dedcb85cb465efd31805da77245b35167b9831ff393572e3` |
| `generation_config.json` | 260 | `a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e` |
| `processor_config.json` | 1,382 | `6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9` |
| `tokenizer_config.json` | 2,102 | `09e6222fd7049dae603d0f61a8b10af5a813f22dca10e6cdc6918f76d7661104` |
| `tokenizer.json` | 32,169,626 | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` |
| `chat_template.jinja` | 17,466 | `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0` |
| `model.safetensors.index.json` | 170,749 | `7c48224f0d8b11a4eb3d0be6bac8fab7069223ca3ac96a259c2b649945c88332` |

At startup the shards are checked by size; the maintainer download helper `prepare_model.sh`
checks their full SHA-256. All values are those of the repository at that commit: the shard hashes
and the `tokenizer.json` hash equal its Git LFS object IDs, and the other six hashes were computed
on the files downloaded from that commit.

## 2. Relation to metask-jev-rain

This package replaces the system `metask-jev-rain`, which was submitted earlier and returned after
the operator's source review, before evaluation. That system's code is
https://github.com/chenyuwang166-oss/metask-jev-rain/tree/af3169c2eb8f0af2bb83db9a2bd077807dc16e9b
and its disclosure is the `DISCLOSURE.md` file at that commit. Both systems come from the same team
and both use Gemma 4 12B instruction-tuned weights without training. The fast path uses the same
fixed instruction folded into the single user turn and the same rendering of context, question and
options (this package adds two-letter option codes for more than 26 options); the options are
listed in a different order (table below).

| | metask-jev-rain (commit above) | metask-jev-rain-12B (this package) |
|---|---|---|
| Weights | `google/gemma-4-12B-it` at revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; bf16 when at least 27 GiB of GPU memory is free, otherwise bitsandbytes int8 (its measurements were int8 on a 24 GB card) | `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` (section 1) |
| Integration | in-process harness adapter `metask_rain`, added to the harness by a patch; transformers | HTTP service for the unchanged `typesafe` adapter; vLLM 0.31.0 |
| Per decision | one forward pass, no generation | one forward pass; for low-margin items within a 15% quota also a short draft (section 4) |
| Calibration | per precision, fitted on a 116-item half of the public items | fitted on all 231 public items (section 3) |
| Reported usage | input tokens only, output 0; cost basis `self_hosted_gpu`, no price | engine-basis input and output tokens; assumed price (section 5) |
| Input length | up to 65,536 tokens, HTTP 422 above | 8,192-token budget with compression; longer items take the fast path only (section 4) |
| Self-test and file operations | `selftest.sh` refreshes its two harness copies with `rm -rf` before each self-test | no harness copy and no recursive deletion; the service creates files only below `--state-dir`, which must be an absolute path and is resolved (symbolic links and `..` removed) before it is checked to lie outside the package and the model folder; the only deletion is `quota.py` removing its own temporary file after a failed write, inside that directory (it also replaces its quota state file atomically); `pkg_check.py` flags the common deleting and replacing calls in the package code and allows only these two and the freezing tool's replacement of a package file (list in `MAINTAINER.md`) |
| Self-test on the public items | 197/231 (its disclosure, item E6) | section 7 |
| Surface forms read on the fast path | the single-token forms among its templates: yes/no `yes`, ` yes` against `no`, ` no` (a subset of the full sets of 12 forms each, chosen by a pre-registered rule because on all 74 public yes/no items it gives the same decisions as the full sets); choice `A`, ` A`, `a`, ` a` for each option letter; score the digit | the single-token forms among the same templates, with the full yes/no sets of 12 forms each (`yes`, `Yes`, `YES`, ` yes`, ` Yes`, ` YES`, `true`, `True`, ` true`, ` True`, `Y`, ` Y` and the matching no-forms); choice templates `A`, ` A`, `a`, ` a`, `A.`, `(A`; score templates `0`, ` 0`, `0.` |
| A label without a single-token form | log-mass −1e9, so probability 0; a choice item with more than 26 options is refused with an error | every label of that item is scored by the log-probability of its forms as token sequences; up to 676 options, with two-letter codes above 26 |
| Option order in the prompt | the order of the item's `labels` (in-process adapter) | the order of the request's `criteria`: the `typesafe` adapter sends `criteria` and no `labels`. Of the 139 public choice items, 119 list their options in a different order than `labels`, and in 101 of them the correct option has a different letter |

Why the system is replaced: the operator's source review of metask-jev-rain asked us to canonicalize
and validate the paths that its self-test deletes with `rm -rf`. We did not change that package. This
package replaces it: it runs as an HTTP service for the unchanged `typesafe` adapter, keeps no harness
copy and performs no recursive deletion (table above).

## 3. Use of public JevBench items

**Data.** The only JevBench data we used are the public items of
https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc,
`datasets/public/`:

| File | Items | SHA-256 of the file at that commit (as in its `datasets/manifest.json`) |
|---|---|---|
| `easy.jsonl` | 48 | `231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b` |
| `original.jsonl` | 72 | `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180` |
| `hard.jsonl` | 111 | `89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb` |

These are the files as committed, with LF line endings. Our earlier runs read copies of the same
files with CRLF line endings (SHA-256 `314cd493a4c080fca5ad90e1c8a7c2b3f24137e66dc06f8fceaf51ea54f38467`
easy, `b2abe9ac8cabd953a3bc452abd4b229da42ca6d0e647bf93f7638076f42db19f` original,
`8c8f1efb5a04b7016f9c8cd8a70a485f58a70a918aac3e3c1107f41172ca91b6` hard); parsed line by line as
JSON, all 231 items are identical to the committed files.

We never had access to the sealed set. We used no ImageJevBench item at any point. On 2026-10-04 we
also ran a separate QLoRA fine-tuning experiment on `google/gemma-4-12B-it` that used the public
items for monitoring; none of its weights is used by this system.

**What was fitted or chosen on these 231 items.** No model weights were trained on them. Labels
were used for the calibration and for every reported accuracy.

1. Routing threshold `2.9999032097890423` (log-ratio margin between the two best labels on the
   fast path): the midpoint between the 35th and 36th smallest fast-path margin of a fast-only run
   over the 231 items, so that 35/231 = 15.15% of these items fall below it. The threshold uses the
   margins only, not the labels.
2. Calibration of the fast, slow and fallback paths: section 6. The fast block comes from the
   fast-only run of item 1 (2026-10-06): its yes/no accuracy, 59/74, is the fitted fast `p_cal`
   0.7973. The slow and fallback blocks come from a routed run of the same service on the same 231
   items on 2026-10-06 at slow-path quota 0.10 (its threshold `2.2081196767983715`): 22 items took
   the slow path, 15 with a parsed draft (slow block: 6 yes/no, 8 choice, 1 score) and 7 without one
   (fallback block: 2, 3, 2). This package runs at quota 0.15, so the slow block was fitted on fewer
   items, all with margins below 2.21, than this configuration sends to the slow path. Both runs used
   the maintainer benchmark tool `bench.py` (not the official harness), which sends the options in
   the harness's `criteria` order.
3. Slow-path quota 0.15 and slow-share ceiling 0.20 (section 4). Quota levels compared on these
   items, with labels: (a) an offline simulation on runs of the Gemma 4 12B weights in 8-bit
   precision that drafted every item with 300 words, sending the lowest-margin 5%, 10%, 15%, 20% or
   25% of the items to the draft: +3, +8, +10, +10 and +12 items correct against the single pass;
   (b) runs of this INT4 service, each with its own refitted threshold: quota 0.20 with the official
   harness (205/231, USD 0.05144 per 1,000 decisions, engine basis), quota 0.10 with the official
   harness (200/231, USD 0.04358) and quota 0.15 (203/231, USD 0.0474). The rule: the largest of these
   quotas whose projected cost, with 23 long items compressed to 8,192 tokens among 1,624
   decisions, stays below 85% of the USD 0.0646 cap. Quota 0.20 projected about 89% of the cap and
   0.15 about 81% to 83% (two projections). Accuracy rose with the quota; the cost rule, not
   accuracy, set 0.15.
4. Draft instruction of at most 300 words, with a hard cap of 512 generated tokens. Draft budgets
   compared on these items, with labels, in offline runs of the Gemma 4 12B weights in 8-bit
   precision that drafted every item (an item without a parsed draft keeps the single-pass answer):
   40 words / 64 tokens 0.848, 80 / 128 0.840, 150 / 256 0.861 and 300 / 512 0.900 accuracy, against
   0.853 for the single pass. The shorter budgets were often cut off before the answer line (73%, 74%
   and 84% of drafts parsed; 92% at 300 / 512). We chose 300 words / 512 tokens, the longest budget
   tried and the one with the largest gain (+11 items); no longer budget was tried.
5. Weights: the community INT4 checkpoint instead of int8 or bf16 of the original weights, and 12B
   instead of a 4B model. For metask-jev-rain (2026-10-02) we scored frozen Qwen3.5-4B, Qwen3.5-9B
   and Gemma 4 12B instruction-tuned weights on these items with the same single-pass readout and
   chose Gemma 4 12B (our local estimate of the Intelligence score: 52.8 and 55.3 for the two Qwen
   models; 70.0 for Gemma 4 12B at 8 bits and 58.2 at 4-bit NF4). On 2026-10-03 the bf16 and 8-bit
   versions of Gemma 4 12B answered 195 and 197 items correctly and agreed on 225 of 231 decisions.
   For this package we chose 4-bit weights (W4A16) on 2026-10-05, before measuring them, because vLLM
   could not run the bf16 12B weights on our 24 GB card and we wanted to measure locally the
   precision we submit. Measured afterwards with the single pass, the checkpoint of section 1
   answered 192/231, against 197/231 at 8 bits; we kept it. metask-jev-rain-12B and
   metask-jev-rain-4B are submitted as separate entries, so neither was chosen over the other.
6. Prompt: the fast-path prompt is the one of metask-jev-rain, whose disclosure (item B3) states it
   was written once on 2026-10-02 and not iterated against Gemma public-item results. The slow-path
   draft instruction (`prompt.cot_instructions` in `config.json`) was written on 2026-10-05 for a
   pre-registered study on synthetic diagnostic items, before any draft had been run on the public
   items: "First reason step by step in at most N words, tracking any intermediate state
   explicitly. Then output a final line ..."; the yes/no, choice and score versions differ only in
   the answer line. On the public items we varied only the word budget N (item 4).

All accuracies, calibration figures and costs on these items are in-sample. There is no held-out
validation.

**Constants from public JevBench sources.** The JevBench v1.6.0 leaderboard page
https://benchmarkheaven.com/jev-models (retrieved 2026-10-05) states the caps of the Jev class as
USD 0.0646 per 1,000 decisions and 1.23 s adjusted median latency, the adjusted latency of
self-hosted systems as 2 × raw + 0.15 s, and that 23 items of the v1.6 draw are very long
(about 77,000 to 81,000 input tokens). In `config.json`, `limits.cost_per_1000_decisions_usd` =
0.0646 and `limits.median_latency_seconds` = 1.23 are these caps; `bench.long_tail` uses 23 items of
79,000 prompt tokens (our midpoint of that range) and 1,624 decisions in total, the v1.5 decision
count stated in
https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/METHOD-v1.5.md
(the v1.6 self-hosted run has 1,500 items); `limits.raw_median_latency_seconds` = 0.54 is the raw
latency (1.23 − 0.15) / 2 that corresponds to the cap, a local threshold, not a measurement. The
service uses only the 0.0646 cap (the cost fuse, section 4); the other values are used only by the
maintainer benchmark tool `bench.py`.

## 4. Serving behaviour

**Fast path (every decision).** The item is rendered with a fixed instruction and one forward pass
gives the next-token log-probabilities. Each label's score is the log-sum-exp over those of its
surface forms that are single tokens (for example `yes`, ` Yes`; option letters; digits). If some
label has no single-token form, every label is scored by the log-probability of its forms as token
sequences.

**Slow path.** If the margin between the two best labels is below the threshold, the item is not a
long item (see Long items below) and the quota admits it, the model generates a draft (greedy
decoding, seed 0) under an instruction to keep it within 300 words, hard-capped at 512 generated
tokens, ending with `ANSWER: <label>`; the label scores are then read after the answer marker. If
the draft has no answer line, the fast-path scores are kept (calibration block "fallback"). If the
draft fails or exceeds 60 s, the fast-path scores are also kept, the answer carries a `warning`
field and the response a top-level `partial_errors` object, still with HTTP 200. When the engine's
token count for such a draft is unknown (timeout or failed engine call), the response is marked
`usage_estimated` and, on every usage basis, the input count is raised to at least the submitted
tokens plus 512 and the output count to at least the generated tokens plus 512; the cost fuse
counts these values. An unrecoverable engine error during a draft is a service failure (below),
never a fast-path answer.

**Quota and order.** The slow-path quota is a budget over the decisions this service process has
answered: a decision may take the slow path only if `slow + 1 <= q × (decisions + 1)`, where `q` is
0.15 or the lower value set by the cost fuse. A second ceiling, the slow-share ceiling
`slow + 1 <= 0.20 × (decisions + 1)`, is checked before the quota; while `q` is at most 0.15 it never
changes a decision, but for an item that would take the slow path among the first four decisions of
a run it is the check that refuses it, so such an item shows `route_reason` `fuse` (this is not the
cost fuse below). Consequently the first six decisions of a run always take the fast path, and the
same item can take a different path at a different position in the order. The values of
`route_reason` are `router` (margin at or above the threshold), `eligible` (slow path taken),
`quota`, `fuse` (the slow-share ceiling), `length` (long item), `oom_retry` (fast path repeated
with a shorter prompt after running out of GPU memory) and `memory_only` (below). A slow attempt is
counted when it starts; a decision is counted when its item completes, so a failed item does not
advance the decision count (one exception under Failures). Quota state is kept per run below
`--state-dir`; every evaluation starts a new service with a new state directory. If the quota state
cannot be read or written at startup, the service does not start. If it cannot be read or written
during a run, the service continues with the fast path only for the rest of that run, still
answering with HTTP 200: each such decision has `route_reason` `memory_only` (or `length` or
`oom_retry` where those apply) and `/health` reports a warning.

**Cost fuse.** Each completed decision's cost is computed from its engine-basis usage (section 5)
and the assumed prices. After 100 decisions, if the running average reaches 85% of 0.0646 USD per
1,000 decisions, the fuse lowers the slow-path quota so that the expected average returns to 80% of
the cap; it releases when the running average falls to 80%. The fast path is never changed. In the
image run on the public items (section 7) the fuse made [[FACTS-PENDING: S6-FINAL cost-fuse transitions, usage.cost_fuse.transitions of the last line of results.jsonl]]
transitions (running cost at the end USD [[FACTS-PENDING: S6-FINAL usage.cost_fuse.running_cost_per_1000 of the last line of results.jsonl]]
per 1,000 decisions). In an earlier official-harness run of this service at slow-path quota 0.20
(2026-10-06) it made 0 transitions.

**Long items.** Prompts above 8,192 tokens are compressed to that budget (rule and question text
are kept, long narrative state is shortened around query terms). Items whose text exceeds 65,536
characters (8 characters per budget token) are first shortened by characters. Such items, and items
whose slow-path prompt would exceed 8,192 tokens, always take the fast path. A prompt still above
16,384 tokens after compression is cut to its first 9,830 and last 6,554 tokens.

**Failures.** No substitute answer (for example a uniform distribution) is ever returned.

- A malformed request returns HTTP 400 `{"error": "Invalid systemone request"}`; a well-formed
  request outside the supported shapes returns HTTP 422 `{"error": "Unsupported systemone request"}`.
- An item the service cannot process, including a response that would contain a non-finite number,
  returns HTTP 422 `{"error": "item could not be processed"}`. That response contains no answer and
  no usage. The official harness scores the item as wrong and does not count a 422 toward its stop
  rule. Because the item has no cost, `jevbench summarize` at the pinned harness commit reports the
  run's `price_per_1000_decisions_usd` as null; the harness ledger keeps the amount it reserved for
  that item, which is not a measured price. A failed item does not advance the decision count or the
  cost fuse, except an item refused for a non-finite value: that check runs after the item's
  decision and cost are recorded.
- The service is marked failed by the third consecutive item failure or by an unrecoverable engine
  error: vLLM's `EngineDeadError`, or another vLLM server-side error except the per-request
  `EngineGenerateError`, anywhere in the exception's cause chain, or the engine's own dead flag. That
  request returns HTTP 503 `{"error": "service failure"}`; every later POST returns 503 without
  inference, and `GET /health` reports `"status": "error"` with the warning `service failure`. Any
  other unexpected error while answering also returns 503 `{"error": "service failure"}`, without
  marking the service failed. A successfully answered item resets the count of consecutive item
  failures. The harness stops a run after three consecutive 503 responses and leaves the remaining
  items unattempted.
- The official `typesafe` adapter sends one question per request; in a request with several
  questions, one failed item fails the whole request.

**Attention backend.** vLLM chooses the attention backend from the GPU and the model; the service
does not set it on its first startup attempt (auto). This checkpoint mixes attention head sizes 256
(sliding-window layers) and 512 (full-attention layers). For such a model vLLM 0.31 selects a
backend based on FlashAttention 4 or FlashInfer where FlashAttention 4 is available (compute
capability 9.x, 10.x or 11.x, for example H100 or B200) and `TRITON_ATTN` everywhere else; on the
RTX 4090 (compute capability 8.9) used for every measurement it therefore selects `TRITON_ATTN`. The
engine log of an earlier run of this service on that GPU (2026-10-06) reads
`Using AttentionBackendEnum.TRITON_ATTN backend.` vLLM also chooses the kernels for the 4-bit
weights by GPU. On other GPU architectures these kernels can differ, so logits, and with them
routing and answers near a decision boundary, can differ slightly; the measurements cover only the
RTX 4090. If the engine fails to start, the service makes one more attempt with `TRITON_ATTN`
requested explicitly, the backend of the measured GPU. `/health` reports the requested and the
observed backend (`info.attention_backend_requested`, `info.attention_backend`); section 7 gives
the observed backend of the image run.

**Interfaces and files.** `POST /v1/systemone` and `GET /health`. `GET /engine_stats` returns
read-only engine counters used for the maintainer usage reconciliation; the evaluation does not use
it. The service listens on 127.0.0.1 (`--bind 0.0.0.0` is accepted for a harness in another
container; the documented run does not use it). Runtime network access: none (`--network none`;
Hugging Face offline mode; usage statistics disabled). The service reads the package and the model
folder and writes files only below `--state-dir` (plus shared memory in `/dev/shm`); it never opens
the harness checkout, the task files or the harness output.

## 5. Usage and price

`usage.input_tokens` and `usage.output_tokens` follow the engine basis (`config.json`
`usage.basis = "engine"`):

- input: prompt tokens the engine actually computed, that is, all tokens submitted minus vLLM
  prefix-cache hits, including hits on prefixes cached by earlier requests;
- output: tokens the engine generated, that is, one readout token per readout call plus the draft
  tokens.

Each response also carries `usage.alternatives`: `submitted` (all submitted tokens, no cache
deduction) and `each_once` (each prompt counted once), plus `usage.input_tokens_details` with the
cached and submitted token counts. A 422 or 503 response carries no usage. In the image run on the
public items (section 7), USD per 1,000 decisions at the prices below: engine
[[FACTS-PENDING: S6-FINAL engine basis, equal to the summarize price]], submitted
[[FACTS-PENDING: S6-FINAL submitted basis]] and each_once [[FACTS-PENDING: S6-FINAL each_once basis]]
(input and output token totals: [[FACTS-PENDING: S6-FINAL sums of usage.alternatives engine, submitted and each_once input_tokens and output_tokens over results.jsonl]]).
In an earlier official-harness run of this service at slow-path quota 0.20 (2026-10-06) the three
values were 0.05144 (engine), 0.10010 (submitted) and 0.05954 (each_once).

Prices are an assumption for local budgeting: USD 0.05 per million input tokens and 0.15 per
million output tokens, the OpenRouter prices of `google/gemma-3-12b-it` as the nearest listed size
class (OpenRouter has no Gemma 4 12B listing). The pinned harness documentation uses the same
reference for a hosted 12B Gemma
(https://github.com/fstandhartinger/jevbench/blob/bb05a335bc809e61b20c0f745d25499a82b326fc/docs/v1.2-additions-winnow.md).
They are not a Gemma 4 tariff or a hardware cost; the official price follows the operator's rules.
Source: the OpenRouter model list https://openrouter.ai/api/v1/models, retrieved 2026-10-08: prompt
0.00000005 and completion 0.00000015 USD per token for `google/gemma-3-12b-it`, and no
`google/gemma-4-12b-it` entry (the only Gemma 4 models listed are `google/gemma-4-26b-a4b-it` and
`google/gemma-4-31b-it`). We first took these prices from the same list on 2026-10-05.

## 6. Calibration

Choice and score: probabilities are `softmax(score / T)` with a floor of 1e-6 and renormalisation.
Yes/no: the winning label gets `p_cal` and the other `1 - p_cal`. Values are stored at full
precision; nothing is rounded.

Fitting (`bench.py`, `calibrate_temperature`): for choice and score, `T` is the value on the grid
`2^(k/8)`, k = −32 … 40 (0.0625 to 32), that minimises the mean negative log-likelihood of the gold
label under `softmax(raw score / T)`; for yes/no, `p_cal` is the share of items whose top raw label
is the gold label, kept within [0.5, 1 − 1e-6].

| Block (source) | Type | Items | Fitted value | Value used | Adjustment |
|---|---|---|---|---|---|
| fast (fast-only run) | yes/no | 74 | p_cal 0.7972972972972973 | 0.81 | raised to the floor 0.81 |
| fast | choice | 139 | T 2.5936791093020193 | 2.5936791093020193 | none |
| fast | score | 18 | T 2.5936791093020193 | 2.5936791093020193 | none |
| slow (routed run) | yes/no | 6 | p_cal 0.999999 | 0.9 | capped at 0.90 (6 items) |
| slow | choice | 8 | T 4.362030930661031 | 4.362030930661031 | none |
| slow | score | 1 | T 0.0625 | 2.5936791093020193 | one item: fast score temperature used |
| fallback (slow-path items answered from the fast scores) | yes/no | 2 | p_cal 0.5 | 0.81 | fast block used |
| fallback | choice | 3 | T 5.187358218604039 | 2.5936791093020193 | fast block used |
| fallback | score | 2 | T 32.0 | 2.5936791093020193 | fast block used |

The fitted inputs are in `fitted.release.json`; `pkg_check.py --release` checks that
`calibration.json` matches them. The status strings `candidate; held-out validation required` in
`calibration.json` are the fitting tool's labels for in-sample fits; the service does not read them.

The yes/no calibration reports every yes/no answer as `p_cal` or `1 - p_cal`. JevBench METHOD v1.5
(section 3.1, linked in section 3) scores a yes/no probability between 0.20 and 0.80 as an
abstention, counted wrong; the fitted fast value 0.7973 would put every fast-path yes/no answer in
that band, so we raised it to 0.81, just outside it. The slow-path yes/no fit is 0.999999 because
all 6 slow-path yes/no items of the fitting run were answered correctly; at that value any later
error on the slow path would be scored as an almost certain wrong answer, so we capped it at 0.90,
a value set by hand. Neither 0.81 nor 0.90 was fitted.

## 7. Results

All results are on the 231 public items (in-sample) with the official harness and the `typesafe`
adapter; accuracy is micro accuracy (correct/231) from `jevbench.cli summarize`.

**Primary: this package in the runtime image.** README steps 1 to 4 on one RTX 4090 24 GB (compute
capability 8.9): network disabled, read-only root filesystem, package and model mounted read-only,
new `WORK`. Run [[FACTS-PENDING: S6-FINAL start and end UTC from run1/manifest.json]]; startup with
empty caches [[FACTS-PENDING: S6-FINAL seconds from docker run to the health line]] s; `/health`
status `ok`, no warnings, attention backend requested `auto`, observed
[[FACTS-PENDING: S6-FINAL info.attention_backend and the Using ... backend line of attempt-1.log]].
231/231 attempted, [[FACTS-PENDING: S6-FINAL failed count]] failed, schema validity
[[FACTS-PENDING: S6-FINAL schema validity]]; accuracy [[FACTS-PENDING: S6-FINAL n_correct]]/231
(macro [[FACTS-PENDING: S6-FINAL macro accuracy]]); ECE [[FACTS-PENDING: S6-FINAL ECE]], Brier
[[FACTS-PENDING: S6-FINAL Brier]]; USD [[FACTS-PENDING: S6-FINAL summarize price]] per 1,000 decisions
(engine basis, assumed prices); raw p50 [[FACTS-PENDING: S6-FINAL raw p50]] s, p95
[[FACTS-PENDING: S6-FINAL raw p95]] s; peak GPU memory [[FACTS-PENDING: S6-FINAL peak GPU memory from the nvidia-smi samples]] MiB;
slow path [[FACTS-PENDING: S6-FINAL slow-path count of 231, parsed and unparsed]]; cost-fuse
transitions [[FACTS-PENDING: S6-FINAL transitions]].

**Changes from commit 06428f09559a0a0536776c91cd544b12b68176f3 of this repository**
(https://github.com/chenyuwang166-oss/metask-jev-rain-12B/tree/06428f09559a0a0536776c91cd544b12b68176f3).
Code: failure handling (HTTP 422 and 503 instead of a substitute answer, section 4); the startup
supervisor (exact version gate, path, model-folder and environment checks, startup attempts auto
and then `TRITON_ATTN`); quota state that must be writable at startup; the runtime information in
`/health`. The request-path functions and constants listed with the SHA-256 of their source in
`regression_checks.py` (`REQUEST_PATH_SOURCE_SHA256`) are unchanged apart from comments. Parameters:
in `config.json` only the two `fitted_on` strings changed and in `calibration.json` only the
`fitted_on` and `status` strings; every number is unchanged (`PARAMETER_SHA256` in
`regression_checks.py` pins all other values). Their SHA-256 at that commit were
`a89b9511475fa7f145ca03f2a4ea518e7fc7753802834cb77c6301148e32a8cb` (`config.json`) and
`33a61924192335b98e7524e58a33ba32d136c8fc880b39e4ff58ed964ad02b40` (`calibration.json`);
`FREEZE.md` gives the current values. The same parameters do not make results bit-for-bit
identical: Python 3.12 in the image sums floating-point numbers differently from Python 3.10 of the
earlier runs, so probabilities can differ at rounding level for the same logits, and GPU inference
varies slightly between runs.

**Earlier runs (history).**

- 2026-10-07 22:51:48 to 22:54:56 UTC (first and last result timestamps; that run wrote no
  manifest): the official-harness run reported in the disclosure at commit
  06428f09559a0a0536776c91cd544b12b68176f3, the previous version of this package, in the Python
  3.10.12 venv on the RTX 4090 24 GB: 231/231 answered, 0 failed, schema validity 1.0, accuracy
  202/231 (macro 0.8699), USD 0.04736 per 1,000 decisions (engine basis, assumed prices), raw p50
  0.052 s, p95 5.71 s, peak GPU memory 20,516 MiB; no response carried a substitute answer. Its
  service log has no attention-backend line (section 4 gives the line of an earlier run).

## 8. Environment and hardware

- Runtime image `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`,
  the linux/amd64 image of vLLM release `v0.31.0`: index digest
  `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`, linux/amd64 manifest
  digest `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b`, config digest
  `sha256:c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3`.
- `launch.py` refuses to start unless these exact versions are installed (a local build suffix such
  as `+cu130` is ignored): vLLM 0.31.0, torch 2.13.0, transformers 5.17.0, tokenizers 0.23.2, triton
  3.7.1, compressed-tensors 0.17.0. They equal the measured venv (`requirements.lock`). Python must
  be 3.10 or newer (the image has 3.12.3; the earlier runs used 3.10.12).
- `gpu_memory_utilization` 0.85 and `max_num_seqs` 1; maximum model length 18,432 tokens. The KV
  cache is a share of the GPU's memory, and prefix-cache hits lower the engine-basis input tokens,
  so the measured cost can vary slightly with GPU memory size.
- Measured hardware: one RTX 4090 24 GB (compute capability 8.9), NVIDIA driver 580.105.08, Ubuntu
  22.04.5. Kernel choice by GPU: section 4.
- The CUDA 13.0 runtime in the image needs NVIDIA driver 580 or newer.

## 9. Limits

- All public-item figures are in-sample; there is no held-out validation.
- Routing depends on arrival order (section 4); results are defined for one run from a freshly
  started service.
- Items longer than 8,192 prompt tokens never take the slow path and are compressed.
- Decisions on the slow path take seconds, decisions on the fast path tens of milliseconds; over all
  decisions the p95 was 5.71 s in the earlier venv run.
- Measurements exist for the RTX 4090 only (section 4).
