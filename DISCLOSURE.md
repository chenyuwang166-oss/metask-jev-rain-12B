# metask-jev-rain-12B — disclosure / 披露

## 中文

系统 `metask-jev-rain-12B` 是 `metask-jev-rain` 的修订版。底座仍为冻结的
`google/gemma-4-12b-it`；本提交方没有训练、微调或更新底座参数，没有适配器。
实际加载权重唯一确定为社区 QAT INT4 仓库
`cyankiwi/gemma-4-12B-it-qat-AWQ-INT4`，来源：
https://huggingface.co/cyankiwi/gemma-4-12B-it-qat-AWQ-INT4 。
社区提供的 QAT 量化权重不等于原始未量化权重；“无训练”指本提交方的工作，
不否认上游仓库的 QAT 制作过程。上游仓库标注 Apache-2.0；该权重同时适用 Gemma 条款，保留上游许可证及通知；
Gemma 条款来源：https://ai.google.dev/gemma/terms 。

加载格式固定为 **compressed-tensors INT4**，vLLM 从 checkpoint 配置自动识别。
程序参数固定为 `--quant awq --backend vllm --model-key gemma-4-12b-awq`；
`awq` 是服务的量化配置入口，不强制把 compressed-tensors 当成经典 AWQ 内核。
系统名称、响应模型名及包版本固定为 `metask-jev-rain-12B`，无候选精度或待选映射。

服务由原来的单次前向升级为两路：快路单次前向读取候选分数；低置信度题在
15% 慢路配额（quota=0.15）及 20% 保险上限（fuse=0.20）约束下生成最多
300 词、512 token 的草稿，再读取答案。阈值约 2.9999，实际固定值为
2.9999032097890423。配额、熔断、解析失败及长题策略会影响实际慢路比例，
不承诺每组请求恰好 15%。长题压缩到 8192 token 输入预算，只走快路；
物理输入保护上限 16384，不能作为长题的服务预算。

usage.basis 固定为 engine，按引擎实际处理量统计输入与输出，包含草稿、
答案读取及重处理，缓存节省以引擎计量为准。正式 harness 采用相同 usage。
阈值及校准在公开 231 题上拟合，仅确定路由比例、概率映射与温度，未训练模型。
校准保留小样本标记：慢路 score 仅 1 个样本，采用快路 score 温度；
慢路二元 p_cal 上限 0.90，fallback 使用快路映射。未完成独立留出集验证。

价格固定采用 **12B 参考价假设**：输入 $0.05 / 百万 token、输出 $0.15 /
百万 token。参考 `google/gemma-3-12b-it` 最近尺寸档；OpenRouter 无
`gemma-4-12b`，因此这不是该模型的实际托管报价，也不是硬件运行成本。
实测硬件为 RTX 4090 24 GB；本次仅打包和离线检查，没有运行模型。
权重 revision 及三个分片哈希按提供值记录；下表 15% 配置冷跑数据由维护者补齐后冻结，不要求评测方作选择。

## English

`metask-jev-rain-12B` revises `metask-jev-rain`. Its base remains the frozen
`google/gemma-4-12b-it`. The submitter performed no training, fine-tuning or base
parameter updates and uses no adapter. The sole deployed weight source is the
community QAT INT4 repository `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4`, linked above.
These community-quantized weights are distinct from the original unquantized
weights. No training refers to the submitter, not the upstream QAT production
process. The upstream repository declares Apache-2.0; the weights also remain subject to the Gemma terms; upstream license notices
are retained. Source and license links are provided above.

Precision is fixed: **compressed-tensors INT4**, auto-detected by vLLM from the
checkpoint. The service uses `--quant awq --backend vllm --model-key gemma-4-12b-awq`;
this profile label does not force a classic AWQ kernel on compressed-tensors.
The public model identity and version are both `metask-jev-rain-12B`.

The former single-forward service now has two paths: one forward readout on the
fast path, and a draft of at most 300 words / 512 tokens for low-confidence items
admitted by quota 0.15 and fuse 0.20. The fixed log-ratio threshold is
2.9999032097890423 (approximately 2.9999). Admission, cost fuse, parsing and long
items may reduce the actual slow share. Long items are compressed to an 8192-token
input budget and always use the fast path; the physical guard is 16384 tokens.

Usage uses actual engine processing, including drafts, answer readouts and
reprocessing; prefix-cache savings follow engine accounting. Thresholds and
calibration were fitted on the public 231 tasks, solely for routing proportions,
probability mapping and temperatures, without model training. Sparse-sample audit
metadata is retained: slow score has one sample and uses the fast score temperature;
slow binary p_cal is capped at 0.90; fallback inherits the fast mapping.
Independent held-out validation has not been performed.

The fixed **12B reference-price assumption** is USD 0.05 input / 0.15 output per
million tokens, using `google/gemma-3-12b-it` as the nearest listed size class.
OpenRouter has no `gemma-4-12b` listing; these are not actual Gemma 4 hosting tariffs
or hardware costs. Measured hardware: RTX 4090 24 GB. This packaging task performed
only offline checks, with no model execution. Maintainers will fill the fields
below before final submission; no precision or model-mapping choice is delegated
to evaluators.

## Freeze evidence / 冻结证据

| Field / 字段 | Value / 值 |
|---|---|
| Quantized repository immutable commit / 量化仓库提交 | 12deda07c573047b191b63f0768c0412120f3ed4 |
| Weight shard 1 SHA256 / 分片 1 | `093872a6b9b1c316c66241f1d8fadf7b6d16f7368914e34fbec0e5343fc10a78` |
| Weight shard 2 SHA256 / 分片 2 | `c43491e1a10ca18f90c7ffd69338124cdbfa47315a1da3d4e49bd0180f73de11` |
| Weight shard 3 SHA256 / 分片 3 | `a93a9a3917fd4c2aad6108b8b0f02c0acd074b9f306039d09447b81fcc092254` |
| Official harness 15% routing cold-run accuracy / 15% 配置冷跑准确率 | 0.8745 (202/231; macro per harness summary), 0 failed, schema validity 1.0, raw p50 0.052 s / p95 5.71 s |
| Official harness 15% routing cold-run total price, USD / 15% 配置冷跑总费用 | $0.04736 per 1,000 decisions (derived_usage_times_tariff, 0.05/0.15 per M assumed reference) |
| Official harness 15% routing cold-run p50 latency, seconds | 0.052 (raw; adjusted 2×0.052+0.15 = 0.25) |
| Official harness 15% routing cold-run p95 latency, seconds | 5.71 (raw) |
| Peak GPU memory, GiB / 峰值显存 | 20,516 MiB on RTX 4090 24 GB |
| Final freeze time, UTC / 最终冻结时间 | 2026-10-07T23:01:02Z |
| Package commit / 包提交哈希 | v1.0-frozen |

Cold run means a fresh service and fresh quota state, with the documented startup
warmup; it does not imply warmup is disabled. Preserve the official harness output,
external PREPARATION.json and WEIGHTS.sha256 alongside the evaluation evidence.

The three shard hashes and revision above were supplied for this release; this
offline task did not download or hash model weights. Shard numbers follow the
provided order; compare the complete set against external WEIGHTS.sha256.
The threshold fit achieved 35/231 (15.1515%) eligibility on public calibration
rows; this is not an observed routed harness slow share or cold-run accuracy.
The retained calibration blocks originate from the earlier routed10 dataset;
changing routing to 15% does not constitute a new calibration or held-out validation.
All cold-run placeholders above require measurements with this 15% configuration.
