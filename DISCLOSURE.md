# metask-jev-rain-12B: disclosure

How to run the service: [README.md](README.md).

## 1. Identity

| Item | Value |
|---|---|
| System name | `metask-jev-rain-12B` (team MeTask) |
| Code | this repository at the commit given in the submission email; MIT license ([LICENSE](LICENSE)) |
| Weights loaded | `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` at https://huggingface.co/cyankiwi/gemma-4-12B-it-qat-AWQ-INT4/tree/12deda07c573047b191b63f0768c0412120f3ed4 (compressed-tensors W4A16, detected by vLLM from the checkpoint) |
| Base chain | `google/gemma-4-12B-it` → `google/gemma-4-12B-it-qat-q4_0-unquantized` (Google's quantization-aware-training release, named as `base_model` in the cyankiwi model card) → `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` (the only weights loaded) |
| Weights license | Apache-2.0, as declared by the cyankiwi model card (`license: apache-2.0`, license link https://ai.google.dev/gemma/docs/gemma_4_license) |
| Training by us | none for this system: no fine-tuning, LoRA, adapter or other parameter update of the weights it loads; `model.adapter` is `null` |

### Training data

This system loads the third-party weights above unchanged; we added no data of our own to them.
Google documents the training data of the Gemma models. The cyankiwi model card names the
calibration data of its 4-bit quantization "STEM and Agentic" (Hugging Face dataset
`cyankiwi/calibration`). We cannot verify whether any of these third-party data contain JevBench
items.


## 3. Use of public JevBench items

**Data.** The only JevBench data we used are the public items of
https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc,
`datasets/public/`.

No model weights were trained on them. Labels were used for the calibration.

We never had access to the sealed set. We used no ImageJevBench item at any point.
