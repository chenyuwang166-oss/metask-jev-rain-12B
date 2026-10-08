#!/usr/bin/env bash
# Maintainer tool: download the pinned checkpoint into a new folder and verify it with the launch checks.
#   bash prepare_model.sh /abs/path/to/new/model/folder
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PY="${PYTHON:-python3}"
MODEL_DIR=${1:?Usage: bash prepare_model.sh /abs/path/to/new/model/folder}
"$PY" -B - "$PKG" "$MODEL_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
package=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(package))
from launch import SHARD_SHA256, check_model
from huggingface_hub import snapshot_download
repo='cyankiwi/gemma-4-12B-it-qat-AWQ-INT4'
rev='12deda07c573047b191b63f0768c0412120f3ed4'
chain='google/gemma-4-12B-it -> google/gemma-4-12B-it-qat-q4_0-unquantized -> cyankiwi/gemma-4-12B-it-qat-AWQ-INT4'
target=Path(sys.argv[2])
if not target.is_absolute():raise SystemExit('Give the model folder as an absolute path')
p=target.resolve()
if p==package or p.is_relative_to(package) or package.is_relative_to(p):raise SystemExit('Model folder must be separate from the package')
if p.exists():raise SystemExit('Use a new model folder')
snapshot_download(repo_id=repo,revision=rev,local_dir=p)
cfg=json.loads((p/'config.json').read_text(encoding='utf-8'))
qc=cfg.get('quantization_config') or {}
if qc.get('quant_method') not in ('compressed-tensors','compressed_tensors'):
    raise SystemExit('Expected compressed-tensors INT4 checkpoint')
groups=qc.get('config_groups') or {}
if not groups or any(g.get('weights',{}).get('num_bits')!=4 for g in groups.values()):
    raise SystemExit('Expected 4-bit weight quantization in all config groups')
try:check_model(p,full=True)
except ValueError as exc:raise SystemExit(str(exc))
weights={name:SHARD_SHA256[name] for name in sorted(SHARD_SHA256)}
weights_list=''.join(f'{h}  {n}\n' for n,h in weights.items())
meta={'model_repo':repo,'model_revision':rev,'base_chain':chain,'weights_sha256':weights,
      'quantization':'compressed-tensors INT4','vllm_quantization':'auto-detect',
      'weights_list_sha256':hashlib.sha256(weights_list.encode()).hexdigest()}
with (p/'WEIGHTS.sha256').open('x',encoding='utf-8',newline='\n') as out:out.write(weights_list)
with (p/'PREPARATION.json').open('x',encoding='utf-8',newline='\n') as out:out.write(json.dumps(meta,indent=2)+'\n')
print('Model folder verified against '+repo+'@'+rev+'; WEIGHTS.sha256 and PREPARATION.json written there.')
PY
