#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python - "$@" <<'PY'
import argparse, hashlib, json
from pathlib import Path
from huggingface_hub import snapshot_download
ap=argparse.ArgumentParser()
ap.add_argument('model_dir')
a=ap.parse_args()
repo='cyankiwi/gemma-4-12B-it-qat-AWQ-INT4'
base='google/gemma-4-12b-it'
rev='12deda07c573047b191b63f0768c0412120f3ed4'
p=Path(a.model_dir).resolve();package=Path.cwd().resolve()
if p.is_relative_to(package) or package.is_relative_to(p):raise SystemExit('Model directory must be separate from package')
if p.exists():raise SystemExit('Use a new model directory')
snapshot_download(repo_id=repo,revision=rev,local_dir=p)
meta={'model_repo':repo,'model_revision':rev,'base_repo':base}
cfg=json.loads((p/'config.json').read_text(encoding='utf-8'))
qc=cfg.get('quantization_config') or {}
if qc.get('quant_method') not in ('compressed-tensors','compressed_tensors'):
    raise SystemExit('Expected compressed-tensors INT4 checkpoint')
groups=qc.get('config_groups') or {}
if not groups or any(g.get('weights',{}).get('num_bits')!=4 for g in groups.values()):
    raise SystemExit('Expected 4-bit weight quantization in all config groups')
assert any(p.glob('*.safetensors')), 'Missing weights'
def sha(f):
    h=hashlib.sha256()
    with f.open('rb') as fh:
        for chunk in iter(lambda:fh.read(1<<24),b''):h.update(chunk)
    return h.hexdigest()
weights={q.name:sha(q) for q in sorted(p.glob('*.safetensors'))}
weight_manifest=''.join(f'{h}  {n}\n' for n,h in weights.items())
(p/'WEIGHTS.sha256').write_text(weight_manifest,encoding='utf-8',newline='\n')
meta.update(weights_sha256=weights,
            quantization='compressed-tensors INT4',vllm_quantization='auto-detect',
            weight_manifest_sha256=hashlib.sha256(weight_manifest.encode()).hexdigest())
(p/'PREPARATION.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Model prepared; external WEIGHTS.sha256 and PREPARATION.json record identity and hashes.')
PY
