"""Maintainer tool: rebuild requirements.lock and environment.resolved.json from the measured freeze.

  python3 -B resolve_env.py --output-dir /abs/new/dir

No resolution and no installation. Both files are created exclusively in a new directory outside
the package; pkg_check.py regenerates them and requires them to equal the packaged files byte for byte.
"""
import argparse
import hashlib
import json
from pathlib import Path
from core import external_path

ROOT = Path(__file__).resolve().parent

def render():
    source = (ROOT/'requirements.measured.lock').read_text(encoding='utf-8')
    versions = {}
    lines = []
    for line in source.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, version = line.split('==')
        if name == 'torch':
            version = '2.13.0+cu130'  # CUDA build of torch in the measured venv.
        if name in versions:
            raise ValueError('Duplicate measured package')
        versions[name] = version
        lines.append(name+'=='+version)
    lock = ('# Measured version lock; versions only, no artifact hashes.\n'
            '# All measured entries retained; torch CUDA build made explicit.\n'
            '--extra-index-url https://download.pytorch.org/whl/cu130\n'
            + '\n'.join(lines)+'\n')
    evidence = {
        'source': 'requirements.measured.lock; historical measured environment',
        'os': 'Ubuntu 22.04.5', 'python': '3.10.12', 'python_required': '>=3.10',
        'nvidia_driver': '580.105.08', 'gpu': 'RTX 4090', 'gpu_memory_gb': 24,
        'torch_cuda_build': 'cu130', 'cuda_toolkit_system': 'not recorded',
        'measured_freeze': source, 'versions': versions, 'removed_entries': [],
        'overrides': {'torch': {'freeze': '2.13.0', 'measured_build': '2.13.0+cu130'}},
        'lock_kind': 'exact versions, not wheel hashes',
        'installation_verified_here': False,
        'lock_sha256': hashlib.sha256(lock.encode()).hexdigest(),
        'measured_freeze_sha256': hashlib.sha256(source.encode()).hexdigest(),
    }
    return {'requirements.lock': lock, 'environment.resolved.json': json.dumps(evidence, indent=2)+'\n'}

def build(output_dir):
    output = external_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name, text in render().items():
        with (output/name).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    print('requirements.lock and environment.resolved.json written to '+str(output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True, type=Path, help='new absolute directory outside the package')
    build(parser.parse_args().output_dir)
