"""Report observed attention implementations; never substitute requested settings."""
import os
from pathlib import Path
import re


def worker_attention_backends(worker):
    """Runs inside the vLLM worker against loaded attention layers."""
    model=worker.model_runner.model
    names=set()
    for module in model.modules():
        backend=getattr(module,'attn_backend',None)
        get_name=getattr(backend,'get_name',None)
        if callable(get_name):names.add(str(get_name()))
    return sorted(names)


def observe_engine_attention(engine):
    try:
        results=engine.collective_rpc(worker_attention_backends)
        return sorted({name for names in results for name in names if isinstance(name,str) and name})
    except Exception:
        # API/layout differences fall back to the actual engine log.
        return []


def attention_health(observed=None):
    requested=os.environ.get('VLLM_ATTENTION_BACKEND') or 'auto'
    names=list(observed or [])
    source='engine_worker' if names else 'unreported'
    if not names:
        path=os.environ.get('JEV_ATTENTION_LOG')
        text=''
        if path:
            try:text=Path(path).read_text(encoding='utf-8',errors='replace')
            except OSError:pass
        text=re.sub(r'\x1b\[[0-9;]*m','',text)
        matches=re.findall(r'\bUsing\s+([A-Za-z][A-Za-z0-9_ -]*?)\s+backend\b',text,re.I)
        aliases={'FLASH_ATTENTION':'FLASH_ATTN','TRITON_ATTENTION':'TRITON_ATTN',
                 'FLASH_INFER':'FLASHINFER','FLASHINFER_ATTENTION':'FLASHINFER'}
        for match in matches:
            name=match.strip().upper().replace('-','_').replace(' ','_')
            names.append(aliases.get(name,name.removesuffix('_ATTENTION')))
        names=list(dict.fromkeys(names))
        if names:source='engine_log'
    return {'attention_backend': names[-1] if names else None,
            'attention_backends': names,
            'attention_backend_requested':requested,
            'attention_backend_source':source}
