"""Foreground supervisor: runtime and model checks, startup attempts, readiness, termination.

The service writes files only below --state-dir (plus shared memory in /dev/shm):
  <state-dir>/<run-id>-startup/   attempts.jsonl, attempt-N.log, state-N/ (quota state), tmp/, home/
  <state-dir>/cache/              compilation caches shared by runs that use the same state directory
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
import urllib.request

ROOT=Path(__file__).resolve().parent
IMAGE='vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b'
MODEL_KEY='gemma-4-12b-awq'
MODEL_REVISION='cyankiwi/gemma-4-12B-it-qat-AWQ-INT4@12deda07c573047b191b63f0768c0412120f3ed4'
# Exact versions on the numeric path; a local build suffix such as +cu130 is ignored.
VERSIONS={'vllm':'0.31.0','torch':'2.13.0','transformers':'5.17.0','tokenizers':'0.23.2',
          'triton':'3.7.1','compressed-tensors':'0.17.0'}
# Files of MODEL_REVISION that fix the architecture, quantization, tokenizer, chat template and generation defaults.
MODEL_SHA256={
    'config.json':'53b52e5eeb829992dedcb85cb465efd31805da77245b35167b9831ff393572e3',
    'generation_config.json':'a8349d9bd64cc5841297fcb5002f0fdc4749c473c8f1b10ea337f9ce4ee7014e',
    'processor_config.json':'6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9',
    'tokenizer_config.json':'09e6222fd7049dae603d0f61a8b10af5a813f22dca10e6cdc6918f76d7661104',
    'tokenizer.json':'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'chat_template.jinja':'36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0',
    'model.safetensors.index.json':'7c48224f0d8b11a4eb3d0be6bac8fab7069223ca3ac96a259c2b649945c88332',
}
# Weight shards: startup checks sizes; prepare_model.sh (maintainers) checks the full hashes.
SHARD_BYTES={
    'model-00001-of-00003.safetensors':5368362132,
    'model-00002-of-00003.safetensors':5357673316,
    'model-00003-of-00003.safetensors':496095448,
}
SHARD_SHA256={
    'model-00001-of-00003.safetensors':'093872a6b9b1c316c66241f1d8fadf7b6d16f7368914e34fbec0e5343fc10a78',
    'model-00002-of-00003.safetensors':'c43491e1a10ca18f90c7ffd69338124cdbfa47315a1da3d4e49bd0180f73de11',
    'model-00003-of-00003.safetensors':'a93a9a3917fd4c2aad6108b8b0f02c0acd074b9f306039d09447b81fcc092254',
}
RUN_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}')
BINDS=('127.0.0.1','0.0.0.0')
STARTUP_TIMEOUT=1800
# vLLM places engine IPC sockets in TMPDIR; Unix socket paths are limited to 107 bytes.
MAX_TMP_PATH=70
# Startup attempts: vLLM's own choice (auto), then TRITON_ATTN. This checkpoint mixes attention
# head sizes 256 and 512; where FlashAttention 4 is unavailable (for example compute capability
# 8.x, the measured RTX 4090) vLLM 0.31 resolves auto to TRITON_ATTN, and FLASH_ATTN cannot start.
ATTEMPTS=(None,'TRITON_ATTN')
# Model folder entries a loader may read; among them only the pinned files are accepted.
LOADER_SUFFIXES=('.json','.safetensors','.bin','.pt','.pth','.gguf','.model','.tiktoken','.jinja','.py','.txt')
# Inherited path variables must stay inside --state-dir; unset ones default below state/cache or HOME.
PATH_VARIABLES=('XDG_CACHE_HOME','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','HF_HOME',
                'CUDA_CACHE_PATH','NUMBA_CACHE_DIR','XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME',
                'VLLM_CONFIG_ROOT','TRITON_HOME','FLASHINFER_WORKSPACE_BASE','TORCH_HOME','TORCH_EXTENSIONS_DIR',
                'MPLCONFIGDIR','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','HF_ASSETS_CACHE')


def check_python(version_info=None):
    version=tuple((version_info or sys.version_info)[:2])
    if version<(3,10):raise ValueError('Python >= 3.10 required')


def check_versions():
    found={}
    for name,expected in VERSIONS.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ValueError(f'{name} is not installed; run in {IMAGE}') from None
        if actual.split('+',1)[0]!=expected:
            raise ValueError(f'{name} {actual} found, {expected} required; run in {IMAGE}')
        found[name]=actual
    return found


def _absolute(value,flag):
    path=Path(value)
    if not path.is_absolute():raise ValueError(f'{flag} must be an absolute path')
    return path.resolve()


def _overlap(a,b):
    return a==b or a.is_relative_to(b) or b.is_relative_to(a)


def check_paths(model,state_dir):
    model=_absolute(model,'--model');state=_absolute(state_dir,'--state-dir')
    for path,flag in ((model,'--model'),(state,'--state-dir')):
        if _overlap(path,ROOT):raise ValueError(f'{flag} must be outside the package and must not contain it')
    if _overlap(model,state):raise ValueError('--model and --state-dir must not contain each other')
    return model,state


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


MODEL_FOLDER_RULE=('MODEL must contain regular files '
                   '(a Hugging Face cache snapshots/<sha> directory cannot be mounted alone)')


def _model_file(model,name):
    path=model/name
    if path.is_symlink():
        try:path.resolve(strict=True)
        except (OSError,RuntimeError):  # RuntimeError: a link loop before Python 3.13
            raise ValueError(f'Model file is a dangling or looping symbolic link: {name}; {MODEL_FOLDER_RULE}') from None
        raise ValueError(f'Model file is a symbolic link: {name}; {MODEL_FOLDER_RULE}')
    if not path.is_file():raise ValueError('Missing model file: '+name)
    return path


def check_model(model,full=False):
    """Read-only identity check of the model folder against MODEL_REVISION; nothing is written.

    The pinned files must be regular files with the pinned SHA-256 (shards: size, or full hash).
    Any other entry a loader may read (weights, tokenizer, processor or template files) is refused.
    """
    model=Path(model)
    if not model.is_dir():raise ValueError('Model folder not found: '+str(model))
    pinned={*MODEL_SHA256,*SHARD_BYTES,'PREPARATION.json'}
    for entry in sorted(model.iterdir()):
        name=entry.name
        if name not in pinned and (name.lower().endswith(LOADER_SUFFIXES) or name.lower().startswith('tokenizer')):
            raise ValueError(f'Unexpected file in model folder: {name}; use the files of {MODEL_REVISION} only')
    for name,expected in MODEL_SHA256.items():
        path=_model_file(model,name)
        if sha256(path)!=expected:raise ValueError(f'Model file differs from {MODEL_REVISION}: {name}')
    for name,size in SHARD_BYTES.items():
        path=_model_file(model,name)
        if path.stat().st_size!=size:raise ValueError(f'Model shard size differs from {MODEL_REVISION}: {name}')
        if full and sha256(path)!=SHARD_SHA256[name]:raise ValueError(f'Model shard differs from {MODEL_REVISION}: {name}')


EXISTING_RUN='run_id already exists in --state-dir; use a new --state-dir (with the README commands: a new WORK)'


def prepare_state(state,run_id=None):
    """Claim <state>/<run-id>-startup with exclusive creation; nothing is overwritten or removed."""
    state=Path(state)
    rid=run_id or datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S%fZ')
    if not RUN_ID.fullmatch(rid):raise ValueError('run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}')
    startup=state/(rid+'-startup')
    if len(os.fsencode(startup/'tmp'))>MAX_TMP_PATH:
        raise ValueError(f'--state-dir path is too long for engine sockets; keep {startup/"tmp"} within {MAX_TMP_PATH} bytes')
    try:state.mkdir(parents=True,exist_ok=True)
    except OSError:raise ValueError('State directory is not writable: '+str(state)) from None
    if (state/rid).exists() or startup.exists():raise ValueError(EXISTING_RUN)
    cache=state/'cache'
    if cache.is_symlink() or (cache.exists() and not cache.is_dir()):
        raise ValueError(f'{cache} must be a directory, not a link or a file')
    try:
        startup.mkdir()
        for name in ('tmp','home'):(startup/name).mkdir()
        cache.mkdir(exist_ok=True)
        with (startup/'attempts.jsonl').open('x',encoding='utf-8') as probe:
            probe.flush();os.fsync(probe.fileno())
    except FileExistsError:raise ValueError(EXISTING_RUN) from None
    except OSError:raise ValueError('State directory is not writable: '+str(state)) from None
    if cache.is_symlink() or not cache.resolve().is_relative_to(state.resolve()):
        raise ValueError(f'{cache} must be a directory inside --state-dir, not a link')
    return rid,startup


def check_environment(base,state):
    """Refuse inherited cache or configuration paths outside --state-dir; nothing is written.

    An inherited path variable is accepted only if it is absolute and inside --state-dir, so
    writes cannot leave the state directory. start() runs this before claiming the run name.
    """
    root=Path(state).resolve()
    for key in PATH_VARIABLES:
        if key not in base:continue
        value=Path(base[key])
        if not value.is_absolute() or not value.resolve().is_relative_to(root):
            raise ValueError(f'{key}={base[key]} is outside --state-dir; do not set cache or configuration paths')


def child_environment(base,state,startup):
    """Engine environment: offline, no telemetry, every write location below --state-dir."""
    check_environment(base,state)
    env=dict(base)
    # vLLM 0.31 no longer reads the legacy attention variable; an inherited value would only log a warning.
    env.pop('VLLM_ATTENTION_BACKEND',None);env.pop('JEV_ATTENTION_BACKEND',None)
    env.update(VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
               VLLM_HOST_IP='127.0.0.1',HOME=str(startup/'home'),TMPDIR=str(startup/'tmp'))
    cache=Path(state)/'cache'
    for key,sub in (('XDG_CACHE_HOME',None),('VLLM_CACHE_ROOT','vllm'),('TRITON_CACHE_DIR','triton'),
                    ('TORCHINDUCTOR_CACHE_DIR','inductor'),('HF_HOME','hf'),('CUDA_CACHE_PATH','nv'),
                    ('NUMBA_CACHE_DIR','numba')):
        env.setdefault(key,str(cache/sub if sub else cache))
    return env


def validate_health(health, no_cache=False):
    bad=health.get('warnings',[])
    if bad or health.get('status','ok')!='ok' or health.get('prefix_caching') is not (not no_cache) or health.get('supports_slow_session') is not True:
        raise RuntimeError('Service readiness rejected: health warnings or cache/session capability; health='+json.dumps(health)[:1500])


def start(model, port, mode='routed', config=None, no_cache=False, state_dir=None, run_id=None, bind='127.0.0.1'):
    check_python()
    check_versions()
    if bind not in BINDS:raise ValueError('--bind must be one of '+', '.join(BINDS))
    if state_dir is None:raise ValueError('--state-dir is required')
    model,state=check_paths(model,state_dir)
    check_environment(os.environ,state)  # an input check: refused before anything is written
    check_model(model)
    from model_setup import report_processor_configs
    report_processor_configs(model)
    cfg=Path(config or ROOT/'config.json').resolve()
    c=json.loads(cfg.read_text(encoding='utf-8'))
    if mode=='routed' and not c.get('package',{}).get('finalized'):
        raise ValueError('Real fitted parameters must be finalized before routed serving')
    try:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            probe.bind((bind,port))
    except OSError:raise ValueError(f'Port {port} on {bind} is not available') from None
    rid,startup=prepare_state(state,run_id)
    tempfile.tempdir=str(startup/'tmp')
    env=child_environment(os.environ,state,startup)
    args=[sys.executable,'-B',str(ROOT/'server.py'),'--config',str(cfg),'--model',str(model),
          '--bind',bind,'--port',str(port),'--backend','vllm','--quant','awq',
          '--mode',mode,'--model-key',MODEL_KEY,'--state-dir',str(state),'--run-id',rid]
    if no_cache:args.append('--no-prefix-caching')
    return start_attempts(args,env,startup,port,mode,no_cache)


def start_attempts(args,env,startup,port,mode='routed',no_cache=False,timeout=STARTUP_TIMEOUT):
    """auto first; TRITON_ATTN only if the engine fails to start. Each attempt has its own deadline."""
    startup=Path(startup)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))  # readiness never uses a proxy
    def event(attempt,backend,result,log=None):
        # Best effort: a record that cannot be written or printed never blocks starting or stopping.
        record=dict(attempt=attempt,requested_backend=backend or 'auto',result=result)
        if log is not None:record['log']=str(log)
        line=json.dumps(record)
        try:
            with (startup/'attempts.jsonl').open('a',encoding='utf-8',newline='\n') as out:
                out.write(line+'\n')
        except OSError:pass
        try:print(line,flush=True)
        except OSError:pass
    for attempt,backend in enumerate(ATTEMPTS,1):
        child_env=dict(env)
        child_env.pop('VLLM_ATTENTION_BACKEND',None);child_env.pop('JEV_ATTENTION_BACKEND',None)
        if backend:child_env['JEV_ATTENTION_BACKEND']=backend
        log_path=startup/f'attempt-{attempt}.log'
        child_env['JEV_ATTENTION_LOG']=str(log_path)
        child_args=list(args)
        # Separate quota state per attempt: a failed warmup never counts toward the served run.
        child_args[child_args.index('--state-dir')+1]=str(startup/f'state-{attempt}')
        event(attempt,backend,'starting',log_path)
        proc=None
        try:
            with log_path.open('xb') as out:
                proc=subprocess.Popen(child_args,cwd=ROOT,env=child_env,
                                      stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                logs=log_path.read_text(encoding='utf-8',errors='replace')
                if 'EngineCore failed to start' in logs:
                    event(attempt,backend,'failed: EngineCore failed to start',log_path)
                    break
                if proc.poll() is not None:
                    event(attempt,backend,'failed: exited before readiness',log_path)
                    break
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                        health=json.load(response)
                        ready=response.status==200 and health['mode']==mode and health['model_key']==MODEL_KEY
                except (OSError,ValueError,KeyError):
                    ready=False
                if ready:
                    # Recheck after the HTTP round trip; failure takes precedence.
                    if proc.poll() is not None or 'EngineCore failed to start' in log_path.read_text(encoding='utf-8',errors='replace'):
                        event(attempt,backend,'failed: engine stopped during readiness',log_path)
                        break
                    validate_health(health,no_cache)
                    event(attempt,backend,'ready',log_path)
                    print(json.dumps(health,ensure_ascii=False),flush=True)
                    return proc
                time.sleep(1)
            else:
                raise TimeoutError(f'Health readiness timed out after {timeout} s; see {log_path}')
        except BaseException as exc:
            # Stop the server first; the record follows even if a stop request arrives meanwhile.
            try:
                if proc is not None:stop(proc)
            finally:
                event(attempt,backend,'aborted: '+type(exc).__name__,log_path)
            raise
        stop(proc)
    raise RuntimeError(f'Both attention backend attempts (auto, TRITON_ATTN) failed before readiness; see {startup}')


KILL_WAIT=60


@contextmanager
def cleanup_signals():
    """Hold SIGTERM, SIGINT and SIGHUP while child processes are being stopped.

    A signal that arrives meanwhile is recorded, not acted on; once the previous handlers are
    restored it is raised as KeyboardInterrupt, so a stop request is never lost.
    """
    saved={};pending=[]
    def hold(number,frame):pending.append(number)
    try:
        if threading.current_thread() is threading.main_thread():
            for name in ('SIGTERM','SIGINT','SIGHUP'):
                number=getattr(signal,name,None)
                if number is not None:saved[number]=signal.signal(number,hold)
        yield
    finally:
        for number,handler in saved.items():
            if handler is not None:signal.signal(number,handler)
    if pending:raise KeyboardInterrupt


def stop(proc):
    """SIGTERM to the server's process group, up to 30 s, then SIGKILL to the group and a wait of
    at most KILL_WAIT s; a signal during this sequence is raised afterwards (cleanup_signals)."""
    with cleanup_signals():
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
        finally:
            # Reap descendants left in the process group before rebinding on retry.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:proc.wait(timeout=KILL_WAIT)
            except subprocess.TimeoutExpired:
                try:print(f'launch: warning: process group {proc.pid} still running {KILL_WAIT} s after SIGKILL',
                          file=sys.stderr,flush=True)
                except OSError:pass


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model',required=True,help='absolute path of the model folder (read only)')
    ap.add_argument('--state-dir',required=True,help='absolute path; the service writes only below it')
    ap.add_argument('--run-id',help='new name for this run; default run-<UTC timestamp>')
    ap.add_argument('--port',type=int,default=8000)
    ap.add_argument('--bind',choices=BINDS,default='127.0.0.1')
    a=ap.parse_args(argv)
    def interrupted(*_):raise KeyboardInterrupt
    for name in ('SIGTERM','SIGINT','SIGHUP'):
        number=getattr(signal,name,None)
        if number is not None:signal.signal(number,interrupted)
    proc=None
    try:
        proc=start(a.model,a.port,state_dir=a.state_dir,run_id=a.run_id,bind=a.bind)
        return proc.wait()
    except KeyboardInterrupt:return 130
    except (ValueError,RuntimeError,OSError) as exc:
        print('launch: error: '+str(exc),file=sys.stderr,flush=True)
        return 1
    finally:
        if proc:
            # A stop request during this final stop is already being carried out.
            try:stop(proc)
            except KeyboardInterrupt:pass

if __name__=='__main__':raise SystemExit(main())
