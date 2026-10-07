"""Foreground supervisor: readiness check, print health, forward termination."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tempfile
from datetime import datetime, timezone
import urllib.request

ROOT=Path(__file__).resolve().parent

def prepare_state(state_dir=None, run_id=None):
    state=Path(state_dir or './state').resolve()
    if state.is_relative_to(ROOT) and not state.is_relative_to(ROOT/'state'):
        raise ValueError('In-package state must be under ./state')
    rid=run_id or datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S%fZ')
    if not rid or Path(rid).name!=rid or rid in ('.','..') or '/' in rid or chr(92) in rid:
        raise ValueError('run_id must be a single directory name')
    state.mkdir(parents=True,exist_ok=True)
    # Actual create/write/flush rather than a permission-bit guess.
    try:
        with tempfile.TemporaryFile(dir=state) as probe:
            probe.write(b'probe');probe.flush();os.fsync(probe.fileno())
    except OSError:
        raise ValueError('Quota state directory is not writable') from None
    if (state/rid).exists():raise ValueError('run_id already exists; choose a fresh run_id')
    return state,rid

def validate_health(health, no_cache=False):
    bad=health.get('warnings',[])
    if bad or health.get('status','ok')!='ok' or health.get('prefix_caching') is not (not no_cache) or health.get('supports_slow_session') is not True:
        raise RuntimeError('Service readiness rejected: health warnings or cache/session capability; health='+json.dumps(health)[:1500])

def start(model, port, mode='routed', config=None, no_cache=False, state_dir=None, run_id=None):
    if sys.version_info[:2]!=(3,10):raise ValueError('Python 3.10 required')
    # Check only the two runtime packages, never the optional full lock.
    for name,major in [('vllm','0'),('torch','2')]:
        if importlib.metadata.version(name).split('.')[0]!=major:
            raise ValueError('Unsupported runtime major version: '+name)
    p=Path(model).resolve()
    for n in ('config.json',):
        if not (p/n).is_file():raise ValueError('Missing model file: '+n)
    from model_setup import ensure_processors
    ensure_processors(p)
    cfg=Path(config or ROOT/'config.json').resolve()
    c=json.loads(cfg.read_text(encoding='utf-8'))
    if mode=='routed' and not c.get('package',{}).get('finalized'):
        raise ValueError('Real fitted parameters must be finalized before routed serving')
    state,rid=prepare_state(state_dir,run_id)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        probe.bind(('127.0.0.1',port))
    env=dict(os.environ,VLLM_USE_FLASHINFER_SAMPLER='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
             VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    args=[sys.executable,'-B',str(ROOT/'server.py'),'--config',str(cfg),'--model',str(p),
          '--bind','127.0.0.1','--port',str(port),'--backend','vllm','--quant','awq',
          '--mode',mode,'--model-key','gemma-4-12b-awq','--state-dir',str(state),'--run-id',rid]
    if no_cache:args.append('--no-prefix-caching')
    return start_attempts(args,env,state,rid,port,mode,no_cache)


def start_attempts(args,env,state,rid,port,mode='routed',no_cache=False,timeout=900):
    # Separate attempt state prevents a failed warmup from contaminating the next run.
    log_dir=Path(state)/(rid+'-startup')
    log_dir.mkdir(exist_ok=False)
    def event(attempt,backend,result):
        record=json.dumps(dict(attempt=attempt,requested_backend=backend or 'auto',result=result))
        with (log_dir/'attempts.jsonl').open('a',encoding='utf-8',newline='\n') as out:
            out.write(record+'\n')
        print(record,flush=True)
    for attempt,backend in enumerate((None,'FLASH_ATTN','TRITON_ATTN'),1):
        child_env=dict(env,VLLM_USE_FLASHINFER_SAMPLER='0',PYTHONUNBUFFERED='1')
        child_env.pop('VLLM_ATTENTION_BACKEND',None)
        if backend:child_env['VLLM_ATTENTION_BACKEND']=backend
        log_path=log_dir/f'attempt-{attempt}.log'
        child_env['JEV_ATTENTION_LOG']=str(log_path)
        child_args=list(args)
        child_args[child_args.index('--state-dir')+1]=str(log_dir/f'state-{attempt}')
        event(attempt,backend,'starting')
        proc=None
        try:
            with log_path.open('wb') as out:
                proc=subprocess.Popen(child_args,cwd=ROOT,env=child_env,
                                      stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                logs=log_path.read_text(encoding='utf-8',errors='replace')
                if 'EngineCore failed to start' in logs:
                    event(attempt,backend,'failed: EngineCore failed to start')
                    break
                if proc.poll() is not None:
                    event(attempt,backend,'failed: exited before readiness')
                    break
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                        health=json.load(response)
                        ready=response.status==200 and health['mode']==mode and health['model_key']=='gemma-4-12b-awq'
                except (OSError,ValueError,KeyError):
                    ready=False
                if ready:
                    # Recheck after the HTTP round trip; failure takes precedence.
                    if proc.poll() is not None or 'EngineCore failed to start' in log_path.read_text(encoding='utf-8',errors='replace'):
                        event(attempt,backend,'failed: engine stopped during readiness')
                        break
                    validate_health(health,no_cache)
                    event(attempt,backend,'ready')
                    print(json.dumps(health,ensure_ascii=False),flush=True)
                    return proc
                time.sleep(1)
            else:
                raise TimeoutError('Health readiness timed out')
        except BaseException as exc:
            event(attempt,backend,'aborted: '+type(exc).__name__)
            if proc is not None:stop(proc)
            raise
        stop(proc)
    raise RuntimeError('All three attention backend attempts failed before readiness')


def stop(proc):
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # Reap descendants left in the process group before rebinding on retry.
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    proc.wait()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--port',type=int,default=8000)
    ap.add_argument('--state-dir');ap.add_argument('--run-id')
    a=ap.parse_args()
    def interrupted(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    proc=None
    try:
        proc=start(a.model,a.port,state_dir=a.state_dir,run_id=a.run_id)
        return proc.wait()
    except KeyboardInterrupt:return 130
    finally:
        if proc:stop(proc)

if __name__=='__main__':raise SystemExit(main())
