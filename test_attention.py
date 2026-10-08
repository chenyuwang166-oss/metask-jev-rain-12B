"""Offline startup-attempt and attention-report tests with real, lightweight fake server subprocesses.

Test directories are created with tempfile.mkdtemp below one per-run directory and are kept.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import urllib.request

import attention
import launch

SCRATCH=Path(tempfile.mkdtemp(prefix='jev12b-attention-'))
OPENER=urllib.request.build_opener(urllib.request.ProxyHandler({}))

FAKE_SERVER = r"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from attention import attention_health
backend=os.environ.get('JEV_ATTENTION_BACKEND')
assert os.environ['VLLM_USE_FLASHINFER_SAMPLER']=='0'
assert 'VLLM_ATTENTION_BACKEND' not in os.environ
if backend is None:
    print('EngineCore failed to start',flush=True)
    time.sleep(60)
    sys.exit(1)
print('Using AttentionBackendEnum.TRITON_ATTN backend.',flush=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        body=json.dumps(dict(status='ok',mode='routed',model_key='gemma-4-12b-awq',
            prefix_caching=True,supports_slow_session=True,warnings=[],info=attention_health())).encode()
        self.send_response(200);self.end_headers();self.wfile.write(body)
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


def host_env():
    """This process's environment without cache or configuration paths (launch refuses those)."""
    return {k:v for k,v in os.environ.items() if k not in launch.PATH_VARIABLES}


def startup_dir():
    path=Path(tempfile.mkdtemp(dir=SCRATCH))/'fresh-startup'
    path.mkdir()
    for name in ('tmp','home'):(path/name).mkdir()
    return path


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));return probe.getsockname()[1]


def fake_stop(proc):
    if proc.poll() is None:proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()


class AttentionTests(unittest.TestCase):
    def test_auto_failure_then_triton_ready(self):
        startup=startup_dir();port=free_port()
        args=[sys.executable,'-B','-c',FAKE_SERVER,str(port),'--state-dir',str(startup.parent),'--run-id','fresh']
        calls=[];sessions=[];popen=subprocess.Popen
        def spawn(args,**kwargs):
            sessions.append(kwargs.get('start_new_session') is True)
            proc=popen(args,**kwargs);calls.append((args,kwargs['env'],proc));return proc
        inherited=dict(host_env(),VLLM_ATTENTION_BACKEND='IGNORED',JEV_ATTENTION_BACKEND='IGNORED',VLLM_USE_FLASHINFER_SAMPLER='1',
                       http_proxy='http://127.0.0.1:9',HTTP_PROXY='http://127.0.0.1:9',no_proxy='',NO_PROXY='')
        env=launch.child_environment(inherited,startup.parent,startup)
        output=io.StringIO()
        try:
            with patch.dict(os.environ,{'http_proxy':'http://127.0.0.1:9','HTTP_PROXY':'http://127.0.0.1:9','no_proxy':'','NO_PROXY':''}):
                with patch('launch.subprocess.Popen',side_effect=spawn),patch('launch.stop',side_effect=fake_stop),contextlib.redirect_stdout(output):
                    proc=launch.start_attempts(args,env,startup,port,timeout=15)
            self.assertEqual([e.get('JEV_ATTENTION_BACKEND') for _,e,_ in calls],[None,'TRITON_ATTN'])
            self.assertTrue(all('VLLM_ATTENTION_BACKEND' not in e for _,e,_ in calls))
            self.assertTrue(all(e['VLLM_USE_FLASHINFER_SAMPLER']=='0' and e['HOME']==str(startup/'home') for _,e,_ in calls))
            self.assertEqual([a[a.index('--state-dir')+1] for a,_,_ in calls],[str(startup/f'state-{n}') for n in (1,2)])
            self.assertTrue(calls[0][2].poll() is not None)
            self.assertTrue(all(kwargs_new_session for kwargs_new_session in sessions))
            with OPENER.open(f'http://127.0.0.1:{port}/health',timeout=5) as response:health=json.load(response)
            self.assertEqual(health['info']['attention_backend'],'TRITON_ATTN')
            self.assertEqual(health['info']['attention_backend_requested'],'TRITON_ATTN')
            records=[json.loads(line) for line in (startup/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['result'] for r in records],['starting','failed: EngineCore failed to start','starting','ready'])
            self.assertEqual([r['requested_backend'] for r in records],['auto','auto','TRITON_ATTN','TRITON_ATTN'])
            self.assertEqual(records[-1]['log'],str(startup/'attempt-2.log'))
            self.assertEqual(json.loads(output.getvalue().splitlines()[-1])['status'],'ok')
        finally:
            for _,_,proc in calls:fake_stop(proc)

    def test_auto_ready_does_not_retry(self):
        startup=startup_dir();port=free_port()
        code=FAKE_SERVER.replace('if backend is None:', 'if False:')
        args=[sys.executable,'-B','-c',code,str(port),'--state-dir',str(startup.parent)]
        proc=None
        try:
            with patch('launch.stop',side_effect=fake_stop),contextlib.redirect_stdout(io.StringIO()):
                proc=launch.start_attempts(args,launch.child_environment(host_env(),startup.parent,startup),startup,port,timeout=15)
            records=[json.loads(line) for line in (startup/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['result'] for r in records],['starting','ready'])
            self.assertEqual(records[-1]['requested_backend'],'auto')
            with OPENER.open(f'http://127.0.0.1:{port}/health',timeout=5) as response:health=json.load(response)
            self.assertEqual((health['info']['attention_backend'],health['info']['attention_backend_requested']),('TRITON_ATTN','auto'))
        finally:
            if proc is not None:fake_stop(proc)

    def test_log_parsing(self):
        log=Path(tempfile.mkdtemp(dir=SCRATCH))/'engine.log'
        cases=[('Using AttentionBackendEnum.FLASH_ATTN backend.','FLASH_ATTN'),
               ('Using FLASHINFER attention backend','FLASHINFER'),
               ('Using Flash Attention backend on V1 engine.','FLASH_ATTN'),
               ('Using FLASH_ATTN attention backend out of potential backends.','FLASH_ATTN'),
               ('\x1b[1mUsing AttentionBackendEnum.TRITON_ATTN backend.\x1b[0m','TRITON_ATTN'),
               ('Using TRITON backend for unquantized MoE',None),('',None)]
        for line,expected in cases:
            log.write_text(line)
            with self.subTest(line=line),patch.dict(os.environ,{'JEV_ATTENTION_LOG':str(log)},clear=True):
                health=attention.attention_health()
                self.assertEqual(health['attention_backend'],expected)
                self.assertEqual(health['attention_backend_requested'],'auto')
                self.assertEqual(health['attention_backend_source'],'engine_log' if expected else 'unreported')
        with patch.dict(os.environ,{'JEV_ATTENTION_BACKEND':'TRITON_ATTN'},clear=True):
            self.assertEqual(attention.attention_health()['attention_backend_requested'],'TRITON_ATTN')
            self.assertIsNone(attention.attention_health()['attention_backend'])

    def test_exited_attempt_is_reported(self):
        startup=startup_dir();port=free_port()
        code=FAKE_SERVER.replace("print('EngineCore failed to start',flush=True)","print('auto failed',flush=True)").replace(
            'time.sleep(60)','pass')
        args=[sys.executable,'-B','-c',code,str(port),'--state-dir',str(startup.parent)]
        proc=None
        try:
            with patch('launch.stop',side_effect=fake_stop),contextlib.redirect_stdout(io.StringIO()):
                proc=launch.start_attempts(args,launch.child_environment(host_env(),startup.parent,startup),startup,port,timeout=15)
            records=[json.loads(line) for line in (startup/'attempts.jsonl').read_text().splitlines()]
            self.assertEqual([r['result'] for r in records],['starting','failed: exited before readiness','starting','ready'])
        finally:
            if proc is not None:fake_stop(proc)

    def test_unhealthy_ready_service_is_rejected(self):
        startup=startup_dir();port=free_port()
        code=FAKE_SERVER.replace('if backend is None:', 'if False:').replace('warnings=[]',"warnings=['quota persistence unavailable']")
        args=[sys.executable,'-B','-c',code,str(port),'--state-dir',str(startup.parent)]
        stopped=[]
        def record_stop(proc):
            stopped.append(proc);fake_stop(proc)
        with patch('launch.stop',side_effect=record_stop),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'readiness rejected'):
                launch.start_attempts(args,launch.child_environment(host_env(),startup.parent,startup),startup,port,timeout=15)
        self.assertEqual(len(stopped),1)
        records=[json.loads(line) for line in (startup/'attempts.jsonl').read_text().splitlines()]
        self.assertEqual([r['result'] for r in records],['starting','aborted: RuntimeError'])

    def test_abort_stops_before_recording(self):
        # The server is stopped first; the 'aborted' record follows, even if the stop itself is interrupted.
        for interrupted in (False,True):
            with self.subTest(interrupted=interrupted):
                startup=startup_dir();seen=[]
                proc=Mock();proc.poll.return_value=None
                def record_stop(proc):
                    seen.append((startup/'attempts.jsonl').read_text().splitlines())
                    if interrupted:raise KeyboardInterrupt
                with patch('launch.subprocess.Popen',return_value=proc),patch('launch.stop',side_effect=record_stop), \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(KeyboardInterrupt if interrupted else TimeoutError):
                        launch.start_attempts(['fake','--state-dir','x'],{},startup,free_port(),timeout=0)
                self.assertEqual([[json.loads(l)['result'] for l in lines] for lines in seen],[['starting']])
                records=[json.loads(line)['result'] for line in (startup/'attempts.jsonl').read_text().splitlines()]
                self.assertEqual(records,['starting','aborted: TimeoutError'])

    def test_attempt_records_are_best_effort(self):
        # An attempts record that cannot be written or printed never blocks the attempts or their cleanup.
        class Closed(io.StringIO):
            def write(self,text):raise OSError('closed output')
        startup=startup_dir();(startup/'attempts.jsonl').mkdir()  # not writable as a file
        proc=Mock();proc.poll.return_value=1
        with patch('launch.subprocess.Popen',return_value=proc),patch('launch.stop') as stop,contextlib.redirect_stdout(Closed()):
            with self.assertRaisesRegex(RuntimeError,'Both attention backend attempts'):
                launch.start_attempts(['fake','--state-dir','x'],{},startup,free_port(),timeout=5)
        self.assertEqual(stop.call_count,len(launch.ATTEMPTS))

    def test_engine_arguments_and_attention_request(self):
        # A stand-in vllm module records the engine arguments: the second startup attempt really
        # requests TRITON_ATTN, the first leaves the choice to vLLM, and the readout settings hold.
        import copy,types
        import backends
        from core import load_config
        recorded=[]
        class RecordingLLM:
            def __init__(self,**kwargs):
                recorded.append(kwargs);raise RuntimeError('stop after recording the arguments')
        fake=types.ModuleType('vllm');fake.LLM=RecordingLLM;fake.SamplingParams=object
        model=Path(tempfile.mkdtemp(dir=SCRATCH))
        (model/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'compressed-tensors'}}),encoding='utf-8')
        config=copy.deepcopy(load_config());config['model']['path']=str(model)
        for requested in ('','TRITON_ATTN'):
            with patch.dict(sys.modules,{'vllm':fake}),patch.dict(os.environ,{'JEV_ATTENTION_BACKEND':requested}), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(backends.BackendError):backends.VLLMBackend(config)
        auto,triton=recorded
        self.assertNotIn('attention_backend',auto)
        self.assertEqual(triton.pop('attention_backend'),'TRITON_ATTN')
        self.assertEqual(auto,triton)
        expected=dict(model=str(model),dtype='bfloat16',quantization=None,seed=0,max_logprobs=-1,
                      logprobs_mode='raw_logprobs',generation_config='vllm',enable_lora=False,disable_log_stats=False,
                      enable_prefix_caching=True,gpu_memory_utilization=0.85,max_num_seqs=1,max_model_len=18432)
        self.assertEqual(auto,expected)

    def test_exhaustion_and_timeout(self):
        for timed_out in (False,True):
            with self.subTest(timed_out=timed_out):
                startup=startup_dir()
                proc=Mock();proc.poll.return_value=1
                with patch('launch.subprocess.Popen',return_value=proc) as spawn,patch('launch.stop') as stop,contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(TimeoutError if timed_out else RuntimeError,re.escape(str(startup))):
                        launch.start_attempts(['fake','--state-dir','x'],{},startup,free_port(),timeout=0 if timed_out else 1)
                    self.assertEqual(spawn.call_count,1 if timed_out else len(launch.ATTEMPTS))
                    self.assertEqual(stop.call_count,spawn.call_count)


if __name__=='__main__':unittest.main()
