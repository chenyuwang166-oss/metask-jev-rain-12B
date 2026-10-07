"""Offline fallback tests with real, lightweight fake server subprocesses."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import attention
import launch

FAKE_SERVER = r"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from attention import attention_health
backend=os.environ.get('VLLM_ATTENTION_BACKEND')
assert os.environ['VLLM_USE_FLASHINFER_SAMPLER']=='0'
if backend is None:
    print('auto failed',flush=True)
    sys.exit(1)
if backend=='FLASH_ATTN':
    print('EngineCore failed to start',flush=True)
    time.sleep(60)
    sys.exit(1)
print('Using TRITON_ATTN attention backend',flush=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        body=json.dumps(dict(mode='routed',model_key='gemma-4-12b-awq',
            prefix_caching=True,supports_slow_session=True,warnings=[],
            **attention_health())).encode()
        self.send_response(200);self.end_headers();self.wfile.write(body)
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


def fake_stop(proc):
    if proc.poll() is None:proc.terminate()
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()


class AttentionTests(unittest.TestCase):
    def test_two_failures_then_ready(self):
        with tempfile.TemporaryDirectory() as temp, socket.socket() as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1];probe.close()
            args=[sys.executable,'-B','-c',FAKE_SERVER,str(port),'--state-dir',temp,'--run-id','fresh']
            calls=[];popen=subprocess.Popen
            def spawn(args,**kwargs):
                proc=popen(args,**kwargs);calls.append((args,kwargs['env'],proc));return proc
            env=dict(os.environ,VLLM_ATTENTION_BACKEND='IGNORED',VLLM_USE_FLASHINFER_SAMPLER='1')
            output=io.StringIO()
            try:
                with patch('launch.subprocess.Popen',side_effect=spawn),patch('launch.stop',side_effect=fake_stop),contextlib.redirect_stdout(output):
                    proc=launch.start_attempts(args,env,Path(temp),'fresh',port,timeout=15)
                self.assertEqual([e.get('VLLM_ATTENTION_BACKEND') for _,e,_ in calls],[None,'FLASH_ATTN','TRITON_ATTN'])
                self.assertTrue(all(e['VLLM_USE_FLASHINFER_SAMPLER']=='0' for _,e,_ in calls))
                self.assertEqual(len({a[a.index('--state-dir')+1] for a,_,_ in calls}),3)
                self.assertTrue(all(pr.poll() is not None for _,_,pr in calls[:2]))
                import urllib.request
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health') as response:health=json.load(response)
                self.assertEqual(health['attention_backend'],'TRITON_ATTN')
                records=[json.loads(line) for line in (Path(temp)/'fresh-startup'/'attempts.jsonl').read_text().splitlines()]
                self.assertEqual([r['result'] for r in records],['starting','failed: exited before readiness','starting','failed: EngineCore failed to start','starting','ready'])
            finally:
                for _,_,proc in calls:fake_stop(proc)

    def test_auto_ready_does_not_retry(self):
        with tempfile.TemporaryDirectory() as temp, socket.socket() as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1];probe.close()
            code=FAKE_SERVER.replace('if backend is None:', 'if False:').replace("if backend=='FLASH_ATTN':",'if False:').replace('Using TRITON_ATTN','Using FLASHINFER')
            args=[sys.executable,'-B','-c',code,str(port),'--state-dir',temp]
            proc=None
            try:
                with patch('launch.stop',side_effect=fake_stop),contextlib.redirect_stdout(io.StringIO()):
                    proc=launch.start_attempts(args,dict(os.environ),Path(temp),'auto',port,timeout=15)
                records=[json.loads(line) for line in (Path(temp)/'auto-startup'/'attempts.jsonl').read_text().splitlines()]
                self.assertEqual([r['result'] for r in records],['starting','ready'])
                self.assertEqual(records[-1]['requested_backend'],'auto')
            finally:
                if proc is not None:fake_stop(proc)

    def test_auto_observation_and_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'engine.log';log.write_text('Using FLASHINFER attention backend')
            with patch.dict(os.environ,{'JEV_ATTENTION_LOG':str(log)},clear=True):
                self.assertEqual(attention.attention_health()['attention_backend'],'FLASHINFER')
                self.assertEqual(attention.attention_health()['attention_backend_requested'],'auto')
                log.write_text('')
                self.assertIsNone(attention.attention_health()['attention_backend'])

    def test_exhaustion_and_timeout(self):
        from unittest.mock import Mock
        for timed_out in (False,True):
            with self.subTest(timed_out=timed_out),tempfile.TemporaryDirectory() as temp:
                proc=Mock();proc.poll.return_value=1
                with patch('launch.subprocess.Popen',return_value=proc) as spawn,patch('launch.stop') as stop,contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(TimeoutError if timed_out else RuntimeError):
                        launch.start_attempts(['fake','--state-dir',temp],{},Path(temp),'fresh',8000,timeout=0 if timed_out else 1)
                    self.assertEqual(spawn.call_count,1 if timed_out else 3)
                    self.assertEqual(stop.call_count,spawn.call_count)


if __name__=='__main__':unittest.main()
