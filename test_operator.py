"""Offline operator contract tests: no model imports or downloads."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import attention
from backends import generation_audit
from core import Service, load_config
from finalize import freeze
import launch
from model_setup import ensure_processors
from pkg_check import check_manifest


class OperatorTests(unittest.TestCase):
    def test_generation_health_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            config=copy.deepcopy(load_config());config['model']['path']=temp
            reference=Path(temp)/'generation_config.json'
            reference.write_text(json.dumps(dict(do_sample=True,temperature=.7,top_p=.9,top_k=64)))
            service=Service.__new__(Service)
            service.config=config;service.warnings=[]
            service.counter=SimpleNamespace(memory_only=False,cost_fuse=SimpleNamespace(snapshot=lambda _:{}))
            service.backend=SimpleNamespace(generation_reference=generation_audit(config))
            health=service.health()
            self.assertEqual(health['status'],'ok');self.assertEqual(health['warnings'],[])
            self.assertEqual(set(health['info']['intended_overrides']),{'do_sample','temperature','top_p','top_k'})
            from server import make_handler
            service.backend.attention_backends=['FLASH_ATTN']
            handler_type=make_handler(service)
            handler=handler_type.__new__(handler_type);handler.path='/health'
            responses=[];handler.respond=lambda code,body:responses.append((code,body))
            handler.do_GET()
            self.assertEqual(responses[-1][1]['status'],'ok')
            self.assertEqual(responses[-1][1]['warnings'],[])
            self.assertEqual(responses[-1][1]['attention_backend_source'],'engine_worker')
            service.backend.attention_backends=[]
            with patch.dict(os.environ,{},clear=True):handler.do_GET()
            self.assertEqual(responses[-1][1]['status'],'ok')
            self.assertNotIn('actual attention backend unavailable',responses[-1][1]['warnings'])
            self.assertIn('attention_backend',responses[-1][1].get('info',{}))
            config['generation']['temperature']=.5
            self.assertFalse(generation_audit(config)['verified'])
            config['generation']['temperature']=1.0
            reference.write_text(json.dumps(dict(repetition_penalty=1.2)))
            service.backend.generation_reference=generation_audit(config)
            self.assertIn('generation reference mismatch',service.health()['warnings'])
            reference.write_text('{')
            self.assertEqual(generation_audit(config)['status'],'unreadable')
            reference.unlink()
            self.assertFalse(generation_audit(config)['verified'])

    def test_backend_observation(self):
        backend=SimpleNamespace(get_name=lambda:'FLASH_ATTN')
        worker=SimpleNamespace(model_runner=SimpleNamespace(model=SimpleNamespace(modules=lambda:[SimpleNamespace(attn_backend=backend)])))
        engine=SimpleNamespace(collective_rpc=lambda fn:[fn(worker)])
        observed=attention.observe_engine_attention(engine)
        with patch.dict(os.environ,{'VLLM_ATTENTION_BACKEND':'TRITON_ATTN'},clear=True):
            health=attention.attention_health(observed)
        self.assertEqual(health['attention_backend'],'FLASH_ATTN')
        self.assertEqual(health['attention_backend_source'],'engine_worker')
        with tempfile.TemporaryDirectory() as temp:
            log=Path(temp)/'engine.log'
            for line in ('Using Flash Attention backend on V1 engine.', 'Using FLASH_ATTN attention backend out of potential backends.'):
                log.write_text(line)
                with patch.dict(os.environ,{'JEV_ATTENTION_LOG':str(log)},clear=True):
                    self.assertEqual(attention.attention_health()['attention_backend'],'FLASH_ATTN')

    def test_processor_check_is_offline(self):
        # Serving must never touch the network: missing optional processor configs are only reported.
        import tempfile, socket
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model'; model.mkdir()
            real=socket.socket
            def blocked(*a,**k): raise AssertionError('network access attempted during serving')
            socket.socket=blocked
            try:
                missing=ensure_processors(model)
            finally:
                socket.socket=real
            self.assertIn('preprocessor_config.json',missing)
            self.assertFalse((model/'preprocessor_config.json').exists())

    def test_clone_metadata_and_default_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'source.txt').write_text('release')
            freeze(root)
            for name in ('.git','__pycache__','state'):
                (root/name).mkdir();(root/name/'generated').write_bytes(bytes([0,255,13]))
            check_manifest(root)
            freeze(root);check_manifest(root)
            old=Path.cwd()
            try:
                os.chdir(root)
                state,rid=launch.prepare_state()
            finally:os.chdir(old)
            self.assertEqual(state,root/'state');self.assertRegex(rid,r'^run-\d{8}T\d{12}Z$')


if __name__=='__main__':unittest.main()
