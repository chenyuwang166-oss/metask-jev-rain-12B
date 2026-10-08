"""Offline operator contract tests: no model imports or downloads.

Test directories are created with tempfile.mkdtemp below one per-run directory and are kept.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from backends import generation_audit
from core import Service, load_config
from finalize import freeze
import launch
from model_setup import report_processor_configs
from pkg_check import check_manifest

SCRATCH=Path(tempfile.mkdtemp(prefix='jev12b-operator-'))


def scratch():
    return Path(tempfile.mkdtemp(dir=SCRATCH))


class OperatorTests(unittest.TestCase):
    def test_generation_health_boundaries(self):
        model=scratch()
        config=copy.deepcopy(load_config());config['model']['path']=str(model)
        reference=model/'generation_config.json'
        reference.write_text(json.dumps(dict(do_sample=True,temperature=.7,top_p=.9,top_k=64)))
        service=Service.__new__(Service)
        service.config=config;service.warnings=[];service.dead=False
        service.counter=SimpleNamespace(memory_only=False,cost_fuse=SimpleNamespace(snapshot=lambda _:{}))
        service.backend=SimpleNamespace(generation_reference=generation_audit(config),runtime_info={'vllm':'0.31.0'})
        health=service.health()
        self.assertEqual(health['status'],'ok');self.assertEqual(health['warnings'],[])
        self.assertEqual(set(health['info']['intended_overrides']),{'do_sample','temperature','top_p','top_k'})
        from server import make_handler
        handler_type=make_handler(service)
        handler=handler_type.__new__(handler_type);handler.path='/health'
        responses=[];handler.respond=lambda code,body:responses.append((code,body))
        log=scratch()/'engine.log';log.write_text('Using AttentionBackendEnum.FLASH_ATTN backend.')
        with patch.dict(os.environ,{'JEV_ATTENTION_LOG':str(log)},clear=True):handler.do_GET()
        code,body=responses[-1]
        self.assertEqual((code,body['status'],body['warnings']),(200,'ok',[]))
        self.assertEqual(body['info']['attention_backend'],'FLASH_ATTN')
        self.assertEqual(body['info']['attention_backend_source'],'engine_log')
        self.assertEqual(body['info']['attention_backend_requested'],'auto')
        self.assertEqual(body['info']['runtime'],{'vllm':'0.31.0'})
        self.assertIn('intended_overrides',body['info'])
        with patch.dict(os.environ,{},clear=True):handler.do_GET()
        code,body=responses[-1]
        self.assertEqual((body['status'],body['warnings'],body['info']['attention_backend']),('ok',[],None))
        config['generation']['temperature']=.5
        self.assertFalse(generation_audit(config)['verified'])
        config['generation']['temperature']=1.0
        reference.write_text(json.dumps(dict(repetition_penalty=1.2)))
        service.backend.generation_reference=generation_audit(config)
        self.assertIn('generation reference mismatch',service.health()['warnings'])
        reference.write_text('{')
        self.assertEqual(generation_audit(config)['status'],'unreadable')
        config['model']['path']=str(scratch())  # a folder without generation_config.json
        self.assertFalse(generation_audit(config)['verified'])

    def test_processor_report_is_offline(self):
        # Serving never touches the network: absent processor configs are only reported.
        import socket
        model=scratch()
        real=socket.socket
        def blocked(*a,**k): raise AssertionError('network access attempted during serving')
        socket.socket=blocked
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                missing=report_processor_configs(model)
        finally:
            socket.socket=real
        self.assertEqual(missing,['processor_config.json'])
        self.assertIn('processor_config.json',out.getvalue())
        self.assertEqual(sorted(p.name for p in model.iterdir()),[])
        # A complete folder prints nothing: Gemma 4 has no preprocessor or video preprocessor config.
        (model/'processor_config.json').write_text('{}')
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(report_processor_configs(model),[])
        self.assertEqual(out.getvalue(),'')

    def test_clone_metadata_and_required_state_dir(self):
        root=scratch();(root/'source.txt').write_text('release')
        freeze(root)
        for name in ('.git','__pycache__','state'):
            (root/name).mkdir();(root/name/'generated').write_bytes(bytes([0,255,13]))
        check_manifest(root)
        freeze(root);check_manifest(root)
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):launch.main(['--model',str(scratch())])
        with self.assertRaisesRegex(ValueError,'--state-dir is required'):
            with patch('launch.check_versions'):launch.start(scratch(),8000)


if __name__=='__main__':unittest.main()
