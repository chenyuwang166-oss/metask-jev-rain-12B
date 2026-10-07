"""Offline regressions for packaging and startup guards; never imports a model."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch
import quota
from core import load_config
from finalize import freeze
from selftest import prepare_public


class OfflineChecks(unittest.TestCase):
    def test_health_contract(self):
        healthy={'prefix_caching':True,'supports_slow_session':True,'warnings':[]}
        launch.validate_health(healthy)
        launch.validate_health(dict(healthy,prefix_caching=False),no_cache=True)
        for changes in ({'prefix_caching':False},{'supports_slow_session':False},
                        {'warnings':['Quota persistence unavailable: memory counts, fast only']},
                        {'warnings':['calibration placeholder: refit required']}):
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                launch.validate_health(dict(healthy,**changes))
        for field in ('prefix_caching','supports_slow_session'):
            bad=dict(healthy);del bad[field]
            with self.assertRaises(RuntimeError):launch.validate_health(bad)

    def test_state_success_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            state,rid=launch.prepare_state(temp,'fresh')
            self.assertEqual(rid,'fresh');self.assertEqual(state,Path(temp).resolve())
            (state/rid).mkdir()
            with self.assertRaises(ValueError):launch.prepare_state(temp,rid)

    def test_unwritable_state(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('launch.tempfile.TemporaryFile',side_effect=PermissionError),self.assertRaises(ValueError):
                launch.prepare_state(temp,'fresh')

    def test_state_paths(self):
        with self.assertRaises(ValueError):launch.prepare_state(launch.ROOT,'fresh')
        with tempfile.TemporaryDirectory() as temp:
            for rid in ('..','a/b','a'+chr(92)+'b'):
                with self.subTest(rid=rid),self.assertRaises(ValueError):launch.prepare_state(temp,rid)

    def test_counter_fallback_is_rejected_by_supervisor(self):
        with tempfile.TemporaryDirectory() as temp:
            config=copy.deepcopy(load_config())
            config['routing'].update(state_dir=temp,run_id='fresh')
            with patch('quota.QuotaCounter',side_effect=PermissionError):
                counter=quota.RuntimeCounter(config)
            self.assertTrue(counter.memory_only)
            with self.assertRaises(RuntimeError):
                launch.validate_health({'prefix_caching':True,'supports_slow_session':True,'warnings':['Quota persistence unavailable; memory counts and fast only']})

    def test_freeze_rejects_runtime_output(self):
        for name in ('selftest-run','artifact.pyc','artifact.tmp'):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/name).touch()
                with self.assertRaises(ValueError):freeze(root)
                self.assertFalse((root/'MANIFEST.sha256').exists())

    def test_public_output_and_count_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);public=root/'public';public.mkdir()
            for filename in ('easy.jsonl','original.jsonl','hard.jsonl'):
                (public/filename).write_text('',encoding='utf-8',newline='\n')
            with self.assertRaises(ValueError):prepare_public(public,launch.ROOT/'invalid-output')
            with self.assertRaises(ValueError):prepare_public(public,public/'output')
            out=root/'output'
            with self.assertRaises(ValueError):prepare_public(public,out)
            self.assertFalse(out.exists())

    def test_public_index_contains_only_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);public=root/'public';public.mkdir()
            for filename,tier,count in (('easy.jsonl','easy',48),('original.jsonl','standard',72),('hard.jsonl','hard',111)):
                rows=[{'id':tier+str(i),'state':'test context','question':{'type':'boolean','instructions':'test question'},'expected':True} for i in range(count)]
                (public/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
            out,index=prepare_public(public,root/'output')
            rows=[json.loads(line) for line in index.read_text().splitlines()]
            self.assertEqual(len(rows),231)
            self.assertTrue(all(set(row)=={'id','type','tier'} for row in rows))
            self.assertEqual(rows[0],{'id':'easy0','type':'noul','tier':'easy'})
            self.assertEqual(rows[-1]['id'],'hard110')
            with self.assertRaises(FileExistsError):prepare_public(public,out)


    def test_compressed_tensors_auto_detection(self):
        from backends import VLLMBackend
        with tempfile.TemporaryDirectory() as temp:
            cfg={'path':temp,'quant':'awq'}
            path=Path(temp)/'config.json'
            for method in ('compressed-tensors','compressed_tensors'):
                path.write_text(json.dumps({'quantization_config':{'quant_method':method}}),encoding='utf-8')
                self.assertIsNone(VLLMBackend._awq_quant_arg(cfg))
            path.write_text(json.dumps({'quantization_config':{'quant_method':'awq'}}),encoding='utf-8')
            self.assertEqual(VLLMBackend._awq_quant_arg(cfg),'awq')

    def test_version_pins_ignore_local_suffix_but_reject_other_releases(self):
        pins={'vllm':'0.31.1','torch':'2.13.1'}
        with tempfile.TemporaryDirectory() as temp:
            missing=Path(temp)/'absent-model'
            with patch.object(launch.sys,'version_info',(3,10)),patch('launch.importlib.metadata.version',side_effect=lambda n:pins[n].split('+')[0]+'+localbuild'):
                # Passing all pins reaches the file guard, before any engine starts.
                with self.assertRaisesRegex(ValueError,'Missing model file'):
                    launch.start(missing,8000)
            with patch.object(launch.sys,'version_info',(3,10)),patch('launch.importlib.metadata.version',return_value='0.0.0+localbuild'):
                with self.assertRaisesRegex(ValueError,'Unsupported runtime major version'):
                    launch.start(missing,8000)

    def test_prepare_embedded_python_and_launch_identity(self):
        import ast
        text=(launch.ROOT/'prepare_model.sh').read_text(encoding='utf-8')
        body=text.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
        compile(body,'prepare_model.sh embedded Python','exec')
        tree=ast.parse((launch.ROOT/'launch.py').read_text(encoding='utf-8'))
        args=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=='args' for target in n.targets))
        pairs={a.value:b.value for a,b in zip(args.elts,args.elts[1:])
               if isinstance(a,ast.Constant) and isinstance(b,ast.Constant)}
        self.assertEqual(pairs['--quant'],'awq')
        self.assertEqual(pairs['--backend'],'vllm')
        self.assertEqual(pairs['--model-key'],'gemma-4-12b-awq')


if __name__=='__main__':unittest.main()
