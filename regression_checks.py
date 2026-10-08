"""Offline regressions for packaging, startup guards and failure semantics; never imports a model.

Test directories are created with tempfile.mkdtemp below one per-run directory and are kept.
"""
import ast
import contextlib
import copy
import hashlib
import http.client
import inspect
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import core
import launch
import quota
from backends import BackendError, new_meter
from core import (ITEM_FAILURE_LIMIT, ItemFailure, Service, ServiceFailure, choice_codes, external_output,
                  external_path, load_config, render, threshold_identity, validate_threshold)
from finalize import freeze
from selftest import prepare_public

SCRATCH=Path(tempfile.mkdtemp(prefix='jev12b-regression-'))
PROMPT_SHA='e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938'
RENDER_SOURCE_SHA='05e748f9d0bc376e9fb1d98351de0d19d9a9c7c889978abcc9dfec2e7e66cf48'
CHOICE_CODES_SOURCE_SHA='2e41db1081038ff27be84e5970ba240c5669a697eeb397302c30bad984305b24'
# SHA-256 of the source text of every request-path function and constant outside the failure
# handling (Service, the fast/slow attempt wrappers) and the backend and counter constructors.
# A change to any of them changes inference and needs a new measurement.
REQUEST_PATH_SOURCE_SHA256={
    'core.normalize_question':'c6f6fb6e87f1d71203f74fff435ff1394fa3fb0437f9ef5ade99e55896da9dc8',
    'core.normalize_request':'fa178d048506c36ec630bb64ee562732f97ee869e814d9a8ec446e86037fa661',
    'core.letter_forms':'503e33c2798da62fccf229024bc5d3965fa6b6f0e0c16045b4d4e99616c75fe3',
    'core.digit_forms':'f943c38ae00fa636e42767aa3972bc4cf039b026b84dca815bea0b71f51d12d8',
    'core.parse_answer':'b11386d4f73b1040edbb0f91b9eecbdf59822200921428ba85fef7495f2be144',
    'core.readout_prefix':'ab7600ff57faacb0f37c8b0facfe9e363cecba0890c560668a57e453ea3e25b1',
    'core.logsumexp':'214f9818b14d8634008880ef6e9630cf75e194cd09a2d71bfb86169e1e071b88',
    'core.candidate_ids':'e7e59bc7e52b9a32f36a970500ca0a122cc20bbda7dd804e17266a8fe804ff51',
    'core.aggregate':'5d9665f0b2256a429cd00715a96612eee26bdbd15bdb1e36bb9ce44f1746bd81',
    'core.validate_calibration':'04fc6bac05a75ff67be1e8e0993a0cbcdbcd2e5a247be65e50e4ff8012df773b',
    'core.probabilities':'fb5f37582d24e8fa52a1c23907343fba4a64ec7cf94a1d1415dcef8616bd2ab6',
    'core.contains_rule_subtree':'2fa14bbfe49d4f24712cb2beceb6100f6f9897842700693bd5eb0292a2069d55',
    'core.lexical_terms':'5bdf52abfe196729d0461db3f9e60b5eef36dbf89e8ba62959a0418fdab655aa',
    'core.shrink_state':'ccba1165030cc9f98c5631596e7bc2ee4ead227db0625fa21004c9c21de108a3',
    'core.prepare_task':'c521abdc828e181927080e488314e7a58a72f2ff86f4a47c10f619756856f06a',
    'core.sequence_scores':'b487266864af7a7cbd3dae41aa5c7abe809b4344d9025e6e2e8f6f40e587c678',
    'core.slow_readout':'d2357763de5c8196c07e4e9fe9243931b7a7830a7effd267b079f45a9b901eff',
    'core.label_margin':'07ece3129c3b0a0e42dd0ead53309a9e969a2f6edd1ba9cd9ca070e84d55b1f2',
    'core.fast_readout':'712aaa56457a946dfc15b5ae546824ecb1e30d839701bd53eda53eaf8c84ac9b',
    'core.release_failed_session':'02cc4be648353b0ef6ae96c0e3d84dc216a0c9240e663bcd7d48ddcfd868f1ff',
    'core.threshold_identity':'54586ffc3cd34a919e59abadebfb7b4c50717db4065a838d987993f89049f62a',
    'core.validate_threshold':'a787da921d9cba38f83bd4ded71fa7045f20398e5a8613bc78d0fc6179bb8c29',
    'core.apply_profile':'22ca6432b71e1d2e81e9f2dc58fd6437def88aee08b260033ee7e9b80645bd29',
    'core.apply_budget_tier':'3e31c710515b97c28005a162219120636659293fdbc405e235d56f19c37fc967',
    'core.validate_config':'911148e6cce630ef441583ae36832e1adc610e8bc52292af977051058bfb404b',
    'core.finite_number':'2e6ee9d75ba16df4c6a0f34cf1f8811ecce79371678dfcac1f8c5fc46952be7c',
    'core.load_config':'378a1e37a44c2e8a0156c07ca590a53dc9fb511f7e16ae486fce00247823fcfd',
    'core.load_json':'f6ddeb333b13e52574e0d0487b1ff643ed20f1a78a79bc6ed41269fb998b12aa',
    'core.YES_FORMS':'48f99b50f17cb541ea35d3cb01ea4c3a7c445c22895eee7cb6cc1146c1c243da',
    'core.NO_FORMS':'2a76e716a4a7aeaecd91b753ee1c359b7af0f7952889299554ba82c59e492c04',
    'core.ALIASES':'846715bc67571c0974d5daf6c0f2abbf4841b6458efee18894fb9867f7772ba4',
    'core._RULE':'ddcc6f9960cb65df96d5e1a33fb2adc61be6ff6fc320eda507cf80191670532a',
    'core._RULE_KEY':'115e0222999981029d58230718d8dafb0c9e74233b411cf5fe35487aef76184a',
    'backends._candidate_ids':'b4880ee85f782b90c557de5321810dc6c0be40ec26d6fb4d42808a7dedda6690',
    'backends._max_new_tokens':'44d01d5155ddd07db0950af91ae74611c993824f4c24384ecab82051f0a49f42',
    'backends._check_scores':'78684b17f8bd6462176319ceb0bd57fd2338bb52e60fa5686a6e5b68b0bc01d7',
    'backends.generation_audit':'d959c664df3417f7feb65dc6613d39a9e4362b35d591e9bdba736ece9e9987a4',
    'backends.check_deadline':'35cd63a4e81971d772f72447bf4c66f1debe914feedede49dec77ccd1f01ae1e',
    'backends.new_meter':'29456891903438e09fc3fe796f3ae957eb0c3ad271adad127b00592b9fb6af2b',
    'backends.create_backend':'4e3363219b3747af33b70bca7fb8f1b3aea92fc958d77f04e3e062ed8b7a37da',
    'backends.TokenizerMixin.scan_digit_tokens':'9ebb2e6c7e4eafe794f68994b968ec3f780327772154bd7a95c4b5198182fedd',
    'backends.TokenizerMixin.encode':'0e2a221960670cae5030f8bc0908b8b1e21f4727313052070c3fd866b5235d5f',
    'backends.TokenizerMixin.decode':'4bbe705062225882ddd6b7fb987a44abca9a8cf5e5cb2485e58c0e7434ed3e8e',
    'backends.TokenizerMixin.prompt_ids':'06f3254bd4b891cdf36a95d332e6435720f520556a850b32ee55f25def1e631f',
    'backends.VLLMBackend._awq_quant_arg':'eb9bdfa1d9fe0734db1dad71d3b7972a51a7e7add6b62dc1262cc5c6b3fad908',
    'backends.VLLMBackend._completion':'228705e2e8d55a12d998e9479694301e92088eb04d21609cbec6075d77c2f2ce',
    'backends.VLLMBackend.next_logits':'9a68bf803e894f1bcfea23f32c064f59e15f7c25bd80c6a44dc7bdf2c2bf2529',
    'backends.VLLMBackend.open_session':'24734742d00197c6e0e4f0ee0c34c022bf6646dd31a80e160d99015a9d98f881',
    'backends.VLLMBackend.slow_session':'3f6fc107df781dd555b22a7ff496699e2f54e12145707a526f8a9f5ac56685fc',
    'backends.VLLMBackend.generate':'5950aae291520c0578eaa891785f5c3c26ad1511709b4082f1916c3ceca4f9b3',
    'backends.CachedSession.__init__':'f21f5facd479978677f6e939608734f864286dcee0b58a8d5458577d68fa759c',
    'backends.CachedSession.advance':'1e9fa1cab2863cf7630017558566dcf1445d1ca3bc99f6e7c9ee85b4e6eead8b',
    'backends.CachedSession.top_token':'5f975ce0d1154e619011661307befc2927ec4c6a84fc051035f2dd679e29cb82',
    'backends.CachedSession.log_probs':'5b2b6f969e391f14b9cbc2ff703bf59f310665900dcef647e9af4ba280239d0a',
    'backends.CachedSession.non_digit_logprob':'df579fdde820a187ccf7f2984935d3cc5e61c906a984b9342a9e6a1bda482cd5',
    'backends.CachedSession.fork':'2e3619c932c2c0a68f313811190fc237ebd96393bdfcbb12787e758409c93dab',
    'backends.VLLMPrefixSession.__init__':'44c359336c10380f6f181f6bf2e885746f6c60064ed5a08e2f6ed3ef899660db',
    'backends.VLLMPrefixSession.logits':'d31489f06e62895b98a834c4935d85db336bcae64636091a8d2f4f045b3eafc5',
    'backends.VLLMPrefixSession.log_probs':'84d075a30db2a4e7c0780e865112c5cef60b2dfaaebfc5a45aa51863ed66ecc5',
    'backends.VLLMPrefixSession.non_digit_logprob':'6b6528981387765e01913db4819a9391f96d647c943a647f849f14286f799ab9',
    'backends.VLLMPrefixSession.fork':'ede3dfd5d82be1f374f6e3f0db1d81f743c578ccee1331180c4cc26c7d392eb9',
    'backends.VLLMPrefixSession.advance':'076936d8f8a9271203db9ec0512a896a1471b1862d538406673c0153ff0b2cfe',
    'quota.cost_settings':'ec6c2b2e51c191b5e31d4fa11a7767f02356ed2c8b46bf50c9065fadc0429644',
    'quota.configure_cost':'f153abe9af3df7aad976e0869163d9acd0f0ccf7fd5537e07a1001a783324eaf',
    'quota.CostFuse.__init__':'a5ee03cb96abf129153673c87445559503447b4bfc86de0e71ee8af73e5d3683',
    'quota.CostFuse.state':'3d751be53513524b2425da5c5288579d96df5e052e6415315adac8c315ac39c7',
    'quota.CostFuse.quota':'d4dc3d718c6e987b4d70f5e508d9f72506bcf5d9a5dea7f8ec5b4696e6ee745f',
    'quota.CostFuse.record':'f790f752c13719e669c56b52fc36bb08847513ee561a2eccf7c171d0ff66f1ec',
    'quota.CostFuse.snapshot':'207012f7f0db48220def19210bebf712e06e868d0181f762ed4ecb61429fd9a0',
    'quota.QuotaCounter._read':'aa83122b3989560859c5c64d72bc8554736bd4bc77b344c8047411cfed49fe7a',
    'quota.QuotaCounter._write':'b2667f3782599733da229fd50f5c75014d7bd6f19b988b8d034fbe95bf9c8e04',
    'quota.QuotaCounter._snapshot':'f608d8257f6fd885f4b8ec45f4095c08a05ed47dc4660ce0f56f6d6813a4bb5f',
    'quota.QuotaCounter.configure_cost':'bd5f718a9fce942efd9809472e6c48c7595e43730e570610d689119ac9cdef72',
    'quota.QuotaCounter.accounting':'1ace5c093e8ee2c857ea861cf47a292553819e4277b9cb713199b68cbea38b26',
    'quota.QuotaCounter.record_cost':'a1dbad2188685c80b4547023a0539743750a88a299c22945035f0bc3090b89fd',
    'quota.QuotaCounter.decision':'ea27223a8503ca4d38dc9f49b72837490da847b3b04fce12be66ead108dd3a5c',
    'quota.RuntimeCounter.configure_cost':'0c98f2884b5736184974345ab7b6f840a7be1551b64af9a23e9a14e6b99ad282',
    'quota.RuntimeCounter.accounting':'23da23b7c0bd4bd8edf027d9777f135ee54d9ce1e9d3b2a192d5dd04482f6338',
    'quota.RuntimeCounter.record_cost':'0bbd0e1346a3893b07a332003c9d09b4dcb2ee911cbfa40f3a7bff5ee0c81061',
    'quota.RuntimeCounter.decision':'3f35f8606af754ecfe1945b1f76d7e8f6602b5f14879c6c4534ae6d4460ce492',
}
# SHA-256 of the canonical JSON (sorted keys) of each parameter file without its descriptive
# fitted_on, status and model_calibration_status strings: every number and setting is covered.
PARAMETER_SHA256={
    'config.json':'15997a9db2cd677059dca15ba279cd8c21894ea4a5e2e23a79bc9bc2f46b8da8',
    'calibration.json':'3fdab2baa38bb7c6fc7c341df676e58e6b4ab2509ad298a9df275cfcaf15cfdd',
    'fitted.release.json':'d15bb1bacba03c0d5d598885f67a75fd12967a4d4cf4ccc579270136eb37ffb2',
}
PARAMETER_LABELS=('fitted_on','status','model_calibration_status')


def parameter_digest(path):
    def strip(value):
        if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k not in PARAMETER_LABELS}
        if isinstance(value,list):return [strip(v) for v in value]
        return value
    data=strip(json.loads(Path(path).read_text(encoding='utf-8')))
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest()


# Synthetic stand-ins for vLLM's exception classes (matched by class name and module).
EngineDeadError=type('EngineDeadError',(RuntimeError,),{})
VLLMServerError=type('VLLMServerError',(RuntimeError,),{'__module__':'vllm.exceptions'})
EngineGenerateError=type('EngineGenerateError',(VLLMServerError,),{'__module__':'vllm.v1.engine.exceptions'})


def scratch():
    return Path(tempfile.mkdtemp(dir=SCRATCH))


def payload():
    return {'state':'test context','questions':{'decision':{'type':'boolean','instructions':'test question'}}}


def source_text(path, dotted):
    """Source lines of a top-level function, class method or assignment, decorators included."""
    text=Path(path).read_text(encoding='utf-8')
    body,node=ast.parse(text).body,None
    for part in dotted.split('.'):
        node=next(n for n in body if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==part)
                  or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==part for t in n.targets)))
        body=getattr(node,'body',[])
    first=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return '\n'.join(text.splitlines()[first-1:node.end_lineno])


def free_port():
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));return probe.getsockname()[1]


def can_symlink(directory):
    probe=directory/'symlink-probe'
    try:probe.symlink_to(directory,target_is_directory=True)
    except (OSError,NotImplementedError):return False
    return True


class NumericIdentity(unittest.TestCase):
    def test_prompt_sha_and_threshold_identity_unchanged(self):
        config=load_config()
        self.assertEqual(threshold_identity(config)['prompt_sha'],PROMPT_SHA)
        self.assertEqual(validate_threshold(config),[])
        self.assertEqual(hashlib.sha256(inspect.getsource(render).encode()).hexdigest(),RENDER_SOURCE_SHA)
        self.assertEqual(hashlib.sha256(inspect.getsource(choice_codes).encode()).hexdigest(),CHOICE_CODES_SOURCE_SHA)

    def test_parameter_files_unchanged(self):
        # Every value of the parameter files except the descriptive fitted_on and status strings; the
        # digests equal those of the same files at commit 06428f09559a0a0536776c91cd544b12b68176f3.
        changed=[name for name,expected in PARAMETER_SHA256.items() if parameter_digest(launch.ROOT/name)!=expected]
        self.assertEqual(changed,[],'parameter values changed; a new measurement is required')

    def test_request_path_source_unchanged(self):
        changed=[]
        for name,expected in REQUEST_PATH_SOURCE_SHA256.items():
            module,dotted=name.split('.',1)
            digest=hashlib.sha256(source_text(launch.ROOT/(module+'.py'),dotted).encode('utf-8')).hexdigest()
            if digest!=expected:changed.append(name)
        self.assertEqual(changed,[],'request-path source changed; a new measurement is required')


class StartupGuards(unittest.TestCase):
    def test_python_gate(self):
        launch.check_python((3,12));launch.check_python((3,10))
        with self.assertRaisesRegex(ValueError,'Python >= 3.10'):launch.check_python((3,9))
        with patch.object(launch.sys,'version_info',(3,12,3)):launch.check_python()

    def test_exact_version_gate(self):
        exact={name:version+'+cu130' for name,version in launch.VERSIONS.items()}
        with patch('launch.importlib.metadata.version',side_effect=lambda n:exact[n]):
            self.assertEqual(set(launch.check_versions()),set(launch.VERSIONS))
        newer=dict(exact,vllm='0.31.1')
        with patch('launch.importlib.metadata.version',side_effect=lambda n:newer[n]):
            with self.assertRaisesRegex(ValueError,'vllm 0.31.1 found, 0.31.0 required'):launch.check_versions()
        def missing(name):
            if name=='compressed-tensors':raise launch.importlib.metadata.PackageNotFoundError(name)
            return exact[name]
        with patch('launch.importlib.metadata.version',side_effect=missing):
            with self.assertRaisesRegex(ValueError,re.escape(launch.IMAGE)):launch.check_versions()

    def test_paths_absolute_and_separate(self):
        base=scratch();model=base/'model';state=base/'state'
        self.assertEqual(launch.check_paths(model,state),(model.resolve(),state.resolve()))
        # '..' components that lead back into the package; a string-prefix check would miss them.
        escape=launch.ROOT.parent/'sibling'/'..'/launch.ROOT.name
        self.assertFalse(str(escape).startswith(str(launch.ROOT)))
        self.assertEqual(escape.resolve(),launch.ROOT.resolve())
        for bad_model,bad_state in ((Path('model'),state),(model,Path('state')),
                                    (model,launch.ROOT/'state'),(model,launch.ROOT.parent),
                                    (launch.ROOT/'model',state),(model,model/'state'),(state/'model',state),
                                    (model,escape/'state'),(escape/'model',state),(model,escape/'..'),
                                    (base/'x'/'..'/'model',base/'model'/'..'/'model'/'s')):
            with self.subTest(model=bad_model,state=bad_state),self.assertRaises(ValueError):
                launch.check_paths(bad_model,bad_state)
        for bad in (escape/'out','relative/../x',escape/'x'/'..'/'..'):
            with self.subTest(external=bad),self.assertRaises(ValueError):external_output(bad)

    def test_symlink_escapes_are_refused(self):
        base=scratch()
        if not can_symlink(base):self.skipTest('symbolic links unavailable on this system')
        link=base/'pkg-link';link.symlink_to(launch.ROOT,target_is_directory=True)
        for bad_model,bad_state in ((base/'m',link/'state'),(link/'model',base/'s'),(base/'m',link)):
            with self.subTest(model=bad_model,state=bad_state),self.assertRaises(ValueError):
                launch.check_paths(bad_model,bad_state)
        for bad in (link/'out',link):
            with self.subTest(external=bad),self.assertRaises(ValueError):external_path(bad)
        # A pre-existing state/cache link to elsewhere is refused before anything is written.
        state=base/'state';state.mkdir();elsewhere=base/'elsewhere';elsewhere.mkdir()
        (state/'cache').symlink_to(elsewhere,target_is_directory=True)
        with patch.object(launch,'MAX_TMP_PATH',4096),self.assertRaisesRegex(ValueError,'not a link'):
            launch.prepare_state(state,'fresh')
        self.assertFalse((state/'fresh-startup').exists())
        # Quota state: a run directory or quota file redirected by a link is refused.
        config=copy.deepcopy(load_config())
        target=base/'target';target.mkdir()
        (state/'linked').symlink_to(target,target_is_directory=True)
        config['routing'].update(state_dir=str(state),run_id='linked',resume=True)
        with self.assertRaisesRegex(quota.QuotaStateError,'inside the state directory'):quota.RuntimeCounter(config)
        run=state/'resumed';run.mkdir();(run/'quota.json').symlink_to(target/'precious.json')
        config['routing'].update(run_id='resumed')
        with self.assertRaisesRegex(quota.QuotaStateError,'inside the run directory'):quota.RuntimeCounter(config)
        self.assertEqual(list(target.iterdir()),[])

    def test_model_identity_is_read_only(self):
        model=scratch()
        files={'config.json':b'{}','tokenizer.json':b'tok','chat_template.jinja':b'tpl'}
        shards={'model-00001-of-00001.safetensors':b'0123456789'}
        for name,data in {**files,**shards}.items():(model/name).write_bytes(data)
        for name in ('README.md','.gitattributes','WEIGHTS.sha256','PREPARATION.json'):(model/name).write_bytes(b'x')
        (model/'.cache').mkdir()
        hashes={n:hashlib.sha256(d).hexdigest() for n,d in files.items()}
        sizes={n:len(d) for n,d in shards.items()}
        full={n:hashlib.sha256(d).hexdigest() for n,d in shards.items()}
        before={p.name:(p.stat().st_mtime_ns,p.read_bytes()) for p in model.iterdir() if p.is_file()}
        with patch.dict(launch.MODEL_SHA256,hashes,clear=True),patch.dict(launch.SHARD_BYTES,sizes,clear=True),patch.dict(launch.SHARD_SHA256,full,clear=True):
            launch.check_model(model);launch.check_model(model,full=True)
            (model/'tokenizer.json').write_bytes(b'other')
            with self.assertRaisesRegex(ValueError,'tokenizer.json'):launch.check_model(model)
            (model/'tokenizer.json').write_bytes(b'tok')
            with patch.dict(launch.SHARD_BYTES,{'model-00001-of-00001.safetensors':11}):
                with self.assertRaisesRegex(ValueError,'shard size'):launch.check_model(model)
            with patch.dict(launch.MODEL_SHA256,{'processor_config.json':'0'*64}):
                with self.assertRaisesRegex(ValueError,'Missing model file: processor_config.json'):launch.check_model(model)
            # Files a loader may read and that are not part of the pinned revision are refused.
            for extra in ('preprocessor_config.json','added_tokens.json','special_tokens_map.json','tokenizer.model',
                          'model.safetensors','pytorch_model.bin','merges.txt','other.jinja'):
                with self.subTest(extra=extra):
                    other=scratch()
                    for name,data in {**files,**shards}.items():(other/name).write_bytes(data)
                    (other/extra).write_bytes(b'{}')
                    with self.assertRaisesRegex(ValueError,'Unexpected file in model folder: '+re.escape(extra)):
                        launch.check_model(other)
            if can_symlink(scratch()):
                rule=re.escape('MODEL must contain regular files (a Hugging Face cache snapshots/<sha> directory cannot be mounted alone)')
                linked,blobs=scratch(),scratch()
                for name,data in {**files,**shards}.items():
                    (blobs/name).write_bytes(data);(linked/name).symlink_to(blobs/name)
                with self.assertRaisesRegex(ValueError,'^Model file is a symbolic link: config.json; '+rule):launch.check_model(linked)
                # A snapshot directory mounted alone: its relative links into ../../blobs dangle.
                dangling=scratch()
                for name in {**files,**shards}:(dangling/name).symlink_to(Path('..')/'..'/'blobs'/name)
                with self.assertRaisesRegex(ValueError,'^Model file is a dangling or looping symbolic link: config.json; '+rule):
                    launch.check_model(dangling)
                looping=scratch()
                for name,data in {**files,**shards}.items():
                    if name!='tokenizer.json':(looping/name).write_bytes(data)
                (looping/'loop').symlink_to(looping/'tokenizer.json');(looping/'tokenizer.json').symlink_to(looping/'loop')
                with self.assertRaisesRegex(ValueError,'^Model file is a dangling or looping symbolic link: tokenizer.json; '+rule):
                    launch.check_model(looping)
                self.assertNotIn('state filesystem',launch.MODEL_FOLDER_RULE)
        after={p.name:(p.stat().st_mtime_ns,p.read_bytes()) for p in model.iterdir() if p.is_file()}
        self.assertEqual(set(after),set(before))
        self.assertEqual({k:v[1] for k,v in after.items()},{k:v[1] for k,v in before.items()})

    def test_pinned_model_constants(self):
        self.assertEqual(set(launch.SHARD_BYTES),set(launch.SHARD_SHA256))
        self.assertTrue(all(re.fullmatch('[0-9a-f]{64}',h) for h in [*launch.MODEL_SHA256.values(),*launch.SHARD_SHA256.values()]))
        self.assertIn('12deda07c573047b191b63f0768c0412120f3ed4',launch.MODEL_REVISION)
        # Values of the Hugging Face files at that revision (raw small files, LFS object IDs of the shards).
        self.assertEqual(launch.MODEL_SHA256['chat_template.jinja'],'36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0')
        self.assertEqual(launch.MODEL_SHA256['model.safetensors.index.json'],'7c48224f0d8b11a4eb3d0be6bac8fab7069223ca3ac96a259c2b649945c88332')
        self.assertEqual(launch.SHARD_BYTES['model-00001-of-00003.safetensors'],5368362132)

    @patch.object(launch,'MAX_TMP_PATH',4096)  # scratch paths can be longer than a deployment path
    def test_state_claim_is_exclusive(self):
        state=scratch()/'state'
        rid,startup=launch.prepare_state(state,'fresh')
        self.assertEqual((rid,startup),('fresh',state/'fresh-startup'))
        self.assertTrue(all((startup/n).is_dir() for n in ('tmp','home')) and (state/'cache').is_dir())
        self.assertEqual((startup/'attempts.jsonl').read_text(),'')
        with self.assertRaisesRegex(ValueError,'run_id already exists in --state-dir; use a new --state-dir'):
            launch.prepare_state(state,'fresh')
        (state/'used').mkdir()
        with self.assertRaisesRegex(ValueError,'run_id already exists'):launch.prepare_state(state,'used')
        rid,_=launch.prepare_state(state)
        self.assertRegex(rid,r'^run-\d{8}T\d{12}Z$')
        other=scratch()/'state';other.mkdir();(other/'cache').write_bytes(b'')
        with self.assertRaisesRegex(ValueError,'must be a directory'):launch.prepare_state(other,'fresh')

    def test_run_id_and_socket_path_rules(self):
        state=scratch()
        for rid in ('..','a/b','a'+chr(92)+'b','-x','.x','x'*65,'a b'):
            with self.subTest(rid=rid),self.assertRaises(ValueError):launch.prepare_state(state,rid)
        with self.assertRaisesRegex(ValueError,'too long'):launch.prepare_state(state/('d'*80),'fresh')
        self.assertFalse((state/('d'*80)).exists())
        # The README paths fit: /work/state/run1-startup/tmp.
        self.assertLessEqual(len(os.fsencode('/work/state/run1-startup/tmp')),launch.MAX_TMP_PATH)

    @patch.object(launch,'MAX_TMP_PATH',4096)
    def test_unwritable_state(self):
        state=scratch()
        with patch('launch.Path.mkdir',side_effect=PermissionError),self.assertRaisesRegex(ValueError,'State directory is not writable'):
            launch.prepare_state(state/'new','fresh')

    def test_child_environment(self):
        state=scratch();startup=state/'r-startup'
        base={'VLLM_ATTENTION_BACKEND':'FLASH_ATTN','JEV_ATTENTION_BACKEND':'X','HOME':'elsewhere',
              'HF_HOME':str(state/'cache'/'own-hf')}
        env=launch.child_environment(base,state,startup)
        self.assertNotIn('VLLM_ATTENTION_BACKEND',env);self.assertNotIn('JEV_ATTENTION_BACKEND',env)
        self.assertEqual((env['HOME'],env['TMPDIR']),(str(startup/'home'),str(startup/'tmp')))
        self.assertEqual(env['VLLM_HOST_IP'],'127.0.0.1')
        for key in ('VLLM_USE_FLASHINFER_SAMPLER',):self.assertEqual(env[key],'0')
        for key in ('VLLM_NO_USAGE_STATS','DO_NOT_TRACK','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE'):
            self.assertEqual(env[key],'1')
        self.assertEqual(env['HF_HOME'],str(state/'cache'/'own-hf'))  # inside --state-dir: kept
        for key in ('XDG_CACHE_HOME','VLLM_CACHE_ROOT','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','NUMBA_CACHE_DIR'):
            self.assertTrue(Path(env[key]).is_relative_to(state/'cache'),key)
        # Any inherited cache or configuration path outside --state-dir is refused.
        for key in launch.PATH_VARIABLES:
            for value in ('relative',str(scratch()),str(state)+'/../outside'):
                with self.subTest(key=key,value=value):
                    with self.assertRaisesRegex(ValueError,key):launch.check_environment({key:value},state)
                    with self.assertRaisesRegex(ValueError,key):launch.child_environment({key:value},state,startup)
        launch.check_environment({'HF_HOME':str(state/'cache'/'own-hf'),'PATH':'/usr/bin'},state)

    @patch.object(launch,'MAX_TMP_PATH',4096)
    def test_environment_refused_before_state_claim(self):
        # A refused path variable claims nothing, so the same state directory (README: the same WORK)
        # can be used again once the variable is removed.
        model=scratch()/'model';model.mkdir();state=scratch()/'state'
        outside=str(scratch()/'hf')
        with patch('launch.check_python'),patch('launch.check_versions'),patch('launch.check_model') as check_model, \
             patch('launch.start_attempts') as attempts, \
             patch.dict(os.environ,{'PATH':os.environ.get('PATH',''),'HF_HOME':outside},clear=True):
            with self.assertRaisesRegex(ValueError,'HF_HOME=.* is outside --state-dir'):
                launch.start(model,free_port(),state_dir=state,run_id='run1')
        check_model.assert_not_called();attempts.assert_not_called()
        self.assertFalse(state.exists())
        source=inspect.getsource(launch.start)
        self.assertLess(source.index('check_environment(os.environ,state)'),source.index('prepare_state(state,run_id)'))

    def test_bind_choices(self):
        self.assertEqual(launch.BINDS,('127.0.0.1','0.0.0.0'))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):launch.main(['--model',str(scratch()),'--state-dir',str(scratch()),'--bind','localhost'])
            with self.assertRaises(SystemExit):launch.main(['--model',str(scratch())])

    def test_start_wiring(self):
        model=scratch()/'model';model.mkdir();(model/'processor_config.json').write_text('{}')
        state=scratch()/'state'
        port=free_port()
        calls=[]
        self.addCleanup(setattr,tempfile,'tempdir',tempfile.tempdir)
        with patch('launch.check_python',side_effect=lambda:calls.append('python')), \
             patch('launch.check_versions',side_effect=lambda:calls.append('versions')), \
             patch('launch.check_model',side_effect=lambda m:calls.append(('model',m))), \
             patch('launch.start_attempts',return_value='proc') as attempts, \
             patch.dict(os.environ,{'PATH':os.environ.get('PATH','')},clear=True), \
             patch.object(launch,'MAX_TMP_PATH',4096):
            self.assertEqual(launch.start(model,port,state_dir=state,run_id='fresh'),'proc')
        self.assertEqual(calls,['python','versions',('model',model.resolve())])
        args,env,startup,used_port,mode,no_cache=attempts.call_args.args
        self.assertEqual((startup,used_port,mode,no_cache),(state.resolve()/'fresh-startup',port,'routed',False))
        pairs=dict(zip(args[3::2],args[4::2]))
        self.assertEqual(args[2],str(launch.ROOT/'server.py'))
        self.assertEqual((pairs['--quant'],pairs['--backend'],pairs['--model-key'],pairs['--bind']),('awq','vllm','gemma-4-12b-awq','127.0.0.1'))
        self.assertEqual((pairs['--model'],pairs['--state-dir'],pairs['--run-id']),(str(model.resolve()),str(state.resolve()),'fresh'))
        self.assertEqual(env['HOME'],str(startup/'home'))
        self.assertEqual(inspect.signature(launch.start_attempts).parameters['timeout'].default,1800)
        self.assertEqual(launch.STARTUP_TIMEOUT,1800)
        self.assertEqual(launch.ATTEMPTS,(None,'TRITON_ATTN'))
        # Unfinalized parameters are refused for routed serving before any state is written.
        config=json.loads((launch.ROOT/'config.json').read_text(encoding='utf-8'));config['package']['finalized']=False
        draft=scratch()/'config.json';draft.write_text(json.dumps(config),encoding='utf-8')
        with patch('launch.check_python'),patch('launch.check_versions'),patch('launch.check_model'), \
             patch('launch.start_attempts') as attempts,self.assertRaisesRegex(ValueError,'finalized'):
            launch.start(model,port,config=draft,state_dir=scratch()/'s2',run_id='fresh')
        attempts.assert_not_called()

    def test_main_registers_signals_and_stops(self):
        handlers={}
        def register(number,handler):handlers[number]=handler
        proc=Mock();proc.wait.return_value=0
        with patch('launch.signal.signal',side_effect=register),patch('launch.start',return_value=proc), \
             patch('launch.stop') as stop:
            self.assertEqual(launch.main(['--model','/m','--state-dir','/s']),0)
        expected={getattr(launch.signal,n) for n in ('SIGTERM','SIGINT','SIGHUP') if hasattr(launch.signal,n)}
        self.assertEqual(set(handlers),expected)
        self.assertEqual(len({id(h) for h in handlers.values()}),1)
        with self.assertRaises(KeyboardInterrupt):next(iter(handlers.values()))(None,None)
        stop.assert_called_once_with(proc)
        with patch('launch.signal.signal'),patch('launch.start',side_effect=ValueError('Missing model file: x')), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(launch.main(['--model','/m','--state-dir','/s']),1)
        self.assertIn('launch: error: Missing model file: x',err.getvalue())

    @staticmethod
    def fake_signals():
        """A stand-in signal module that records installed handlers; the previous handler is 'old'."""
        installed={}
        def register(number,handler):
            previous=installed.get(number,'old');installed[number]=handler;return previous
        fake=SimpleNamespace(SIGTERM=15,SIGKILL=9,SIGINT=2,SIGHUP=1,signal=Mock(side_effect=register))
        return fake,installed

    def test_stop_always_kills_group(self):
        fake,installed=self.fake_signals()
        sent=[]
        proc=Mock();proc.pid=4242
        proc.wait.side_effect=[KeyboardInterrupt(),0]  # the first wait is interrupted by an exception
        with patch.object(launch,'signal',fake),patch.object(launch.os,'killpg',side_effect=lambda pid,sig:sent.append((pid,sig)),create=True):
            with self.assertRaises(KeyboardInterrupt):launch.stop(proc)
        self.assertEqual(sent,[(4242,15),(4242,9)])
        self.assertEqual(proc.wait.call_count,2)
        self.assertEqual([c.kwargs.get('timeout') for c in proc.wait.call_args_list],[30,launch.KILL_WAIT])
        held=[c.args for c in fake.signal.call_args_list if callable(c.args[1])]
        self.assertEqual(sorted(n for n,_ in held),[1,2,15])
        self.assertEqual(installed,{1:'old',2:'old',15:'old'})  # previous handlers restored

    def test_signal_during_stop_is_raised_afterwards(self):
        # A stop request that arrives while the group is being stopped is neither lost nor allowed to
        # skip SIGKILL: it is raised as KeyboardInterrupt once the sequence is complete.
        fake,installed=self.fake_signals()
        sent=[];timeouts=[]
        proc=Mock();proc.pid=4242
        def wait(timeout=None):
            timeouts.append(timeout)
            if len(timeouts)==1:
                installed[15](15,None)  # SIGTERM arrives during the first wait
                raise launch.subprocess.TimeoutExpired('server',timeout)
            return -9
        proc.wait.side_effect=wait
        with patch.object(launch,'signal',fake),patch.object(launch.os,'killpg',side_effect=lambda pid,sig:sent.append((pid,sig)),create=True):
            with self.assertRaises(KeyboardInterrupt):launch.stop(proc)
            self.assertEqual(sent,[(4242,15),(4242,9)])
            self.assertEqual(timeouts,[30,launch.KILL_WAIT])
            self.assertEqual(installed,{1:'old',2:'old',15:'old'})
            # Without a signal, stop returns normally; a group that outlives SIGKILL is reported, not awaited forever.
            sent.clear();proc.wait.side_effect=[launch.subprocess.TimeoutExpired('server',30),launch.subprocess.TimeoutExpired('server',60)]
            with contextlib.redirect_stderr(io.StringIO()) as err:launch.stop(proc)
        self.assertEqual(sent,[(4242,15),(4242,9)])
        self.assertIn(f'still running {launch.KILL_WAIT} s after SIGKILL',err.getvalue())
        self.assertEqual(launch.KILL_WAIT,60)

    def test_main_returns_after_interrupted_final_stop(self):
        proc=Mock();proc.wait.return_value=0
        with patch('launch.signal.signal'),patch('launch.start',return_value=proc), \
             patch('launch.stop',side_effect=KeyboardInterrupt) as stop:
            self.assertEqual(launch.main(['--model','/m','--state-dir','/s']),0)
        stop.assert_called_once_with(proc)

    def test_launch_identity_in_server_arguments(self):
        tree=ast.parse((launch.ROOT/'launch.py').read_text(encoding='utf-8'))
        args=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=='args' for target in n.targets))
        pairs={a.value:b for a,b in zip(args.elts,args.elts[1:]) if isinstance(a,ast.Constant)}
        self.assertEqual(pairs['--quant'].value,'awq');self.assertEqual(pairs['--backend'].value,'vllm')
        self.assertEqual(pairs['--model-key'].id,'MODEL_KEY');self.assertEqual(launch.MODEL_KEY,'gemma-4-12b-awq')
        text=(launch.ROOT/'prepare_model.sh').read_text(encoding='utf-8')
        compile(text.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],'prepare_model.sh embedded Python','exec')

    def test_compressed_tensors_auto_detection(self):
        from backends import VLLMBackend
        model=scratch();cfg={'path':str(model),'quant':'awq'}
        for method in ('compressed-tensors','compressed_tensors'):
            (model/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':method}}),encoding='utf-8')
            self.assertIsNone(VLLMBackend._awq_quant_arg(cfg))
        (model/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'awq'}}),encoding='utf-8')
        self.assertEqual(VLLMBackend._awq_quant_arg(cfg),'awq')
        self.assertIsNone(VLLMBackend._awq_quant_arg({'path':str(model),'quant':'bf16'}))

    def test_shell_scripts_use_python3(self):
        for script in launch.ROOT.glob('*.sh'):
            text=script.read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'(?m)(^|[\s;&|(])python([\s;]|$)',text),script.name)
            self.assertNotIn('VLLM_ATTENTION_BACKEND',text)
        self.assertIn('PY="${PYTHON:-python3}"',(launch.ROOT/'serve.sh').read_text(encoding='utf-8'))

    def test_health_contract(self):
        healthy={'prefix_caching':True,'supports_slow_session':True,'warnings':[]}
        launch.validate_health(healthy)
        launch.validate_health(dict(healthy,prefix_caching=False),no_cache=True)
        for changes in ({'prefix_caching':False},{'supports_slow_session':False},
                        {'warnings':['quota persistence unavailable: memory counts, fast only']},
                        {'warnings':['calibration placeholder: refit required']},{'status':'warning'},{'status':'error'}):
            with self.subTest(changes=changes),self.assertRaises(RuntimeError):
                launch.validate_health(dict(healthy,**changes))
        for field in ('prefix_caching','supports_slow_session'):
            bad=dict(healthy);del bad[field]
            with self.assertRaises(RuntimeError):launch.validate_health(bad)


class QuotaState(unittest.TestCase):
    def test_direct_counter_startup_is_fatal(self):
        config=copy.deepcopy(load_config())
        config['routing'].update(state_dir=str(scratch()),run_id='fresh')
        with patch('quota.QuotaCounter',side_effect=PermissionError):
            with self.assertRaisesRegex(quota.QuotaStateError,'refusing startup'):quota.RuntimeCounter(config)
        for state_dir in (None,'relative/state'):
            config['routing'].update(state_dir=state_dir,run_id='other')
            with self.subTest(state_dir=state_dir),self.assertRaisesRegex(quota.QuotaStateError,'required'):
                quota.RuntimeCounter(config)
        config['routing'].update(state_dir=str(launch.ROOT/'state'),run_id='other')
        with self.assertRaisesRegex(quota.QuotaStateError,'separate from the package'):quota.RuntimeCounter(config)
        with self.assertRaisesRegex(quota.QuotaStateError,'absolute'):quota.QuotaCounter('relative.json',quota=.15,fuse=.2)

    def test_server_requires_state_dir(self):
        import server
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):server.main(['--model',str(scratch())])
            with self.assertRaises(SystemExit):server.main(['--model',str(scratch()),'--state-dir','relative'])


class OutputPaths(unittest.TestCase):
    def test_external_paths(self):
        out=scratch()/'run'/'prefix'
        self.assertEqual(external_output(out),out.resolve());self.assertTrue(out.parent.is_dir())
        for bad in ('relative/prefix',launch.ROOT/'prefix',launch.ROOT,launch.ROOT.parent,launch.ROOT/'..'/launch.ROOT.name/'x'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):external_path(bad)

    def test_public_output_and_count_guards(self):
        root=scratch();public=root/'public';public.mkdir()
        for filename in ('easy.jsonl','original.jsonl','hard.jsonl'):
            (public/filename).write_text('',encoding='utf-8',newline='\n')
        with self.assertRaises(ValueError):prepare_public(public,launch.ROOT/'invalid-output')
        with self.assertRaises(ValueError):prepare_public(public,public/'output')
        out=root/'output'
        with self.assertRaisesRegex(ValueError,'231'):prepare_public(public,out)
        self.assertFalse(out.exists())

    def test_selftest_bench_reconcile_chain(self):
        import bench,reconcile_usage
        root=scratch();public=root/'public';public.mkdir()
        for filename,tier,count in (('easy.jsonl','easy',48),('original.jsonl','standard',72),('hard.jsonl','hard',111)):
            rows=[{'id':tier+str(i),'state':'test context','question':{'type':'boolean','instructions':'test question'},'expected':True} for i in range(count)]
            (public/filename).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
        out,index=prepare_public(public,root/'output')
        rows=[json.loads(line) for line in index.read_text().splitlines()]
        self.assertEqual(len(rows),231);self.assertTrue(all(set(row)=={'id','type','tier'} for row in rows))
        self.assertEqual(rows[0],{'id':'easy0','type':'noul','tier':'easy'})
        with self.assertRaises(FileExistsError):prepare_public(public,out)
        rows_path,summary_path=bench.output_paths(out/'fast')
        self.assertEqual((rows_path.parent,summary_path.name),(out,'fast.summary.json'))
        self.assertEqual(bench.output_paths(out/'margin',calibration_only=True).name,'margin.margin_calibration.json')
        self.assertEqual(reconcile_usage.resolve(out/'fast'),(out/'fast').resolve())
        for bad in (launch.ROOT/'fast','fast'):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):bench.output_paths(bad)
                with self.assertRaises(ValueError):reconcile_usage.resolve(bad)
                with self.assertRaises(ValueError):prepare_public(public,bad)
        with self.assertRaises(ValueError):prepare_public(public,public/'output')

    def test_freeze_rejects_runtime_output(self):
        for name in ('selftest-run','artifact.pyc','artifact.tmp','debug.log'):
            root=scratch();(root/name).touch()
            (root/'.gitignore').write_bytes((launch.ROOT/'.gitignore').read_bytes())  # *.log is ignored by git
            with self.assertRaises(ValueError):freeze(root)
            self.assertFalse((root/'MANIFEST.sha256').exists())


class PackageScan(unittest.TestCase):
    """pkg_check holds generic rules only; fixtures are assembled at run time because the package
    scan reads this file as text."""
    def test_generic_text_rules(self):
        import pkg_check
        email='x'+'@'+'example.com';address='203.0.113'+'.5';short='1c2d'+'3e4f'
        for text in ('contact '+email,'host '+address,'ref '+short,'TO'+'DO later','FIX'+'ME',
                     'ssh '+'host','/ho'+'me/name/x'):
            with self.subTest(text=text):self.assertTrue(pkg_check.text_problems('f.md',text))
        version='.'.join('1234')  # a four-part package version is not an address
        for text in ('bind 127.0.0.1 or 0.0.0.0','sha256 '+'ab'*32,'commit '+'0a'*20,'package '+version,'plain words'):
            with self.subTest(text=text):self.assertEqual(pkg_check.text_problems('f.md',text,{version}),[])
        self.assertTrue(pkg_check.text_problems('f.md','package '+version))

    def test_document_rules(self):
        import pkg_check
        marker='[[FACTS-'+'PENDING: S6-FINAL a value]]'
        self.assertEqual(pkg_check.document_text_problems('DISCLOSURE.md','Accuracy '+marker+'.'),
                         ['DISCLOSURE.md: 1 [[FACTS-PENDING]] marker(s) remain'])
        pinned='https://github.com/o/r/tree/'+'0a'*20
        self.assertEqual(pkg_check.document_text_problems('README.md','Code: '+pinned+'.'),[])
        for name,text in (('DISCLOSURE.md','https://github.com/o/r/tree/main'),('DISCLOSURE.md','huggingface.co/o/m'),
                          ('DISCLOSURE.md','a place'+'holder value'),('DISCLOSURE.md','still pen'+'ding'),
                          ('DISCLOSURE.md','commit '+'1c2d'+'3e4f'),('README.md','an option'+'al step')):
            with self.subTest(text=text):self.assertTrue(pkg_check.document_text_problems(name,text))
        self.assertEqual(pkg_check.document_text_problems('DISCLOSURE.md','an option'+'al step'),[])

    def test_no_specific_denylist_in_package(self):
        import pkg_check
        for name in ('SCAN_ORDER_SHA256','SCAN_COMMERCIAL','SCAN_INTERNAL','commercial_findings'):
            self.assertFalse(hasattr(pkg_check,name),name)
        # The only full-length hex values in pkg_check are the fixed public identifiers it checks.
        known={pkg_check.PROMPT_SHA,pkg_check.JEVBENCH_COMMIT,*pkg_check.PUBLIC_DATA_SHA256.values(),
               pkg_check.IMAGE_INDEX.split(':')[1],pkg_check.IMAGE_CONFIG.split(':')[1]}
        source=(launch.ROOT/'pkg_check.py').read_text(encoding='utf-8')
        self.assertLessEqual(set(re.findall(r'(?<![0-9a-f])[0-9a-f]{40,64}(?![0-9a-f])',source)),known)

    def test_deletion_scan_rules(self):
        import pkg_check
        for source in ('Path(a).replace(b)','Path(a).rename(b)','target.replace(other)','shutil.rmtree(d)','os.remove(p)',
                       "subprocess.run(['rm','-rf',d])",'p.unlink()'):
            with self.subTest(source=source):self.assertTrue(pkg_check.python_deletions('probe source',ast.parse(source)))
        for source in ("text.replace('a','b')","'a-b'.replace('-')",'now.replace(tzinfo=None)'):
            with self.subTest(source=source):self.assertEqual(pkg_check.python_deletions('probe source',ast.parse(source)),[])
        for line in ('rmdir "$D"','unlink "$F"','truncate -s 0 "$F"','shred "$F"','rm -rf "$W"'):
            with self.subTest(line=line):self.assertRegex(line,pkg_check.SCAN_SHELL_DELETION)
        self.assertEqual(pkg_check.deletion_findings(launch.ROOT),[])
        self.assertEqual({name for name,_ in pkg_check.ALLOWED_FILE_CALLS},{'quota.py','finalize.py'})


class FailureSemantics(unittest.TestCase):
    """HTTP 422 for one item, HTTP 503 for the service; never a substitute answer."""
    sequence=0

    def serve(self,service):
        from server import QuietHTTPServer,make_handler
        httpd=QuietHTTPServer(('127.0.0.1',0),make_handler(service))
        thread=threading.Thread(target=httpd.serve_forever,kwargs={'poll_interval':0.01},daemon=True);thread.start()
        self.addCleanup(httpd.server_close);self.addCleanup(thread.join,3);self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    def request(self,port,method='POST',path='/v1/systemone',body=None):
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
        try:
            data=json.dumps(payload() if body is None else body).encode() if method=='POST' else None
            conn.request(method,path,body=data,headers={'Content-Type':'application/json'})
            response=conn.getresponse();return response.status,json.loads(response.read())
        finally:conn.close()

    def service(self,error=None,*,slow=False):
        """A real Service whose backend raises `error` when the prompt is tokenized."""
        type(self).sequence+=1
        class FailedBackend:
            def __init__(self,error):self.error,self.calls=error,0
            def prompt_ids(self,*_):
                self.calls+=1;raise self.error
            def slow_session(self,*_):raise AssertionError('unexpected unpatched slow readout')
        config=copy.deepcopy(load_config())
        config['routing'].update(state_dir=str(SCRATCH/'failures'),run_id='case-'+str(self.sequence),mode='fast_only')
        if slow:config['routing'].update(mode='uniform',quota=1,fuse=1)
        calibration=json.loads((launch.ROOT/'calibration.json').read_text(encoding='utf-8'))
        backend=FailedBackend(error or RuntimeError('synthetic item failure'))
        return Service(config,calibration,backend,quota.RuntimeCounter(config))

    @contextlib.contextmanager
    def fast_success(self):
        length={key:False for key in ('truncated','character_precompressed','protected_content_exceeds_budget','physical_truncated')}
        length['original_prompt_tokens']=1
        with patch('core.prepare_task',return_value=(None,(None,{},[1],[1]),length)), \
             patch('core.fast_attempt_with_retry',return_value=({'no':0.,'yes':-1.},0,None)):
            yield

    def assert_failure(self,actual,code):
        self.assertEqual(actual,(code,{'error':'service failure' if code==503 else 'item could not be processed'}))

    @staticmethod
    def wrapped(exc):
        error=BackendError('synthetic wrapped backend failure');error.__cause__=exc
        return error

    def test_limit_is_three(self):
        self.assertEqual(ITEM_FAILURE_LIMIT,3)

    def test_http_status_mapping(self):
        errors=[]
        def answer(_):
            raise errors.pop(0)
        service=SimpleNamespace(config=load_config(),answer=answer)
        port=self.serve(service)
        errors.append(ItemFailure('x'))
        with contextlib.redirect_stdout(io.StringIO()):self.assert_failure(self.request(port),422)
        for exc in (ServiceFailure('x'),RuntimeError('x'),KeyError('x'),quota.QuotaStateError('x')):
            errors.append(exc)
            with contextlib.redirect_stdout(io.StringIO()):self.assert_failure(self.request(port),503)
        # A non-finite value reaching the HTTP layer is a service fault; items are checked earlier (422).
        service.answer=lambda _:{'answers':{},'usage':{'input_tokens':float('nan')}}
        with contextlib.redirect_stdout(io.StringIO()):self.assert_failure(self.request(port),503)
        self.assertEqual(self.request(port,body={'state':'x'})[0],400)

    def test_exception_graph_and_vllm_classes(self):
        fatal=self.wrapped(EngineDeadError('engine stopped'))
        branch=RuntimeError('wrapper with both links');branch.__cause__=RuntimeError('recoverable cause')
        branch.__context__=fatal;fatal.__context__=branch
        server_subclass=type('WorkerFailure',(VLLMServerError,),{})
        for error in (fatal,branch,server_subclass('fatal'),ServiceFailure('fatal')):
            with self.subTest(error=type(error).__name__):self.assertTrue(core.unrecoverable_engine_error(error))
        cycle=RuntimeError('cycle');cycle.__cause__=cycle
        for error in (cycle,EngineGenerateError('single item'),type('VLLMServerError',(RuntimeError,),{})('unrelated class')):
            with self.subTest(error=type(error).__name__):self.assertFalse(core.unrecoverable_engine_error(error))

    def test_engine_death_returns_503_and_health_error(self):
        for error in (self.wrapped(EngineDeadError('engine stopped')),self.wrapped(VLLMServerError('fatal'))):
            with self.subTest(error=type(error.__cause__).__name__):
                service=self.service(error);port=self.serve(service)
                with contextlib.redirect_stdout(io.StringIO()):self.assert_failure(self.request(port),503)
                status,health=self.request(port,'GET','/health')
                self.assertEqual((status,health['status']),(200,'error'))
                self.assertIn('service failure',health['warnings'])
                self.assertTrue(service.dead)
                calls=service.backend.calls
                for path,body in (('/v1/systemone',None),('/unknown',{}),('/v1/systemone',{})):
                    self.assert_failure(self.request(port,path=path,body=body),503)
                self.assertEqual(service.backend.calls,calls)
                with self.assertRaises(ServiceFailure):service.answer(payload())

    def test_engine_dead_flag_blocks_before_inference(self):
        service=self.service()
        service.backend.engine=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(resources=SimpleNamespace(engine_dead=True))))
        port=self.serve(service)
        self.assertEqual(self.request(port,'GET','/health')[1]['status'],'error')
        self.assert_failure(self.request(port),503)
        self.assertEqual(service.backend.calls,0)

    def test_third_consecutive_item_failure_latches_503(self):
        service=self.service();port=self.serve(service)
        with contextlib.redirect_stdout(io.StringIO()):
            for expected in (422,422,503,503):self.assert_failure(self.request(port),expected)
        self.assertEqual(service.backend.calls,3)
        self.assertEqual(service.item_failure_streak,ITEM_FAILURE_LIMIT)
        self.assertEqual(service.health()['status'],'error')
        self.assertEqual(service.counter.snapshot()['total'],0)  # failed items are never counted as decisions

    def test_success_resets_item_failure_streak(self):
        service=self.service();port=self.serve(service)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):self.assert_failure(self.request(port),422)
            with self.fast_success():
                status,response=self.request(port)
            self.assertEqual(status,200)
            self.assertNotIn('warning',response['answers']['decision'])
            self.assertEqual(service.item_failure_streak,0)
            for expected in (422,422,503):self.assert_failure(self.request(port),expected)

    def test_success_and_failure_in_one_request(self):
        service=self.service()
        for _ in range(2):
            with self.assertRaises(ItemFailure):service.answer(payload())
        with self.fast_success():
            answer,detail=service._one_impl(core.normalize_request(payload(),8)[0][1])
        two={'state':'test context','questions':{'a':payload()['questions']['decision'],'b':payload()['questions']['decision']}}
        with patch.object(service,'_one_impl',side_effect=[(answer,detail),ItemFailure('item')]),self.assertRaises(ItemFailure):
            service.answer(two)
        self.assertEqual(service.item_failure_streak,1)
        self.assertFalse(service.dead)

    def test_vllm_item_generation_error_stays_422(self):
        service=self.service(self.wrapped(EngineGenerateError('single item')));port=self.serve(service)
        with contextlib.redirect_stdout(io.StringIO()):self.assert_failure(self.request(port),422)
        self.assertFalse(service.dead)
        self.assertNotEqual(service.health()['status'],'error')

    def test_non_finite_answer_is_an_item_failure(self):
        service=self.service()
        with self.fast_success():
            answer,detail=service._one_impl(core.normalize_request(payload(),8)[0][1])
        bad=dict(answer,noul=float('nan'))
        with patch.object(service,'_one_impl',return_value=(bad,detail)),self.assertRaises(ItemFailure):
            service.answer(payload())
        self.assertEqual(service.item_failure_streak,1)

    def test_fast_readout_engine_death_is_not_downgraded(self):
        service=self.service();port=self.serve(service)
        with self.fast_success(),patch('core.fast_attempt_with_retry',side_effect=self.wrapped(EngineDeadError('dead'))), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assert_failure(self.request(port),503)
        self.assertTrue(service.dead)

    def test_fatal_oom_message_is_not_retried(self):
        fatal=EngineDeadError('out of memory')
        with patch('core.fast_readout',side_effect=fatal) as readout,patch('core.release_failed_session') as release:
            with self.assertRaises(EngineDeadError):
                core.fast_attempt_with_retry(object(),{},{},[],[],{},new_meter())
        self.assertEqual(readout.call_count,1)
        release.assert_not_called()

    def test_slow_engine_death_is_not_returned_as_fast_answer(self):
        service=self.service(slow=True);port=self.serve(service)
        with self.fast_success(),patch('core.slow_readout',side_effect=self.wrapped(EngineDeadError('dead'))), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assert_failure(self.request(port),503)
        self.assertTrue(service.dead)
        self.assertEqual(service.health()['status'],'error')

    def test_recoverable_slow_failure_keeps_fast_answer(self):
        service=self.service(slow=True);port=self.serve(service)
        with self.fast_success(),patch('core.slow_readout',side_effect=RuntimeError('recoverable')):
            status,response=self.request(port)
        self.assertEqual(status,200)
        self.assertEqual(response['answers']['decision']['warning'],'slow_failed: fast answer kept')
        self.assertIn('decision',response['partial_errors'])
        self.assertFalse(service.dead)

    def test_slow_timeout_keeps_estimated_fast_answer(self):
        service=self.service(slow=True);port=self.serve(service)
        with self.fast_success(),patch('core.slow_readout',side_effect=TimeoutError('item timeout')):
            status,response=self.request(port)
        self.assertEqual(status,200)
        self.assertEqual(response['answers']['decision']['warning'],'slow_timeout: fast answer kept')
        self.assertTrue(response['usage']['usage_estimated'])
        self.assertFalse(service.dead)

    def test_late_engine_death_in_draining_worker_marks_service_failed(self):
        service=self.service(slow=True)
        service.config['generation']['wall_timeout_seconds']=0.02
        entered,release,marked=threading.Event(),threading.Event(),threading.Event()
        mark_dead=service._mark_dead
        def delayed_failure(*_):
            entered.set()
            if not release.wait(3):raise AssertionError('test worker was not released')
            raise self.wrapped(EngineDeadError('late engine failure'))
        def record_death():
            mark_dead();marked.set()
        try:
            with self.fast_success(),patch('core.slow_readout',side_effect=delayed_failure), \
                 patch.object(service,'_mark_dead',side_effect=record_death):
                response=service.answer(payload())
                self.assertTrue(entered.is_set())
                self.assertTrue(response['usage']['usage_estimated'])
                release.set()
                self.assertTrue(marked.wait(3))
            self.assertEqual(service.health()['status'],'error')
            with self.assertRaises(ServiceFailure):service.answer(payload())
        finally:
            release.set()

    def test_dead_service_answers_expect_continue_with_503(self):
        service=self.service();service.dead=True
        port=self.serve(service)
        conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
        try:
            conn.putrequest('POST','/v1/systemone')
            conn.putheader('Expect','100-continue');conn.putheader('Content-Length','999999999')
            conn.endheaders()
            response=conn.getresponse()
            self.assert_failure((response.status,json.loads(response.read())),503)
        finally:conn.close()

    def test_health_line_starts_with_status(self):
        # README step 2 waits for a log line beginning {"status": "ok"; launch prints json.dumps(health).
        service=self.service()
        line=json.dumps(service.health(),ensure_ascii=False)
        self.assertTrue(line.startswith('{"status": "ok"'),line[:40])

    def test_http_workers_are_daemon_threads(self):
        from server import QuietHTTPServer
        self.assertTrue(QuietHTTPServer.daemon_threads)

    def test_no_uniform_fallback_in_source(self):
        source=(launch.ROOT/'core.py').read_text(encoding='utf-8')+(launch.ROOT/'server.py').read_text(encoding='utf-8')
        for text in ('emergency_response','uniform distribution','1/len(labels)','1 / len(labels)'):
            self.assertNotIn(text,source)


class ReadinessProbe(unittest.TestCase):
    def test_probe_and_bench_ignore_proxy_settings(self):
        from http.server import BaseHTTPRequestHandler,HTTPServer
        import bench
        class Health(BaseHTTPRequestHandler):
            def log_message(self,*_):pass
            def reply(self):
                body=json.dumps({'status':'ok'}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            def do_GET(self):self.reply()
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length','0')));self.reply()
        httpd=HTTPServer(('127.0.0.1',0),Health)
        threading.Thread(target=httpd.serve_forever,daemon=True).start()
        self.addCleanup(httpd.server_close);self.addCleanup(httpd.shutdown)
        import urllib.request
        endpoint=f'http://127.0.0.1:{httpd.server_address[1]}'
        dead={'http_proxy':'http://127.0.0.1:9','HTTP_PROXY':'http://127.0.0.1:9','no_proxy':'','NO_PROXY':''}
        with patch.dict(os.environ,dead):
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(endpoint+'/health',timeout=5) as response:
                self.assertEqual(json.load(response),{'status':'ok'})
            self.assertEqual(bench.get_health(endpoint,5),{'status':'ok'})
            self.assertEqual(bench.post_json(endpoint,payload(),5),{'status':'ok'})
        self.assertIn('ProxyHandler({})',(launch.ROOT/'launch.py').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
