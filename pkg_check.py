"""Offline package checks (maintainer tool). No network, GPU imports or inference.

  python3 -B pkg_check.py             structure, references, sample finalize rehearsal, scans, tests
  python3 -B pkg_check.py --release   additionally: release identity, freeze record and document checks

The finalize rehearsal runs in a copy of the package in a new directory created by tempfile.mkdtemp;
the copy is kept (its path is printed). This tool deletes nothing.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from package_files import package_files

if not __debug__:
    raise SystemExit('Run pkg_check without -O or PYTHONOPTIMIZE: its checks use assert')

ROOT=Path(__file__).resolve().parent
REQUIRED=('server.py core.py backends.py quota.py bench.py reconcile_usage.py attention.py model_setup.py launch.py '
          'package_files.py selftest.py finalize.py pkg_check.py resolve_env.py regression_checks.py test_attention.py '
          'test_operator.py serve.sh selftest.sh install.sh prepare_model.sh config.json calibration.json '
          'fitted.release.json fitted.example.json requirements.txt requirements.lock requirements.measured.lock '
          'environment.resolved.json README.md DISCLOSURE.md MAINTAINER.md FREEZE.md LICENSE MANIFEST.sha256 '
          '.gitattributes .gitignore').split()
GENERATED={'requirements.lock','environment.resolved.json'}
DOCS=('README.md','DISCLOSURE.md','MAINTAINER.md','FREEZE.md')
PROMPT_SHA='e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938'
JEVBENCH_COMMIT='bb05a335bc809e61b20c0f745d25499a82b326fc'
# Raw bytes (LF) of the public files at JEVBENCH_COMMIT, as listed in its datasets/manifest.json.
PUBLIC_DATA_SHA256={
    'easy.jsonl':'231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b',
    'original.jsonl':'5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180',
    'hard.jsonl':'89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb',
}
IMAGE_INDEX='sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3'
IMAGE_CONFIG='sha256:c76d0e2225a4b1cb1e2109ace39639f55e714abd1a7a427acc8b0bbd7f6a83b3'

# Scan policy: generic rules only. The SCAN_* assignments below are plain literals; together with the
# frozen source of core.render and core.choice_codes they are the only text the scans skip (see scan_view).
SCAN_UNFINISHED=('TODO','FIXME')
SCAN_FACTS='[[FACTS-PENDING'
SCAN_DOC_WORDS=(r'<fill',r'(?<!FACTS-)\bpending\b',r'\bplaceholders?\b',r'\bconfirm',r'Maintainers must',r'v1\.0-frozen')
SCAN_README_WORDS=(r'\boptional',r'\balternatively\b',r'\beither\b')
SCAN_PROCESS=(r'\bso\s+far\b',r'\bbefore\s+publication\b',r'\bonce\b[^.!?\]\n]*\bavailable\b',r'\bmust\s+supply\b',
              r'\bnot\s+yet\b',r'\buser-supplied\b',r'\bnot\s+(?:separately\s+)?supplied\b',r'\bwere\s+not\s+supplied\b')
# Hex tokens of 7-39 characters with both digits and letters read as abbreviated commits, hashes or
# identifiers; full 40- and 64-character values are not matched.
SCAN_SHORT_HEX=r'(?i)(?<![0-9a-z])[0-9a-f]{7,39}(?![0-9a-z])'
SCAN_LEGACY='VLLM_ATTENTION_BACKEND'
SCAN_SHELL_DELETION=(r'(^|[^\w-])(\\|/bin/|/usr/bin/)?rm\b|\s-delete\b|\bunlink\b|\bshred\b|\brmdir\b|\bgit\s+clean\b'
                     r'|\brmtree\b|\.unlink\(|\bmv\s+-f\b|\btruncate\b')
RUNTIME_PY=('launch.py','server.py','core.py','backends.py','attention.py','quota.py','model_setup.py')
# The common calls that delete or replace a file (a static scan, not a proof); the allowed ones are
# listed exactly. Path.replace(target) and Path.rename(target) are matched by method name with one
# positional argument on a receiver that is not a string literal (str.replace takes two).
DELETING_METHODS={'unlink','rmtree','remove','removedirs','rmdir','truncate','TemporaryDirectory','TemporaryFile','SpooledTemporaryFile'}
REPLACING_CALLS={'os.remove','os.unlink','os.replace','os.rename','os.renames','os.truncate','shutil.move','shutil.rmtree'}
PATH_MOVING_METHODS={'replace','rename'}
# Outside the offline tests (which write fixtures into their own new directories), files are only
# created exclusively ('x'), appended to, or written through a new temporary file.
OVERWRITING_METHODS={'write_text','write_bytes'}
TEST_FILES=('regression_checks.py','test_attention.py','test_operator.py')
ALLOWED_FILE_CALLS={('quota.py','Path(temporary).unlink(missing_ok=True)'),
                    ('quota.py','os.replace(temporary, self.state_path)'),
                    ('finalize.py','os.replace(tmp, path)')}


def check_manifest(root):
    assert b'\r' not in (root/'MANIFEST.sha256').read_bytes(),'MANIFEST must use LF for sha256sum -c'
    expected={}
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        sha,name=line.split('  ',1)
        p=root/name
        assert re.fullmatch('[0-9a-f]{64}',sha),'Invalid manifest digest'
        assert not Path(name).is_absolute() and p.resolve().is_relative_to(root.resolve()) and p.is_file(), 'Unsafe/missing manifest member'
        assert name not in expected,'Duplicate manifest member'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,'Hash mismatch: '+name
        expected[name]=sha
    actual={p.relative_to(root).as_posix() for p in package_files(root) if p.is_file() and p.name!='MANIFEST.sha256'}
    assert actual==set(expected),'Manifest membership mismatch'


def scan_view(path,text):
    """Text of a package file for the scans, line numbers kept: in .py files the SCAN_* policy
    assignments of this file and the frozen render/choice_codes source of core.py are blanked."""
    if path.suffix!='.py':return text
    lines=text.splitlines()
    for node in ast.parse(text).body:
        protected=path.name=='core.py' and isinstance(node,ast.FunctionDef) and node.name in ('render','choice_codes')
        policy=(path.name=='pkg_check.py' and isinstance(node,ast.Assign)
                and all(isinstance(t,ast.Name) and t.id.startswith('SCAN_') for t in node.targets))
        if protected or policy:
            lines[node.lineno-1:node.end_lineno]=['']*(node.end_lineno-node.lineno+1)
    return '\n'.join(lines)


def embedded_python(path,text):
    """Python run from shell heredocs (<<'PY' ... PY), for the deletion audit."""
    if path.suffix!='.sh':return []
    return re.findall(r"<<'PY'\n(.*?)\nPY\n",text,re.S)


def dotted_name(node,aliases):
    if isinstance(node,ast.Name):return aliases.get(node.id,node.id)
    if isinstance(node,ast.Attribute):return dotted_name(node.value,aliases)+'.'+node.attr
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='getattr' and len(node.args)>=2:
        def literal(value):
            if isinstance(value,ast.Constant) and isinstance(value.value,str):return value.value
            if isinstance(value,ast.BinOp) and isinstance(value.op,ast.Add):return literal(value.left)+literal(value.right)
            return '?'
        return dotted_name(node.args[0],aliases)+'.'+literal(node.args[1])
    return ''


def python_deletions(name,tree):
    """The common calls in one Python module that delete, truncate, overwrite or replace files
    (import aliases resolved); see DELETING_METHODS, REPLACING_CALLS and PATH_MOVING_METHODS."""
    aliases={}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):aliases.update({a.asname or a.name:a.name for a in node.names})
        elif isinstance(node,ast.ImportFrom):aliases.update({a.asname or a.name:(node.module or '')+'.'+a.name for a in node.names})
    findings=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        call=dotted_name(node.func,aliases);method=call.rsplit('.',1)[-1]
        keeps=any(k.arg=='delete' and isinstance(k.value,ast.Constant) and k.value.value is False for k in node.keywords)
        path_move=(method in PATH_MOVING_METHODS and isinstance(node.func,ast.Attribute) and len(node.args)==1
                   and not (isinstance(node.func.value,ast.Constant) and isinstance(node.func.value.value,str)))
        flagged=(method in DELETING_METHODS or call in REPLACING_CALLS or path_move or (method=='NamedTemporaryFile' and not keeps))
        if name not in TEST_FILES and method!='NamedTemporaryFile':
            modes=[n.value for n in [*node.args[:2],*(k.value for k in node.keywords if k.arg=='mode')]
                   if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            flagged=flagged or method in OVERWRITING_METHODS or (method=='open' and any(re.fullmatch(r'[rbt+]*w[rbt+]*',m) for m in modes))
        if call.startswith(('subprocess.','os.system','os.popen','os.exec','os.spawn')):
            literals=[n.value for arg in [*node.args,*(k.value for k in node.keywords)] for n in ast.walk(arg)
                      if isinstance(n,ast.Constant) and isinstance(n.value,str)]
            flagged=flagged or any(re.search(SCAN_SHELL_DELETION,value) for value in literals)
        if flagged and (name,ast.unparse(node)) not in ALLOWED_FILE_CALLS:
            findings.append((name,node.lineno,ast.unparse(node)[:80]))
    return findings


def deletion_findings(root):
    """The common file deletion, truncation and replacement calls in package code (Python, shell lines
    and Python in <<'PY' heredocs); only ALLOWED_FILE_CALLS pass."""
    findings=[]
    for path in package_files(root):
        if path.suffix=='.py':
            findings+=python_deletions(path.name,ast.parse(path.read_text(encoding='utf-8')))
        elif path.suffix=='.sh':
            text=path.read_text(encoding='utf-8')
            for block in embedded_python(path,text):
                findings+=python_deletions(path.name+' (embedded Python)',ast.parse(block))
            for number,line in enumerate(text.splitlines(),1):
                if re.search(SCAN_SHELL_DELETION,line):findings.append((path.name,number,line.strip()))
    return findings


IDENTIFIERS=(r'[A-Za-z]:[\\/](?:Users|home)[\\/]',r'/(?:home|Users|root)/[\w.-]+',r'(?<!\w)~[/\\]',
             r'(?i)\bssh\s+(?:-[^\s]+\s+)*[\w.-]+',r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',r'\b(?:\d{1,3}\.){3}\d{1,3}\b')


def short_hex_tokens(text):
    """Abbreviated hex values: SCAN_SHORT_HEX tokens that contain both digits and letters."""
    return [t for t in re.findall(SCAN_SHORT_HEX,text) if re.search(r'[0-9]',t) and re.search(r'(?i)[a-f]',t)]


def text_problems(name,text,version_tokens=()):
    """Generic rules for the text of one package file: unfinished-work markers, abbreviated hex
    values, private paths, e-mail and IPv4 addresses (loopback and wildcard binds excepted) and
    remote-shell commands. Numeric package versions are not addresses."""
    problems=[f'{name}: unfinished-work marker {m}' for m in SCAN_UNFINISHED if m in text]
    problems+=[f'{name}: abbreviated hex value {t}' for t in sorted(set(short_hex_tokens(text)))]
    for version in sorted(version_tokens,key=len,reverse=True):
        text=re.sub(r'(?<![\w.])'+re.escape(version)+r'(?![\w.])','VERSION',text)
    for pattern in IDENTIFIERS:
        hits=[h for h in re.findall(pattern,text) if h not in ('127.0.0.1','0.0.0.0')]
        if hits:problems.append(f'{name}: private path, address or remote-shell command')
    return problems


def scan_text(root):
    """Text rules for every package file (render and choice_codes source excluded)."""
    measured=(root/'requirements.measured.lock').read_text(encoding='utf-8')
    version_tokens={line.split('==',1)[1] for line in measured.splitlines() if '==' in line}
    for f in package_files(root):
        if not f.is_file():continue
        assert b'\r' not in f.read_bytes(),'Package text must use LF: '+f.name
        text=scan_view(f,f.read_text(encoding='utf-8'))
        found=text_problems(f.name,text,version_tokens)
        assert not found,'; '.join(found)
        if f.suffix=='.py':
            tree=ast.parse(f.read_text(encoding='utf-8'))
            legacy=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==SCAN_LEGACY]
            removed={id(n.args[0]) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                     and n.func.attr=='pop' and n.args}
            if f.name in RUNTIME_PY:
                # vLLM 0.31 ignores this variable; runtime code may only remove inherited values.
                assert all(id(n) in removed for n in legacy) and text.count(SCAN_LEGACY)==len(legacy),'VLLM_ATTENTION_BACKEND set in '+f.name
    for script in sorted(root.glob('*.sh')):
        t=script.read_text(encoding='utf-8')
        assert not re.search(r'(?m)(^|[\s;&|(])python([\s;]|$)',t),'Bare python call in '+script.name
        assert SCAN_LEGACY not in t,'VLLM_ATTENTION_BACKEND in '+script.name
    for name in ('serve.sh','selftest.sh'):
        assert 'VLLM_USE_FLASHINFER_SAMPLER=0' in (root/name).read_text(encoding='utf-8')


def release_problems(root,c):
    """Release checks; every problem is collected so the full list is visible at once."""
    from core import load_json,threshold_identity,validate_threshold
    import launch
    problems=[]
    def need(condition,message):
        if not condition:problems.append(message)
    need(c['package']['finalized'] and not c['package'].get('sample_only'),'Real fitted parameters missing')
    need(threshold_identity(c)['prompt_sha']==PROMPT_SHA,'prompt_sha changed')
    need(validate_threshold(c)==[],'threshold identity degraded')
    cal=load_json(root/c['calibration_file'])['models']['gemma-4-12b-awq']
    need(not cal.get('placeholder') and all(not cal[b].get('placeholder') for b in ('fast','slow','fallback')),'Calibration placeholder')
    for name in GENERATED:need((root/name).is_file(),'Missing release environment artifact: '+name)
    for f in package_files(root):
        rel=f.relative_to(root).parts
        need(not f.is_symlink(),'Symlink in release: '+f.name)
        need('__pycache__' not in rel and not rel[0].startswith('selftest-') and f.suffix not in ('.pyc','.tmp'),'Runtime output in release: '+f.name)
    env=load_json(root/'environment.resolved.json')
    need(env['python']=='3.10.12' and env['python_required']=='>=3.10','Measured environment record changed')
    need(env['lock_sha256']==hashlib.sha256((root/'requirements.lock').read_bytes()).hexdigest(),'Lock hash differs from environment record')
    need(env['measured_freeze_sha256']==hashlib.sha256((root/'requirements.measured.lock').read_bytes()).hexdigest(),'Measured freeze hash differs')
    pins=dict(line.split('==',1) for line in (root/'requirements.lock').read_text().splitlines() if line and not line.startswith(('#','--')))
    need(pins==env['versions'],'Lock versions differ from measured evidence')
    measured=dict(line.split('==',1) for line in (root/'requirements.measured.lock').read_text().splitlines() if line)
    measured['torch']='2.13.0+cu130'
    need(pins==measured,'Unexplained change to measured freeze')
    for name,version in launch.VERSIONS.items():
        need(pins.get(name,'').split('+')[0]==version,'Version gate differs from measured lock: '+name)
    need(cal['fast']['by_type']['score']['status']=='insufficient','fast score status')
    need(cal['slow']['by_type']['score']['count']==1,'slow score count')
    for b in ('fast','slow','fallback'):
        need(cal[b]['held_out_validated'] is False,'held_out_validated '+b)
        need(cal[b]['validation_status']=='candidate; held-out validation required','validation_status '+b)
    need(c['model']['name']==c['model']['version']==c['package']['version']=='metask-jev-rain-12B','System name')
    need(c['model']['model_key']=='gemma-4-12b-awq'==launch.MODEL_KEY and c['model']['quant']=='awq' and c['model']['backend']=='vllm','Model identity')
    need(c['routing']['quota']==.15 and c['routing']['fuse']==.20 and c['routing']['target_slow_share']==.15,'Quota')
    need(c['routing']['margin_threshold']==2.9999032097890423,'Threshold')
    need(c['generation']['reasoning_words']==300 and c['generation']['max_new_tokens']==512,'Draft budget')
    need(c['limits']['max_prompt_tokens']==8192 and c['limits']['physical_max_prompt_tokens']==16384,'Prompt limits')
    need(c['routing']['tiers']['long']=={'max_prompt_tokens':8192,'fast_only':True},'Long tier')
    need(c['routing']['long_item_slow_policy']=='fast_only' and c['usage']['basis']=='engine','Long policy or usage basis')
    need(c['bench']['target_slow_share']==.15,'Bench share')
    need(c['pricing']['model_key']=='12B' and set(c['pricing']['models'])=={'12B'},'Pricing key')
    need(c['pricing']['models']['12B']['input_per_million']==.05 and c['pricing']['models']['12B']['output_per_million']==.15,'Prices')
    fitted=load_json(root/'fitted.release.json')
    for block in ('fast','slow','fallback'):
        expected=fitted['calibration'][block]
        need(cal[block]['noul_calibration']['p_cal']==expected['p_cal'],'p_cal differs from fitted.release.json: '+block)
        need(cal[block]['temperature_by_kind']['choice']==expected['T_choice'],'T_choice differs from fitted.release.json: '+block)
        need(cal[block]['temperature_by_kind']['score']==expected['T_score'],'T_score differs from fitted.release.json: '+block)
        need(cal[block]['by_type']==expected['by_type'],'by_type differs from fitted.release.json: '+block)
    need(cal.get('status')==fitted.get('model_calibration_status'),'Calibration status differs from fitted.release.json')
    need((root/'.gitattributes').read_text().strip()=='* -text','.gitattributes')
    freeze=(root/'FREEZE.md').read_text(encoding='utf-8')
    record=dict(re.findall(r'(?m)^(config\.json SHA256|calibration\.json SHA256|prompt_sha|Runtime image|Status): (\S.*)$',freeze))
    need(record.get('config.json SHA256')==hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),'FREEZE.md config.json hash differs from the file')
    need(record.get('calibration.json SHA256')==hashlib.sha256((root/'calibration.json').read_bytes()).hexdigest(),'FREEZE.md calibration.json hash differs from the file')
    need(record.get('prompt_sha')==PROMPT_SHA,'FREEZE.md prompt_sha')
    need(record.get('Runtime image')==launch.IMAGE,'FREEZE.md runtime image')
    need(record.get('Status')=='PARAMETERS FITTED','FREEZE.md status')
    problems+=document_problems(root,launch)
    return problems


def document_problems(root,launch):
    """Wording, links, abbreviated hashes and the fixed identifiers quoted in the documents."""
    problems=[]
    def need(condition,message):
        if not condition:problems.append(message)
    texts={}
    for name in DOCS:
        problems+=document_text_problems(name,(root/name).read_text(encoding='utf-8'))
        text=re.sub(re.escape(SCAN_FACTS)+r'[^\]]*\]\]','',(root/name).read_text(encoding='utf-8'))
        texts[name]=text
        for image in re.findall(r'vllm/vllm-openai[@:][\w.:]+',text):
            need(image==launch.IMAGE,f'{name}: runtime image reference differs from launch.IMAGE: {image}')
        for digest in re.findall(r'sha256:[0-9a-f]{64}',text):
            need(digest in (launch.IMAGE.split('@',1)[1],IMAGE_INDEX,IMAGE_CONFIG),f'{name}: unknown image digest {digest}')
        for repo,revision in (('cyankiwi/gemma-4-12B-it-qat-AWQ-INT4',launch.MODEL_REVISION.split('@',1)[1]),
                              ('fstandhartinger/jevbench',JEVBENCH_COMMIT)):
            for url in re.findall(r'https://(?:huggingface\.co|github\.com)/'+re.escape(repo)+r'/[^\s)>\]`]+',text):
                need(revision in url,f'{name}: {repo} link not at {revision}')
    disclosure=texts['DISCLOSURE.md']
    for filename,digest in PUBLIC_DATA_SHA256.items():
        row=re.search(r'(?m)^\|[^\n]*`'+re.escape(filename)+r'`[^\n]*$',disclosure)
        need(row is not None and digest in row[0],'DISCLOSURE.md: public data hash differs for '+filename)
    for filename,digest in launch.MODEL_SHA256.items():
        row=re.search(r'(?m)^\| `'+re.escape(filename)+r'` \| ([\d,]+) \| `([0-9a-f]{64})` \|$',disclosure)
        need(row is not None and row[2]==digest,'DISCLOSURE.md: model file table differs from launch.py for '+filename)
    for filename,size in launch.SHARD_BYTES.items():
        row=re.search(r'(?m)^\| `'+re.escape(filename)+r'` \| ([\d,]+) \| `([0-9a-f]{64})` \|$',disclosure)
        need(row is not None and int(row[1].replace(',',''))==size and row[2]==launch.SHARD_SHA256[filename],
             'DISCLOSURE.md: model shard table differs from launch.py for '+filename)
    for name in ('README.md','DISCLOSURE.md'):
        for digest in (launch.IMAGE.split('@',1)[1],IMAGE_INDEX,IMAGE_CONFIG):
            need(digest in texts[name],f'{name}: image digest missing: {digest}')
    repo,revision=launch.MODEL_REVISION.split('@',1)
    downloads=re.findall(r'hf download (\S+) --revision (\S+)',texts['README.md'])
    need(downloads==[(repo,revision)],'README.md: model download command differs from launch.MODEL_REVISION')
    prices=json.loads((root/'config.json').read_text(encoding='utf-8'))['pricing']['models']['12B']
    flags=re.findall(r'--price-in-per-m (\S+) --price-out-per-m (\S+)',texts['README.md'])
    need([(float(a),float(b)) for a,b in flags]==[(prices['input_per_million'],prices['output_per_million'])],
         'README.md: harness price flags differ from config.json pricing')
    return problems


def document_text_problems(name,text):
    """Release rules for one document: fact markers, process wording, links to Hugging Face or
    GitHub not pinned to a 40-character commit, and abbreviated hex values."""
    problems=[]
    pending=text.count(SCAN_FACTS)
    if pending:problems.append(f'{name}: {pending} {SCAN_FACTS}]] marker(s) remain')
    text=re.sub(re.escape(SCAN_FACTS)+r'[^\]]*\]\]','',text)  # reported above; scan the rest
    for pattern in SCAN_DOC_WORDS+SCAN_PROCESS+(SCAN_README_WORDS if name=='README.md' else ()):
        if re.search('(?i)'+pattern,text):problems.append(f'{name}: wording not allowed in release documents: {pattern}')
    for url in re.findall(r'(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co|github\.com)/[^\s)>\]`\'"]+',text):
        url=url.rstrip('.,;:')
        if re.fullmatch(r'https://(?:huggingface\.co|github\.com)/[\w.-]+/[\w.-]+/(?:tree|blob)/[0-9a-f]{40}(?:/[\w./-]+)?',url) is None:
            problems.append(f'{name}: link not pinned to a 40-character commit: {url}')
    problems+=[f'{name}: abbreviated hash {t}' for t in short_hex_tokens(text)]
    return problems


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release',action='store_true');a=ap.parse_args()
    for name in REQUIRED:assert (ROOT/name).is_file(),'Missing package file: '+name
    check_manifest(ROOT)
    modules={p.stem for p in ROOT.glob('*.py')}
    external={'torch','transformers','vllm','peft','huggingface_hub'}
    for path in ROOT.glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                top=name.split('.')[0]
                assert top in modules or top in sys.stdlib_module_names or top in external, 'Unresolved Python import: '+name
        # Literal references to packaged scripts must exist.
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and re.fullmatch(r'[\w.-]+\.py',node.value):
                assert (ROOT/node.value).is_file(),'Missing script reference: '+node.value
    found=deletion_findings(ROOT)
    assert not found,'File deletion or replacement in package code: '+repr(found)
    scan_text(ROOT)
    import resolve_env
    for name,text in resolve_env.render().items():
        assert (ROOT/name).read_text(encoding='utf-8')==text,name+' differs from resolve_env.py output'
    from core import load_config,load_json,threshold_identity,validate_calibration
    c=load_config();assert c['model']['path']==''
    assert set(c['profiles'])=={'gemma-4-12b-awq'}
    assert (ROOT/c['calibration_file']).is_file()
    for block in ('fast','slow','fallback'):validate_calibration(load_json(ROOT/c['calibration_file']),'gemma-4-12b-awq',block)
    scratch=Path(tempfile.mkdtemp(prefix='jev12b-pkg-check-'))
    clone=scratch/'pkg';shutil.copytree(ROOT,clone,ignore=shutil.ignore_patterns('.git','__pycache__','state'))
    sample=json.loads((clone/'fitted.example.json').read_text())
    assert sample['prompt_sha']==threshold_identity(c)['prompt_sha']
    sample['calibration']['slow'].update(by_type={'score':{'count':5,'status':'insufficient'}},held_out_validated=False,validation_status='candidate; held-out validation required')
    with (scratch/'fitted.sample.json').open('x',encoding='utf-8',newline='\n') as out:out.write(json.dumps(sample)+'\n')
    cmd=[sys.executable,'-B',str(clone/'finalize.py'),str(scratch/'fitted.sample.json'),'--sample']
    subprocess.run(cmd,check=True,capture_output=True)
    check_manifest(clone)
    new=json.loads((clone/'config.json').read_text());cal=json.loads((clone/'calibration.json').read_text())
    assert not new['package']['finalized'] and new['package']['sample_only']
    assert 'Status: SAMPLE ONLY' in (clone/'FREEZE.md').read_text()
    for r in (new['routing'],new['profiles']['gemma-4-12b-awq']['routing']):
        assert r['margin_threshold']==sample['threshold'] and r['quota']==.15 and r['fuse']==.20
    for name,b in sample['calibration'].items():
        v=cal['models']['gemma-4-12b-awq'][name]
        assert v['noul_calibration']['p_cal']==b['p_cal']
        assert v['temperature_by_kind']['choice']==b['T_choice']
        assert v['temperature_by_kind']['score']==b['T_score']
    preserved=cal['models']['gemma-4-12b-awq']['slow']
    for field in ('by_type','held_out_validated','validation_status'):
        assert preserved[field]==sample['calibration']['slow'][field],'Lost calibration audit metadata'
    before=(clone/'config.json').read_bytes()
    sample['prompt_sha']='0'*64
    with (scratch/'bad.json').open('x',encoding='utf-8',newline='\n') as out:out.write(json.dumps(sample))
    bad=subprocess.run([sys.executable,'-B',str(clone/'finalize.py'),str(scratch/'bad.json'),'--sample'],capture_output=True)
    assert bad.returncode!=0 and (clone/'config.json').read_bytes()==before
    # Every Python file compiles; compiled in memory, nothing is written.
    for source in ROOT.glob('*.py'):compile(source.read_text(encoding='utf-8'),str(source),'exec')
    print('Scratch copy kept at '+str(scratch))
    for test in ('regression_checks.py','test_attention.py','test_operator.py'):
        subprocess.run([sys.executable,'-B',str(ROOT/test)],check=True)
    print('PASS: references, config, sample finalize, rejection before mutation, hashes, compilation, scans, tests')
    if a.release:
        problems=release_problems(ROOT,c)
        for problem in problems:print('RELEASE FAIL: '+problem)
        if problems:raise SystemExit(1)
        print('PASS: release identity, freeze record and documents')
    print('Offline packaging check only; no dependency installation or GPU validation performed.')

if __name__=='__main__':main()
