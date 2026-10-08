# Maintainer notes

Tools for maintainers of this package. The evaluation itself follows README.md only. None of these
tools deletes files; every output path is absolute, outside the package, and created exclusively
(`finalize.py` is the only tool that rewrites package files).

## Package layout

| File | Role |
|---|---|
| `serve.sh`, `launch.py` | supervisor: Python and exact version gate, path and model-folder checks, startup attempts, readiness, signals |
| `server.py`, `core.py`, `backends.py`, `quota.py`, `attention.py`, `model_setup.py` | the service |
| `config.json`, `calibration.json` | frozen parameters; `fitted.release.json` holds the fitted inputs, `fitted.example.json` sample values for the offline checks only |
| `bench.py`, `reconcile_usage.py`, `selftest.py`, `selftest.sh` | public-data selftest and usage reconciliation |
| `finalize.py`, `pkg_check.py`, `package_files.py`, `resolve_env.py` | freezing and offline checks |
| `regression_checks.py`, `test_attention.py`, `test_operator.py` | offline tests (no GPU, no model) |
| `install.sh`, `prepare_model.sh`, `requirements.txt` | maintainer venv and model download |
| `requirements.lock`, `requirements.measured.lock`, `environment.resolved.json` | record of the measured venv |
| `README.md`, `DISCLOSURE.md`, `MAINTAINER.md`, `FREEZE.md`, `LICENSE`, `MANIFEST.sha256` | documents; `FREEZE.md` and `MANIFEST.sha256` are generated |

## Venv and model folder

```bash
bash install.sh /abs/path/to/new/venv
bash prepare_model.sh /abs/path/to/new/model/folder
```

`install.sh` creates a new venv outside the package with Python 3.10 or newer and installs
`requirements.txt`: the exact versions `launch.py` enforces (vLLM 0.31.0, torch 2.13.0+cu130,
transformers 5.17.0, tokenizers 0.23.2, triton 3.7.1, compressed-tensors 0.17.0) and
huggingface-hub; pip resolves the remaining dependencies. `requirements.lock` is the complete
201-package freeze of the measured venv (Python 3.10.12); it records versions, not artifact hashes.
`python3 -B resolve_env.py --output-dir /abs/new/dir` rebuilds `requirements.lock` and
`environment.resolved.json` from `requirements.measured.lock` into a new directory without resolving
anything; `pkg_check.py` requires the packaged files to equal that output byte for byte.

`prepare_model.sh` downloads `cyankiwi/gemma-4-12B-it-qat-AWQ-INT4` at
`12deda07c573047b191b63f0768c0412120f3ed4` into a new folder, checks the compressed-tensors 4-bit
metadata and runs the startup identity check with full shard hashes, then writes `WEIGHTS.sha256`
and `PREPARATION.json` into that folder. The evaluation needs neither file; the startup identity
check accepts both, the model card, `.gitattributes` and the download cache directory, and refuses
any other file a loader may read and any symbolic link among the pinned files.

Run the service from the venv exactly as in the image, with absolute paths:

```bash
PYTHON=/abs/path/to/venv/bin/python bash serve.sh --model /abs/model --state-dir /abs/state --run-id NAME
```

`--bind 0.0.0.0` makes the service reachable from another container; readiness is always probed on
127.0.0.1, without any HTTP proxy. Startup refuses inherited cache or configuration path variables
(`HF_HOME`, `XDG_*`, `TRITON_*`, `TORCH_HOME` and the others in `launch.PATH_VARIABLES`) that point
outside `--state-dir`; unset them in the maintainer shell.

## Public-data selftest

```bash
PYTHON=/abs/path/to/venv/bin/python bash selftest.sh --model /abs/model \
  --public /abs/path/to/jevbench/datasets/public --output-dir /abs/path/to/new/output-dir
```

It starts the service three times (fast-only; routed; routed without prefix caching), runs
`bench.py` against each, reconciles the reported usage with the engine counters
(`reconcile_usage.py`), and writes a calibration candidate `calibration.candidate.json`. Everything
is written below `--output-dir`; the package is never modified. The selftest uses the GPU
exclusively; run nothing else on it at the same time.

## Freezing

```bash
python3 -B finalize.py --freeze-record     # after any change to a package file
python3 -B finalize.py fitted.release.json # only to re-inject fitted parameters; ends with a freeze record
```

`--freeze-record` writes `FREEZE.md` (status, UTC time, runtime image, SHA-256 of `config.json` and
`calibration.json`, prompt_sha) from the files as they are, then rebuilds `MANIFEST.sha256` over
every package file except itself, in byte order of the POSIX path (clone metadata, `__pycache__` and
`state` are excluded). It refuses to run while the package holds a file that `.gitignore` excludes,
so the manifest of the working tree is the manifest of the commit.

After the release commit, verify the committed bytes in a fresh `git clone` of that commit and in
its `git archive` export: run `sha256sum -c MANIFEST.sha256` and `python3 -B pkg_check.py --release`
in both. Use an explicitly configured GitHub noreply identity for any new commit and check the
author and committer of the commit before it is pushed.

## Checks

```bash
python3 -B pkg_check.py            # structure, references, sample finalize rehearsal, scans, all offline tests
python3 -B pkg_check.py --release  # additionally: release identity, freeze record and document checks
sha256sum -c MANIFEST.sha256
```

`pkg_check.py` refuses to run under `-O`. Its scan policy is the `SCAN_*` constants at its top,
written as plain literals and limited to generic rules; the scans skip only those lines and the
frozen source of `core.render` and `core.choice_codes`. Checks for specific names or identifiers
are not part of the package. It flags:

- the common file deletion, truncation and replacement calls in package code (a static scan, not a
  proof). Python, with import aliases resolved: `unlink`, `remove`, `removedirs`, `rmdir`,
  `truncate`, `rmtree`, the temporary file and directory classes that delete on close, `os.remove`,
  `os.unlink`, `os.replace`, `os.rename`, `os.renames`, `os.truncate`, `shutil.move`, `Path.replace`
  and `Path.rename` with one argument and, outside the offline tests, `write_text`, `write_bytes`
  and `open` with a `w` mode; the same rules apply to Python in `<<'PY'` heredocs of shell scripts.
  Shell lines, and literal arguments to `subprocess`, `os.system`, `os.popen`, `os.exec*` and
  `os.spawn*`: `rm`, `rmdir`, `unlink`, `truncate`, `shred`, `find -delete`, `git clean` and
  `mv -f`. The only allowed calls are `quota.py` replacing its quota state file with a temporary
  file and removing that temporary file after a failed write, and `finalize.py` replacing a package
  file through an exclusively created temporary file;
- in every package file: unfinished-work markers (`SCAN_UNFINISHED`), abbreviated hex values (7 to
  39 hex characters with both digits and letters), private paths, e-mail and IPv4 addresses (except
  `127.0.0.1` and `0.0.0.0`) and remote-shell commands; bare `python` in shell scripts;
- with `--release`: unresolved fact markers, process wording and links to Hugging Face or GitHub
  that are not pinned to a 40-character commit in the documents; any difference between the
  documents and the fixed identifiers (model file table and `launch.py`, public data hashes,
  runtime image digests, model and harness revisions, the model download command and the harness
  price flags in README.md).

The finalize rehearsal runs in a copy below a new `tempfile.mkdtemp` directory, which is kept. The
offline tests also keep their directories below the system temporary directory.

Numeric invariants checked by the offline tests (`regression_checks.py` unless noted):

- `prompt_sha` `e18ed7a06eb52f1b351458f384710a2c240aaf2b4f64d916b776c6d18ca3f938`: it covers the
  source of `core.render` and `core.choice_codes`, `prompt.system`, `prompt.enable_thinking` and
  the generation settings other than `max_new_tokens`, `reasoning_words` and
  `wall_timeout_seconds`; `validate_threshold(load_config()) == []`;
- the source hashes of `render` and `choice_codes`, and of the functions and constants listed in
  `REQUEST_PATH_SOURCE_SHA256`. Not listed: `Service` (routing predicate, calibration block choice,
  usage bases), the fast and slow attempt wrappers, the backend and counter constructors and
  `server.py`;
- `PARAMETER_SHA256`: every value of `config.json`, `calibration.json` and `fitted.release.json`
  except the descriptive `fitted_on`, `status` and `model_calibration_status` strings, which covers
  `prompt.cot_instructions`, `prompt.answer_marker`, the limits, the routing, the cost fuse, the
  prices and the engine options;
- the vLLM engine arguments, including `attention_backend` only on the second startup attempt
  (`test_attention.py`, with a stand-in `vllm` module).

A change to any of them requires a new GPU measurement. Code outside these lists is checked only by
the behavioural tests.

## GPU comparison evidence

Compare a new run with a reference run item by item: `fast_prompt_tokens`, `kept_prompt_tokens` and
`original_prompt_tokens` in `usage.questions` must be equal, and so must the slow-path prompt tokens
where both runs take the slow path; report the largest absolute difference of `fast_margin`, every
change of route or argmax, all three usage totals and prices, failures, fuse transitions, latency
and peak GPU memory. Probabilities computed from identical logits can differ at rounding level
between Python 3.10 and 3.12; compare them with a tolerance such as 1e-12 and assess GPU run-to-run
variation separately. Record the files created under the writable mount: `state/run1-startup/`
(`attempts.jsonl`, `attempt-N.log`, `state-N/`, `tmp/`, `home/`), `state/cache/` and the harness
output `run1/`.
