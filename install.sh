#!/usr/bin/env bash
# Maintainer tool: create a new venv with the version-gated runtime (see MAINTAINER.md).
#   bash install.sh /abs/path/to/new/venv
set -euo pipefail
PKG=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PY="${PYTHON:-python3}"
VENV=${1:?Usage: bash install.sh /abs/path/to/new/venv}
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo 'Python >= 3.10 required' >&2; exit 2
fi
case "$VENV" in
  /*) ;;
  *) echo 'Give the venv as an absolute path' >&2; exit 2 ;;
esac
if [[ -e "$VENV" ]]; then
  echo 'Use a new venv path' >&2; exit 2
fi
if ! "$PY" -c 'import pathlib, sys; v = pathlib.Path(sys.argv[1]).resolve(); p = pathlib.Path(sys.argv[2]).resolve(); raise SystemExit(1 if v == p or v.is_relative_to(p) or p.is_relative_to(v) else 0)' "$VENV" "$PKG"; then
  echo 'The venv must be outside the package' >&2; exit 2
fi
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install -r "$PKG/requirements.txt"
"$VENV/bin/python" -m pip check
