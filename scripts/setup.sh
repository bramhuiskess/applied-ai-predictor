#!/usr/bin/env bash
# Create .venv with a Python >= 3.11 and install the project.
# Usage: ./scripts/setup.sh [extras]     (default extras: train,dev)
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRAS="${1:-train,dev,datasets}"
MIN_MINOR=11

ok() { "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3, $MIN_MINOR) else 1)" 2>/dev/null; }

PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1 && ok "$cand"; then
    PY="$(command -v "$cand")"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3.$MIN_MINOR or newer is required, but none was found." >&2
  echo "macOS's built-in python3 (3.9) is too old. Install one with:  brew install python@3.12" >&2
  exit 1
fi

# Throw away a venv that was made with an older Python.
if [ -x .venv/bin/python ] && ! ok .venv/bin/python; then
  echo "Removing .venv (built with $(.venv/bin/python --version 2>&1))"
  rm -rf .venv
fi

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install -e ".[$EXTRAS]"

echo
echo "Done with $(.venv/bin/python --version). Activate with:  . .venv/bin/activate"
