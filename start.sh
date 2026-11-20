#!/usr/bin/env bash
# One-click launcher for macOS and Linux: finds (or installs) Python, builds .venv, starts the setup.
#
#   ./start.sh                 guided: asks before installing anything, then offers to open the dashboard
#   ./start.sh --yes --no-start   no questions, no dashboard (for scripts and CI)
#   ./start.sh --dry-run       print what would happen, change nothing
#
# Flags: --yes  --no-start  --minimal (no training libraries)  --extras LIST  --dry-run  --help
# Exit codes: 0 ok, 1 no usable Python and none installed, 2 an install step failed.
set -euo pipefail
cd "$(dirname "$0")"

MIN_MINOR=11
PREFERRED="$(tr -d '[:space:]' < .python-version 2>/dev/null || true)"
PREFERRED="${PREFERRED:-3.12}"
UV_URL="https://astral.sh/uv/install.sh"
YES=0 NO_START=0 DRY=0 EXTRAS="train,datasets"

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --no-start) NO_START=1 ;;
    --minimal) EXTRAS="" ;;
    --extras) shift; EXTRAS="${1:-}" ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '%s\n' "$*"; }
plan() { say "[dry-run] would: $*"; }

# ask "question": yes (0) or no (1). Default yes. --yes answers yes, EOF (no terminal) answers the default.
ask() {
  if [ "$DRY" = 1 ]; then say "[dry-run] would ask: $1"; return 0; fi
  [ "$YES" = 1 ] && return 0
  printf '%s [Y/n] ' "$1"
  read -r answer || answer=""
  case "$answer" in n|N|no|NO|No) return 1 ;; *) return 0 ;; esac
}

py_ok() { "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3, $MIN_MINOR) else 1)" >/dev/null 2>&1; }

manual_help() {
  say ""
  say "Install Python $PREFERRED yourself, then run this file again:"
  say "  macOS:    brew install python@$PREFERRED      (or https://www.python.org/downloads/)"
  say "  Debian:   sudo apt install python$PREFERRED python$PREFERRED-venv"
  say "  Fedora:   sudo dnf install python$PREFERRED"
  say "  Windows:  winget install Python.Python.$PREFERRED"
}

# ---------------------------------------------------------------- 1. find a Python
PY=""
for cand in "python$PREFERRED" python3.13 python3.12 python3.11 python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then PY="$(command -v "$cand")"; break; fi
done

UV=""
for cand in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then UV="$cand"; break; fi
done

USE_UV=0
if [ -n "$PY" ]; then
  say "Python: $("$PY" --version 2>&1) ($PY)"
else
  found="$(python3 --version 2>&1 || true)"
  say "No Python 3.$MIN_MINOR or newer found${found:+ (found: $found)}."
  if ! ask "Install Python $PREFERRED for this project with uv (no admin rights; the system Python is not touched)?"; then
    manual_help
    exit 1
  fi
  USE_UV=1
  if [ -z "$UV" ]; then
    say "uv is a small tool by Astral that can download Python. It will be installed in ~/.local/bin"
    say "from the official installer: $UV_URL"
    if [ "$DRY" = 1 ]; then
      plan "download and run $UV_URL, then 'uv venv --python $PREFERRED .venv'"
      UV="$HOME/.local/bin/uv"
    else
      tmp="$(mktemp)"
      if command -v curl >/dev/null 2>&1; then curl -LsSf "$UV_URL" -o "$tmp" || true
      elif command -v wget >/dev/null 2>&1; then wget -q "$UV_URL" -O "$tmp" || true; fi
      if [ ! -s "$tmp" ] || ! UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$HOME/.local/bin" sh "$tmp"; then
        rm -f "$tmp"
        say "Could not install uv (offline, or blocked?)." >&2
        manual_help
        exit 2
      fi
      rm -f "$tmp"
      UV="$HOME/.local/bin/uv"  # this shell's PATH does not know it yet, so call it by its full path
    fi
  fi
fi

# ---------------------------------------------------------------- 2. the virtual environment
NEW_VENV=0
[ -d .venv ] || NEW_VENV=1
if [ -x .venv/bin/python ] && ! py_ok .venv/bin/python; then
  say ".venv was built with an older Python ($(.venv/bin/python --version 2>&1))."
  if ask "Recreate .venv?"; then
    NEW_VENV=1
    if [ "$DRY" = 1 ]; then plan "remove .venv and create it again"; else rm -rf .venv; fi
  else
    say "Cannot continue with that .venv." >&2
    exit 1
  fi
fi

SPEC=".[${EXTRAS}]"
[ -z "$EXTRAS" ] && SPEC="."
STAMP="$(cksum < pyproject.toml) $EXTRAS"
if [ -x .venv/bin/predictor ] && [ "$(cat .venv/.installed 2>/dev/null || true)" = "$STAMP" ]; then
  say "Environment: .venv is already installed and up to date."
else
  if [ "$NEW_VENV" = 1 ]; then
    if [ "$USE_UV" = 1 ]; then plan "create .venv with Python $PREFERRED (uv)"; else plan "create .venv with $PY"; fi
    if [ "$DRY" != 1 ]; then
      if [ "$USE_UV" = 1 ]; then "$UV" venv --python "$PREFERRED" .venv || exit 2; else "$PY" -m venv .venv || exit 2; fi
    fi
  fi
  say "Installing the project ($SPEC) ..."
  if [ "$DRY" = 1 ]; then
    plan "pip install -e \"$SPEC\" into .venv"
  else
    if .venv/bin/python -m pip --version >/dev/null 2>&1; then
      .venv/bin/python -m pip install --quiet --upgrade pip && .venv/bin/python -m pip install -e "$SPEC" || exit 2
    elif [ -n "$UV" ]; then
      "$UV" pip install --python .venv/bin/python -e "$SPEC" || exit 2
    else
      say "This .venv has no pip and uv is not installed." >&2
      exit 2
    fi
    printf '%s' "$STAMP" > .venv/.installed
  fi
fi

# ---------------------------------------------------------------- 3. start
if [ "$NO_START" = 1 ]; then
  say "Done. Next: . .venv/bin/activate && predictor setup"
  exit 0
fi
if ask "Open the dashboard now? (starts the guided setup in your browser)"; then
  if [ "$DRY" = 1 ]; then plan "run .venv/bin/predictor setup"; exit 0; fi
  exec .venv/bin/predictor setup
fi
say ""
say "Whenever you want it:"
say "  . .venv/bin/activate"
say "  predictor setup        # guided, in the browser (add --cli for the terminal)"
say "  predictor run --open   # straight to the dashboard once setup is done"
