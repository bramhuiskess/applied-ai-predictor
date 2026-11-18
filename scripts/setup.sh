#!/usr/bin/env bash
# Kept for older instructions: create .venv and install, without starting anything.
# Usage: ./scripts/setup.sh [extras]     (default extras: train,dev,datasets)
exec "$(dirname "$0")/../start.sh" --no-start --extras "${1:-train,dev,datasets}"
