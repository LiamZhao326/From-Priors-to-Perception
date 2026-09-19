#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CATEGORY INPUT_DIR OUTPUT_JSON" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${ROOT}/01_scene_construction/construct_scenarios.py" \
  --category "$1" \
  --input-dir "$2" \
  --output "$3"
