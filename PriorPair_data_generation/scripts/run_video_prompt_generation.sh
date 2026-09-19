#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 METADATA_DIR OUTPUT_JSON" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${ROOT}/03_generative_generation/generate_video_prompts.py" \
  --metadata-dir "$1" \
  --output "$2"
