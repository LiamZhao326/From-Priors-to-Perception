#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CATEGORY METADATA VERIFIED_ANNOTATIONS POSITIVE_DIR NEGATIVE_DIR OUTPUT_JSON" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${ROOT}/04_label_generation/generate_oav_labels.py" \
  --category "$1" \
  --metadata "$2" \
  --verified-annotations "$3" \
  --positive-dir "$4" \
  --negative-dir "$5" \
  --output "$6"
