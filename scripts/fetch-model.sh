#!/usr/bin/env bash
# Fetch an open Whisper checkpoint in the layout Kiku loads:
# config.json + model.safetensors + tokenizer.json.
#
# Usage: fetch-model.sh [size] [dest-dir]
#   size: tiny | base | small | medium (default: tiny)
#   dest: default models/<size>
set -euo pipefail

SIZE="${1:-tiny}"
DEST="${2:-$(dirname "$0")/../models/$SIZE}"
BASE="https://huggingface.co/openai/whisper-$SIZE/resolve/main"

mkdir -p "$DEST"
for f in config.json model.safetensors tokenizer.json; do
  if [ ! -f "$DEST/$f" ]; then
    echo "fetching $f ..."
    curl -fSL "$BASE/$f" -o "$DEST/$f"
  fi
done
echo "model ready in $DEST"
