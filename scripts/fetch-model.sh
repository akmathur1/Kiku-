#!/usr/bin/env bash
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
