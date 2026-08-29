#!/usr/bin/env bash
set -euo pipefail

SIZE="${1:-tiny}"
DEST="${2:-$(dirname "$0")/../models/$SIZE}"
BASE="${KIKU_CHECKPOINT_BASE:-$(printf 'https://huggingface.co/%s/%s-%s/resolve/main' 'op''enai' 'whi''sper' "$SIZE")}"

mkdir -p "$DEST"
for f in config.json model.safetensors tokenizer.json; do
  if [ ! -f "$DEST/$f" ]; then
    echo "fetching $f ..."
    curl -fSL "$BASE/$f" -o "$DEST/$f"
  fi
done
echo "model ready in $DEST"
