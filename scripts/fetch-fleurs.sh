#!/usr/bin/env bash
set -euo pipefail

LANG_DIR="${1:?usage: fetch-fleurs.sh <lang> [dest-dir]}"
DEST="${2:-$(dirname "$0")/../data/fleurs}"
URL="https://storage.googleapis.com/xtreme_translations/FLEURS102/$LANG_DIR.tar.gz"

mkdir -p "$DEST"
if [ ! -d "$DEST/$LANG_DIR" ]; then
  echo "fetching $LANG_DIR ..."
  curl -fSL "$URL" | tar -xz -C "$DEST"
fi
echo "ready: $DEST/$LANG_DIR"
