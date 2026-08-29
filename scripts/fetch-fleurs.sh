#!/usr/bin/env bash
# Fetch a FLEURS language for the eval-fleurs harness (audio ships as WAV;
# nothing downloaded is committed).
#
# Usage: fetch-fleurs.sh <lang> [dest-dir]
#   lang: a FLEURS language directory name, e.g. ko_kr, de_de, cmn_hans_cn
#         (the full list is in the reference Multilingual_ASR notebook)
#   dest: default data/fleurs (the language extracts to <dest>/<lang>)
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
