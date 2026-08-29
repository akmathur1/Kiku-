#!/usr/bin/env bash
# Fetch a LibriSpeech split and convert its FLAC utterances to WAV for the
# eval-librispeech harness (kiku reads WAV; ffmpeg handles FLAC).
#
# Usage: fetch-librispeech.sh [split] [dest-dir]
#   split: test-clean (default) | test-other | dev-clean | dev-other
#   dest:  default kiku/data (the split extracts to <dest>/LibriSpeech/<split>)
set -euo pipefail

SPLIT="${1:-test-clean}"
DEST="${2:-$(dirname "$0")/../data}"
URL="https://www.openslr.org/resources/12/$SPLIT.tar.gz"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

mkdir -p "$DEST"
if [ ! -d "$DEST/LibriSpeech/$SPLIT" ]; then
  echo "fetching $SPLIT ..."
  curl -fSL "$URL" | tar -xz -C "$DEST"
fi

echo "converting FLAC -> WAV ..."
find "$DEST/LibriSpeech/$SPLIT" -name '*.flac' | while read -r flac; do
  wav="${flac%.flac}.wav"
  [ -f "$wav" ] || ffmpeg -loglevel error -n -i "$flac" "$wav" </dev/null
done
echo "ready: $DEST/LibriSpeech/$SPLIT"
