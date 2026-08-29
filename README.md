# Kiku

Kiku is Molterra's speech recognition module: an automatic speech recognition
(ASR) system implemented as an encoder-decoder Transformer following the
Whisper architecture (Radford et al., 2022, *Robust Speech Recognition via
Large-Scale Weak Supervision*), written from scratch in Rust.

Input audio is split into 30-second chunks, converted into an 80-channel
log-Mel spectrogram, and passed into the encoder. The decoder predicts the
corresponding text intermixed with special tokens that direct the single model
to perform language identification, phrase-level timestamps, multilingual
speech transcription, and to-English speech translation.

Kiku loads open Whisper checkpoints (the Hugging Face safetensors layout) as
its starting weights, so it transcribes for real today while the architecture,
frontend, and decoding loop are fully ours to evolve for meeting audio —
low-volume speech, background chatter, technical vocabulary — and, later, our
own training runs.

## Layout

| module | what it is |
|---|---|
| `audio` | 16 kHz resampling, 80-channel log-Mel frontend (25 ms window / 10 ms hop, N_FFT 400, hop 160), 30 s chunk padding |
| `model` | Conv1D+GELU stem (second conv stride 2), sinusoidal-position encoder, learned-position decoder with cross-attention, tied output projection |
| `tokenizer` | byte-level BPE decoding + the multitask special-token map, read from the checkpoint's `tokenizer.json` |
| `decode` | the multitask loop: SOT → language → task → timestamps → text → EOT, with the paper's decoding heuristics |

Decoding implements the reliability heuristics validated in the paper:

- **Voice activity detection**: a window is treated as non-speech only when
  P(`<|nospeech|>`) > 0.6 *and* the average log-probability of the decoded
  text is < −1 — the no-speech probability alone is not sufficient (§4.5).
- **Timestamp grammar**: timestamps appear in pairs, never decrease, and the
  initial timestamp of each window is constrained to its first second so the
  model cannot ignore the opening words.
- **Long-form audio**: windows advance by the last predicted timestamp, so a
  segment never straddles a window boundary.

Every segment carries evidence: start/end times, the identified language, the
average log-probability, and the no-speech probability. Downstream consumers
gate on that evidence — a low-confidence segment is display material, not
trusted reasoning input.

## Usage

```bash
# Fetch an open checkpoint (tiny/base/small/medium):
./scripts/fetch-model.sh tiny

# Transcribe a WAV file (any sample rate / channel count):
cargo run --release --bin transcribe -- models/tiny audio.wav

# Force a language, or translate X→English:
cargo run --release --bin transcribe -- models/tiny audio.wav --language de
cargo run --release --bin transcribe -- models/tiny audio.wav --translate
```

As a library:

```rust
let t = kiku::Transcriber::load(Path::new("models/tiny"))?;
let segments = t.transcribe(&samples_16k_mono, &kiku::Options::default())?;
```

## Evaluation: LibriSpeech word error rate

The Rust counterpart of OpenAI's LibriSpeech evaluation notebook: transcribe a
LibriSpeech split, normalize hypothesis and reference with the English
normalizer (`normalize` module), and compute pooled corpus WER (`wer` module).

```bash
# Fetch a split and convert FLAC → WAV (needs ffmpeg; ~350 MB for test-clean):
./scripts/fetch-librispeech.sh test-clean

# Full run, or subsample for a quick pass:
cargo run --release --bin eval_librispeech -- models/tiny data/LibriSpeech/test-clean
cargo run --release --bin eval_librispeech -- models/tiny data/LibriSpeech/test-clean --every 50
cargo run --release --bin eval_librispeech -- models/tiny data/LibriSpeech/test-clean --limit 100 --tsv out.tsv
```

The normalizer is a full port of Whisper's `EnglishTextNormalizer` — the
number normalizer (currencies, ordinals, decades, "double oh seven"), the
British→American spelling dictionary (`assets/english.json`, upstream's
table with its one corrupted entry — `archaeology` → `archeology</span>` — fixed), contractions and title abbreviations, and Unicode symbol/diacritic
removal — with upstream's normalizer test suite ported alongside it
(`tests/normalizers.rs`). Evaluation output is measurement, never trusted
meeting memory.

## Evaluation: FLEURS multilingual transcription and translation

The Rust counterpart of OpenAI's Multilingual_ASR notebook: transcribe a
FLEURS language in its own language (forced, as the notebook does), normalize
hypothesis and reference with the language-agnostic basic normalizer
(`normalize::normalize_basic`), and compute pooled corpus WER — or CER for
languages written without spaces (zh, ja, th, lo, my, km). `--translate`
additionally runs X→English translation; FLEURS ships no English reference
for it, so translations go to the TSV for inspection rather than being scored.

```bash
# Fetch a language (audio ships as WAV; e.g. ko_kr, de_de, es_419, cmn_hans_cn):
./scripts/fetch-fleurs.sh es_419

cargo run --release --bin eval_fleurs -- models/tiny data/fleurs/es_419 --limit 20
cargo run --release --bin eval_fleurs -- models/tiny data/fleurs/es_419 --limit 20 --translate --tsv out.tsv
```

## What Kiku deliberately does not do

Kiku never infers *who* is speaking. The paper documents Whisper confidently
guessing speaker names from transcript context; in Molterra, speaker identity
belongs to the structural channel attribution and meeting-scoped diarization
in `capture/`, and an ASR model's guess is never identity evidence. Kiku's
output is transcription evidence — text, times, language, confidence — and the
existing hearing pipeline (record pass, reconciliation, hard-word repair,
acoustic-regime gating) remains the trust layer above any ASR backend.

## Relationship to `capture/`

Kiku is a standalone crate (its own workspace, like `capture/`). The capture
pipeline currently uses a hosted ASR backend; Kiku is the seam for a local,
open backend behind the same evidence contract. Integration lands as its own
slice: an ASR-backend trait in `capture/` with Kiku as one implementation,
selected per session, with the H4–H8 stages unchanged above it.

## Status

- Implemented: frontend, encoder-decoder model, checkpoint loading, KV-cached
  decoding with the timestamp grammar, language ID, translation, VAD, the
  temperature fallback ladder (compression-ratio repetition detection +
  log-probability gating, §4.5), long-form windowing, WAV CLI. Verified
  end-to-end with `whisper-tiny` on real synthesized speech and silence.
  Checkpoint facts and inherited limitations: [MODEL-CARD.md](MODEL-CARD.md).
- Not yet: beam search, previous-text conditioning (the tenant keyterm
  boosting hook), word-level timestamps (cross-attention DTW), streaming,
  our own training runs. These land as follow-up slices.
