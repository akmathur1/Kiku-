# Kiku model card

Kiku loads openly published, MIT-licensed checkpoints into our own Rust
implementation of the architecture. This card states the facts that matter
for Molterra: what our runtime adds and what it does not change.

## Checkpoints

| size | parameters | multilingual | notes |
|---|---|---|---|
| tiny | 39 M | yes (also `tiny.en`) | development/smoke default |
| base | 74 M | yes (also `base.en`) | |
| small | 244 M | yes (also `small.en`) | |
| medium | 769 M | yes (also `medium.en`) | |
| large (v1 to v3) | 1550 M | yes | best accuracy |
| large-v3-turbo | 809 M | yes | pruned decoder; **not trained for translation** |

The `.en` variants perform better for English-only audio, most noticeably at
tiny/base size. `turbo` keeps returning the source language even when asked
to translate, so use a full multilingual checkpoint for translation into
English.

## Training data and language coverage

The weights were trained on 680,000 hours of weakly-supervised web audio
(~117k hours non-English across ~98 languages, ~125k hours of translation
into English).
Accuracy varies widely by language: high-resource
languages (English, Spanish, German, …) reach strong WER; low-resource
languages can be an order of magnitude worse, and some are only usable with
the large checkpoints. Kiku's FLEURS harness (`eval_fleurs`) measures this
per language on our own runtime.

## Known limitations (inherited from the checkpoints)

- **No word-accuracy guarantee.** State of the art is *low* error, not zero
  error; evidence fields (avg logprob, no-speech prob) exist so downstream
  consumers gate rather than trust.
- **Hallucination.** A sequence-to-sequence decoder can emit text that was
  never spoken, especially on non-speech audio, long silence, or heavy
  noise. Kiku counters with the VAD rule, the compression-ratio
  repetition detector, and the temperature fallback ladder, and Molterra's
  hearing pipeline gates on evidence above that.
- **Repetition loops.** Autoregressive decoding can loop; detected via zlib
  compression ratio > 2.4 and retried at higher temperature.
- **Uneven demographic/language performance.** Accuracy differs by language,
  accent, acoustic conditions, and demographic dimensions of the speaker.
- **Speaker identity.** The model will guess speaker names from context.
  Kiku discards this: speaker identity belongs to structural channel
  attribution and diarization in `capture/`, never to ASR text.

## Out-of-scope uses

No surveillance of individuals without consent, no
subjective classification of speakers, no high-risk decision-making from
raw transcripts. In Molterra, recording is consent-gated and transcripts
are evidence-carrying inputs to a gated pipeline, not ground truth.
