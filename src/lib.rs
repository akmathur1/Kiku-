//! Kiku — Molterra's open-source speech recognition module.
//!
//! An automatic speech recognition system implemented as an encoder-decoder
//! Transformer following the Whisper architecture (Radford et al., 2022):
//! input audio is split into 30-second chunks, converted into an 80-channel
//! log-Mel spectrogram, and passed to the encoder; the decoder predicts text
//! intermixed with special tokens that direct the single model to perform
//! language identification, phrase-level timestamps, multilingual speech
//! transcription, and to-English speech translation.
//!
//! Kiku loads open Whisper checkpoints (Hugging Face safetensors layout) as
//! its starting weights, so it transcribes for real today while the
//! architecture, decoding loop, and frontend are fully ours to evolve.
//!
//! What Kiku deliberately does NOT do: it never infers *who* is speaking.
//! Speaker identity in Molterra stays with the structural channel attribution
//! and meeting-scoped diarization in `capture` — the paper documents Whisper
//! confidently guessing speaker names from transcript context, and that
//! failure mode stays out of our trust path by construction.

pub mod audio;
pub mod decode;
pub mod model;
pub mod normalize;
pub mod tokenizer;
pub mod wer;

pub use decode::{Options, Segment, Task, Transcriber};
