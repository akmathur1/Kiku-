//! The multitask decoding loop.
//!
//! The decoder is steered with the Whisper token format:
//! `<|startoftranscript|> → language → task → (timestamps | <|notimestamps|>)
//! → text → <|endoftext|>`. Language identification reads the distribution
//! over language tokens at the first step; voice activity detection combines
//! the <|nospeech|> probability with the average log-probability of the
//! decoded text (0.6 / -1.0, the thresholds validated in the paper); long
//! audio is windowed by the predicted timestamps, with the initial timestamp
//! constrained to the first second of each window.
//!
//! Decoding is KV-cached (each step feeds one token, not the whole prefix)
//! and runs the paper's temperature fallback: greedy first, then
//! progressively hotter sampling whenever the result shows the failure
//! signatures — a compression ratio above 2.4 (looping/repetition) or an
//! average log-probability below −1 (low evidence).

use candle_core::{Device, IndexOp, Tensor};

use crate::audio;
use crate::model::{DecoderCache, Kiku};
use crate::tokenizer::Tokenizer;

/// 20 ms per timestamp token.
const TIME_PRECISION: f32 = 0.02;
/// The initial timestamp of a window is constrained to [0, 1] s (§4.5).
const MAX_INITIAL_TIMESTAMP: u32 = 50;
/// VAD thresholds from the paper: no-speech prob > 0.6 AND avg logprob < -1.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const LOGPROB_THRESHOLD: f32 = -1.0;
/// Decoded text whose zlib compression ratio exceeds this is degenerate
/// repetition (§4.5) and triggers the temperature fallback.
const COMPRESSION_RATIO_THRESHOLD: f32 = 2.4;
/// The fallback ladder from the paper: greedy, then hotter sampling.
const TEMPERATURES: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Transcribe,
    /// X→English speech translation.
    Translate,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub task: Task,
    /// Force a language tag (e.g. "en"); `None` runs language identification
    /// on each window.
    pub language: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            task: Task::Transcribe,
            language: None,
        }
    }
}

/// One decoded segment, bounded by predicted timestamp tokens.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Seconds from the start of the input audio.
    pub start: f32,
    pub end: f32,
    pub text: String,
    /// Identified (or forced) language tag for the window.
    pub language: String,
    /// Mean log-probability of the decoded tokens — the caller's evidence
    /// signal; low values mean the text should not be trusted downstream.
    pub avg_logprob: f32,
    /// P(<|nospeech|>) at the first decode step of the window.
    pub no_speech_prob: f32,
}

pub struct Transcriber {
    pub model: Kiku,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl Transcriber {
    pub fn load(dir: &std::path::Path) -> anyhow::Result<Self> {
        let device = Device::Cpu;
        let model = Kiku::load(dir, &device)?;
        let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Transcribe 16 kHz mono samples of any length. Windows of 30 s are
    /// decoded in sequence, each window advanced by its last predicted
    /// timestamp so segments never straddle a window boundary.
    pub fn transcribe(&self, samples: &[f32], opts: &Options) -> anyhow::Result<Vec<Segment>> {
        let mut segments = Vec::new();
        let mut seek = 0usize;
        while seek < samples.len() {
            let window = &samples[seek..(seek + audio::N_SAMPLES).min(samples.len())];
            let offset = seek as f32 / audio::SAMPLE_RATE as f32;
            let (window_segments, advance) = self.decode_window(window, offset, opts)?;
            segments.extend(window_segments);
            seek += advance.max(1);
        }
        Ok(segments)
    }

    fn encode(&self, window: &[f32]) -> anyhow::Result<Tensor> {
        let n_mels = self.model.config.num_mel_bins;
        let mel = audio::log_mel_spectrogram(window, n_mels);
        let mel = Tensor::from_vec(mel, (1, n_mels, audio::N_FRAMES), &self.device)?;
        Ok(self.model.encoder.forward(&mel)?)
    }

    /// Next-token logits for tokens not yet in the cache.
    fn step(
        &self,
        new_tokens: &[u32],
        encoder_out: &Tensor,
        cache: &mut DecoderCache,
    ) -> anyhow::Result<Vec<f32>> {
        let input = Tensor::new(new_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .decoder
            .forward_cached(&input, encoder_out, cache)?;
        Ok(logits.i((0, new_tokens.len() - 1))?.to_vec1()?)
    }

    /// One rung of the fallback ladder: decode the window at a temperature.
    /// Returns the decoded tokens (prefix excluded) and their average
    /// log-probability under the untempered distribution.
    fn decode_with_temperature(
        &self,
        prefix: &[u32],
        encoder_out: &Tensor,
        temperature: f32,
        seed: u64,
    ) -> anyhow::Result<(Vec<u32>, f32)> {
        let sp = self.tokenizer.special;
        let mut cache = DecoderCache::default();
        let mut rng = Rng::new(seed);
        let max_len = self.model.config.max_target_positions / 2;

        let mut tokens: Vec<u32> = Vec::new();
        let mut sum_logprob = 0.0f32;
        let mut next_input: Vec<u32> = prefix.to_vec();
        while prefix.len() + tokens.len() < max_len {
            let mut logits = self.step(&next_input, encoder_out, &mut cache)?;
            self.apply_timestamp_rules(&mut logits, &tokens);
            let probs = softmax(&logits);

            // The paper's timestamp heuristic: if the total probability mass
            // on timestamp tokens exceeds the best text token, emit a
            // timestamp.
            let ts_mass: f32 = probs[sp.timestamp_begin as usize..].iter().sum();
            let best_text = argmax(&probs[..sp.timestamp_begin as usize]);
            let next = if ts_mass > probs[best_text] {
                sp.timestamp_begin as usize
                    + if temperature > 0.0 {
                        sample(
                            &logits[sp.timestamp_begin as usize..],
                            temperature,
                            &mut rng,
                        )
                    } else {
                        argmax(&probs[sp.timestamp_begin as usize..])
                    }
            } else if temperature > 0.0 {
                sample(
                    &logits[..sp.timestamp_begin as usize],
                    temperature,
                    &mut rng,
                )
            } else {
                best_text
            } as u32;

            // Evidence is always the untempered probability of the choice.
            sum_logprob += probs[next as usize].max(f32::MIN_POSITIVE).ln();
            if next == sp.eot {
                break;
            }
            tokens.push(next);
            next_input = vec![next];
        }
        let avg_logprob = sum_logprob / (tokens.len() + 1) as f32;
        Ok((tokens, avg_logprob))
    }

    /// Decode one 30-second window. Returns its segments and how many
    /// samples to advance the window by.
    fn decode_window(
        &self,
        window: &[f32],
        offset: f32,
        opts: &Options,
    ) -> anyhow::Result<(Vec<Segment>, usize)> {
        let sp = self.tokenizer.special;
        let encoder_out = self.encode(window)?;

        // First step from <|startoftranscript|> alone: read the no-speech
        // probability and identify the language.
        let mut lid_cache = DecoderCache::default();
        let logits = self.step(&[sp.sot], &encoder_out, &mut lid_cache)?;
        let probs = softmax(&logits);
        let no_speech_prob = probs[sp.no_speech as usize];
        let language_token = match &opts.language {
            Some(tag) => self.language_token_for(tag)?,
            None => {
                let range = sp.language_begin..sp.language_begin + sp.language_count;
                range
                    .clone()
                    .max_by(|&a, &b| probs[a as usize].partial_cmp(&probs[b as usize]).unwrap())
                    .unwrap_or(sp.language_begin)
            }
        };
        let language = self
            .tokenizer
            .language_tag(language_token)
            .unwrap_or("en")
            .to_string();
        let task_token = match opts.task {
            Task::Transcribe => sp.transcribe,
            Task::Translate => sp.translate,
        };
        let prefix = [sp.sot, language_token, task_token];

        // The temperature fallback ladder (§4.5): accept the first decode
        // that is neither degenerate repetition (compression ratio) nor
        // low-evidence (average log-probability); otherwise retry hotter.
        // When every rung fails the checks, keep the best-evidence
        // non-repetitive attempt — its weak evidence travels with the
        // segment for downstream gating.
        let mut tokens: Vec<u32> = Vec::new();
        let mut avg_logprob = f32::NEG_INFINITY;
        let mut last_attempt: Option<(Vec<u32>, f32)> = None;
        for (attempt, &temperature) in TEMPERATURES.iter().enumerate() {
            let (t, lp) = self.decode_with_temperature(
                &prefix,
                &encoder_out,
                temperature,
                // Deterministic per window and per rung, so a transcription
                // is reproducible end to end.
                (offset.to_bits() as u64) ^ ((attempt as u64) << 32),
            )?;
            let text = self.tokenizer.decode(&t);
            let repetitive = compression_ratio(&text) > COMPRESSION_RATIO_THRESHOLD;
            if !repetitive && (lp > avg_logprob || tokens.is_empty()) {
                tokens = t.clone();
                avg_logprob = lp;
            }
            last_attempt = Some((t, lp));
            let needs_fallback = repetitive || lp < LOGPROB_THRESHOLD;
            // A window the VAD gate will drop anyway (both arms: high
            // no-speech AND low evidence) cannot be rescued by temperature.
            let is_silence = no_speech_prob > NO_SPEECH_THRESHOLD && lp < LOGPROB_THRESHOLD;
            if !needs_fallback || is_silence {
                break;
            }
        }
        if tokens.is_empty() {
            // Every rung was repetitive: keep the last attempt with its weak
            // evidence rather than silently dropping a speech window.
            if let Some((t, lp)) = last_attempt {
                tokens = t;
                avg_logprob = lp;
            }
        }
        let decoded = &tokens[..];

        // Voice activity detection: silence/non-speech skips the window.
        if no_speech_prob > NO_SPEECH_THRESHOLD && avg_logprob < LOGPROB_THRESHOLD {
            return Ok((Vec::new(), audio::N_SAMPLES));
        }

        let ts_seconds = |id: u32| -> f32 { (id - sp.timestamp_begin) as f32 * TIME_PRECISION };
        let window_seconds = window.len() as f32 / audio::SAMPLE_RATE as f32;
        let mut segments = Vec::new();
        let mut seg_start = 0.0f32;
        let mut seg_tokens: Vec<u32> = Vec::new();
        let mut last_ts = 0.0f32;
        for &id in decoded {
            if id >= sp.timestamp_begin {
                // A closing timestamp can point past a short final window's
                // real audio; clamp so segments never outlive the recording.
                let t = ts_seconds(id).min(window_seconds);
                last_ts = t;
                if seg_tokens.is_empty() {
                    seg_start = t;
                } else {
                    segments.push(Segment {
                        start: offset + seg_start,
                        end: offset + t,
                        text: self.tokenizer.decode(&seg_tokens).trim().to_string(),
                        language: language.clone(),
                        avg_logprob,
                        no_speech_prob,
                    });
                    seg_tokens.clear();
                }
            } else {
                seg_tokens.push(id);
            }
        }
        if !seg_tokens.is_empty() {
            segments.push(Segment {
                start: offset + seg_start,
                end: offset + window_seconds,
                text: self.tokenizer.decode(&seg_tokens).trim().to_string(),
                language,
                avg_logprob,
                no_speech_prob,
            });
            last_ts = window_seconds;
        }
        segments.retain(|s| !s.text.is_empty());

        // Advance the window to the last closed timestamp so the next window
        // starts where this one's transcription genuinely ended.
        let advance = if last_ts > 0.0 {
            ((last_ts * audio::SAMPLE_RATE as f32) as usize).min(audio::N_SAMPLES)
        } else {
            audio::N_SAMPLES
        };
        Ok((segments, advance.max(audio::SAMPLE_RATE / 2)))
    }

    fn language_token_for(&self, tag: &str) -> anyhow::Result<u32> {
        let sp = self.tokenizer.special;
        (sp.language_begin..sp.language_begin + sp.language_count)
            .find(|&id| self.tokenizer.language_tag(id) == Some(tag))
            .ok_or_else(|| anyhow::anyhow!("unknown language tag: {tag}"))
    }

    /// Whisper's timestamp grammar, applied as logit suppression:
    /// specials never sample (except <|endoftext|>), timestamps appear in
    /// pairs and never decrease, and the first timestamp of a window falls
    /// within its first second.
    fn apply_timestamp_rules(&self, logits: &mut [f32], sampled: &[u32]) {
        let sp = self.tokenizer.special;
        // Suppress every special token below the timestamp range except EOT.
        for id in sp.sot..sp.timestamp_begin {
            if id != sp.eot {
                logits[id as usize] = f32::NEG_INFINITY;
            }
        }

        let is_ts = |id: u32| id >= sp.timestamp_begin;
        let last_was_ts = sampled.last().copied().map(is_ts).unwrap_or(false);
        let penultimate_was_ts = sampled.len() < 2
            || sampled
                .get(sampled.len() - 2)
                .copied()
                .map(is_ts)
                .unwrap_or(false);
        if last_was_ts {
            if penultimate_was_ts {
                // A closed pair: the next token must be text (or EOT).
                for l in logits[sp.timestamp_begin as usize..].iter_mut() {
                    *l = f32::NEG_INFINITY;
                }
            } else {
                // An open timestamp after text must close: only timestamps
                // or EOT may follow.
                for id in 0..sp.timestamp_begin {
                    if id != sp.eot {
                        logits[id as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Timestamps never decrease. When the last token closed a segment
        // (a timestamp right after text), that value stays legal so the next
        // segment may open on the shared boundary; otherwise the next
        // timestamp must strictly advance.
        if let Some(&max_ts) = sampled.iter().filter(|&&id| is_ts(id)).max() {
            let allow_equal = last_was_ts && !penultimate_was_ts;
            let end = if allow_equal { max_ts } else { max_ts + 1 };
            for id in sp.timestamp_begin..end {
                logits[id as usize] = f32::NEG_INFINITY;
            }
        }

        // The first sampled token is constrained to an initial timestamp
        // within the first second of the window.
        if sampled.is_empty() {
            for (i, l) in logits.iter_mut().enumerate() {
                let id = i as u32;
                let allowed =
                    id >= sp.timestamp_begin && id <= sp.timestamp_begin + MAX_INITIAL_TIMESTAMP;
                if !allowed {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// zlib compression ratio of the text — degenerate token loops compress far
/// better than natural language (§4.5's repetition detector).
fn compression_ratio(text: &str) -> f32 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 0.0;
    }
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    use std::io::Write;
    encoder.write_all(bytes).expect("in-memory zlib write");
    let compressed = encoder.finish().expect("in-memory zlib finish");
    bytes.len() as f32 / compressed.len() as f32
}

/// A small deterministic xorshift64* generator — sampling stays reproducible
/// for a given audio input without pulling in an RNG dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let bits = x.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }
}

/// Sample an index from softmax(logits / temperature).
fn sample(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    let tempered: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    let probs = softmax(&tempered);
    let target = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if target < cumulative {
            return i;
        }
    }
    argmax(&probs)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one_and_orders_by_logit() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn argmax_picks_the_peak() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3]), 1);
    }

    #[test]
    fn compression_ratio_flags_repetition() {
        let looped = "the meeting the meeting the meeting ".repeat(20);
        assert!(compression_ratio(&looped) > COMPRESSION_RATIO_THRESHOLD);
        let natural = "Revenue grew eight percent while churn held steady across Europe.";
        assert!(compression_ratio(natural) < COMPRESSION_RATIO_THRESHOLD);
        assert_eq!(compression_ratio(""), 0.0);
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed_and_respects_suppression() {
        let logits = vec![0.0, f32::NEG_INFINITY, 2.0, 1.0];
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..50 {
            let i = sample(&logits, 0.8, &mut a);
            assert_eq!(i, sample(&logits, 0.8, &mut b));
            assert_ne!(i, 1, "a suppressed token must never sample");
        }
    }
}
