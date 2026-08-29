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

use candle_core::{Device, IndexOp, Tensor};

use crate::audio;
use crate::model::Kiku;
use crate::tokenizer::Tokenizer;

/// 20 ms per timestamp token.
const TIME_PRECISION: f32 = 0.02;
/// The initial timestamp of a window is constrained to [0, 1] s (§4.5).
const MAX_INITIAL_TIMESTAMP: u32 = 50;
/// VAD thresholds from the paper: no-speech prob > 0.6 AND avg logprob < -1.
const NO_SPEECH_THRESHOLD: f32 = 0.6;
const LOGPROB_THRESHOLD: f32 = -1.0;

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

    /// Next-token logits for the current prefix.
    fn step(&self, tokens: &[u32], encoder_out: &Tensor) -> anyhow::Result<Vec<f32>> {
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.decoder.forward(&input, encoder_out)?;
        Ok(logits.i((0, tokens.len() - 1))?.to_vec1()?)
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
        let logits = self.step(&[sp.sot], &encoder_out)?;
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

        let mut tokens = vec![sp.sot, language_token, task_token];
        let sample_begin = tokens.len();
        let max_len = self.model.config.max_target_positions / 2;
        let mut sum_logprob = 0.0f32;

        while tokens.len() < max_len {
            let mut logits = self.step(&tokens, &encoder_out)?;
            self.apply_timestamp_rules(&mut logits, &tokens[sample_begin..]);
            let probs = softmax(&logits);

            // The paper's timestamp heuristic: if the total probability mass
            // on timestamp tokens exceeds the best text token, emit the most
            // probable timestamp.
            let ts_mass: f32 = probs[sp.timestamp_begin as usize..].iter().sum();
            let best_text = argmax(&probs[..sp.timestamp_begin as usize]);
            let next = if ts_mass > probs[best_text] {
                sp.timestamp_begin as usize + argmax(&probs[sp.timestamp_begin as usize..])
            } else {
                best_text
            } as u32;

            sum_logprob += probs[next as usize].max(f32::MIN_POSITIVE).ln();
            if next == sp.eot {
                break;
            }
            tokens.push(next);
        }

        let decoded = &tokens[sample_begin..];
        let avg_logprob = sum_logprob / (decoded.len() + 1) as f32;

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
                let t = ts_seconds(id);
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
}
