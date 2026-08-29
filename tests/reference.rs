//! The Rust port of the reference (openai/whisper) test suite, minus the
//! parts that test features Kiku has not built yet (word-level timestamps via
//! cross-attention DTW and its median filter — those tests land with that
//! feature).
//!
//! Two tiers, mirroring the repo's has_db pattern honestly:
//! - fixture-only tests run unconditionally (`tests/fixtures/jfk.wav`);
//! - checkpoint-backed tests return early when `models/tiny` is absent
//!   (CI has no checkpoint), so a green run there proves nothing — the real
//!   execution is local with a fetched model. Run `scripts/fetch-model.sh
//!   tiny` first to make them real.

use std::path::{Path, PathBuf};

use kiku::audio::{self, HOP_LENGTH, N_FFT, SAMPLE_RATE};
use kiku::tokenizer::Tokenizer;
use kiku::{Options, Transcriber};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn model_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/tiny");
    dir.join("model.safetensors").exists().then_some(dir)
}

fn load_jfk() -> Vec<f32> {
    let mut reader = hound::WavReader::open(fixture("jfk.wav")).expect("jfk.wav fixture");
    let spec = reader.spec();
    assert_eq!(spec.channels, 1);
    assert_eq!(spec.sample_rate as usize, SAMPLE_RATE);
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

/// Port of test_audio: the fixture decodes to 10–12 s of mono 16 kHz audio
/// with sane amplitude, and the log-Mel output obeys the reference dynamic
/// range (clamped to 8 below the peak, scaled by /4 → max − min ≤ 2.0).
#[test]
fn audio_and_log_mel_invariants() {
    let samples = load_jfk();
    assert!(SAMPLE_RATE * 10 < samples.len() && samples.len() < SAMPLE_RATE * 12);

    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let std =
        (samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / samples.len() as f32).sqrt();
    assert!(0.0 < std && std < 1.0);

    // The frontend pads/trims to a full 30 s chunk, so the output length is
    // fixed regardless of the input duration.
    let mel = audio::log_mel_spectrogram(&samples, 80);
    assert_eq!(mel.len(), 80 * audio::N_FRAMES);
    let max = mel.iter().cloned().fold(f32::MIN, f32::max);
    let min = mel.iter().cloned().fold(f32::MAX, f32::min);
    assert!(max - min <= 2.0, "dynamic range {} > 2.0", max - min);
    assert!(mel.iter().all(|v| v.is_finite()));

    // Deterministic: the same audio produces the same spectrogram.
    let again = audio::log_mel_spectrogram(&samples, 80);
    assert_eq!(mel, again);

    // The frontend constants the model depends on.
    assert_eq!(N_FFT, 400);
    assert_eq!(HOP_LENGTH, 160);
}

/// Port of test_tokenizer: language tokens form a contiguous run after SOT,
/// every language tag looks like a language (2–3 lowercase letters), no
/// control token is counted as a language, and all language tokens sit below
/// the timestamp range.
#[test]
fn tokenizer_language_token_invariants() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no checkpoint at models/tiny (run scripts/fetch-model.sh tiny)");
        return;
    };
    let tok = Tokenizer::load(&dir.join("tokenizer.json")).expect("tokenizer loads");
    let sp = tok.special;

    assert!(sp.language_count > 0);
    assert_eq!(sp.language_begin, sp.sot + 1);
    for i in 0..sp.language_count {
        let id = sp.language_begin + i;
        let tag = tok
            .language_tag(id)
            .unwrap_or_else(|| panic!("token {id} in the language run has no tag"));
        assert!(
            matches!(tag.len(), 2..=3) && tag.chars().all(|c| c.is_ascii_lowercase()),
            "language tag {tag:?} is not 2-3 lowercase letters"
        );
        assert!(id < sp.timestamp_begin);
    }
    // Control tokens are not language tags.
    for id in [sp.transcribe, sp.translate, sp.no_speech, sp.no_timestamps] {
        assert!(
            tok.language_tag(id).is_none(),
            "control token {id} counted as language"
        );
        assert!(!(sp.language_begin..sp.language_begin + sp.language_count).contains(&id));
    }
    // Special/timestamp tokens decode to no text.
    assert_eq!(
        tok.decode(&[sp.sot, sp.language_begin, sp.transcribe, sp.timestamp_begin]),
        ""
    );
}

/// Port of test_transcribe: real end-to-end decoding of the JFK fixture with
/// an open checkpoint — correct language, the famous words present, segments
/// starting at 0.00 with monotonic, ordered timestamps.
#[test]
fn transcribe_jfk_end_to_end() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no checkpoint at models/tiny (run scripts/fetch-model.sh tiny)");
        return;
    };
    let transcriber = Transcriber::load(&dir).expect("model loads");
    let samples = load_jfk();
    let segments = transcriber
        .transcribe(&samples, &Options::default())
        .expect("transcription succeeds");
    assert!(!segments.is_empty());

    let text = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    assert!(text.contains("my fellow americans"), "got: {text}");
    assert!(text.contains("your country"), "got: {text}");
    assert!(text.contains("do for you"), "got: {text}");

    assert_eq!(segments[0].language, "en");
    assert_eq!(segments[0].start, 0.0);
    let mut prev_end = 0.0f32;
    for s in &segments {
        assert!(s.start <= s.end, "segment {} > {}", s.start, s.end);
        assert!(s.start >= prev_end, "segments out of order");
        prev_end = s.end;
        assert!(s.no_speech_prob < 0.6, "speech flagged as no-speech");
        assert!(
            s.avg_logprob > -1.0,
            "low-confidence decode of clean speech"
        );
    }
}
