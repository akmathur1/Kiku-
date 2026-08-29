//! `eval-fleurs` — multilingual transcription (and X→English translation)
//! over a FLEURS language, the Rust counterpart of the reference
//! Multilingual_ASR notebook.
//!
//! Usage:
//!   eval-fleurs <model-dir> <fleurs-lang-dir> [--limit N] [--every N] [--tsv out.tsv] [--translate]
//!
//! `<fleurs-lang-dir>` is an extracted FLEURS language directory (e.g.
//! `data/fleurs/ko_kr`, from kiku/scripts/fetch-fleurs.sh) holding `test.tsv`
//! and `audio/test/*.wav`. Each utterance is transcribed in its own language
//! (forced, as the notebook does) and scored against the reference
//! transcription after `normalize_basic` on both sides — pooled corpus WER,
//! or CER for languages written without spaces. With `--translate`, each
//! utterance is additionally translated to English; FLEURS ships no English
//! reference for the translation task, so translations are written to the
//! TSV for inspection rather than scored.
//!
//! Evaluation output is measurement only — it never becomes meeting memory.

use std::io::Write;
use std::path::{Path, PathBuf};

use kiku::{audio, normalize, wer, Options, Task, Transcriber};

struct Utterance {
    wav: PathBuf,
    reference: String,
}

/// FLEURS directory name → Whisper language code. The regular case is the
/// prefix before the first underscore; the exceptions are spelled out.
fn whisper_language(fleurs_lang: &str) -> anyhow::Result<&str> {
    Ok(match fleurs_lang {
        "cmn_hans_cn" | "yue_hant_hk" => "zh",
        "fil_ph" => "tl",
        "nb_no" => "no",
        other => other
            .split('_')
            .next()
            .filter(|p| !p.is_empty())
            .ok_or_else(|| anyhow::anyhow!("cannot derive a language code from {other:?}"))?,
    })
}

/// Languages written without spaces are scored per character, as the
/// reference notebook does (word splits are not meaningful there).
fn scores_by_character(language: &str) -> bool {
    matches!(language, "zh" | "ja" | "th" | "lo" | "my" | "km")
}

fn collect_utterances(lang_dir: &Path) -> anyhow::Result<Vec<Utterance>> {
    let tsv_path = lang_dir.join("test.tsv");
    let audio_dir = lang_dir.join("audio").join("test");
    let tsv = std::fs::read_to_string(&tsv_path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", tsv_path.display()))?;

    let mut utterances = Vec::new();
    for (n, line) in tsv.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        // Columns (no header): id, file_name, raw_transcription, transcription, …
        let cols: Vec<&str> = line.split('\t').collect();
        anyhow::ensure!(
            cols.len() >= 4,
            "{} line {}: expected ≥4 tab-separated columns, got {}",
            tsv_path.display(),
            n + 1,
            cols.len()
        );
        let wav = audio_dir.join(cols[1]);
        anyhow::ensure!(
            wav.exists(),
            "{} lists {} but {} is missing — an incomplete corpus would silently \
             skew the error rate; re-run scripts/fetch-fleurs.sh",
            tsv_path.display(),
            cols[1],
            wav.display()
        );
        utterances.push(Utterance {
            wav,
            reference: cols[3].to_string(),
        });
    }
    utterances.sort_by(|a, b| a.wav.cmp(&b.wav));
    Ok(utterances)
}

fn read_wav_16k(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let channels = spec.channels as usize;
    let mono: Vec<f32> = raw
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    Ok(audio::resample_to_16k(&mono, spec.sample_rate as usize))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut limit = usize::MAX;
    let mut every = 1usize;
    let mut tsv: Option<PathBuf> = None;
    let mut translate = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                i += 1;
                limit = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--limit needs a value"))?
                    .parse()?;
            }
            "--every" => {
                i += 1;
                every = args
                    .get(i)
                    .ok_or_else(|| anyhow::anyhow!("--every needs a value"))?
                    .parse::<usize>()?
                    .max(1);
            }
            "--tsv" => {
                i += 1;
                tsv = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--tsv needs a path"))?
                        .into(),
                );
            }
            "--translate" => translate = true,
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let [model_dir, lang_dir] = positional.as_slice() else {
        anyhow::bail!(
            "usage: eval-fleurs <model-dir> <fleurs-lang-dir> [--limit N] [--every N] [--tsv out.tsv] [--translate]"
        );
    };

    let lang_dir = Path::new(lang_dir);
    let fleurs_lang = lang_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| {
            anyhow::anyhow!("cannot read a language name from {}", lang_dir.display())
        })?;
    let language = whisper_language(&fleurs_lang)?.to_string();
    let by_character = scores_by_character(&language);

    let all = collect_utterances(lang_dir)?;
    anyhow::ensure!(
        !all.is_empty(),
        "no utterances found under {} — expected test.tsv with audio/test/*.wav (run scripts/fetch-fleurs.sh)",
        lang_dir.display()
    );
    let selected: Vec<&Utterance> = all.iter().step_by(every).take(limit).collect();
    anyhow::ensure!(
        !selected.is_empty(),
        "--limit/--every selected zero utterances — refusing to report an error rate over nothing"
    );
    eprintln!(
        "{} ({language}): {} utterances in corpus; evaluating {} (every {}, limit {})",
        fleurs_lang,
        all.len(),
        selected.len(),
        every,
        if limit == usize::MAX {
            "none".to_string()
        } else {
            limit.to_string()
        }
    );

    let transcriber = Transcriber::load(Path::new(model_dir))?;
    let transcribe_opts = Options {
        language: Some(language.clone()),
        ..Options::default()
    };
    let translate_opts = Options {
        language: Some(language.clone()),
        task: Task::Translate,
    };

    let mut pairs: Vec<(String, String)> = Vec::with_capacity(selected.len());
    let mut tsv_out = tsv
        .as_ref()
        .map(|p| -> anyhow::Result<_> {
            let mut f = std::fs::File::create(p)?;
            writeln!(
                f,
                "wav\treference\thypothesis\tnormalized_reference\tnormalized_hypothesis{}",
                if translate { "\ttranslation" } else { "" }
            )?;
            Ok(f)
        })
        .transpose()?;

    for (n, utt) in selected.iter().enumerate() {
        let samples = read_wav_16k(&utt.wav)?;
        let segments = transcriber.transcribe(&samples, &transcribe_opts)?;
        let hypothesis: String = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let norm_ref = normalize::normalize_basic(&utt.reference);
        let norm_hyp = normalize::normalize_basic(&hypothesis);
        eprintln!(
            "[{}/{}] {}\n  ref: {}\n  hyp: {}",
            n + 1,
            selected.len(),
            utt.wav.file_name().unwrap_or_default().to_string_lossy(),
            norm_ref,
            norm_hyp
        );

        let translation = if translate {
            let segments = transcriber.transcribe(&samples, &translate_opts)?;
            let text: String = segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("  en:  {}", text.trim());
            Some(text)
        } else {
            None
        };

        if let Some(f) = tsv_out.as_mut() {
            write!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                utt.wav.display(),
                utt.reference.replace('\t', " "),
                hypothesis.replace('\t', " "),
                norm_ref,
                norm_hyp
            )?;
            if let Some(t) = &translation {
                write!(f, "\t{}", t.replace('\t', " "))?;
            }
            writeln!(f)?;
        }

        pairs.push((norm_ref, norm_hyp));
    }

    let (metric, value) = if by_character {
        ("CER", wer::corpus_cer(&pairs))
    } else {
        ("WER", wer::corpus_wer(&pairs))
    };
    println!(
        "{fleurs_lang} ({language}) {metric}: {:.2} % ({} utterances)",
        value * 100.0,
        pairs.len()
    );
    Ok(())
}
