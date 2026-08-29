use std::io::Write;
use std::path::{Path, PathBuf};

use kiku::normalize::normalize_english;
use kiku::wer::corpus_wer;
use kiku::{audio, Options, Transcriber};

struct Utterance {
    wav: PathBuf,
    reference: String,
}

fn collect_utterances(root: &Path) -> anyhow::Result<Vec<Utterance>> {
    let mut utterances = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)?.collect::<Result<_, _>>()?;
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".trans.txt"))
            {
                let parent = path.parent().unwrap_or(root).to_path_buf();
                for (lineno, line) in std::fs::read_to_string(&path)?.lines().enumerate() {
                    anyhow::ensure!(
                        !line.is_empty(),
                        "{} has a blank transcript record at line {} — a dropped \
                         record would report WER over an undisclosed subset",
                        path.display(),
                        lineno + 1
                    );
                    let Some((id, text)) = line.split_once(' ') else {
                        anyhow::bail!(
                            "{} has a malformed transcript line {line:?} — a dropped \
                             record would report WER over an undisclosed subset",
                            path.display()
                        );
                    };
                    let wav = parent.join(format!("{id}.wav"));
                    anyhow::ensure!(
                        wav.exists(),
                        "{} lists utterance {id} but {} is missing — an incomplete corpus \
                         would silently skew WER; re-run scripts/fetch-librispeech.sh",
                        path.display(),
                        wav.display()
                    );
                    utterances.push(Utterance {
                        wav,
                        reference: text.to_string(),
                    });
                }
            }
        }
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
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let [model_dir, split_dir] = positional.as_slice() else {
        anyhow::bail!(
            "usage: eval-librispeech <model-dir> <librispeech-split-dir> [--limit N] [--every N] [--tsv out.tsv]"
        );
    };

    let all = collect_utterances(Path::new(split_dir))?;
    anyhow::ensure!(
        !all.is_empty(),
        "no utterances found under {split_dir} — expected .trans.txt files with .wav audio beside them (run scripts/fetch-librispeech.sh)"
    );
    let selected: Vec<&Utterance> = all.iter().step_by(every).take(limit).collect();
    anyhow::ensure!(
        !selected.is_empty(),
        "--limit/--every selected zero utterances — refusing to report a WER over nothing"
    );
    eprintln!(
        "{} utterances in corpus; evaluating {} (every {}, limit {})",
        all.len(),
        selected.len(),
        every,
        if limit == usize::MAX {
            "none".to_string()
        } else {
            limit.to_string()
        }
    );

    let opts = Options {
        language: Some("en".to_string()),
        ..Options::default()
    };
    let transcriber = Transcriber::load(Path::new(model_dir))?;

    let mut tsv_file = tsv.map(std::fs::File::create).transpose()?;
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(selected.len());
    for (n, utt) in selected.iter().enumerate() {
        let samples = read_wav_16k(&utt.wav)?;
        let segments = transcriber.transcribe(&samples, &opts)?;
        let hypothesis = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let ref_clean = normalize_english(&utt.reference);
        let hyp_clean = normalize_english(&hypothesis);
        if let Some(f) = tsv_file.as_mut() {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                utt.wav.display(),
                utt.reference,
                hypothesis,
                ref_clean,
                hyp_clean
            )?;
        }
        eprintln!(
            "[{}/{}] {}\n  ref: {}\n  hyp: {}",
            n + 1,
            selected.len(),
            utt.wav.file_stem().unwrap_or_default().to_string_lossy(),
            ref_clean,
            hyp_clean
        );
        pairs.push((ref_clean, hyp_clean));
    }

    let wer = corpus_wer(&pairs);
    println!("WER: {:.2} % ({} utterances)", wer * 100.0, pairs.len());
    Ok(())
}
