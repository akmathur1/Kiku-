//! `kiku-transcribe` — transcribe a WAV file from the command line.
//!
//! Usage: kiku-transcribe <model-dir> <audio.wav> [--language en] [--translate]
//!
//! The model directory holds `config.json`, `model.safetensors`, and
//! `tokenizer.json` (the Hugging Face Whisper checkpoint layout); see
//! kiku/scripts/fetch-model.sh.

use kiku::{audio, Options, Task, Transcriber};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut positional = Vec::new();
    let mut opts = Options::default();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--language" => {
                i += 1;
                opts.language = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--language needs a value"))?
                        .clone(),
                );
            }
            "--translate" => opts.task = Task::Translate,
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    let [model_dir, wav_path] = positional.as_slice() else {
        anyhow::bail!(
            "usage: kiku-transcribe <model-dir> <audio.wav> [--language en] [--translate]"
        );
    };

    let mut reader = hound::WavReader::open(wav_path)?;
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
    // Downmix to mono, then resample to 16 kHz.
    let channels = spec.channels as usize;
    let mono: Vec<f32> = raw
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    let samples = audio::resample_to_16k(&mono, spec.sample_rate as usize);

    let transcriber = Transcriber::load(std::path::Path::new(model_dir))?;
    let segments = transcriber.transcribe(&samples, &opts)?;
    for seg in &segments {
        println!(
            "[{:7.2} --> {:7.2}] ({}, lp {:.2}, ns {:.2}) {}",
            seg.start, seg.end, seg.language, seg.avg_logprob, seg.no_speech_prob, seg.text
        );
    }
    if segments.is_empty() {
        println!("(no speech detected)");
    }
    Ok(())
}
