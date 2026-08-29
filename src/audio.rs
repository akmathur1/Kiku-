pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_SECONDS: usize = 30;
pub const N_SAMPLES: usize = SAMPLE_RATE * CHUNK_SECONDS;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

pub fn resample_to_16k(samples: &[f32], src_rate: usize) -> Vec<f32> {
    if src_rate == SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = src_rate as f64 / SAMPLE_RATE as f64;
    let out_len = (samples.len() as f64 / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos.floor() as usize;
            let frac = (pos - idx as f64) as f32;
            let a = samples[idx.min(samples.len() - 1)];
            let b = samples[(idx + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

fn hz_to_mel(hz: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        3.0 * hz / 200.0
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep).exp()
    } else {
        200.0 * mel / 3.0
    }
}

pub fn mel_filterbank(n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let fmax = SAMPLE_RATE as f64 / 2.0;
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_max * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64)
        .collect();
    (0..n_mels)
        .map(|m| {
            let (lower, center, upper) = (mel_points[m], mel_points[m + 1], mel_points[m + 2]);
            let norm = 2.0 / (upper - lower);
            fft_freqs
                .iter()
                .map(|&f| {
                    let up = (f - lower) / (center - lower);
                    let down = (upper - f) / (upper - center);
                    (norm * up.min(down).max(0.0)) as f32
                })
                .collect()
        })
        .collect()
}

pub fn log_mel_spectrogram(samples: &[f32], n_mels: usize) -> Vec<f32> {
    let mut audio = samples.to_vec();
    audio.resize(N_SAMPLES, 0.0);

    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(audio.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| audio[i]));
    padded.extend_from_slice(&audio);
    padded.extend(
        (audio.len() - pad - 1..audio.len() - 1)
            .rev()
            .map(|i| audio[i]),
    );

    let window: Vec<f32> = (0..N_FFT)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos()))
        .collect();

    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let n_freqs = N_FFT / 2 + 1;

    let mut power = vec![0.0f32; N_FRAMES * n_freqs];
    let mut buf = vec![rustfft::num_complex::Complex::new(0.0f32, 0.0); N_FFT];
    for frame in 0..N_FRAMES {
        let start = frame * HOP_LENGTH;
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = rustfft::num_complex::Complex::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        for (k, v) in buf.iter().take(n_freqs).enumerate() {
            power[frame * n_freqs + k] = v.norm_sqr();
        }
    }

    let filters = mel_filterbank(n_mels);
    let mut mel = vec![0.0f32; n_mels * N_FRAMES];
    let mut max_val = f32::MIN;
    for (m, filter) in filters.iter().enumerate() {
        for frame in 0..N_FRAMES {
            let mut acc = 0.0f32;
            for (k, &w) in filter.iter().enumerate() {
                if w > 0.0 {
                    acc += w * power[frame * n_freqs + k];
                }
            }
            let v = acc.max(1e-10).log10();
            mel[m * N_FRAMES + frame] = v;
            max_val = max_val.max(v);
        }
    }
    for v in &mut mel {
        *v = (v.max(max_val - 8.0) + 4.0) / 4.0;
    }
    mel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_shape_and_coverage() {
        let fb = mel_filterbank(80);
        assert_eq!(fb.len(), 80);
        assert_eq!(fb[0].len(), N_FFT / 2 + 1);
        for f in &fb {
            assert!(f.iter().any(|&w| w > 0.0));
        }
    }

    #[test]
    fn slaney_mel_round_trips() {
        for hz in [100.0, 900.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-6, "{hz} -> {back}");
        }
    }

    #[test]
    fn silence_maps_to_scaled_floor() {
        let mel = log_mel_spectrogram(&[0.0; SAMPLE_RATE], 80);
        assert_eq!(mel.len(), 80 * N_FRAMES);
        let first = mel[0];
        assert!(mel.iter().all(|&v| (v - first).abs() < 1e-5));
    }

    #[test]
    fn tone_concentrates_energy_in_the_right_mel_band() {
        let tone: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let mel = log_mel_spectrogram(&tone, 80);
        let col = |m: usize| mel[m * N_FRAMES + 50];
        let peak_band = (0..80)
            .max_by(|&a, &b| col(a).partial_cmp(&col(b)).unwrap())
            .unwrap();
        let center = |m: usize| {
            let mel_max = hz_to_mel(8000.0);
            mel_to_hz(mel_max * (m + 1) as f64 / 81.0)
        };
        assert!(
            (center(peak_band) - 1000.0).abs() < 150.0,
            "peak band {peak_band} centered at {:.0} Hz",
            center(peak_band)
        );
    }

    #[test]
    fn resample_halves_length() {
        let samples = vec![0.5f32; 32_000];
        let out = resample_to_16k(&samples, 32_000);
        assert_eq!(out.len(), 16_000);
        assert!(out.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }
}
