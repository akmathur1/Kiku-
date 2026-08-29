//! The Kiku model: an encoder-decoder Transformer over log-Mel input,
//! following the Whisper architecture.
//!
//! Encoder: 2x Conv1D(width 3) + GELU stem (second conv stride 2), fixed
//! sinusoidal positions, pre-activation residual blocks, final LayerNorm.
//! Decoder: learned positions, pre-activation residual blocks with
//! cross-attention into the encoder, output projection tied to the token
//! embedding. Weight names follow the Hugging Face Whisper checkpoint
//! layout so open checkpoints load directly.

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    conv1d, embedding, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, Embedding,
    LayerNorm, Linear, Module, VarBuilder,
};
use serde::Deserialize;

/// The architecture hyperparameters, read from the checkpoint's config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,
    pub vocab_size: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_layers: usize,
    pub decoder_attention_heads: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
}

struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    n_heads: usize,
}

impl MultiHeadAttention {
    fn load(dim: usize, n_heads: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            q: linear(dim, dim, vb.pp("q_proj"))?,
            // The key projection carries no bias in the Whisper architecture.
            k: linear_no_bias(dim, dim, vb.pp("k_proj"))?,
            v: linear(dim, dim, vb.pp("v_proj"))?,
            out: linear(dim, dim, vb.pp("out_proj"))?,
            n_heads,
        })
    }

    /// `x`: (batch, q_len, dim); `kv`: keys/values source (self-attention
    /// passes `x` itself, cross-attention passes the encoder output).
    fn forward(
        &self,
        x: &Tensor,
        kv: &Tensor,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, q_len, dim) = x.dims3()?;
        let kv_len = kv.dim(1)?;
        let head_dim = dim / self.n_heads;
        let shape_heads = |t: Tensor, len: usize| -> candle_core::Result<Tensor> {
            t.reshape((b, len, self.n_heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = shape_heads(self.q.forward(x)?, q_len)?;
        let k = shape_heads(self.k.forward(kv)?, kv_len)?;
        let v = shape_heads(self.v.forward(kv)?, kv_len)?;
        let scale = (head_dim as f64).powf(-0.5);
        let mut scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        if let Some(mask) = mask {
            scores = scores.broadcast_add(mask)?;
        }
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, q_len, dim))?;
        self.out.forward(&ctx)
    }
}

struct ResidualBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    fc1: Linear,
    fc2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualBlock {
    fn load(dim: usize, n_heads: usize, cross: bool, vb: VarBuilder) -> candle_core::Result<Self> {
        let cross_attn = if cross {
            Some((
                MultiHeadAttention::load(dim, n_heads, vb.pp("encoder_attn"))?,
                layer_norm(dim, 1e-5, vb.pp("encoder_attn_layer_norm"))?,
            ))
        } else {
            None
        };
        Ok(Self {
            attn: MultiHeadAttention::load(dim, n_heads, vb.pp("self_attn"))?,
            attn_ln: layer_norm(dim, 1e-5, vb.pp("self_attn_layer_norm"))?,
            cross_attn,
            fc1: linear(dim, dim * 4, vb.pp("fc1"))?,
            fc2: linear(dim * 4, dim, vb.pp("fc2"))?,
            mlp_ln: layer_norm(dim, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        encoder_out: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        // Pre-activation residual: LayerNorm feeds the sublayer, not its sum.
        let normed = self.attn_ln.forward(x)?;
        let mut x = (x + self.attn.forward(&normed, &normed, mask)?)?;
        if let Some((cross, cross_ln)) = &self.cross_attn {
            let enc = encoder_out.expect("cross-attention block requires encoder output");
            let normed = cross_ln.forward(&x)?;
            x = (&x + cross.forward(&normed, enc, None)?)?;
        }
        let normed = self.mlp_ln.forward(&x)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&normed)?.gelu()?)?;
        &x + mlp
    }
}

pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positions: Tensor,
    blocks: Vec<ResidualBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn load(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = cfg.d_model;
        let conv_cfg = |stride| Conv1dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualBlock::load(
                    dim,
                    cfg.encoder_attention_heads,
                    false,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            conv1: conv1d(cfg.num_mel_bins, dim, 3, conv_cfg(1), vb.pp("conv1"))?,
            conv2: conv1d(dim, dim, 3, conv_cfg(2), vb.pp("conv2"))?,
            // The sinusoidal table ships in the checkpoint; loading it beats
            // recomputing and keeps parity with the reference bit-for-bit.
            positions: vb.get((cfg.max_source_positions, dim), "embed_positions.weight")?,
            blocks,
            ln_post: layer_norm(dim, 1e-5, vb.pp("layer_norm"))?,
        })
    }

    /// `mel`: (batch, n_mels, n_frames) → (batch, n_frames/2, dim).
    pub fn forward(&self, mel: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.conv1.forward(mel)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let mut x = x.transpose(1, 2)?; // (batch, positions, dim)
        let seq = x.dim(1)?;
        x = x.broadcast_add(&self.positions.i(..seq)?)?;
        for block in &self.blocks {
            x = block.forward(&x, None, None)?;
        }
        self.ln_post.forward(&x)
    }
}

pub struct TextDecoder {
    token_embedding: Embedding,
    positions: Tensor,
    blocks: Vec<ResidualBlock>,
    ln: LayerNorm,
}

impl TextDecoder {
    fn load(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let dim = cfg.d_model;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualBlock::load(
                    dim,
                    cfg.decoder_attention_heads,
                    true,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: embedding(cfg.vocab_size, dim, vb.pp("embed_tokens"))?,
            positions: vb.get((cfg.max_target_positions, dim), "embed_positions.weight")?,
            blocks,
            ln: layer_norm(dim, 1e-5, vb.pp("layer_norm"))?,
        })
    }

    /// Returns next-token logits for every position: (batch, len, vocab).
    /// The output projection is tied to the token embedding.
    pub fn forward(&self, tokens: &Tensor, encoder_out: &Tensor) -> candle_core::Result<Tensor> {
        let (_, len) = tokens.dims2()?;
        let mut x = self
            .token_embedding
            .forward(tokens)?
            .broadcast_add(&self.positions.i(..len)?)?;
        let mask = causal_mask(len, x.device())?;
        for block in &self.blocks {
            x = block.forward(&x, Some(encoder_out), Some(&mask))?;
        }
        let x = self.ln.forward(&x)?;
        x.broadcast_matmul(&self.token_embedding.embeddings().t()?)
    }
}

fn causal_mask(len: usize, device: &Device) -> candle_core::Result<Tensor> {
    let data: Vec<f32> = (0..len)
        .flat_map(|q| (0..len).map(move |k| if k > q { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_vec(data, (len, len), device)
}

pub struct Kiku {
    pub config: Config,
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
}

impl Kiku {
    /// Load from a checkpoint directory holding `config.json` and
    /// `model.safetensors` in the Hugging Face Whisper layout.
    pub fn load(dir: &std::path::Path, device: &Device) -> anyhow::Result<Self> {
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let weights = dir.join("model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
        let vb = vb.pp("model");
        let encoder = AudioEncoder::load(&config, vb.pp("encoder"))?;
        let decoder = TextDecoder::load(&config, vb.pp("decoder"))?;
        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_blocks_future_positions() {
        let mask = causal_mask(3, &Device::Cpu).unwrap();
        let rows: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        assert_eq!(rows[0][0], 0.0);
        assert_eq!(rows[0][1], f32::NEG_INFINITY);
        assert_eq!(rows[2][0], 0.0);
        assert_eq!(rows[2][2], 0.0);
    }
}
