use std::collections::HashMap;

use anyhow::Context;
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizerFile {
    added_tokens: Vec<AddedToken>,
    model: TokenizerModel,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub sot: u32,
    pub eot: u32,
    pub transcribe: u32,
    pub translate: u32,
    pub no_timestamps: u32,
    pub no_speech: u32,
    pub timestamp_begin: u32,
    pub language_begin: u32,
    pub language_count: u32,
}

pub struct Tokenizer {
    id_to_bytes: Vec<Option<Vec<u8>>>,
    added: HashMap<u32, String>,
    pub special: SpecialTokens,
}

fn unicode_to_byte() -> HashMap<char, u8> {
    let mut direct: Vec<u8> = (b'!'..=b'~').collect();
    direct.extend(0xA1..=0xAC_u8);
    direct.extend(0xAE..=0xFF_u8);
    let mut map = HashMap::new();
    for &b in &direct {
        map.insert(b as char, b);
    }
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !direct.contains(&b) {
            map.insert(char::from_u32(256 + n).unwrap(), b);
            n += 1;
        }
    }
    map
}

impl Tokenizer {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let file: TokenizerFile = serde_json::from_str(&std::fs::read_to_string(path)?)
            .context("parsing tokenizer.json")?;
        let table = unicode_to_byte();
        let max_id = file
            .model
            .vocab
            .values()
            .chain(file.added_tokens.iter().map(|t| &t.id))
            .max()
            .copied()
            .unwrap_or(0) as usize;
        let mut id_to_bytes: Vec<Option<Vec<u8>>> = vec![None; max_id + 1];
        for (token, &id) in &file.model.vocab {
            let bytes: Vec<u8> = token
                .chars()
                .filter_map(|c| table.get(&c).copied())
                .collect();
            id_to_bytes[id as usize] = Some(bytes);
        }
        let added: HashMap<u32, String> = file
            .added_tokens
            .iter()
            .map(|t| (t.id, t.content.clone()))
            .collect();

        let find = |name: &str| -> anyhow::Result<u32> {
            file.added_tokens
                .iter()
                .find(|t| t.content == name)
                .map(|t| t.id)
                .or_else(|| file.model.vocab.get(name).copied())
                .with_context(|| format!("special token {name} missing from tokenizer"))
        };
        let sot = find("<|startoftranscript|>")?;
        let is_lang = |content: &str| {
            content.starts_with("<|")
                && content.ends_with("|>")
                && matches!(content.len() - 4, 2..=3)
                && content[2..content.len() - 2]
                    .chars()
                    .all(|c| c.is_ascii_lowercase())
        };
        let mut language_count = 0u32;
        while let Some(content) = added.get(&(sot + 1 + language_count)) {
            if is_lang(content) {
                language_count += 1;
            } else {
                break;
            }
        }
        let special = SpecialTokens {
            sot,
            eot: find("<|endoftext|>")?,
            transcribe: find("<|transcribe|>")?,
            translate: find("<|translate|>")?,
            no_timestamps: find("<|notimestamps|>")?,
            no_speech: find("<|nospeech|>").or_else(|_| find("<|nocaptions|>"))?,
            timestamp_begin: find("<|notimestamps|>")? + 1,
            language_begin: sot + 1,
            language_count,
        };
        Ok(Self {
            id_to_bytes,
            added,
            special,
        })
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if self.added.contains_key(&id) || id >= self.special.timestamp_begin {
                continue;
            }
            if let Some(Some(b)) = self.id_to_bytes.get(id as usize) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn language_tag(&self, id: u32) -> Option<&str> {
        let sp = &self.special;
        if !(sp.language_begin..sp.language_begin + sp.language_count).contains(&id) {
            return None;
        }
        let content = self.added.get(&id)?;
        content.strip_prefix("<|")?.strip_suffix("|>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_is_a_bijection_over_all_bytes() {
        let table = unicode_to_byte();
        assert_eq!(table.len(), 256);
        let mut seen = [false; 256];
        for &b in table.values() {
            assert!(!seen[b as usize]);
            seen[b as usize] = true;
        }
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        let table = unicode_to_byte();
        assert_eq!(table[&'A'], b'A');
        assert_eq!(table[&'~'], b'~');
        assert_eq!(table[&'Ġ'], b' ');
    }
}
