//! Word error rate.
//!
//! WER = (substitutions + deletions + insertions) / reference words, with the
//! edit counts from word-level Levenshtein distance — the same definition
//! `jiwer.wer` computes in the reference evaluation. Corpus WER pools edits
//! and reference lengths over all utterance pairs before dividing, so long
//! utterances weigh proportionally (again matching jiwer).

/// Token-level edit distance between a reference and a hypothesis.
pub fn edit_distance<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> usize {
    let (r, h) = (reference.len(), hypothesis.len());
    let mut prev: Vec<usize> = (0..=h).collect();
    let mut curr = vec![0usize; h + 1];
    for i in 1..=r {
        curr[0] = i;
        for j in 1..=h {
            let sub = prev[j - 1] + usize::from(reference[i - 1] != hypothesis[j - 1]);
            curr[j] = sub.min(prev[j] + 1).min(curr[j - 1] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[h]
}

/// Pooled corpus WER over (reference, hypothesis) pairs of *normalized* text.
pub fn corpus_wer(pairs: &[(String, String)]) -> f64 {
    let mut edits = 0usize;
    let mut ref_words = 0usize;
    for (reference, hypothesis) in pairs {
        let r: Vec<&str> = reference.split_whitespace().collect();
        let h: Vec<&str> = hypothesis.split_whitespace().collect();
        edits += edit_distance(&r, &h);
        ref_words += r.len();
    }
    if ref_words == 0 {
        return 0.0;
    }
    edits as f64 / ref_words as f64
}

/// Pooled corpus CER over (reference, hypothesis) pairs of *normalized* text
/// — the character-level analogue of [`corpus_wer`], for languages written
/// without spaces (the reference notebook evaluates those per character).
pub fn corpus_cer(pairs: &[(String, String)]) -> f64 {
    let mut edits = 0usize;
    let mut ref_chars = 0usize;
    for (reference, hypothesis) in pairs {
        let r: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();
        let h: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();
        edits += edit_distance(&r, &h);
        ref_chars += r.len();
    }
    if ref_chars == 0 {
        return 0.0;
    }
    edits as f64 / ref_chars as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_zero() {
        assert_eq!(edit_distance(&["a", "b", "c"], &["a", "b", "c"]), 0);
    }

    #[test]
    fn counts_substitution_deletion_insertion() {
        assert_eq!(edit_distance(&["a", "b", "c"], &["a", "x", "c"]), 1); // sub
        assert_eq!(edit_distance(&["a", "b", "c"], &["a", "c"]), 1); // del
        assert_eq!(edit_distance(&["a", "b"], &["a", "x", "b"]), 1); // ins
    }

    #[test]
    fn corpus_wer_pools_by_reference_length() {
        // 1 edit over 3 ref words + 0 edits over 1 ref word = 1/4.
        let pairs = vec![
            ("a b c".to_string(), "a x c".to_string()),
            ("d".to_string(), "d".to_string()),
        ];
        assert!((corpus_wer(&pairs) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn corpus_cer_ignores_whitespace_and_pools_by_chars() {
        // "안녕 세상" vs "안녕 세상아": 1 insertion over 4 reference chars.
        let pairs = vec![("안녕 세상".to_string(), "안녕 세상아".to_string())];
        assert!((corpus_cer(&pairs) - 0.25).abs() < 1e-9);
    }
}
