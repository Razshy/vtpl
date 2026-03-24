use std::collections::BTreeSet;

/// Extract character-level n-grams from text.
/// Lowercases and strips non-alphanumeric characters before extraction.
pub fn extract_ngrams(text: &str, n: usize) -> BTreeSet<String> {
    let normalized: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();

    let mut grams = BTreeSet::new();
    for word in normalized.split_whitespace() {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < n {
            grams.insert(word.to_string());
            continue;
        }
        for window in chars.windows(n) {
            grams.insert(window.iter().collect());
        }
    }
    grams
}

#[inline]
pub fn trigrams(text: &str) -> BTreeSet<String> {
    extract_ngrams(text, 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trigrams_basic() {
        let gs = trigrams("concurrent");
        assert!(gs.contains("con"));
        assert!(gs.contains("onc"));
        assert!(gs.contains("ncu"));
        assert!(gs.contains("ent"));
    }

    #[test]
    fn short_words_kept_whole() {
        let gs = trigrams("go is ok");
        assert!(gs.contains("go"));
        assert!(gs.contains("is"));
        assert!(gs.contains("ok"));
    }

    #[test]
    fn punctuation_stripped() {
        let gs = trigrams("it's a test!");
        assert!(gs.contains("it"));
        assert!(gs.contains("tes"));
        assert!(gs.contains("est"));
    }
}
