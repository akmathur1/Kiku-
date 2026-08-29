use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

const ADDITIONAL_DIACRITICS: &[(char, &str)] = &[
    ('œ', "oe"),
    ('Œ', "OE"),
    ('ø', "o"),
    ('Ø', "O"),
    ('æ', "ae"),
    ('Æ', "AE"),
    ('ß', "ss"),
    ('ẞ', "SS"),
    ('đ', "d"),
    ('Đ', "D"),
    ('ð', "d"),
    ('Ð', "D"),
    ('þ', "th"),
    ('Þ', "th"),
    ('ł', "l"),
    ('Ł', "L"),
];

fn remove_symbols_and_diacritics(s: &str, keep: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.nfkd() {
        if keep.contains(c) {
            out.push(c);
        } else if let Some((_, rep)) = ADDITIONAL_DIACRITICS.iter().find(|(k, _)| *k == c) {
            out.push_str(rep);
        } else {
            use unicode_properties::GeneralCategory::NonspacingMark;
            if c.general_category() == NonspacingMark {
            } else {
                match c.general_category_group() {
                    GeneralCategoryGroup::Mark
                    | GeneralCategoryGroup::Symbol
                    | GeneralCategoryGroup::Punctuation => out.push(' '),
                    _ => out.push(c),
                }
            }
        }
    }
    out
}

const ONES: &[&str] = &[
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: &[(&str, u128)] = &[
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
];

const MULTIPLIERS: &[(&str, u128)] = &[
    ("hundred", 100),
    ("thousand", 1_000),
    ("million", 1_000_000),
    ("billion", 1_000_000_000),
    ("trillion", 1_000_000_000_000),
    ("quadrillion", 1_000_000_000_000_000),
    ("quintillion", 1_000_000_000_000_000_000),
    ("sextillion", 1_000_000_000_000_000_000_000),
    ("septillion", 1_000_000_000_000_000_000_000_000),
    ("octillion", 1_000_000_000_000_000_000_000_000_000),
    ("nonillion", 1_000_000_000_000_000_000_000_000_000_000),
    ("decillion", 1_000_000_000_000_000_000_000_000_000_000_000),
];

fn is_zero(w: &str) -> bool {
    matches!(w, "o" | "oh" | "zero")
}

fn ones_value(w: &str) -> Option<u128> {
    ONES.iter().position(|&n| n == w).map(|i| i as u128 + 1)
}

fn ones_suffixed(w: &str) -> Option<(u128, &'static str)> {
    match w {
        "zeroth" => return Some((0, "th")),
        "first" => return Some((1, "st")),
        "second" => return Some((2, "nd")),
        "third" => return Some((3, "rd")),
        "fifth" => return Some((5, "th")),
        "twelfth" => return Some((12, "th")),
        _ => {}
    }
    for (i, &name) in ONES.iter().enumerate() {
        let value = i as u128 + 1;
        let plural = if name == "six" {
            "sixes".to_string()
        } else {
            format!("{name}s")
        };
        if w == plural {
            return Some((value, "s"));
        }
        if value > 3 && value != 5 && value != 12 {
            let ordinal = if name.ends_with('t') {
                format!("{name}h")
            } else {
                format!("{name}th")
            };
            if w == ordinal {
                return Some((value, "th"));
            }
        }
    }
    None
}

fn tens_value(w: &str) -> Option<u128> {
    TENS.iter().find(|(n, _)| *n == w).map(|(_, v)| *v)
}

fn tens_suffixed(w: &str) -> Option<(u128, &'static str)> {
    for &(name, value) in TENS {
        if w == name.replace('y', "ies") {
            return Some((value, "s"));
        }
        if w == name.replace('y', "ieth") {
            return Some((value, "th"));
        }
    }
    None
}

fn multiplier_value(w: &str) -> Option<u128> {
    MULTIPLIERS.iter().find(|(n, _)| *n == w).map(|(_, v)| *v)
}

fn multiplier_suffixed(w: &str) -> Option<(u128, &'static str)> {
    for &(name, value) in MULTIPLIERS {
        if w == format!("{name}s") {
            return Some((value, "s"));
        }
        if w == format!("{name}th") {
            return Some((value, "th"));
        }
    }
    None
}

fn preceding_prefixer(w: &str) -> Option<char> {
    match w {
        "minus" | "negative" => Some('-'),
        "plus" | "positive" => Some('+'),
        _ => None,
    }
}

fn following_prefixer(w: &str) -> Option<char> {
    match w {
        "pound" | "pounds" => Some('£'),
        "euro" | "euros" => Some('€'),
        "dollar" | "dollars" => Some('$'),
        "cent" | "cents" => Some('¢'),
        _ => None,
    }
}

fn is_prefix_symbol(c: char) -> bool {
    matches!(c, '-' | '+' | '£' | '€' | '$' | '¢')
}

fn is_decimal_word(w: &str) -> bool {
    ones_value(w).is_some() || tens_value(w).is_some() || is_zero(w)
}

fn is_special(w: &str) -> bool {
    matches!(w, "and" | "double" | "triple" | "point")
}

fn is_number_word(w: &str) -> bool {
    is_zero(w)
        || ones_value(w).is_some()
        || ones_suffixed(w).is_some()
        || tens_value(w).is_some()
        || tens_suffixed(w).is_some()
        || multiplier_value(w).is_some()
        || multiplier_suffixed(w).is_some()
        || preceding_prefixer(w).is_some()
        || following_prefixer(w).is_some()
        || w == "per"
        || w == "percent"
        || is_special(w)
}

fn is_numeric(w: &str) -> bool {
    let mut parts = w.splitn(2, '.');
    let int = parts.next().unwrap_or("");
    if int.is_empty() || !int.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(frac) => !frac.is_empty() && frac.chars().all(|c| c.is_ascii_digit()),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Value {
    Int(u128),
    Str(String),
}

impl Value {
    fn render(&self) -> String {
        match self {
            Value::Int(v) => v.to_string(),
            Value::Str(s) => s.clone(),
        }
    }
}

fn decimal_times_multiplier(s: &str, multiplier: u128) -> Option<u128> {
    let (int_part, frac_part) = match s.split_once('.') {
        Some((i, f)) => (i, f),
        None => (s, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !int_part.chars().all(|c| c.is_ascii_digit())
        || !frac_part.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let mantissa: u128 = format!("{int_part}{frac_part}").parse().ok()?;
    let denom = 10u128.checked_pow(frac_part.len() as u32)?;
    let product = mantissa.checked_mul(multiplier)?;
    if product % denom == 0 {
        Some(product / denom)
    } else {
        None
    }
}

struct NumberMachine {
    prefix: Option<char>,
    value: Option<Value>,
    out: Vec<String>,
}

impl NumberMachine {
    fn output_raw(&mut self, result: String) {
        let result = match self.prefix.take() {
            Some(p) => format!("{p}{result}"),
            None => result,
        };
        self.value = None;
        self.out.push(result);
    }

    fn output_value(&mut self) {
        if let Some(v) = self.value.take() {
            self.output_raw(v.render());
        }
    }

    fn value_or_empty(&self) -> String {
        match &self.value {
            None | Some(Value::Int(0)) => String::new(),
            Some(v) => v.render(),
        }
    }
}

fn process_words(words: &[String]) -> Vec<String> {
    let mut m = NumberMachine {
        prefix: None,
        value: None,
        out: Vec::with_capacity(words.len()),
    };
    let mut skip = false;

    for idx in 0..words.len() {
        if skip {
            skip = false;
            continue;
        }
        let prev = if idx > 0 {
            Some(words[idx - 1].as_str())
        } else {
            None
        };
        let current = words[idx].as_str();
        let next = words.get(idx + 1).map(|s| s.as_str());

        let next_is_numeric = next.is_some_and(is_numeric);
        let first = current.chars().next();
        let has_prefix = first.is_some_and(is_prefix_symbol);
        let current_without_prefix = if has_prefix {
            &current[first.unwrap().len_utf8()..]
        } else {
            current
        };
        let prev_is_ones = prev.is_some_and(|p| ones_value(p).is_some());
        let prev_is_tens = prev.is_some_and(|p| tens_value(p).is_some());

        if is_numeric(current_without_prefix) {
            if let Some(v) = m.value.take() {
                if let Value::Str(s) = &v {
                    if s.ends_with('.') {
                        m.value = Some(Value::Str(format!("{s}{current}")));
                        continue;
                    }
                }
                m.value = Some(v);
                m.output_value();
            }
            if has_prefix {
                m.prefix = first;
            }
            m.value = if current_without_prefix.contains('.') {
                Some(Value::Str(current_without_prefix.to_string()))
            } else {
                Some(Value::Int(current_without_prefix.parse().unwrap()))
            };
        } else if !is_number_word(current) {
            m.output_value();
            m.output_raw(current.to_string());
        } else if is_zero(current) {
            let s = m.value_or_empty();
            m.value = Some(Value::Str(format!("{s}0")));
        } else if let Some(ones) = ones_value(current) {
            match m.value.take() {
                None => m.value = Some(Value::Int(ones)),
                Some(Value::Str(s)) => {
                    if prev_is_tens && ones < 10 {
                        debug_assert!(s.ends_with('0'));
                        m.value = Some(Value::Str(format!("{}{ones}", &s[..s.len() - 1])));
                    } else {
                        m.value = Some(Value::Str(format!("{s}{ones}")));
                    }
                }
                Some(Value::Int(v)) if prev_is_ones => {
                    if prev_is_tens && ones < 10 {
                        let s = v.to_string();
                        m.value = Some(Value::Str(format!("{}{ones}", &s[..s.len() - 1])));
                    } else {
                        m.value = Some(Value::Str(format!("{v}{ones}")));
                    }
                }
                Some(Value::Int(v)) => {
                    if ones < 10 {
                        if v % 10 == 0 {
                            m.value = Some(Value::Int(v + ones));
                        } else {
                            m.value = Some(Value::Str(format!("{v}{ones}")));
                        }
                    } else {
                        if v % 100 == 0 {
                            m.value = Some(Value::Int(v + ones));
                        } else {
                            m.value = Some(Value::Str(format!("{v}{ones}")));
                        }
                    }
                }
            }
        } else if let Some((ones, suffix)) = ones_suffixed(current) {
            match m.value.take() {
                None => m.output_raw(format!("{ones}{suffix}")),
                Some(Value::Str(s)) => {
                    if prev_is_tens && ones < 10 {
                        debug_assert!(s.ends_with('0'));
                        m.output_raw(format!("{}{ones}{suffix}", &s[..s.len() - 1]));
                    } else {
                        m.output_raw(format!("{s}{ones}{suffix}"));
                    }
                }
                Some(Value::Int(v)) if prev_is_ones => {
                    if prev_is_tens && ones < 10 {
                        let s = v.to_string();
                        m.output_raw(format!("{}{ones}{suffix}", &s[..s.len() - 1]));
                    } else {
                        m.output_raw(format!("{v}{ones}{suffix}"));
                    }
                }
                Some(Value::Int(v)) => {
                    if ones < 10 {
                        if v % 10 == 0 {
                            m.output_raw(format!("{}{suffix}", v + ones));
                        } else {
                            m.output_raw(format!("{v}{ones}{suffix}"));
                        }
                    } else if v % 100 == 0 {
                        m.output_raw(format!("{}{suffix}", v + ones));
                    } else {
                        m.output_raw(format!("{v}{ones}{suffix}"));
                    }
                }
            }
        } else if let Some(tens) = tens_value(current) {
            match m.value.take() {
                None => m.value = Some(Value::Int(tens)),
                Some(Value::Str(s)) => m.value = Some(Value::Str(format!("{s}{tens}"))),
                Some(Value::Int(v)) => {
                    if v % 100 == 0 {
                        m.value = Some(Value::Int(v + tens));
                    } else {
                        m.value = Some(Value::Str(format!("{v}{tens}")));
                    }
                }
            }
        } else if let Some((tens, suffix)) = tens_suffixed(current) {
            match m.value.take() {
                None => m.output_raw(format!("{tens}{suffix}")),
                Some(Value::Str(s)) => m.output_raw(format!("{s}{tens}{suffix}")),
                Some(Value::Int(v)) => {
                    if v % 100 == 0 {
                        m.output_raw(format!("{}{suffix}", v + tens));
                    } else {
                        m.output_raw(format!("{v}{tens}{suffix}"));
                    }
                }
            }
        } else if let Some(multiplier) = multiplier_value(current) {
            match m.value.take() {
                None => m.value = Some(Value::Int(multiplier)),
                Some(Value::Str(s)) => match decimal_times_multiplier(&s, multiplier) {
                    Some(p) => m.value = Some(Value::Int(p)),
                    None => {
                        m.value = Some(Value::Str(s));
                        m.output_value();
                        m.value = Some(Value::Int(multiplier));
                    }
                },
                Some(Value::Int(0)) => {
                    m.value = Some(Value::Int(0));
                }
                Some(Value::Int(v)) => {
                    let before = v / 1000 * 1000;
                    let residual = v % 1000;
                    m.value = Some(Value::Int(before + residual * multiplier));
                }
            }
        } else if let Some((multiplier, suffix)) = multiplier_suffixed(current) {
            match m.value.take() {
                None => m.output_raw(format!("{multiplier}{suffix}")),
                Some(Value::Str(s)) => match decimal_times_multiplier(&s, multiplier) {
                    Some(p) => m.output_raw(format!("{p}{suffix}")),
                    None => {
                        m.value = Some(Value::Str(s));
                        m.output_value();
                        m.output_raw(format!("{multiplier}{suffix}"));
                    }
                },
                Some(Value::Int(v)) => {
                    let before = v / 1000 * 1000;
                    let residual = v % 1000;
                    m.output_raw(format!("{}{suffix}", before + residual * multiplier));
                }
            }
        } else if let Some(p) = preceding_prefixer(current) {
            m.output_value();
            if next.is_some_and(is_number_word) || next_is_numeric {
                m.prefix = Some(p);
            } else {
                m.output_raw(current.to_string());
            }
        } else if let Some(p) = following_prefixer(current) {
            if m.value.is_some() {
                m.prefix = Some(p);
                m.output_value();
            } else {
                m.output_raw(current.to_string());
            }
        } else if current == "per" || current == "percent" {
            match m.value.take() {
                Some(v) if current == "percent" => m.output_raw(format!("{}%", v.render())),
                Some(v) if next == Some("cent") => {
                    m.output_raw(format!("{}%", v.render()));
                    skip = true;
                }
                Some(v) => {
                    m.output_raw(v.render());
                    m.output_raw(current.to_string());
                }
                None => m.output_raw(current.to_string()),
            }
        } else if is_special(current) {
            let next_in_words = next.is_some_and(is_number_word);
            if !next_in_words && !next_is_numeric {
                m.output_value();
                m.output_raw(current.to_string());
            } else if current == "and" {
                if !prev.is_some_and(|p| multiplier_value(p).is_some()) {
                    m.output_value();
                    m.output_raw(current.to_string());
                }
            } else if current == "double" || current == "triple" {
                let next_ones = next.and_then(ones_value);
                let next_zero = next.is_some_and(is_zero);
                if next_ones.is_some() || next_zero {
                    let repeats = if current == "double" { 2 } else { 3 };
                    let ones = next_ones.unwrap_or(0);
                    let s = m.value_or_empty();
                    m.value = Some(Value::Str(format!(
                        "{s}{}",
                        ones.to_string().repeat(repeats)
                    )));
                    skip = true;
                } else {
                    m.output_value();
                    m.output_raw(current.to_string());
                }
            } else if current == "point" && (next.is_some_and(is_decimal_word) || next_is_numeric) {
                let s = m.value_or_empty();
                m.value = Some(Value::Str(format!("{s}.")));
            }
        } else {
            unreachable!("number word not covered: {current}");
        }
    }

    m.output_value();
    m.out
}

fn number_regexes() -> &'static (Regex, Regex, Regex, Regex, Regex, Regex, Regex) {
    static RE: OnceLock<(Regex, Regex, Regex, Regex, Regex, Regex, Regex)> = OnceLock::new();
    RE.get_or_init(|| {
        (
            Regex::new(r"\band\s+a\s+half\b").unwrap(),
            Regex::new(r"([a-z])([0-9])").unwrap(),
            Regex::new(r"([0-9])([a-z])").unwrap(),
            Regex::new(r"([0-9])\s+(st|nd|rd|th|s)\b").unwrap(),
            Regex::new(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b").unwrap(),
            Regex::new(r"[€£$]0.([0-9]{1,2})\b").unwrap(),
            Regex::new(r"\b1(s?)\b").unwrap(),
        )
    })
}

fn number_preprocess(s: &str) -> String {
    let (half_re, letter_digit, digit_letter, suffix_re, ..) = {
        let r = number_regexes();
        (&r.0, &r.1, &r.2, &r.3)
    };

    let segments: Vec<&str> = half_re.split(s).collect();
    let mut results: Vec<String> = Vec::new();
    let n = segments.len();
    for (i, segment) in segments.iter().enumerate() {
        if segment.trim().is_empty() {
            continue;
        }
        if i == n - 1 {
            results.push(segment.to_string());
        } else {
            results.push(segment.to_string());
            let last_word = segment.split_whitespace().last().unwrap_or("");
            if is_decimal_word(last_word) || multiplier_value(last_word).is_some() {
                results.push("point five".to_string());
            } else {
                results.push("and a half".to_string());
            }
        }
    }
    let mut s = results.join(" ");

    s = letter_digit.replace_all(&s, "$1 $2").into_owned();
    s = digit_letter.replace_all(&s, "$1 $2").into_owned();
    s = suffix_re.replace_all(&s, "$1$2").into_owned();
    s
}

fn number_postprocess(s: &str) -> String {
    let (cents_re, extract_re, ones_re) = {
        let r = number_regexes();
        (&r.4, &r.5, &r.6)
    };

    let mut s = cents_re
        .replace_all(s, |caps: &regex::Captures| {
            let cents: u32 = caps[3].parse().unwrap();
            format!("{}{}.{:02}", &caps[1], &caps[2], cents)
        })
        .into_owned();
    s = extract_re
        .replace_all(&s, |caps: &regex::Captures| {
            let cents: u32 = caps[1].parse().unwrap();
            format!("¢{cents}")
        })
        .into_owned();

    s = ones_re.replace_all(&s, "one$1").into_owned();
    s
}

pub fn normalize_english_numbers(text: &str) -> String {
    let s = number_preprocess(text);
    let words: Vec<String> = s.split_whitespace().map(str::to_string).collect();
    let s = process_words(&words).join(" ");
    number_postprocess(&s)
}

fn spelling_map() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/english.json"))
            .expect("assets/english.json parses")
    })
}

pub fn normalize_english_spelling(text: &str) -> String {
    text.split_whitespace()
        .map(|w| spelling_map().get(w).map_or(w, String::as_str))
        .collect::<Vec<_>>()
        .join(" ")
}

const REPLACERS: &[(&str, &str)] = &[
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\blet's\b", "let us"),
    (r"\bain't\b", "aint"),
    (r"\by'all\b", "you all"),
    (r"\bwanna\b", "want to"),
    (r"\bgotta\b", "got to"),
    (r"\bgonna\b", "going to"),
    (r"\bi'ma\b", "i am going to"),
    (r"\bimma\b", "i am going to"),
    (r"\bwoulda\b", "would have"),
    (r"\bcoulda\b", "could have"),
    (r"\bshoulda\b", "should have"),
    (r"\bma'am\b", "madam"),
    (r"\bmr\b", "mister "),
    (r"\bmrs\b", "missus "),
    (r"\bst\b", "saint "),
    (r"\bdr\b", "doctor "),
    (r"\bprof\b", "professor "),
    (r"\bcapt\b", "captain "),
    (r"\bgov\b", "governor "),
    (r"\bald\b", "alderman "),
    (r"\bgen\b", "general "),
    (r"\bsen\b", "senator "),
    (r"\brep\b", "representative "),
    (r"\bpres\b", "president "),
    (r"\brev\b", "reverend "),
    (r"\bhon\b", "honorable "),
    (r"\basst\b", "assistant "),
    (r"\bassoc\b", "associate "),
    (r"\blt\b", "lieutenant "),
    (r"\bcol\b", "colonel "),
    (r"\bjr\b", "junior "),
    (r"\bsr\b", "senior "),
    (r"\besq\b", "esquire "),
    (r"'d been\b", " had been"),
    (r"'s been\b", " has been"),
    (r"'d gone\b", " had gone"),
    (r"'s gone\b", " has gone"),
    (r"'d done\b", " had done"),
    (r"'s got\b", " has got"),
    (r"n't\b", " not"),
    (r"'re\b", " are"),
    (r"'s\b", " is"),
    (r"'d\b", " would"),
    (r"'ll\b", " will"),
    (r"'t\b", " not"),
    (r"'ve\b", " have"),
    (r"'m\b", " am"),
];

struct TextRegexes {
    brackets: Regex,
    parens: Regex,
    fillers: Regex,
    space_apostrophe: Regex,
    replacers: Vec<(Regex, &'static str)>,
    digit_commas: Regex,
    periods: Regex,
    symbol_cleanup: Regex,
    percent_cleanup: Regex,
    whitespace: Regex,
}

fn text_regexes() -> &'static TextRegexes {
    static RE: OnceLock<TextRegexes> = OnceLock::new();
    RE.get_or_init(|| TextRegexes {
        brackets: Regex::new(r"[<\[][^>\]]*[>\]]").unwrap(),
        parens: Regex::new(r"\(([^)]+?)\)").unwrap(),
        fillers: Regex::new(r"\b(hmm|mm|mhm|mmm|uh|um)\b").unwrap(),
        space_apostrophe: Regex::new(r"\s+'").unwrap(),
        replacers: REPLACERS
            .iter()
            .map(|(p, r)| (Regex::new(p).unwrap(), *r))
            .collect(),
        digit_commas: Regex::new(r"(\d),(\d)").unwrap(),
        periods: Regex::new(r"\.([^0-9]|$)").unwrap(),
        symbol_cleanup: Regex::new(r"[.$¢€£]([^0-9])").unwrap(),
        percent_cleanup: Regex::new(r"([^0-9])%").unwrap(),
        whitespace: Regex::new(r"\s+").unwrap(),
    })
}

pub fn normalize_english(text: &str) -> String {
    let re = text_regexes();
    let mut s = text.to_lowercase();

    s = re.brackets.replace_all(&s, "").into_owned();
    s = re.parens.replace_all(&s, "").into_owned();
    s = re.fillers.replace_all(&s, "").into_owned();
    s = re.space_apostrophe.replace_all(&s, "'").into_owned();

    for (pattern, replacement) in &re.replacers {
        s = pattern.replace_all(&s, *replacement).into_owned();
    }

    s = re.digit_commas.replace_all(&s, "$1$2").into_owned();
    s = re.periods.replace_all(&s, " $1").into_owned();
    s = remove_symbols_and_diacritics(&s, ".%$¢€£");

    s = normalize_english_numbers(&s);
    s = normalize_english_spelling(&s);

    s = re.symbol_cleanup.replace_all(&s, " $1").into_owned();
    s = re.percent_cleanup.replace_all(&s, "$1 ").into_owned();
    s = re.whitespace.replace_all(&s, " ").into_owned();

    s.trim().to_string()
}

fn remove_symbols(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.nfkc() {
        match c.general_category_group() {
            GeneralCategoryGroup::Mark
            | GeneralCategoryGroup::Symbol
            | GeneralCategoryGroup::Punctuation => out.push(' '),
            _ => out.push(c),
        }
    }
    out
}

pub fn normalize_basic(text: &str) -> String {
    let re = text_regexes();
    let mut s = text.to_lowercase();

    s = re.brackets.replace_all(&s, "").into_owned();
    s = re.parens.replace_all(&s, "").into_owned();
    s = remove_symbols(&s).to_lowercase();
    s = re.whitespace.replace_all(&s, " ").into_owned();

    s.trim().to_string()
}
