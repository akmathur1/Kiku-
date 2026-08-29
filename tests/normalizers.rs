use kiku::normalize::{
    normalize_basic, normalize_english, normalize_english_numbers, normalize_english_spelling,
};

fn both(input: &str, expected: &str) {
    assert_eq!(
        normalize_english_numbers(input),
        expected,
        "number normalizer: {input:?}"
    );
    assert_eq!(
        normalize_english(input),
        expected,
        "text normalizer: {input:?}"
    );
}

#[test]
fn number_normalizer() {
    both("two", "2");
    both("thirty one", "31");
    both("five twenty four", "524");
    both("nineteen ninety nine", "1999");
    both("twenty nineteen", "2019");

    both("two point five million", "2500000");
    both("four point two billions", "4200000000s");
    both("200 thousand", "200000");
    both("200 thousand dollars", "$200000");
    both("$20 million", "$20000000");
    both("€52.4 million", "€52400000");
    both("£77 thousands", "£77000s");

    both("two double o eight", "2008");

    both("three thousand twenty nine", "3029");
    both("forty three thousand two hundred sixty", "43260");
    both("forty three thousand two hundred and sixty", "43260");

    both("nineteen fifties", "1950s");
    both("thirty first", "31st");
    both(
        "thirty three thousand and three hundred and thirty third",
        "33333rd",
    );

    both("three billion", "3000000000");
    both("millions", "1000000s");

    both("july third twenty twenty", "july 3rd 2020");
    both("august twenty sixth twenty twenty one", "august 26th 2021");
    both("3 14", "3 14");
    both("3.14", "3.14");
    both("3 point 2", "3.2");
    both("3 point 14", "3.14");
    both("fourteen point 4", "14.4");
    both("two point two five dollars", "$2.25");
    both("two hundred million dollars", "$200000000");
    both("$20.1 million", "$20100000");

    both("ninety percent", "90%");
    both("seventy six per cent", "76%");

    both("double oh seven", "007");
    both("double zero seven", "007");
    both("nine one one", "911");
    both("nine double one", "911");
    both("one triple oh one", "10001");

    both("two thousandth", "2000th");
    both("thirty two thousandth", "32000th");

    both("minus 500", "-500");
    both("positive twenty thousand", "+20000");

    both("two dollars and seventy cents", "$2.70");
    both("3 cents", "¢3");
    both("$0.36", "¢36");
    both("three euros and sixty five cents", "€3.65");

    both("three and a half million", "3500000");
    both("forty eight and a half dollars", "$48.5");
    both("b747", "b 747");
    both("10 th", "10th");
    both("10th", "10th");
}

#[test]
fn basic_normalizer_diacritics() {
    assert_eq!(
        normalize_basic("caf\u{e9} na\u{ef}ve"),
        "caf\u{e9} na\u{ef}ve"
    );
    assert_eq!(normalize_basic("cafe\u{301}"), "caf\u{e9}");
    assert_eq!(normalize_basic("q\u{301}z"), "q z");
}

#[test]
fn spelling_normalizer() {
    assert_eq!(normalize_english_spelling("mobilisation"), "mobilization");
    assert_eq!(normalize_english_spelling("cancelation"), "cancellation");
}

#[test]
fn text_normalizer() {
    assert_eq!(normalize_english("Let's"), "let us");
    assert_eq!(normalize_english("he's like"), "he is like");
    assert_eq!(normalize_english("she's been like"), "she has been like");
    assert_eq!(normalize_english("10km"), "10 km");
    assert_eq!(normalize_english("10mm"), "10 mm");
    assert_eq!(normalize_english("RC232"), "rc 232");

    assert_eq!(
        normalize_english("Mr. Park visited Assoc. Prof. Kim Jr."),
        "mister park visited associate professor kim junior"
    );
}
