import pytest
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    UsSsnRecognizer,
)

from detectors.presidio import (
    fuzzy_pii_recognizer,
    presidio_pii_analyzer,
)

# Build recognizers once at module level to avoid repeated regex compilation
_iban_recognizers = fuzzy_pii_recognizer(
    recognizers=[(IbanRecognizer(exact_match=True), [dict(substitutions=1)])]
)
_ssn_recognizers = fuzzy_pii_recognizer(
    recognizers=[(UsSsnRecognizer(), [
        dict(deletions=1),
        dict(deletions=2),
        dict(substitutions=1),
        dict(substitutions=1, deletions=1),
    ])]
)
_email_recognizers = fuzzy_pii_recognizer(
    recognizers=[(EmailRecognizer(), [
        dict(deletions=1),
        dict(substitutions=1),
    ])]
)
_credit_card_recognizers = fuzzy_pii_recognizer(
    recognizers=[(CreditCardRecognizer(), [
        dict(deletions=1),
        dict(deletions=3),
        dict(substitutions=3),
        dict(substitutions=4),
        dict(deletions=2, substitutions=2),
    ])]
)


def _build_analyzer(recognizers):
    """Build a single AnalyzerEngine with fuzzy recognizers, reusable across test cases."""
    analyzer = AnalyzerEngine(supported_languages=["en"])
    for recognizer in recognizers:
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


# One AnalyzerEngine per recognizer set — avoids reloading spaCy per test case
_iban_analyzer = _build_analyzer(_iban_recognizers)
_ssn_analyzer = _build_analyzer(_ssn_recognizers)
_email_analyzer = _build_analyzer(_email_recognizers)
_cc_analyzer = _build_analyzer(_credit_card_recognizers)


def _analyze(analyzer, text):
    return presidio_pii_analyzer(text=text, recognizers=[], _analyzer_override=analyzer)


@pytest.mark.parametrize(
    "text, expected_count",
    [
        # Exact match
        ("My IBAN is GB82 WEST 1234 5698 7654 32.", 1),
        ("My IBAN is GB82 WEST 1234 5698 7654 32", 1),
        # Fuzzy match
        ("My IBAN is GB82 WEST 1234 5698 7654 32 1", 1),
        # Not detected
        ("My IBAN is GB82 WEST 1234 5698 7654 3B", 0),
        ("My IBAN is GB82 WEST 1234 5698 32", 0),
        ("My IBAN is of 2024 and what genres", 0),
        ("My IBAN is in 2023 that have captivated audiences and", 0),
        ("of 3D printing in", 0),
        ("My IBAN is explore potential challenges like", 0),
        ("My IBAN is GB82 1234 32 1", 0),
        ("My IBAN is 1234 32", 0),
    ],
)
def test_fuzzy_iban_detection(text, expected_count):
    detected = _analyze(_iban_analyzer, text)
    assert len(detected) == expected_count


@pytest.mark.parametrize(
    "text, expected_count",
    [
        # Exact match
        ("My SSN is 219-09-9999.", 1),
        ("My SSN is 219-09-9999", 1),
        ("My SSN is 219 09 9999", 1),
        # Fuzzy match
        ("My SSN is 078-05-1120", 1),
        ("My SSN is 078-05-112-123", 1),
        ("My SSN is 666-00-1234", 1),
        ("My SSN is 078-00-9999", 1),
        ("My SSN is 219 09-9999", 1),
        ("My SSN is 999-99-9999", 1),
        # Not detected
        ("My SSN is 999-99-9", 0),
    ],
)
def test_fuzzy_ssn_detection(text, expected_count):
    detected = _analyze(_ssn_analyzer, text)
    assert len(detected) == expected_count


@pytest.mark.parametrize(
    "text, expected_count",
    [
        # Exact match
        ("My email is here: john@gmail.com", 1),
        # Fuzzy match
        ("My email is here: john@.com", 1),
        ("My email is here: john@.gcom", 1),
        ("My email is here: john@com", 1),
        ("My email is here: john@gmail.con mistake", 1),
        ("My email is here: john@@gmail.com", 1),
        ("My email is here: john@gmail.colm", 1),
        ("My email is here: john.com", 1),
        # Not detected
        ("My email is here: john.", 0),
        ("My email is here: john@", 0),
        ("My email is here: john at gmail", 0),
    ],
)
def test_fuzzy_email_detection(text, expected_count):
    detected = _analyze(_email_analyzer, text)
    assert len(detected) == expected_count


@pytest.mark.parametrize(
    "text, expected_count",
    [
        # Exact match
        ("My credit card number is 4111 1111 1111 1111.", 1),
        ("My credit card number is 4111 1111 1111 1111 and 5500 0000 0000 0004.", 2),
        ("Here is 4111-1111-1111-1111 and also 5500 0000 0000 0004.", 2),
        # Fuzzy match
        ("My credit card number is 4111 1111 1111 11112.", 1),
        ("My fake card number is 1234 5678 9876 5432.", 1),
        ("My partial card number is 4111 1111 1111 11", 1),
        ("My partial card is 4111 1111 1111 11", 1),
        ("My credit card number is 4111_1111_1111_1111", 1),
        ("My partial card is 4111 1111 XXXX 1111 11", 1),
        # Not detected
        ("My very partial is 411 111", 0),
        ("No card numbers in this text.", 0),
        ("No card numbers in 411 111 bank and 123 dollar 234. bank", 0),
    ],
)
def test_fuzzy_credit_card_detection(text, expected_count):
    detected = _analyze(_cc_analyzer, text)
    assert len(detected) == expected_count


def test_empty_error_types_raises():
    with pytest.raises(ValueError, match="No error types"):
        fuzzy_pii_recognizer([(IbanRecognizer(), [])])


def test_insertions_raises():
    with pytest.raises(ValueError, match="Insertions are not allowed"):
        fuzzy_pii_recognizer([(IbanRecognizer(), [dict(insertions=1)])])


def test_get_fuzzy_recognizers_returns_tuple():
    from detectors.presidio import get_fuzzy_recognizers
    result = get_fuzzy_recognizers()
    assert isinstance(result, (list, tuple))
    assert len(result) == 4  # iban, ssn, email, credit_card
