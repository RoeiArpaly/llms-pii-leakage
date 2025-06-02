import pytest
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    UsSsnRecognizer,
)

from detectors.fuzzy_match import fuzzy_pii_recognizer
from detectors.presidio_detector import presidio_pii_analyzer


# TODO: Smart cache...

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
    """Test IBAN detection with fuzzy matching."""
    recognizers = [(IbanRecognizer(exact_match=True), [dict(substitutions=1)])]
    fuzzy_recognizers = fuzzy_pii_recognizer(recognizers=recognizers)
    detected = presidio_pii_analyzer(text=text, recognizers=fuzzy_recognizers, use_cache=False)

    print(f"Input: {text}\nDetected Entities: {detected}")
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
    """Test SSN detection with fuzzy matching."""
    recognizers = [(UsSsnRecognizer(), [
        dict(deletions=1),
        dict(deletions=2),
        dict(substitutions=1),
        dict(substitutions=1, deletions=1),
    ])]
    fuzzy_recognizers = fuzzy_pii_recognizer(recognizers=recognizers)
    detected = presidio_pii_analyzer(text=text, recognizers=fuzzy_recognizers, use_cache=False)

    print(f"Input: {text}\nDetected Entities: {detected}")
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
    """Test email detection with fuzzy matching."""
    recognizers = [(EmailRecognizer(), [
        dict(deletions=1),
        dict(substitutions=1),
    ])]
    fuzzy_recognizers = fuzzy_pii_recognizer(recognizers=recognizers)
    detected = presidio_pii_analyzer(text=text, recognizers=fuzzy_recognizers, use_cache=False)

    print(f"Input: {text}\nDetected Entities: {detected}")
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
        ("My partial is 4111 1111 1111 11", 1),
        ("My credit card number is 4111_1111_1111_1111", 1),
        ("My partial is 4111 1111 XXXX 1111 11", 1),
        # Not detected
        ("My very partial is 411 111", 0),
        ("No card numbers in this text.", 0),
        ("No card numbers in 411 111 bank and 123 dollar 234. bank", 0),
    ],
)
def test_fuzzy_credit_card_detection(text, expected_count):
    """Test credit card number detection with fuzzy matching."""
    recognizers = [(CreditCardRecognizer(), [
        dict(deletions=1),
        dict(deletions=3),
        dict(substitutions=3),
        dict(substitutions=4),
        dict(deletions=2, substitutions=2),
    ])]
    fuzzy_recognizers = fuzzy_pii_recognizer(recognizers=recognizers)
    detected = presidio_pii_analyzer(text=text, recognizers=fuzzy_recognizers, use_cache=False)

    print(f"Input: {text}\nDetected Entities: {detected}")
    assert len(detected) == expected_count
