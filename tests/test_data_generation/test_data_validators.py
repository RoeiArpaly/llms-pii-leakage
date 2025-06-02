import pytest

from data_generation.pii_validators import (
    is_valid_iban,
    is_valid_credit_card,
)
from data_generation.template_validators import contain_pii_template


@pytest.mark.parametrize(
    "text, contains_pii, expected",
    [
        ("My phone number is {{phone_number}}", True, None),
        ("My phone number is {{phone_number}}", False, ValueError),
        ("My phone number is {{phone_number", True, ValueError),
        ("My phone number is {{phone_number", False, ValueError),
        ("My phone number is {{hello_world}}", True, ValueError),
        ("My phone number is {{hello_world}}", False, ValueError),
        ("My phone number is 1234567890", False, ValueError),
        ("My phone number is 1234567890", True, ValueError),
    ],
)
def test_contain_pii_template_parametrized(text, contains_pii, expected):
    if expected is None:
        assert contain_pii_template(text, contains_pii=contains_pii) is None
    else:
        with pytest.raises(ValueError):
            contain_pii_template(text, contains_pii=contains_pii)


@pytest.mark.parametrize(
    "card_number, expected",
    [
        ("356938035643809", True),
        ("213174120250531", True),
        ("3590520199312676", True),
        ("371080715633510", True),
        ("4585115860744434064", True),
        ("4222222222222222", False),
        ("1234567890", False),
    ],
)
def test_luhn_verify(card_number, expected):
    assert is_valid_credit_card(card_number) == expected


@pytest.mark.parametrize("iban, expected", [
    # Valid IBANs
    ("DE44500105175407324931", True),  # Germany
    ("GB82WEST12345698765432", True),  # UK
    ("FR1420041010050500013M02606", True),  # France
    ("GR1601101250000000012300695", True),  # Greece
    ("ES9121000418450200051332", True),  # Spain
    ("NL91ABNA0417164300", True),  # Netherlands

    # Valid but with spaces
    ("DE44 5001 0517 5407 3249 31", True),
    ("GB82 WEST 1234 5698 7654 32", True),

    # Invalid checksums
    ("DE44500105175407324932", False),
    ("GB00WEST12345698765432", False),

    # Invalid structure
    ("DExx500105175407324931", False),  # 'xx' not digits
    ("DE4450010517540732", False),  # Too short
    ("DE4450010517540732493123456789012345678901234567890", False),  # Too long

    # Non-IBAN strings
    ("explore potential challenges like", False),
    ("of 2024 and what genres", False),
    ("", False),
    ("1234567890", False),
    ("ABCDEFGH", False),
])
def test_is_valid_iban(iban, expected):
    assert is_valid_iban(iban) == expected
