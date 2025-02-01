import pytest

from data_generation.data_validators import (
    contain_pii_template,
    luhn_verify,
)


@pytest.mark.parametrize(
    "text, contains_pii, expected",
    [
        ("My phone number is {{phone_number}}", True, None),
        ("My phone number is 1234567890", False, None),
        ("My phone number is {{phone_number}}", False, ValueError),
        ("My phone number is {{phone_number", True, ValueError),
        ("My phone number is {{phone_number", False, ValueError),
        ("My phone number is {{hello_world}}", True, ValueError),
        ("My phone number is {{hello_world}}", False, ValueError),
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
    "string, expected",
    [
        ("356938035643809", None),
        ("213174120250531", None),
        ("3590520199312676", None),
        ("371080715633510", None),
        ("4585115860744434064", None),
        ("4222222222222222", ValueError),
        ("1234567890", ValueError),
    ],
)
def test_luhn_verify(string, expected):
    if expected is None:
        assert luhn_verify(string) is None
    else:
        with pytest.raises(ValueError):
            luhn_verify(string)
