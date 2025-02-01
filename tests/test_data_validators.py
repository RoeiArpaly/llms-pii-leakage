import pytest

from data_validators import contain_pii_template


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My phone number is {{phone_number}}", True),
        ("My phone number is {{phone_number", ValueError),
        ("My phone number is {{hello_world}}", ValueError),
        ("My phone number is 1234567890", False),
    ],
)
def test_contain_pii_template_parametrized(text, expected):
    if isinstance(expected, bool):
        assert contain_pii_template(text) == expected
    else:
        with pytest.raises(ValueError):
            contain_pii_template(text)
