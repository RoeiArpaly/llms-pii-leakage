import pytest

from data_manipulation.pii import word_symbols


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "john34@gmail.com",
            "john34(at)gmail(dot)com"
        ),
        (
            "My phone number is 123-456-7890",
            "My phone number is 123(dash)456(dash)7890"
        ),
    ],
)
def test_word_symbols(text, expected):
    assert word_symbols(text=text) == expected
