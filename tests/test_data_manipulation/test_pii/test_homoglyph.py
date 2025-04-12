import pytest

from data_manipulation.pii import homoglyph


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "1234",
            "１２３４",
        ),
        (
            "john34@gmail.com",
            "јоһп３４@ɡмаіӏ.сом",
        ),
        (
            "JOHN34@gmail.com",
            "ЈОНＮ３４@ɡмаіӏ.сом",
        ),
    ],
)
def test_homoglyph(text, expected):
    assert homoglyph(text=text) == expected
