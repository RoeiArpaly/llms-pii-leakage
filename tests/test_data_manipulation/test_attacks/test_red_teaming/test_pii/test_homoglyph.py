import pytest

from data_manipulation.attacks.red_teaming import homoglyph


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "1234",
            "₁₂₃₄",
        ),
        (
            "john34@gmail.com",
            "јоһп₃₄＠ɡмаіӏ．сом",
        ),
        (
            "JOHN34@gmail.com",
            "ЈОНＮ₃₄＠ɡмаіӏ．сом",
        ),
    ],
)
def test_homoglyph(text, expected):
    assert homoglyph(text=text) == expected
