import pytest

from data_manipulation.pii import number_to_roman


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "john94@gmail.com",
            "johnXCIV@gmail.com",
        ),
        (
            "1990",
            "MCMXC",
        ),
        (
            # 19901234567890 is too large to convert to Roman numerals
            "19901234567890",
            "19901234567890",
        ),

    ],
)
def test_number_to_roman(text, expected):
    assert number_to_roman(text=text) == expected
