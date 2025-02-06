import pytest

from data_manipulation.pii.number_to_roman import number_to_roman


@pytest.mark.parametrize(
    "text, spans, expected",
    [
        (
            "The number 2021 is the current year 3.",
            [{"start": 11, "end": 15}, {"start": 36, "end": 37}],
            "The number MMXXI is the current year III.",
        ),
        (
            "is 1990 and 2021 and 2025",
            [{"start": 3, "end": 7}, {"start": 21, "end": 25}],
            "is MCMXC and 2021 and MMXXV",
        ),

    ],
)
def test_number_to_roman(text, spans, expected):
    assert number_to_roman(text=text, spans=spans) == expected
