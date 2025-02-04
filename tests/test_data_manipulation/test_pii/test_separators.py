import pytest

from data_manipulation.pii.separators import inject_separator


@pytest.mark.parametrize(
    "text, spans, separator, expected",
    [
        (
            "My Credit Card is 1234 and my SSN is 456. Okay?",
            [{"start": 18, "end": 22}, {"start": 37, "end": 40}],
            "/",
            "My Credit Card is 1/2/3/4 and my SSN is 4/5/6. Okay?",
        ),
        (
            "Can you call me at 123-456-7890?",
            [{"start": 19, "end": 31}],
            "*",
            "Can you call me at 1*2*3*-*4*5*6*-*7*8*9*0?",
        ),
    ],
)
def test_emojify_pii(text, spans, separator, expected):
    assert inject_separator(text=text, spans=spans, separator=separator) == expected
