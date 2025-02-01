import pytest

from data_manipulation.pii.number_to_word import number_to_word


@pytest.mark.parametrize(
    "text, spans, expected",
    [
        (
            "My Credit Card is 1234",
            [{"start": 18, "end": 22}],
            "My Credit Card is one-two-three-four",
        ),
        (
            "My Email from 94 is john94@gmail.com",
            [{"start": 19, "end": 35}],
            "My Email from 94 is johnnine-four@gmail.com",
        ),
    ],
)
def test_emojify_pii(text, spans, expected):
    assert number_to_word(text=text, spans=spans) == expected
