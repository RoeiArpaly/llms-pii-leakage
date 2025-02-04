import pytest

from data_manipulation.pii.number_to_word import number_to_word


@pytest.mark.parametrize(
    "text, spans, lang, expected",
    [
        (
            "My Credit Card is 1234",
            [{"start": 18, "end": 22}],
            "english",
            "My Credit Card is one-two-three-four",
        ),
        (
            "My Credit Card is 1234",
            [{"start": 18, "end": 22}],
            "spanish",
            "My Credit Card is uno-dos-tres-cuatro",
        ),
        (
            "My Email from 94 is john94@gmail.com",
            [{"start": 19, "end": 35}],
            "english",
            "My Email from 94 is johnnine-four@gmail.com",
        ),
        (
            "My Email from 94 is john94@gmail.com",
            [{"start": 19, "end": 35}],
            "hebrew",
            "My Email from 94 is johnתשע-ארבע@gmail.com",
        ),
    ],
)
def test_emojify_pii(text, spans, lang, expected):
    assert number_to_word(text=text, spans=spans, lang=lang) == expected
