import pytest

from data_manipulation.pii import number_to_word


@pytest.mark.parametrize(
    "text, lang, expected",
    [
        (
            "123456789",
            "english",
            "one two three four five six seven eight nine",
        ),
        (
            "123456789",
            "spanish",
            "uno dos tres cuatro cinco seis siete ocho nueve",
        ),
        (
            "john94@gmail.com",
            "english",
            "johnnine four@gmail.com"
        ),
    ],
)
def test_number_to_word(text, lang, expected):
    assert number_to_word(text=text, lang=lang) == expected
