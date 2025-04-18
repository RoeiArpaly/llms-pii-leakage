import pytest

from data_manipulation.pii import char_to_word


@pytest.mark.parametrize(
    "text, lang, expected",
    [
        (
            "123456789",
            "english",
            "(one)(two)(three)(four)(five)(six)(seven)(eight)(nine)",
        ),
        (
            "123456789",
            "spanish",
            "(uno)(dos)(tres)(cuatro)(cinco)(seis)(siete)(ocho)(nueve)",
        ),
        (
            "john94@gmail.com",
            "english",
            "john(nine)(four)(at)gmail(dot)com",
        ),
        (
            "john34@gmail.com",
            "english",
            "john(three)(four)(at)gmail(dot)com",
        ),
        (
            "My phone number is 123-456-78",
            "english",
            "My phone number is (one)(two)(three)(dash)(four)(five)(six)(dash)(seven)(eight)",
        ),
    ],
)
def test_number_to_word(text, lang, expected):
    assert char_to_word(text=text, lang=lang) == expected
