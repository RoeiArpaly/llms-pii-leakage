import pytest

from data_manipulation.attacks.red_teaming import chunking
from data_manipulation.attacks.red_teaming.pii.chunking import smart_split


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "john.doe@gmail.com",
            ["john", ".", "doe", "@", "gmail", ".", "com"],
        ),
        (
            "johndoe@example.com",
            ["johndoe", "@", "example", ".", "com"],
        ),
        (
            "123-456-7890",
            ["123", "-", "456", "-", "7890"],
        ),

    ],
)
def test_smart_split(text, expected):
    assert smart_split(text=text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "john.doe@gmail.com",
            '"john" + "." + "doe" + "@" + "gmail" + "." + "com"',

        ),
        (
          "GB11CDHV69994209133897",
          '"GB1" + "1CD" + "HV6" + "999" + "420" + "913" + "3897"'
        ),
        (
            "123-456-7890",
            '"123" + "-" + "456" + "-" + "7890"',
        ),
        (
            "123456789",
            '"123" + "456" + "789"',
        ),
        (
            "1234567",
            '"123" + "4567"',
        ),
        (
            "123456",
            '"123" + "456"',
        ),
        (
            "12345",
            '"12" + "345"',
        ),
        (
            "1234",
            '"12" + "34"',
        ),
        (
            "123",
            '"1" + "23"',
        ),
        (
            "12",
            '"1" + "2"',
        ),
        (
            "1",
            "1",
        ),
    ],
)
def test_chunking(text, expected):
    assert chunking(text=text) == expected
