import pytest

from data_manipulation.pii import chunking
from data_manipulation.pii.chunking import smart_split


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "john.doe@gmail.com",
            ["john", ".", "doe", "@", "gmail", ".", "com"],
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
            "123-456-7890",
            '"123" + "-" + "456" + "-" + "7890"',
        ),
        (
            "123456789",
            '"1234" + "56789"',
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
