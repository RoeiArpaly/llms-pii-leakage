import pytest

from data_manipulation.attacks.red_teaming import inject_separator


@pytest.mark.parametrize(
    "text, separator, expected",
    [
        (
            "1234",
            "/",
            "1/2/3/4",
        ),
        (
            "123-456-7890",
            "*",
            "1*2*3*-*4*5*6*-*7*8*9*0",
        ),
    ],
)
def test_inject_separator(text, separator, expected):
    assert inject_separator(text=text, separator=separator) == expected
