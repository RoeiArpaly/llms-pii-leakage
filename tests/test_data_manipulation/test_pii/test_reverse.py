import pytest

from data_manipulation.pii import reverse_pii


@pytest.mark.parametrize(
    "text, expected",
    [("1234", "4321")],
)
def test_inject_separator(text, expected):
    assert reverse_pii(text=text) == expected
