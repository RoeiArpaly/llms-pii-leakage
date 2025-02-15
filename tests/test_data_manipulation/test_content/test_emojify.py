import pytest

from data_manipulation.content.supportive_context import supportive_context


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "My Credit Card is 1234",
            "My 💳 is 1234",
        ),
        (
            "My Email is john@gmail.com",
            "My 📧 is john@gmail.com",
        ),
    ],
)
def test_emojify_text(text, expected):
    assert supportive_context(text=text) == expected
