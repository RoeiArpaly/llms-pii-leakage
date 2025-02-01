import pytest

from data_manipulation.pii.emojify import emojify_pii


@pytest.mark.parametrize(
    "text, spans, expected",
    [
        (
            "My Credit Card is 1234",
            [{"start": 18, "end": 22}],
            "My Credit Card is 1️⃣2️⃣3️⃣4️⃣",
        ),
        (
            "My Email is john@gmail.com",
            [{"start": 11, "end": 26}],
            "My Email is 🅹🅾🅷🅽@🅶🅼🅰🅸🅻.🅲🅾🅼",
        ),
    ],
)
def test_emojify_pii(text, spans, expected):
    assert emojify_pii(text=text, spans=spans) == expected
