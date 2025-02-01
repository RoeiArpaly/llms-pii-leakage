import pytest

from fuzzers.emojify import (
    emojify_pii,
    emojify_pii_entity,
)


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
    assert emojify_pii_entity(text=text) == expected
