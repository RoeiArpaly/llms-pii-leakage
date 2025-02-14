import pytest

from data_manipulation.pii import emojify_pii


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "1234",
            "1️⃣2️⃣3️⃣4️⃣",
        ),
        (
            "john34@gmail.com",
            "🅹🅾🅷🅽3️⃣4️⃣@🅶🅼🅰🅸🅻.🅲🅾🅼",
        ),
        (
            "JOHN34@gmail.com",
            "🅹🅾🅷🅽3️⃣4️⃣@🅶🅼🅰🅸🅻.🅲🅾🅼",
        ),

    ],
)
def test_emojify_pii(text, expected):
    assert emojify_pii(text=text) == expected
