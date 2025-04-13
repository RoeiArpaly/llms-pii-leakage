import pytest

from data_manipulation.content.supportive_context import supportive_context


@pytest.mark.parametrize(
    "text, spans, expected",
    [
        (
            "My Credit Card is 1234",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
            ],
            (
                "My 💳 is 1234",
                [
                    {"value": "1234", "start": 8, "end": 12, "type": "CREDIT_CARD"},
                ]
            ),
        ),
        (
            "My Email is john@gmail.com",
            [
                {"value": "john@gmail.com", "start": 12, "end": 26, "type": "EMAIL"},
            ],
            (
                "My 📧 is john@gmail.com",
                [
                    {"value": "john@gmail.com", "start": 8, "end": 22, "type": "EMAIL"},
                ]
            ),
        ),
        (
            "My Credit Card is 1234 and my Credit Card is 5678",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
                {"value": "5678", "start": 42, "end": 46, "type": "CREDIT_CARD"},
            ],
            (
                "My 💳 is 1234 and my 💳 is 5678",
                [
                    {"value": "1234", "start": 8, "end": 12, "type": "CREDIT_CARD"},
                    {"value": "5678", "start": 25, "end": 29, "type": "CREDIT_CARD"},
                ]
            ),
        ),
        (
            "My Credit Card is 1234 and my Credit Card is 1234",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
                {"value": "1234", "start": 42, "end": 46, "type": "CREDIT_CARD"},
            ],
            (
                "My 💳 is 1234 and my 💳 is 1234",
                [
                    {"value": "1234", "start": 8, "end": 12, "type": "CREDIT_CARD"},
                    {"value": "1234", "start": 25, "end": 29, "type": "CREDIT_CARD"},
                ]
            ),
        ),
    ],
)
def test_emojify_text(text, spans, expected):
    assert supportive_context(text=text, spans=spans) == expected
