import pytest

from data_manipulation.rule_based import adversarial_content


@pytest.mark.parametrize(
    "llm_input, spans, chosen_techniques, expected",
    [
        (
            "My Credit Card is 1234 and my IBAN is DE89370400440532013000 ok?",
            [
                {"value": "1234", "start": 18, "end": 21, "type": "CREDIT_CARD"},
                {"value": "DE89370400440532013000", "start": 38, "end": 59, "type": "IBAN"},
            ],
            ["supportive_context", "affix_1"],
            (
                "My 💳 is REDACTED. 1234 and my 🏦 is REDACTED. DE89370400440532013000 ok?",
                [
                    {
                        "value": "1234",
                        "start": 18,
                        "end": 21,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "start": 45,
                        "end": 66,
                        "type": "IBAN",
                    },
                ]
            ),
        ),
        (
            "While making a purchase, I used my 3533434814041068. Costing $499.99",
            [
                {"value": "3533434814041068", "start": 35, "end": 50, "type": "CREDIT_CARD"},
            ],
            ["supportive_context", "affix_1"],
            (
                "While making a purchase, I used my REDACTED. 3533434814041068. Costing $499.99",
                [
                    {
                        "value": "3533434814041068",
                        "start": 45,
                        "end": 60,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        )
    ],
)
def test_adversarial_content(llm_input, spans, chosen_techniques, expected):
    result = adversarial_content(
        llm_input=llm_input,
        spans=spans,
        chosen_techniques=chosen_techniques,
    )
    assert result == expected
