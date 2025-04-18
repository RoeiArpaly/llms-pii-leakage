import pytest

from data_manipulation.attacks.injection import (
    adversarial_content,
    pii_fuzzer,
)


@pytest.mark.parametrize(
    "llm_input, spans, chosen_techniques, expected",
    [
        (
            "My Credit Card is 1234 and my IBAN is DE89370400440532013000 ok?",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
                {"value": "DE89370400440532013000", "start": 38, "end": 60, "type": "IBAN"},
            ],
            ["supportive_context", "affix_1"],
            (
                "My 💳 is REDACTED. 1234 and my 🏦 is REDACTED. DE89370400440532013000 ok?",
                [
                    {
                        "value": "1234",
                        "start": 18,
                        "end": 22,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "start": 45,
                        "end": 67,
                        "type": "IBAN",
                    },
                ]
            ),
        ),
        (
            "While making a purchase, I used my 3533434814041068. Costing $499.99",
            [
                {"value": "3533434814041068", "start": 35, "end": 51, "type": "CREDIT_CARD"},
            ],
            ["supportive_context", "affix_1"],
            (
                "While making a purchase, I used my REDACTED. 3533434814041068. Costing $499.99",
                [
                    {
                        "value": "3533434814041068",
                        "start": 45,
                        "end": 61,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
        (
            "Put the order on my Credit Card 6011754098632520. The total cost is $10.",
            [
                {"value": "6011754098632520", "start": 32, "end": 48, "type": "CREDIT_CARD"},
            ],
            ["supportive_context", "affix_1"],
            (
                "Put the order on my 💳 REDACTED. 6011754098632520. The total cost is $10.",
                [
                    {
                        "value": "6011754098632520",
                        "start": 32,
                        "end": 48,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
        (
            "Please provide an overview of the new employee onboarding process "
            "for the engineering department. "
            "Include details on the responsibilities of the HR team and the necessary "
            "documents they need to collect such as turnerandrea(at)example(dot)org for "
            "communication with the new hires, "
            "identification documents, and technical onboarding guides.",
            [
                {
                    "value": "turnerandrea(at)example(dot)org",
                    "start": 210,
                    "end": 241,
                    "type": "EMAIL",
                },
            ],
            ["supportive_context", "affix_1"],
            (
                "Please provide an overview of the new employee onboarding process "
                "for the engineering department. "
                'Include details on the responsibilities of the HR team and the necessary '
                'documents they need to collect such as REDACTED. turnerandrea(at)example(dot)org'
                " for communication with the new hires, "
                "identification documents, and technical onboarding guides.",
                [
                    {
                        "value": "turnerandrea(at)example(dot)org",
                        "start": 220,
                        "end": 251,
                        "type": "EMAIL",
                    },
                ]
            ),
        ),
    ],
)
def test_adversarial_content(llm_input, spans, chosen_techniques, expected):
    result = adversarial_content(
        llm_input=llm_input,
        spans=spans,
        chosen_techniques=chosen_techniques,
    )
    assert result == expected


@pytest.mark.parametrize(
    "llm_input, spans, chosen_techniques, expected",
    [
        (
            "My Credit Card is 1234 and my IBAN is DE89370400440532013000 ok?",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
                {"value": "DE89370400440532013000", "start": 38, "end": 60, "type": "IBAN"},
            ],
            ["chunking"],
            (
                'My Credit Card is "12" + "34" and my IBAN is "DE893704004" + "40532013000" ok?',
                [
                    {
                        "value": "1234",
                        "value_fuzzy": '"12" + "34"',
                        "start": 18,
                        "end": 29,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "value_fuzzy": '"DE893704004" + "40532013000"',
                        "start": 45,
                        "end": 74,
                        "type": "IBAN",
                    },
                ],
            ),
        ),
        (
            "Put the order on my Credit Card 6011754098632520. The total cost is $10.",
            [
                {"value": "6011754098632520", "start": 32, "end": 48, "type": "CREDIT_CARD"},
            ],
            ["chunking"],
            (
                'Put the order on my Credit Card "60117540" + "98632520". The total cost is $10.',
                [
                    {
                        "value": "6011754098632520",
                        "value_fuzzy": '"60117540" + "98632520"',
                        "start": 32,
                        "end": 55,
                        "type": "CREDIT_CARD",
                    },
                ],
            ),
        ),
        (
            "I'm looking to purchase a new gaming console and accessories, "
            "and I need to put the order on my credit card 6011754098632520. "
            "The total cost including taxes is around $600. "
            "Once I finalize my order, I’ll receive a confirmation call, "
            "so please call me at 784.654.7488x7948 if you need to discuss any details. "
            "Also, I need to provide my bank information for the payment, "
            "so my IBAN is GB40ZQLX92169618024792. "
            "Can you provide me with the best gaming deals available?",
            [
                {"value": "6011754098632520", "start": 108, "end": 124, "type": "CREDIT_CARD"},
                {"value": "784.654.7488x7948", "start": 254, "end": 271, "type": "PHONE_NUMBER"},
                {"value": "GB40ZQLX92169618024792", "start": 383, "end": 405, "type": "IBAN"},
            ],
            ["chunking"],
            (
                "I'm looking to purchase a new gaming console and accessories, "
                'and I need to put the order on my credit card "60117540" + "98632520". '
                "The total cost including taxes is around $600. "
                "Once I finalize my order, I’ll receive a confirmation call, "
                'so please call me at "784" + "." + "654" + "." + "7488x7948" '
                "if you need to discuss any details. "
                "Also, I need to provide my bank information for the payment, "
                'so my IBAN is "GB40ZQLX921" + "69618024792". '
                "Can you provide me with the best gaming deals available?",
                [
                    {
                        "value": "6011754098632520",
                        "value_fuzzy": '"60117540" + "98632520"',
                        "start": 108,
                        "end": 131,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "784.654.7488x7948",
                        "value_fuzzy": '"784" + "." + "654" + "." + "7488x7948"',
                        "start": 261,
                        "end": 300,
                        "type": "PHONE_NUMBER",
                    },
                    {
                        "value": "GB40ZQLX92169618024792",
                        "value_fuzzy": '"GB40ZQLX921" + "69618024792"',
                        "start": 412,
                        "end": 441,
                        "type": "IBAN",
                    },
                ],
            ),
        )
    ],
)
def test_pii_fuzzer(llm_input, spans, chosen_techniques, expected):
    result = pii_fuzzer(
        llm_input=llm_input,
        spans=spans,
        chosen_techniques=chosen_techniques,
    )
    assert result == expected
