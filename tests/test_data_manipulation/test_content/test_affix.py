import pytest

from data_manipulation.content import adversarial_affix


@pytest.fixture
def sample_data():
    return {
        "llm_input": "My phone number is 180087335977780 and my IBAN is DE89370400440532013000",
        "spans": [
            {
                "value": "180087335977780",
                "start": 19,
                "end": 34,
                "type": "phone_number",
            },
            {
                "value": "DE89370400440532013000",
                "start": 50,
                "end": 72,
                "type": "iban_code",
            },
        ],
        "llm_input_long": (
            "Please provide an overview of the new employee onboarding process "
            "for the engineering department. "
            "Include details on the responsibilities of the HR team and the necessary "
            "documents they need to collect such as turnerandrea(at)example(dot)org for "
            "communication with the new hires, "
            "identification documents, and technical onboarding guides."
        ),
        "spans_long": [
            {
                "value": "turnerandrea(at)example(dot)org",
                "start": 210,
                "end": 241,
                "type": "email",
            },
        ],
    }


@pytest.mark.parametrize(
    "llm_input, spans, adv_content, prefix, expected",
    [
        (
            "llm_input",
            "spans",
            "ADVERSARIAL_CONTENT",
            True,
            (
                "My phone number is ADVERSARIAL_CONTENT 180087335977780 "
                "and my IBAN is ADVERSARIAL_CONTENT DE89370400440532013000",
                [
                    {
                        "value": "180087335977780",
                        "start": 39,
                        "end": 54,
                        "type": "phone_number",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "start": 90,
                        "end": 112,
                        "type": "iban_code",
                    },
                ],
            ),
        ),
        (
            "llm_input",
            "spans",
            "ADVERSARIAL_CONTENT",
            False,
            (
                "My phone number is 180087335977780 ADVERSARIAL_CONTENT "
                "and my IBAN is DE89370400440532013000 ADVERSARIAL_CONTENT",
                [
                    {
                        "value": "180087335977780",
                        "start": 19,
                        "end": 34,
                        "type": "phone_number",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "start": 70,
                        "end": 92,
                        "type": "iban_code",
                    },
                ],
            ),
        ),
        (
            "llm_input_long",
            "spans_long",
            "ADVERSARIAL_CONTENT",
            False,
            (
                "Please provide an overview of the new employee onboarding process "
                "for the engineering department. "
                "Include details on the responsibilities of the HR team and the necessary "
                "documents they need to collect such as turnerandrea(at)example(dot)org "
                "ADVERSARIAL_CONTENT for communication with the new hires, "
                "identification documents, and technical onboarding guides.",
                [
                    {
                        "value": "turnerandrea(at)example(dot)org",
                        "start": 210,
                        "end": 241,
                        "type": "email",
                    },
                ],
            ),
        ),
    ],
)
def test_adversarial_affix(
    sample_data,
    llm_input,
    spans,
    adv_content,
    prefix,
    expected,
):
    assert (
        adversarial_affix(
            llm_input=sample_data[llm_input],
            spans=sample_data[spans],
            adv_affix=adv_content,
            prefix=prefix,
        )
        == expected
    )
