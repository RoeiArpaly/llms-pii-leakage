import pytest

from pii_shield import guard


@pytest.mark.parametrize(
    "text",
    [
        """
        For my monthly budget,
        I spent a total of $2,500 on various expenses using my credit card ending
        in 6506-7724-7051-8481.
        """,
        """
        For my monthly budget,
        I spent a total of $2,500 on various expenses using my credit card ending
        in 6506-7724-7051-848A.
        """,
    ],
)
def test_guard(text):
    result = guard(text=text, perplexity_threshold=5.0)
    assert result["detected"], f"PII should be detected in the text: {text}"


def test_guard_no_pii():
    result = guard(
        text="What are the best practices for code review?",
        perplexity_threshold=5.0,
    )
    assert not result["detected"]


def test_guard_empty():
    result = guard(text="", perplexity_threshold=5.0)
    assert not result["detected"]
