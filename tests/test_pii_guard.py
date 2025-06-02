import pytest

from config import Config
from pii_guard import guard


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
    result = guard(text=text, perplexity_threshold=Config.PERPLEXITY_THRESHOLD)
    assert result["detected"], f"PII should be detected in the text: {text}"
