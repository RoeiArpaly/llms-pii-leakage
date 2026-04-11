"""Perplexity calculation and logprob extraction for PII detection confidence."""
from math import exp


def calculate_perplexity(logprobs: list[float]) -> float | None:
    """Calculate the perplexity of a sequence of log probabilities."""
    if not logprobs:
        return None
    return exp(-sum(logprobs) / len(logprobs))


def filter_pii_logprobs(logprobs_content: list[dict]) -> list[float]:
    """Extract logprobs of the true/false token following the 'pii_detected' key."""
    key_sequence = ["pi", "i", "_detect", "ed", '":']
    n = len(key_sequence)

    for i in range(len(logprobs_content) - n):
        if [t["token"] for t in logprobs_content[i:i + n]] == key_sequence:
            value_token = logprobs_content[i + n]
            if value_token["token"] in ["true", "false"]:
                return [value_token["logprob"]]
            raise ValueError(
                f"Unexpected value token found: {value_token['token']}. "
                f"Expected 'true' or 'false'."
            )
    raise ValueError("No 'pii_detected' key found in logprobs content.")
