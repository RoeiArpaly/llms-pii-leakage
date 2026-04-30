"""Perplexity calculation from log probabilities."""
from math import exp


def calculate_perplexity(logprobs: list[float]) -> float | None:
    """Calculate the perplexity of a sequence of log probabilities."""
    if not logprobs:
        return None
    return exp(-sum(logprobs) / len(logprobs))
