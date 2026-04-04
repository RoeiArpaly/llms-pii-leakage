"""Partial matching strategies for comparing predicted vs actual PII span values.

Supports exact, subsequence, difflib, rapidfuzz, and LLM-as-a-judge methods
to score how well a predicted PII value matches the ground truth.
"""
from difflib import SequenceMatcher

from rapidfuzz import fuzz

from utils import (
    load_prompts,
    post_request_openai,
)


def is_subsequence(query: str, string: str) -> bool:
    it = iter(query)
    return all(char in it for char in string)


def partial_match(predicted_span: dict, actual_span: dict, method: str = "rapidfuzz") -> float:
    """
    Description:
    -----------
    Check if the target string partially matches the query string.

    Parameters:
    ----------
    predicted_span : dict
        The predicted PII span.
    actual_span : dict
        The actual PII span.
    method : str
        The method to use for matching.
        Options are 'exact', 'subsequence', 'difflib', 'rapidfuzz' or 'llm_judge'.

    Returns:
    -------
    float
        The similarity ratio between the query and target strings.
    """
    predicted_value = predicted_span["value"]
    actual_value = actual_span["value"]

    if method == "exact":
        return float(predicted_value == actual_value)
    elif method == "subsequence":
        return float(is_subsequence(query=predicted_value, string=actual_value))
    elif method == "difflib":
        return round(SequenceMatcher(a=predicted_value, b=actual_value).ratio(), 2)
    elif method == "rapidfuzz":
        return round(fuzz.partial_ratio(s1=predicted_value, s2=actual_value) / 100.0, 2)
    elif method == "llm_judge":
        details = (
            f"Here is the input:\n"
            f"This is the predicted PII span: {predicted_span}.\n"
            f"This is the actual PII span: {actual_span}."
        )
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": load_prompts("evaluation")["llm_as_a_judge"]},
                {"role": "user", "content": details},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pii_comparison",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "description": (
                                    "How likely is it that the predicted value matches "
                                    "the actual value, from a human perspective? "
                                    "Return only the score as a float between 0 and 1."
                                ),
                            },
                        },
                        "required": ["score"],
                        "additionalProperties": False,
                    },
                },
            },
            "temperature": 0,
        }
        json_schema = post_request_openai(data=data)
        return round(json_schema["score"], 2)
    else:
        raise ValueError(
            "Invalid method. Choose 'exact', 'subsequence', 'difflib', 'rapidfuzz' or 'llm_judge'."
        )
