import yaml

from difflib import SequenceMatcher
from importlib.resources import files
from rapidfuzz import fuzz

from utils import post_request_openai


PROMPTS_PATH = files("evaluation").joinpath("prompts.yaml")

with PROMPTS_PATH.open("r") as f:
    PROMPTS = yaml.safe_load(f)


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

    if method == "exact":
        query = predicted_span["value"]
        string = actual_span["value"]
        return float(query == string)
    elif method == "subsequence":
        query = predicted_span["value"]
        string = actual_span["value"]
        return float(is_subsequence(query=query, string=string))
    elif method == "difflib":
        a = predicted_span["value"]
        b = actual_span["value"]
        return round(SequenceMatcher(a=a, b=b).ratio(), 2)
    elif method == "rapidfuzz":
        s1 = predicted_span["value"]
        s2 = actual_span["value"]
        return round(fuzz.partial_ratio(s1=s1, s2=s2) / 100.0, 2)
    elif method == "llm_judge":
        details = (
            f"Here is the input:\n"
            f"This is the predicted PII span: {predicted_span}.\n"
            f"This is the actual PII span: {actual_span}."
        )
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": PROMPTS["llm_as_a_judge"]},
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
