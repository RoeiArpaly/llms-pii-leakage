import csv
import functools
import json
import os
import time

import requests

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from math import exp

from pandas import Series

from logger import logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def calculate_perplexity(logprobs: list[float]) -> float or None:
    """Calculate the perplexity of a sequence of log probabilities."""
    _N = len(logprobs)
    if _N == 0:
        return  # Avoid division by zero
    return exp(-sum(logprobs) / _N)


def filter_pii_logprobs(logprobs_content: list[dict]) -> list[float]:
    """
    Extracts the logprobs of tokens corresponding to the value of the 'pii_detected' key.
    """
    key_sequence = ["pi", "i", "_detect", "ed", '":']  # Tokens forming 'pii_detected":'
    n = len(key_sequence)

    for i in range(len(logprobs_content) - n):
        if [t["token"] for t in logprobs_content[i:i + n]] == key_sequence:
            # Now get the value token(s) after the key (likely one token like 'true' or 'false')
            value_token = logprobs_content[i + n]
            if value_token["token"] in ["true", "false"]:
                return [value_token["logprob"]]
            raise ValueError(
                f"Unexpected value token found: {value_token['token']}. Expected 'true' or 'false'."
            )
    raise ValueError("No 'pii_detected' key found in logprobs content.")


def retry(
        times: int = 5,
        delay: int = 1,
        increment_param: str = None,
        increment_value: float = None,
        excluded_models: list[str] = None,
):
    """
    Retries a function up to `times` times.
    If `increment_param` is set, it increases that parameter in the `data` dictionary
    on retry, UNLESS the model is in `excluded_models`.
    """
    if excluded_models is None:
        excluded_models = []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt + 1 < times:
                        logger.warning(
                            f"Retrying {func.__name__}, attempt {attempt + 1}/{times}. Error: {e}"
                        )
                        data = kwargs.get("data", {})
                        model = data.get("model")
                        if increment_param in data and model not in excluded_models:
                            data[increment_param] += increment_value
                            data[increment_param] = min(data[increment_param], 2)
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator


@retry(
    times=5,
    increment_param="temperature",
    increment_value=0.1,
    excluded_models=["gpt-5-mini"],
)
def post_request_openai(
        data: dict, logprobs: bool = False, structured_output: bool = True,
) -> dict or str:

    response = requests.post(
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=data,
    )
    if response.status_code == 200:
        content = response.json().get("choices", [{}])[0]
        message_content = content.get("message", {}).get("content", {})
        if not structured_output:
            return message_content
        perplexity = None
        if logprobs:
            logprobs_content = content.get("logprobs", {}).get("content", {})
            pii_logprobs = filter_pii_logprobs(logprobs_content=logprobs_content)
            perplexity = calculate_perplexity(logprobs=pii_logprobs)
        try:
            json_schema = json.loads(message_content)
            json_schema["perplexity"] = perplexity
            return json_schema
        except json.JSONDecodeError as e:
            logger.error(content)
            raise ValueError(f"Invalid JSON format.\nError: {e}")
    raise ValueError(f"Invalid response from OpenAI API.\n{response.text}")


def parallel_apply(func: callable, series: Series, max_workers: int = 8, **kwargs) -> list:
    func_with_kwargs = partial(func, **kwargs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func_with_kwargs, series))
    return results


def cast_to_json(column: Series) -> Series:
    return column.apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
    )


def infer_json(column: Series) -> Series:
    return column.apply(parse_json, column_name=column.name)


def parse_json(value, column_name):
    # TODO: improve logic to detect JSON
    if any(v in column_name for v in ["span", "prediction", "techniques", "result"]):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def csv_batch_writer(batch: list[dict], filename):
    """
    Appends a single batch of dicts to a CSV.
    Automatically handles header creation on the first call.
    """
    if not batch:
        return
    # Check if we need to write a header (file doesn't exist or is empty)
    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
    fieldnames = batch[0].keys()
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(batch)
