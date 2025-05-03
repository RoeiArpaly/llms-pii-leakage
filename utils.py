import functools
import json
import os
import time

import requests

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from pandas import Series

from logger import logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def retry(times: int = 5, delay: int = 1):
    """Retries a function up to `times` times with a `delay` in seconds between attempts."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt + 1 < times:
                        logger.warning(f"Retrying {func.__name__}, {attempt + 1}/{times}...")
                        time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return decorator


@retry(times=5)
def post_request_openai(data: dict) -> dict:
    response = requests.post(
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=data,
    )

    if response.status_code == 200:
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", {})
        )
        try:
            json_schema = json.loads(content)
            return json_schema
        except json.JSONDecodeError:
            logger.error(content)
            raise ValueError("Invalid JSON format.")
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
    if "span" in column_name or "prediction" in column_name or "techniques" in column_name:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value
