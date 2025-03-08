import json
import os

import requests

from pandas import Series

from logger import logger


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def cast_to_json(column: Series) -> Series:
    return column.apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)


def infer_json(column: Series) -> Series:
    return column.apply(parse_json, column_name=column.name)


def parse_json(value, column_name):
    # TODO: improve
    if "span" in column_name or "prediction" in column_name or "techniques" in column_name:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value
