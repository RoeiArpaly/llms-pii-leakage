"""OpenAI API client with retry logic and structured output parsing."""
import functools
import json
import os
import time

import requests

from config import Config
from logger import logger
from utils.perplexity import calculate_perplexity


class AuthenticationError(RuntimeError):
    """Raised on 401/403 from the LLM provider — never retried."""


def _resolve_provider(model: str) -> tuple[str, str, str]:
    """Pick provider from env at call time. Returns (url, token, model)."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        resolved = model if "/" in model else f"openai/{model}"
        return (
            "https://openrouter.ai/api/v1/chat/completions",
            openrouter_key,
            resolved,
        )
    return (
        "https://api.openai.com/v1/chat/completions",
        os.getenv("OPENAI_API_KEY"),
        model,
    )


def retry(
        times: int = 5,
        delay: int = 1,
        increment_param: str = None,
        increment_value: float = None,
        excluded_models: list[str] = None,
):
    """Retry a function up to `times` times with optional parameter incrementing."""
    if excluded_models is None:
        excluded_models = []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except AuthenticationError:
                    raise
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
) -> dict | str:
    if Config.DRYRUN:
        from detectors.llm.mock import mock_openai_response
        return mock_openai_response(
            data=data, logprobs=logprobs, structured_output=structured_output,
        )

    url, token, resolved_model = _resolve_provider(model=data.get("model", ""))
    if not token:
        raise AuthenticationError(
            "no API key set. Export OPENROUTER_API_KEY or OPENAI_API_KEY "
            "(or pass --api-key on the CLI). For smoke tests without a "
            "real LLM, set Config.DRYRUN=True or use --dryrun.",
        )
    data = {**data, "model": resolved_model}
    response = requests.post(
        url=url,
        headers={
            "Authorization": f"Bearer {token}",
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
            from detectors.llm.detector import filter_pii_logprobs
            pii_logprobs = filter_pii_logprobs(logprobs_content=logprobs_content)
            perplexity = calculate_perplexity(logprobs=pii_logprobs)
        try:
            json_schema = json.loads(message_content)
            json_schema["perplexity"] = perplexity
            return json_schema
        except json.JSONDecodeError as e:
            logger.error(content)
            raise ValueError(f"Invalid JSON format.\nError: {e}")
    if response.status_code in (401, 403):
        raise AuthenticationError(
            f"API authentication failed ({response.status_code}) at {url}: "
            f"{response.text.strip()[:300]}",
        )
    raise ValueError(
        f"Invalid response from API ({response.status_code}).\n{response.text}",
    )
