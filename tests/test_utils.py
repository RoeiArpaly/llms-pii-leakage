import pandas as pd
import pytest

from utils import (
    filter_pii_logprobs,
    parallel_apply,
    post_request_openai,
)


sample_data = pd.DataFrame({"llm_input": ["Dogs", "Cats"]})


def fetch_openai_response(input_text):
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": f"Tell me a joke about: {input_text}",
            },
        ],
    }
    return post_request_openai(data)


# A mock function to simulate the OpenAI API response.
def mock_post_request_openai(data):
    input_text = data["messages"][0]["content"].split(": ")[-1]
    return f"{input_text} joke."


def test_fetch_openai_response(mocker):
    # Patch the post_request_openai function for a direct call.
    mock_post = mocker.patch(
        __name__ + ".post_request_openai", side_effect=mock_post_request_openai
    )
    response = fetch_openai_response("Dogs")
    assert response == "Dogs joke."
    assert mock_post.call_count == 1


@pytest.mark.parametrize("max_workers", [2, 4, 8])
def test_parallel_apply(mocker, max_workers):
    # Patch 'post_request_openai' in this module's namespace.
    mock_post = mocker.patch(
        __name__ + ".post_request_openai", side_effect=mock_post_request_openai
    )
    data = sample_data.copy()
    data["response"] = parallel_apply(
        func=fetch_openai_response,
        series=data["llm_input"],
        max_workers=max_workers,
    )
    for i, response in enumerate(data["response"]):
        llm_input = data["llm_input"].iloc[i]
        assert response == f"{llm_input} joke."
    # Verify that our patched post_request_openai was called once per row.
    assert mock_post.call_count == len(data)


@pytest.mark.parametrize(
    "logprobs_content, expected",
    [
        # Test case 1: Basic example with valid PII tokens
        (
            [
                {"token": '{"', "logprob": 0.0},
                {"token": "value", "logprob": 0.0},
                {"token": '":"', "logprob": -0.001},
                {"token": "123", "logprob": -0.01},
                {"token": "456", "logprob": -0.02},
                {"token": '","', "logprob": 0.0},
                {"token": "start", "logprob": 0.0},
            ],
            [-0.01, -0.02],
        ),
        # Test case 2: Ends with "start" token instead of '","'
        (
            [
                {"token": "value", "logprob": 0.0},
                {"token": '":"', "logprob": -0.002},
                {"token": "789", "logprob": -0.03},
                {"token": "start", "logprob": 0.0},
                {"token": '","', "logprob": 0.0},
                {"token": "start", "logprob": 0.0},
            ],
            [-0.03, 0.0],
        ),
        # Test case 3: No matching "value" or ':' pattern
        (
            [
                {"token": "other", "logprob": -0.5},
                {"token": "data", "logprob": -0.6},
            ],
            [],
        ),
        # Test case 4: "value" present but missing ':' token
        (
            [
                {"token": "value", "logprob": -0.1},
                {"token": "123", "logprob": -0.2},
                {"token": '","', "logprob": 0.0},
            ],
            [],
        ),
        # Test case 5: Ignores entries without logprob
        (
            [
                {"token": "value"},
                {"token": '":"', "logprob": -0.001},
                {"token": "111", "logprob": -0.01},
                {"token": "222", "logprob": -0.03},
                {"token": '","', "logprob": 0.0},
            ],
            [-0.01, -0.03],
        ),
        # Test case 6: PII with comma in the value
        (
            [
                {"token": "value", "logprob": 0.0},
                {"token": '":"', "logprob": -0.001},
                {"token": "111", "logprob": -0.01},
                {"token": "222", "logprob": -0.03},
                {"token": '","', "logprob": 0.0},
                {"token": "333", "logprob": -0.04},
                {"token": '","', "logprob": 0.0},
                {"token": "start", "logprob": 0.0},
            ],
            [-0.01, -0.03, -0.04],
        ),
        # Test case 7: multiple "value" tokens (multiple spans)
        (
            [
                {"token": "value", "logprob": 0.0},
                {"token": '":"', "logprob": -0.001},
                {"token": "111", "logprob": -0.01},
                {"token": "222", "logprob": -0.03},
                {"token": '","', "logprob": 0.0},
                {"token": "start", "logprob": 0.0},
                {"token": "value", "logprob": 0.0},
                {"token": '":"', "logprob": -0.002},
                {"token": "333", "logprob": -0.04},
                {"token": "444", "logprob": -0.05},
                {"token": '","', "logprob": 0.0},
                {"token": "start", "logprob": 0.0},
            ],
            [-0.01, -0.03, -0.04, -0.05],
        ),
    ]
)
def test_filter_pii_logprobs(logprobs_content, expected):
    assert filter_pii_logprobs(logprobs_content) == expected
