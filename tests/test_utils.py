import pandas as pd
import pytest

from utils import (
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
