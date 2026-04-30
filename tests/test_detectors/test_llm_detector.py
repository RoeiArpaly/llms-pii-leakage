from detectors.llm.detector import llm_pii_detector
from utils import load_prompts


class TestGetPrompts:

    def test_returns_dict_with_spans_detector(self):
        prompts = load_prompts("detectors.llm")
        assert isinstance(prompts, dict)
        assert "spans_detector" in prompts

    def test_cached(self):
        assert load_prompts("detectors.llm") is load_prompts("detectors.llm")


class TestLlmPiiDetector:

    def test_none_returns_empty_spans(self):
        result = llm_pii_detector(None)
        assert isinstance(result, dict)
        assert result["spans"] == []

    def test_calls_openai_with_correct_schema(self, mocker):
        mock_post = mocker.patch("detectors.llm.detector.post_request_openai")
        mock_post.return_value = {
            "result": {"pii_detected": True, "predicted_proba": 0.9},
            "spans": [{"value": "123-45-6789", "start": 0, "end": 11, "type": "ssn"}],
            "perplexity": None,
        }
        result = llm_pii_detector("my ssn is 123-45-6789")

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data") or call_kwargs[0][0]
        if isinstance(data, dict):
            assert data["model"] == "gpt-4o-mini"
            assert data["temperature"] == 0
            assert data["max_tokens"] == 3_000
            assert len(data["messages"]) == 2
            assert data["messages"][0]["role"] == "system"
            assert data["messages"][1]["content"] == "my ssn is 123-45-6789"
        assert result["result"]["pii_detected"] is True

    def test_logprobs_flag_passed(self, mocker):
        mock_post = mocker.patch("detectors.llm.detector.post_request_openai")
        mock_post.return_value = {"result": {}, "spans": [], "perplexity": None}
        llm_pii_detector("text", logprobs=True)

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("logprobs") is True or call_kwargs[1].get("logprobs") is True
        data = call_kwargs.kwargs.get("data") or call_kwargs[1].get("data") or call_kwargs[0][0]
        if isinstance(data, dict):
            assert data.get("logprobs") is True

    def test_custom_model(self, mocker):
        mock_post = mocker.patch("detectors.llm.detector.post_request_openai")
        mock_post.return_value = {"result": {}, "spans": [], "perplexity": None}
        llm_pii_detector("text", model="gpt-5-mini")

        data = mock_post.call_args.kwargs.get("data") or mock_post.call_args[0][0]
        if isinstance(data, dict):
            assert data["model"] == "gpt-5-mini"
