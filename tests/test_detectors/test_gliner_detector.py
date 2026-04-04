
from detectors.gliner.detector import (
    _filter_spans,
    gliner_pii_detector,
    gliner_pii_detector_batch,
)


class TestFilterSpans:

    def test_renames_keys(self):
        spans = [{"text": "123-45-6789", "label": "ssn", "start": 0, "end": 11}]
        result = _filter_spans(spans)
        assert result == [{"value": "123-45-6789", "type": "ssn", "start": 0, "end": 11}]

    def test_filters_invalid_values(self):
        spans = [
            {"text": "email address here", "label": "email", "start": 0, "end": 18},
            {"text": "e-addy here", "label": "email", "start": 0, "end": 11},
            {"text": "john@test.com", "label": "email", "start": 0, "end": 13},
        ]
        result = _filter_spans(spans)
        assert len(result) == 1
        assert result[0]["value"] == "john@test.com"

    def test_empty_list(self):
        assert _filter_spans([]) == []


class TestGlinerPiiDetector:

    def test_returns_list(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.predict_entities.return_value = [
            {"text": "123-45-6789", "label": "ssn", "start": 10, "end": 21, "score": 0.9},
        ]
        mocker.patch("detectors.gliner.detector.get_gliner_model", return_value=mock_model)

        result = gliner_pii_detector("my ssn is 123-45-6789")
        assert len(result) == 1
        assert result[0]["value"] == "123-45-6789"
        assert result[0]["type"] == "ssn"

    def test_no_detections(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.predict_entities.return_value = []
        mocker.patch("detectors.gliner.detector.get_gliner_model", return_value=mock_model)

        assert gliner_pii_detector("hello world") == []


class TestGlinerPiiDetectorBatch:

    def test_batch_returns_list_of_lists(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.inference.return_value = [
            [{"text": "123-45-6789", "label": "ssn", "start": 0, "end": 11}],
            [],
        ]
        mocker.patch("detectors.gliner.detector.get_gliner_model", return_value=mock_model)

        results = gliner_pii_detector_batch(["ssn text", "safe text"])
        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[0][0]["value"] == "123-45-6789"
        assert results[1] == []

    def test_batch_empty(self, mocker):
        mock_model = mocker.MagicMock()
        mock_model.inference.return_value = [[], []]
        mocker.patch("detectors.gliner.detector.get_gliner_model", return_value=mock_model)

        results = gliner_pii_detector_batch(["a", "b"])
        assert results == [[], []]
