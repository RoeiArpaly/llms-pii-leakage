
from pandas import Series

from detectors.guards.utils import guard_pii_detector, guard_pii_detector_batch


PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}


class TestGuardPiiDetector:

    def test_none_text_returns_empty(self):
        assert guard_pii_detector(None, classifier=lambda t: True) == []

    def test_classifier_returns_false(self):
        assert guard_pii_detector("hello", classifier=lambda t: False) == []

    def test_classifier_returns_true(self):
        result = guard_pii_detector("my ssn is 123-45-6789", classifier=lambda t: True)
        assert result == [PII_SPAN]

    def test_classifier_receives_text(self):
        received = []
        guard_pii_detector("test input", classifier=lambda t: received.append(t) or False)
        assert received == ["test input"]

    def test_kwargs_forwarded_to_classifier(self):
        received_kwargs = {}

        def classifier(text, **kwargs):
            received_kwargs.update(kwargs)
            return False

        guard_pii_detector("text", classifier, model_name="test-model")
        assert received_kwargs == {"model_name": "test-model"}


class TestGuardPiiDetectorBatch:

    def test_all_safe(self):
        data = Series(["hello", "world"])
        result = guard_pii_detector_batch(data, lambda texts: [False] * len(texts))
        assert result.tolist() == [[], []]

    def test_all_pii(self):
        data = Series(["ssn: 123-45-6789", "card: 4111"])
        result = guard_pii_detector_batch(data, lambda texts: [True] * len(texts))
        assert result.tolist() == [[PII_SPAN], [PII_SPAN]]

    def test_mixed(self):
        data = Series(["ssn: 123", "safe text", "card: 4111"])
        result = guard_pii_detector_batch(
            data, lambda texts: [True, False, True],
        )
        assert result.tolist() == [[PII_SPAN], [], [PII_SPAN]]

    def test_none_values_skipped(self):
        data = Series([None, "ssn: 123", None])
        result = guard_pii_detector_batch(data, lambda texts: [True])
        assert result.tolist() == [[], [PII_SPAN], []]

    def test_preserves_index(self):
        data = Series(["a", "b"], index=[10, 20])
        result = guard_pii_detector_batch(data, lambda texts: [False, False])
        assert list(result.index) == [10, 20]

    def test_batch_size_splits_correctly(self):
        data = Series(["a", "b", "c", "d", "e"])
        call_sizes = []

        def classifier(texts):
            call_sizes.append(len(texts))
            return [False] * len(texts)

        guard_pii_detector_batch(data, classifier, batch_size=2)
        assert call_sizes == [2, 2, 1]

    def test_kwargs_forwarded(self):
        data = Series(["text"])
        received = {}

        def classifier(texts, **kwargs):
            received.update(kwargs)
            return [False]

        guard_pii_detector_batch(data, classifier, model_name="test")
        assert received == {"model_name": "test"}

    def test_all_none_batch(self):
        data = Series([None, None, None])
        called = []

        def classifier(texts):
            called.append(True)
            return [False] * len(texts)

        result = guard_pii_detector_batch(data, classifier)
        assert result.tolist() == [[], [], []]
        assert called == []
