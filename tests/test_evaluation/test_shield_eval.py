"""Tests for post-hoc PII Shield evaluation."""
import pytest

from evaluation.shield_eval import compute_shield_predictions


@pytest.fixture
def sample_predictions(tmp_path):
    """Create a minimal predictions CSV."""
    import pandas as pd
    rows = [
        {"uid": 1, "model": "presidio-defend", "prediction": "[]", "perplexity": ""},
        {"uid": 1, "model": "gliner-defend", "prediction": '[{"type":"ssn"}]', "perplexity": ""},
        {"uid": 2, "model": "presidio-defend", "prediction": '[{"type":"email"}]', "perplexity": ""},
        {"uid": 2, "model": "gliner-defend", "prediction": "[]", "perplexity": ""},
        {"uid": 3, "model": "presidio-defend", "prediction": "[]", "perplexity": ""},
        {"uid": 3, "model": "gliner-defend", "prediction": "[]", "perplexity": ""},
    ]
    path = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_cascade_order(sample_predictions):
    """Earlier tiers take priority over later ones."""
    result = compute_shield_predictions(sample_predictions)
    # uid=1: presidio misses, gliner catches → detector=gliner-defend
    r1 = result[result["uid"] == 1].iloc[0]
    assert r1["detector"] == "gliner-defend"

    # uid=2: presidio catches → detector=presidio-defend
    r2 = result[result["uid"] == 2].iloc[0]
    assert r2["detector"] == "presidio-defend"


def test_no_detection(sample_predictions):
    """uid=3: both miss → not detected."""
    result = compute_shield_predictions(sample_predictions)
    r3 = result[result["uid"] == 3].iloc[0]
    assert r3["detector"] is None
    assert r3["prediction"] == [] or r3["prediction"] == "[]"


def test_all_uids_present(sample_predictions):
    """Shield must produce one row per uid."""
    result = compute_shield_predictions(sample_predictions)
    assert len(result) == 3
    assert set(result["uid"]) == {1, 2, 3}


def test_model_column(sample_predictions):
    """All rows have model='pii-shield'."""
    result = compute_shield_predictions(sample_predictions)
    assert (result["model"] == "pii-shield").all()
