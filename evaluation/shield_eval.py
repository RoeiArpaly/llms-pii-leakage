"""Post-hoc PII Shield evaluation from existing per-model predictions.

Simulates the cascading defense by checking, for each sample, whether
ANY model in the hardcoded cascade detected PII. Reports which tier
caught each sample.

Cascade (hardcoded, not derived from Config.MODELS):
    1. Presidio (defend)
    2. Presidio-Fuzzy (defend)
    3. GLiNER (defend)
    4. Qwen Guard 0.6B (defend)

This avoids re-running models — it uses the predictions already in
predictions.csv. Tiers that don't have predictions are skipped.
"""
from pandas import DataFrame, read_csv

from detectors.validators import validate_pii_spans
from utils import infer_json

# Tiers whose predictions pass through the validation layer.
_VALIDATED_TIERS = {"gliner-defend", "gliner-nvidia-defend"}

# Hardcoded cascade order. The shield tries each in order and returns
# on the first detection. Only -defend variants are used (with
# defensive preprocessing applied).
SHIELD_CASCADE = [
    "presidio-defend",
    "presidio-fuzzy-defend",
    "gliner-defend",
    "qwen-guard-0.6b-defend",
]


def compute_shield_predictions(
    predictions_path: str = "datasets/predictions.csv",
) -> DataFrame:
    """Compute PII Shield predictions from existing per-model results.

    For each uid, the shield cascade checks models in the hardcoded
    tier order. The first model that detects PII determines the
    shield result. Tiers without predictions are skipped.

    Returns
    -------
    DataFrame
        Columns: uid, model ("pii-shield"), prediction, perplexity,
        detector (which tier caught it).
    """
    predictions = read_csv(predictions_path).apply(infer_json)
    available = set(predictions["model"].unique())

    cascade = [m for m in SHIELD_CASCADE if m in available]
    if not cascade:
        return DataFrame()

    uids = predictions["uid"].unique()
    detections = {}
    for m in cascade:
        m_preds = predictions[predictions["model"] == m].set_index("uid")
        if m in _VALIDATED_TIERS:
            detections[m] = m_preds["prediction"].apply(
                lambda x: (
                    len(validate_pii_spans(x)) > 0
                    if isinstance(x, list) else False
                ),
            )
        else:
            detections[m] = m_preds["prediction"].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False,
            )

    rows = []
    for uid in uids:
        detected = False
        detector = None
        for m in cascade:
            if uid in detections[m].index and detections[m].loc[uid]:
                detected = True
                detector = m
                break

        if detected:
            m_row = predictions[
                (predictions["uid"] == uid)
                & (predictions["model"] == detector)
            ].iloc[0]
            spans = m_row["prediction"]
            if detector in _VALIDATED_TIERS and isinstance(spans, list):
                spans = validate_pii_spans(spans)
            perp = m_row.get("perplexity")
        else:
            spans = []
            perp = None

        rows.append({
            "uid": uid,
            "model": "pii-shield",
            "prediction": spans,
            "perplexity": perp,
            "detector": detector,
        })

    return DataFrame(rows)
