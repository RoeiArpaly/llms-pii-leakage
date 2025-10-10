from gliner import GLiNER

from constants import (
    GLINER_INVALID_VALUES,
    PII_ENTITIES,
)


_model = None


def get_gliner_model():
    global _model
    if _model is None:
        _model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1", max_length=4096)
    return _model


def gliner_pii_detector(text: str, threshold: float = 0.5):

    model = get_gliner_model()
    pii_labels = list(PII_ENTITIES.values())

    spans = model.predict_entities(text=text, labels=pii_labels, threshold=threshold)
    results = []
    for span in spans:  # Rename the label key to type
        span["value"] = span.pop("text")
        span["type"] = span.pop("label")
        if all([val not in span["value"].lower() for val in GLINER_INVALID_VALUES]):
            results.append(span)
    return results
