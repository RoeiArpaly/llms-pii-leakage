import gc

import torch

from detectors.gliner import gliner_pii_detector
from detectors.guards import (
    QWEN_GUARD_GEN_MODELS,
    QWEN_GUARD_STREAM_MODELS,
    guard_pii_detector,
    guard_pii_detector_batch,
    llama_guard_classify_pii,
    llama_guard_classify_pii_batch,
    llama_guard_pii_detector,
    classify_pii_gen,
    classify_pii_gen_batch,
    detect_pii_stream,
    qwen_guard_gen_pii_detector,
    qwen_guard_stream_pii_detector,
)
from detectors.llm import llm_pii_detector
from detectors.presidio import (
    filter_results,
    fuzzy_pii_recognizer,
    get_fuzzy_recognizers,
    get_presidio_model,
    presidio_pii_analyzer,
)


def unload_models():
    """Clear every detector model cache and free GPU/CPU memory."""
    # Import caches lazily to avoid triggering heavy module loads
    # (torch, transformers, gliner) at package import time.
    from detectors.guards import utils as guard_utils
    from detectors.guards import qwen_guard
    from detectors.guards import nemotron_guard
    from detectors.gliner import detector as gliner_det

    for cache in [
        guard_utils._model_cache,
        qwen_guard._model_cache,
        nemotron_guard._cache,
        gliner_det._model_cache,
    ]:
        cache.clear()

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
