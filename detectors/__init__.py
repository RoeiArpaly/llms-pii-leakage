import gc

import torch


def unload_models():
    """Clear every detector model cache and free GPU/CPU memory."""
    from detectors.gliner import detector as gliner_det
    from detectors.guards import utils as guard_utils
    from detectors.guards import granite_guardian
    from detectors.guards import nemotron_guard
    from detectors.guards import qwen_guard
    from detectors.privacy_filter import detector as privacy_filter_det
    from detectors.slm import llama as llama_slm

    for cache in [
        guard_utils._model_cache,
        granite_guardian._model_cache,
        qwen_guard._model_cache,
        nemotron_guard._cache,
        gliner_det._model_cache,
        privacy_filter_det._model_cache,
        llama_slm._model_cache,
    ]:
        cache.clear()

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
