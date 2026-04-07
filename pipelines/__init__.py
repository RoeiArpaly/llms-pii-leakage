"""Pipeline package — re-exports for backward compatibility."""
from pipelines.detection import (  # noqa: F401
    PREDICTIONS_PATH,
    _SLM_MODELS,
    pii_detection_pipeline,
    process_predictions,
)
from pipelines.generation import (  # noqa: F401
    DATASET_PATH,
    generate_baseline_dataset,
    generate_fuzzy_adv_dataset,
    generate_fuzzy_dataset,
)
