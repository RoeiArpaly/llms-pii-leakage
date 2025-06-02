from pandas import (
    read_csv,
    Series,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from detectors.presidio_detector import presidio_pii_analyzer
from pii_guard import guard
from utils import (
    cast_to_json,
    infer_json,
    parallel_apply,
)


# data = read_csv("datasets/fuzzy_adv_dataset.csv").apply(infer_json)
data = read_csv("datasets/baseline_dataset.csv").apply(infer_json)

result = Series(parallel_apply(func=guard, series=data["llm_input"], perplexity_threshold=1.05))
# result = data["llm_input"].apply(guard, perplexity_threshold=1.05)
result_baseline = Series(parallel_apply(func=presidio_pii_analyzer, series=data["llm_input"]))
# result_baseline = data["llm_input"].apply(presidio_pii_analyzer)

y_true = data["pii_spans"].apply(lambda x: len(x) > 0)
y_pred = result.apply(lambda x: x["detected"])
y_pred_baseline = result_baseline.apply(lambda x: len(x) > 0)

print(list(zip(y_true, y_pred))[:5])

print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"F1 Score: {f1_score(y_true, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("Classification Report:")
print(classification_report(y_true, y_pred, zero_division=0))

print(f"Baseline Accuracy: {accuracy_score(y_true, y_pred_baseline)}")
print(f"Baseline F1 Score: {f1_score(y_true, y_pred_baseline)}")
print("Baseline Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_baseline))
print("Baseline Classification Report:")
print(classification_report(y_true, y_pred_baseline, zero_division=0))

data["pii_guard_result"] = result
data["pii_baseline_result"] = result_baseline
data["pii_guard_detected"] = y_pred
data["pii_baseline_detected"] = y_pred_baseline
data.apply(cast_to_json).to_csv("datasets/pii_guard_benchmark_baseline.csv", index=False)
