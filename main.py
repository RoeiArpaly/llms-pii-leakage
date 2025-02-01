import random
import pandas as pd

from config import Config
from logger import logger
from pipelines import (
    fuzzy_pii_adv_content_generation,
    fuzzy_pii_generation,
    llm_detector,
    llm_input_generation,
)

results = []
for i in range(Config.NUMBER_OF_SAMPLES):
    logger.info(f"Generating LLM input sample {i + 1}/{Config.NUMBER_OF_SAMPLES}")
    contains_pii = random.random() < Config.PII_PROBABILITY
    result = llm_input_generation(contains_pii=contains_pii)
    results.append(result)

df = pd.DataFrame(results)
df.index.name = "input_id"

df.to_csv(path_or_buf="datasets/llm_input_generation_results_01.csv", index=True)
logger.info("LLM input generation completed successfully")

data = llm_detector(data=df)
cols = [
    "llm_input",
    "pii_spans_generator",
    "pii_spans_llm_detector",
    "pii_amount_llm_detector",
    "spans_score",
]
data[cols].to_csv(path_or_buf="datasets/llm_detection_results_02.csv", index=True)
logger.info("LLM detection completed successfully")

data = fuzzy_pii_generation(data=data)
cols = [
    "llm_input",
    "llm_input_template",
    "pii_spans_generator",
    "fuzzy_techniques",
    "fuzzy_llm_input",
    "fuzzy_analyzer",
    "fuzzy_llm_restored",
    "fuzzy_llm_restored_analyzer",
    "fuzzy_pii_amount_analyzer",
    "fuzzy_pii_amount_llm_restored_analyzer",
    "spans_score",
]
data[cols].to_csv(path_or_buf="datasets/fuzzy_pii_generation_results_03.csv", index=True)
logger.info("Fuzzy PII generation completed successfully")

data = fuzzy_pii_adv_content_generation(
    data=data,
)
cols = ["llm_input", "llm_input_template", "pii_spans_generator"]
data[cols].to_csv(
    path_or_buf="datasets/fuzzy_pii_adv_content_generation_results_04.csv", index=False
)
logger.info("Fuzzy PII with adversarial content generation completed successfully")


if __name__ == "__main__":
    logger.info("SUCCESS")
