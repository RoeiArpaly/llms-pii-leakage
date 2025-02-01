# 1 - Generate LLM input templates with OpenAI
# 2 - Validate the generated templates
# 3 - Inject PII into the templates with Presidio Data Generator
# 4 - Validate the generated PII with Presidio
# PII IN (CREDIT_CARD_NUMBER, IBAN_CODE, SSN, PHONE_NUMBER)

import random
import pandas as pd

from constants import (
    PII_PROBABILITY,
    NUMBER_OF_SAMPLES,
)
from logger import logger
from pipelines import (
    fuzzy_pii_adv_content_generation,
    fuzzy_pii_generation,
    llm_detector,
    llm_input_generation,
)

results = []
for i in range(NUMBER_OF_SAMPLES):
    logger.info(f"Generating LLM input sample {i + 1}/{NUMBER_OF_SAMPLES}")
    contains_pii = random.random() < PII_PROBABILITY
    result = llm_input_generation(contains_pii=contains_pii)
    results.append(result)

df = pd.DataFrame(results)
df.index.name = "input_id"

df.to_csv(path_or_buf="datasets/llm_input_generation_results_01.csv", index=True)
logger.info("LLM input generation completed successfully")

data = llm_detector(data=df.copy())
data.to_csv(path_or_buf="datasets/llm_detection_results_02.csv", index=True)
logger.info("LLM detection completed successfully")

data = fuzzy_pii_generation(data=df.copy())
data.to_csv(path_or_buf="datasets/fuzzy_pii_generation_results_03.csv", index=True)
logger.info("Fuzzy PII generation completed successfully")

data = fuzzy_pii_adv_content_generation(
    data=df[["llm_input", "llm_input_template", "pii_spans_generator"]].copy(),
)
data.to_csv(
    path_or_buf="datasets/fuzzy_pii_adv_content_generation_results_04.csv", index=False
)
logger.info("Fuzzy PII with adversarial content generation completed successfully")


if __name__ == "__main__":
    logger.info("SUCCESS")
