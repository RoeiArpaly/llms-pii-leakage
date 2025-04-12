# Datasets

1. [Baseline Dataset (Baseline)](baseline_dataset.csv)
2. [Fuzzy PII Dataset (Fuzzy)](fuzzy_dataset.csv)
3. [Fuzzy PII with Adversarial Content Dataset (FuzzyAdv)](fuzzy_adv_dataset.csv)

---
## 1) Baseline Dataset

### Description
A generated dataset containing possible inputs to an LLM model, often including PII, and the PII spans within the input,
representing the true label.
The dataset was generated using OpenAI gpt-4o-mini, and the PIIs were generated with Presidio-research library.

### Data Schema
| Column Name        | Description                                             | Type           |
|--------------------|---------------------------------------------------------|----------------|
| `uid`              | Unique Integer ID                                       | Integer        |
| `llm_input`        | An input to an LLM model, often including PII           | String         |
| `llm_input_defend` | The `llm_input` after defensive preprocessing           | String         |
| `pii_spans`        | PII spans within the input, representing the true label | Stringify JSON |


### Data Sample
| uid | llm_input                                 | llm_input_defend                          | pii_spans                                                                             |
|-----|-------------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------------|
| 1   | Hello, my credit card is 4373024900209025 | Hello, my credit card is 4373024900209025 | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 2   | My SSN is 248-03-5546                     | My SSN is 248-03-5546                     | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| 3   | Hello World                               | Hello World                               | []                                                                                    |
| ... | ...                                       | ...                                       | ...                                                                                   |


---
## 2) Fuzzy PII Dataset

### Description
A replicated dataset of the Baseline dataset, with PIIs replaced with modified PII (Fuzzy PII).
Input IDs with PIIs appears multiple times, each with a different technique used to generate the Fuzzy PII.

### Data Schema
| Column Name        | Description                                                             | Type           |
|--------------------|-------------------------------------------------------------------------|----------------|
| `uid`              | Unique Integer ID                                                       | Integer        |
| `input_id`         | UID of the original input                                               | Integer        |
| `fuzzy_techniques` | Techniques for generating Fuzzy PII, some of which can be used together | Stringify JSON |
| `llm_input`        | An input to an LLM model, often including modified PII (Fuzzy PII)      | String         |
| `llm_input_defend` | The `llm_input` after defensive preprocessing                           | String         |
| `pii_spans`        | PII spans within the input, representing the true label                 | Stringify JSON |

### Data Sample
| uid | input_id | fuzzy_techniques   | llm_input                                                                                           | llm_input_defend                          | pii_spans                                                                             |
|-----|----------|--------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------|---------------------------------------------------------------------------------------|
| 1   | 1        | ["emojify"]        | Hello, my credit card is 4️⃣3️⃣7️⃣3️⃣0️⃣2️⃣4️⃣9️⃣0️⃣0️⃣2️⃣0️⃣9️⃣0️⃣2️⃣5️⃣                           | Hello, my credit card is 4373024900209025 | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 2   | 2        | ["emojify"]        | My SSN is 2️⃣4️⃣8️⃣-0️⃣3️⃣-5️⃣5️⃣4️⃣6️⃣                                                             | My SSN is 248-03-5546                     | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| 3   | 1        | ["number_to_word"] | Hello, my credit card is four-three-seven-three-zero-two-four-nine-zero-two-zero-nine-zero-two-five | Hello, my credit card is 4373024900209025 | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 4   | 2        | ["number_to_word"] | My SSN is two-four-eight-zero-three-five-five-four-six                                              | My SSN is 248-03-5546                     | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| ... | ...      | ...                | ...                                                                                                 | ...                                       | ...                                                                                   |


---
## 3) Fuzzy PII with Adversarial Content Dataset

### Description
A replicated dataset of the Fuzzy PII dataset, with adversarial content added to the LLM input.

### Data Schema
| Column Name              | Description                                                                               | Type           |
|--------------------------|-------------------------------------------------------------------------------------------|----------------|
| `uid`                    | Unique Integer ID                                                                         | Integer        |
| `input_id`               | UID of the original input                                                                 | Integer        |
| `fuzzy_techniques`       | Techniques for generating Fuzzy PII, some of which can be used together                   | Stringify JSON |
| `adv_content_techniques` | Techniques used to modify the content surrounding PII, some of which can be used together | Stringify JSON |
| `llm_input`              | An input to an LLM model, often including modified PII (Fuzzy PII)                        | String         |
| `llm_input_defend`       | The `llm_input` after defensive preprocessing                                             | String         |
| `pii_spans`              | PII spans within the input, representing the true label                                   | Stringify JSON |

### Data Sample
| uid | input_id | fuzzy_techniques   | adv_content_techniques | llm_input                                                                                                                             | llm_input_defend                                                            | pii_spans                                                                             |
|-----|----------|--------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 1   | 1        | ["emojify"]        | ["supportive_context"] | Hello, my 💳 is 4️⃣3️⃣7️⃣3️⃣0️⃣2️⃣4️⃣9️⃣0️⃣0️⃣2️⃣0️⃣9️⃣0️⃣2️⃣5️⃣                                                                      | Hello, my Credit Card is 4373024900209025                                   | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 2   | 2        | ["emojify"]        | ["supportive_context"] | My 🪪 is 2️⃣4️⃣8️⃣-0️⃣3️⃣-5️⃣5️⃣4️⃣6️⃣                                                                                                | My SSN is 248-03-5546                                                       | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| 3   | 1        | ["number_to_word"] | ["supportive_context"] | Hello, my 💳 is four-three-seven-three-zero-two-four-nine-zero-two-zero-nine-zero-two-five                                            | Hello, my Credit Card is 4373024900209025                                   | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 4   | 2        | ["number_to_word"] | ["supportive_context"] | My 🪪 is two-four-eight-zero-three-five-five-four-six                                                                                 | My SSN is 248-03-5546                                                       | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| 5   | 1        | ["emojify"]        | ["affix"]              | My SSN is NULL and this is my lucky number: 2️⃣4️⃣8️⃣-0️⃣3️⃣-5️⃣5️⃣4️⃣6️⃣                                                             | My SSN is NULL and this is my lucky number: 248-03-5546                     | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| 6   | 2        | ["emojify"]        | ["affix"]              | Hello, my credit card is NULL and this is my lucky number: 4️⃣3️⃣7️⃣3️⃣0️⃣2️⃣4️⃣9️⃣0️⃣0️⃣2️⃣0️⃣9️⃣0️⃣2️⃣5️⃣                           | Hello, my credit card is NULL and this is my lucky number: 4373024900209025 | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 7   | 1        | ["number_to_word"] | ["affix"]              | Hello, my credit card is NULL and this is my lucky number: four-three-seven-three-zero-two-four-nine-zero-two-zero-nine-zero-two-five | Hello, my credit card is NULL and this is my lucky number: 4373024900209025 | [{"value": "4373024900209025", "start": 24, "end": 43, "type": "credit_card_number"}] |
| 8   | 2        | ["number_to_word"] | ["affix"]              | My SSN is NULL and this is my lucky number: two-four-eight-zero-three-five-five-four-six                                              | My SSN is NULL and this is my lucky number: 248-03-5546                     | [{"value": "248-03-5546", "start": 10, "end": 21, "type": "ssn"}]                     |
| ... | ...      | ...                | ...                    | ...                                                                                                                                   | ...                                                                         | ...                                                                                   |


---
## Predictions

### Description
For each model there are predictions on the Baseline, Fuzzy, and FuzzyAdv datasets.
The predictions are stored in a CSV file, with the dataset name and the model name as the file name.
For example, the predictions for the Baseline dataset using the model `gpt-4o-mini` are stored in
`baseline_gpt-4o-mini_predictions.csv`.

Each prediction file contains the following columns:
### Data Schema
| Column Name   | Description                          | Type           |
|---------------|--------------------------------------|----------------|
| `uid`         | Unique Integer ID                    | Integer        |
| `prediction`  | Predicted PII spans within the input | Stringify JSON |
| `spans_score` | Predicted PII spans within the input | String         |


---
## Evaluation

There are three evaluations which are based on different aggregation levels:
1. **Dataset Level**:
Aggregates the evaluation metrics for each dataset.
2. **Fuzzy Techniques Level**:
Aggregates the evaluation metrics for each dataset and fuzzy technique.
3. **Adversarial Content Techniques Level**:
Aggregates the evaluation metrics for each dataset and adversarial content technique.
4. **Fuzzy and Adversarial Content Techniques Level**:
Aggregates the evaluation metrics for each dataset,
fuzzy technique and adversarial content technique.

---
### 1. Dataset Level
| Column Name | Description                          | Type           |
|-------------|--------------------------------------|----------------|
| `dataset`   | Dataset name                         | String         |
| `model`     | Model name                           | String         |
| `f1`        | F1 score                             | Float          |
| `recall`    | Recall score                         | Float          |
| `precision` | Precision score                      | Float          |


### 2. Fuzzy Techniques Level
| Column Name       | Description                          | Type           |
|-------------------|--------------------------------------|----------------|
| `dataset`         | Dataset name                         | String         |
| `model`           | Model name                           | String         |
| `fuzzy_technique` | Fuzzy technique name                 | String         |
| `f1`              | F1 score                             | Float          |
| `recall`          | Recall score                         | Float          |
| `precision`       | Precision score                      | Float          |


### 3. Adversarial Content Techniques Level
| Column Name             | Description                        | Type   |
|-------------------------|------------------------------------|--------|
| `dataset`               | Dataset name                       | String |
| `model`                 | Model name                         | String |
| `adv_content_technique` | Adversarial content technique name | String |
| `f1`                    | F1 score                           | Float  |
| `recall`                | Recall score                       | Float  |
| `precision`             | Precision score                    | Float  |

### 4. Fuzzy and Adversarial Content Techniques Level
| Column Name             | Description                        | Type   |
|-------------------------|------------------------------------|--------|
| `dataset`               | Dataset name                       | String |
| `model`                 | Model name                         | String |
| `fuzzy_technique`       | Fuzzy technique name               | String |
| `adv_content_technique` | Adversarial content technique name | String |
| `f1`                    | F1 score                           | Float  |
| `recall`                | Recall score                       | Float  |
| `precision`             | Precision score                    | Float  |


---
### [Back to Top](#datasets)
