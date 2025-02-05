# LLMs PII Leakage

## Description
This project contains adversarial attacks against LLM PII detection Guardrails.

## Project Structure
```
llms-pii-leakage
├── data_generation
│   ├── data_validators.py
│   ├── llm_input_generator.py
│   └── pii_generator.py
│
├── data_manipulation
│   ├── content
│   │   ├── affix.py
│   │   ├── emojify.py
│   │   └── utils.py
│   ├── pii
│   │   ├── emojify.py
│   │   ├── number_to_word.py
│   │   └── separators.py
│   ├── llm.py
│   ├── rule_based.py
│   └── constants.py
│
├── datasets
├── detectors
│   ├── llm.py
│   └── presidio.py
├── evaluation
│   ├── reports.py
│   ├── spans.py
│   └── visualizations.py
├── tests
├── config.py
├── constants.py
├── logger.py
├── main.py
├── pipelines.py
└── utils.py
```

## Setup
1. CD to the project directory
    ```
    cd <project_directory>
    ```

2. Clone the project
    ```bash
    git clone https://github.com/RoeiArpaly/llms-pii-leakage.git
    ```

3. Install virtual environment
    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment
    ```bash
    source venv/bin/activate
    ```

5. Install requirements
    ```bash
    pip install -r requirements-dev.txt
    ```

## Steps
1. Generate LLM input templates with OpenAI
2. Validate the generated templates
3. Inject PII into the templates with Presidio Data Generator
4. Validate the generated PII with Presidio
