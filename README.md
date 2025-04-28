# LLMs PII Leakage

## Description
This project contains adversarial attacks against LLM PII detection Guardrails.

## Project Structure
```
llms-pii-leakage
├── data_generation
│   ├── data_validators.py
│   ├── llm_input_generator.py
│   ├── pii_generator.py
│   └── prompts.py
│
├── data_manipulation
│   ├── attacks
│   │   ├── neural_prompt_to_prompt
│   │   │   ├── llm.py
│   │   │   └── prompts.yaml
│   │   ├── red_teaming
│   │   │   ├── content
│   │   │   │   ├── supportive_context.py
│   │   │   │   └── utils.py
│   │   │   └── pii
│   │   │       ├── char_to_word.py
│   │   │       ├── chunking.py
│   │   │       ├── emojify.py
│   │   │       ├── homoglyph.py
│   │   │       ├── reverse.py
│   │   │       └── separators.py
│   │   ├── template_based
│   │   │   ├── affix.py
│   │   │   └── prompt_injection.py
│   │   └── injection.py
│   ├── defenses
│   │   └── preprocess.py
│   └── constants.py
│
├── datasets
├── detectors
│   ├── fuzzy_match.py
│   ├── gliner_detector.py
│   ├── llm_detector.py
│   ├── presidio_detector.py
│   └── prompts.yaml
├── evaluation
│   ├── constants.py
│   ├── reports.py
│   ├── spans.py
│   └── visualizations.py
│
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
