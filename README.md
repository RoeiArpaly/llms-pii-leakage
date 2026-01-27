# PII Under Attack: Adversarial Threats and Detector Resilience in the Era of LLMs

## Description
This repository contains the source code and datasets accompanying the paper, which studies adversarial threats against Personally Identifiable Information (PII) detection systems in the era of Large Language Models (LLMs).

The codebase implements an experimental framework for evaluating PII detectors under adversarial conditions. It includes tools for generating synthetic yet realistic user prompts containing high-risk PII types (e.g., IBANs, SSNs, and credit card numbers) using an LLM-guided pipeline, as well as implementations and wrappers for multiple classes of PII detectors.

Specifically, the repository supports the evaluation of rule-based, transformer-based, and LLM-based PII detection approaches, and reproduces the paper’s findings showing that detection recall can degrade severely, and in some cases collapse to zero, when subjected to adaptive adversarial attacks.

In addition, the repository includes the implementation of PII Shield, a modular defense framework that combines prevention and detection components to improve robustness against targeted adversarial inputs. The code enables systematic experimentation, benchmarking, and analysis of PII detection robustness across benign and adversarial settings.

## Project Structure
```
llms-pii-leakage
├── data_generation
│   ├── llm_input_generator.py
│   ├── pii_generator.py
│   ├── pii_validators.py
│   ├── prompts.yaml
│   └── template_validators.py
│
├── data_manipulation
│   ├── attacks
│   │   ├── neural_prompt_to_prompt
│   │   │   ├── adaptive_attacks
│   │   │   │   ├── attacker.py
│   │   │   │   ├── constants.py
│   │   │   │   ├── loop.py
│   │   │   │   └── run.py
│   │   │   ├── llm.py
│   │   │   └── prompts.yaml
│   │   │
│   │   ├── red_teaming
│   │   │   ├── content
│   │   │   │   ├── supportive_context.py
│   │   │   │   └── utils.py
│   │   │   └── pii
│   │   │       ├── char_to_word.py
│   │   │       ├── chunking.py
│   │   │       ├── emojify.py
│   │   │       ├── homoglyph.py
│   │   │       ├── separators.py
│   │   │       └── utils.py
│   │   │
│   │   ├── template_based
│   │   │   ├── affix.py
│   │   │   └── prompt_injection.py
│   │   └── injection.py
│   │
│   ├── defenses
│   │   └── preprocess.py
│   └── constants.py
│
├── datasets
│
├── detectors
│   ├── fuzzy_match.py
│   ├── gliner_detector.py
│   ├── llm_detector.py
│   ├── presidio_detector.py
│   └── prompts.yaml
│
├── evaluation
│   ├── constants.py
│   ├── partial_matching.py
│   ├── prompts.yaml
│   ├── reports.py
│   ├── spans.py
│   └── visualizations.py
│
├── tests
├── config.py
├── constants.py
├── logger.py
├── main.py
├── pii_shield.py
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
    git clone https://github.com/{author}/llms-pii-leakage.git
    ```

3. Install virtual environment
    ```bash
    python3.9 -m venv venv
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
