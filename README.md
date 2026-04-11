# PII Under Attack: Adversarial Threats and Detector Resilience in the Era of LLMs

## Description
This repository contains the source code and datasets accompanying the paper, which studies adversarial threats against Personally Identifiable Information (PII) detection systems in the era of Large Language Models (LLMs).

The codebase implements an experimental framework for evaluating PII detectors under adversarial conditions. It includes tools for generating synthetic yet realistic user prompts containing high-risk PII types (e.g., IBANs, SSNs, and credit card numbers) using an LLM-guided pipeline, as well as implementations and wrappers for multiple classes of PII detectors.

Specifically, the repository supports the evaluation of rule-based, transformer-based, and LLM-based PII detection approaches, and reproduces the paper’s findings showing that detection recall can degrade severely, and in some cases collapse to zero, when subjected to adaptive adversarial attacks.

In addition, the repository includes the implementation of PII Shield, a modular defense framework that combines prevention and detection components to improve robustness against targeted adversarial inputs. The code enables systematic experimentation, benchmarking, and analysis of PII detection robustness across benign and adversarial settings.

## Project Structure
```
llms-pii-leakage
├── data_generation/           # LLM-based prompt generation + PII injection
│   ├── llm_input_generator.py
│   ├── pii_generator.py
│   ├── pii_validators.py
│   └── template_validators.py
│
├── data_manipulation/
│   ├── attacks/
│   │   ├── neural_prompt_to_prompt/  # LLM-based adaptive attacks
│   │   ├── red_teaming/
│   │   │   ├── content/              # Content-level attacks
│   │   │   └── pii/                  # PII-level fuzzing (homoglyph, chunking, etc.)
│   │   ├── template_based/           # Affix + prompt injection templates
│   │   └── injection.py              # Attack dispatch
│   └── defenses/
│       └── preprocess.py             # Defensive preprocessing pipeline
│
├── detectors/
│   ├── gliner/                # GLiNER transformer NER detectors
│   ├── guards/                # SLM guard models (Qwen, Llama Guard, Granite, WildGuard, Nemotron)
│   ├── llm/                   # LLM-based detectors (GPT-4o-mini via API)
│   ├── slm/                   # Instruction-tuned SLM detectors (Llama 3.2 1B)
│   ├── presidio/              # Rule-based Presidio + fuzzy matching
│   └── validators.py          # Post-detection PII span validators
│
├── evaluation/
│   ├── report/                # HTML report generator + config + styling
│   ├── visualizations/        # Heatmaps, radar, line charts, perplexity
│   ├── partial_matching.py    # Span-level fuzzy matching
│   ├── scoring.py             # Evaluation metrics
│   ├── shield_eval.py         # Post-hoc PII Shield cascade evaluation
│   └── spans.py               # Span scoring (precision/recall/F1)
│
├── pipelines/
│   ├── checkpoint.py          # Atomic JSON checkpoint manager
│   ├── cli.py                 # CLI display helpers + spinner
│   ├── detection.py           # PII detection pipeline + batching
│   ├── generation.py          # Dataset generation stages
│   └── runner.py              # Stage orchestration + archiving
│
├── utils/
│   ├── api.py                 # OpenAI API client + retry logic
│   ├── data.py                # CSV/JSON serialization helpers
│   ├── parallel.py            # Parallel execution for I/O-bound calls
│   ├── perplexity.py          # Perplexity calculation
│   └── prompts.py             # YAML prompt loading
│
├── tests/
├── config.py                  # Central configuration
├── constants.py               # PII entities, attack techniques, dataset schema
├── logger.py                  # Logging setup
├── main.py                    # CLI entrypoint
└── pii_shield.py              # Cascading PII defense framework
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone the project
    ```bash
    git clone https://github.com/{author}/llms-pii-leakage.git
    cd llms-pii-leakage
    ```

3. Install dependencies
    ```bash
    uv sync
    ```

## Running

```bash
uv run python main.py                          # run full pipeline
uv run python main.py run --skip-gen            # skip baseline generation, reuse dataset
uv run python main.py run --sample 1000         # cap negatives, keep all clean positives
uv run python main.py run --models presidio gliner  # run subset of models
uv run python main.py status                    # show checkpoint state
uv run python main.py reset                     # clear all checkpoints
```
