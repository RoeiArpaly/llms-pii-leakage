# AdvPIIBench: An Adversarial Benchmark for PII Detection in the Era of LLMs

## Description
AdvPIIBench is a benchmark for evaluating Personally Identifiable Information (PII) detectors under adversarial conditions. It accompanies the paper studying how adaptive adversarial techniques degrade the recall of PII detection systems in the era of Large Language Models (LLMs).

The framework generates synthetic yet realistic user prompts containing high-risk PII types (IBANs, SSNs, credit card numbers, phone numbers, and email addresses) via an LLM-guided pipeline, injects them with Microsoft Presidio, and then applies a taxonomy of adversarial attacks. It ships wrappers for a broad set of PII detectors spanning rule-based, transformer-NER, small-language-model (SLM) safety guards, and LLM-based approaches, and reproduces the paper's finding that detection recall can degrade severely — in some cases collapsing to zero — under adaptive attacks.

## Detectors

AdvPIIBench evaluates the following base detectors (configured in `Config.MODELS`):

- **Rule-based** — Presidio, Presidio-Fuzzy (fuzzy-matching recognizers)
- **Transformer NER** — GLiNER, GLiNER-NV, OpenAI Privacy Filter
- **SLM safety guards** — Qwen Guard (0.6B / 4B), Llama Guard 3 (1B / 8B), Nemotron Content Safety 4B, WildGuard 7B, Granite Guardian 8B
- **Instruction-tuned SLM** — Llama 3.2 1B
- **LLM** — GPT-4o-mini (via API)

## Attack Taxonomy

Attacks are applied in two layers:

- **PII-level fuzzing** (`data_manipulation/attacks/red_teaming/pii/`) — homoglyph substitution, chunking, emojify, char-to-word, invisible characters, and separators, applied directly to the PII value.
- **Content-level attacks** (`data_manipulation/attacks/red_teaming/content/` + `template_based/`) — supportive context, prompt injection variants, and adversarial affixes wrapped around the PII-carrying text.
- **Neural prompt-to-prompt** (`data_manipulation/attacks/neural_prompt_to_prompt/`) — an LLM-driven adaptive attacker that iteratively rewrites inputs to evade a target detector.

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
│   └── constants.py                  # Attack lookup tables (homoglyph, emoji, etc.)
│
├── detectors/
│   ├── gliner/                # GLiNER transformer NER detectors
│   ├── guards/                # SLM guard models (Qwen, Llama Guard, Granite, WildGuard, Nemotron)
│   ├── llm/                   # LLM-based detectors (GPT-4o-mini via API)
│   ├── slm/                   # Instruction-tuned SLM detectors (Llama 3.2 1B)
│   ├── presidio/              # Rule-based Presidio + fuzzy matching
│   ├── privacy_filter/        # OpenAI privacy-filter (HF token-classification model)
│   ├── hard_negatives.py      # Lookalike-format suppression (UUID, MAC, hashes, etc.)
│   └── validators.py          # Post-detection PII span validators
│
├── evaluation/
│   ├── visualizations/        # Paper figure generator (orthogonality heatmap)
│   ├── partial_matching.py    # Span-level fuzzy matching
│   ├── scoring.py             # Evaluation metrics
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
└── main.py                    # CLI entrypoint
```

## Pipeline

The pipeline (`main.py` → `pipelines/`) runs sequentially:

1. **Baseline dataset generation** — LLM generates realistic user prompts; Presidio injects synthetic PII.
2. **Fuzzy dataset generation** — applies PII-level adversarial transformations.
3. **Fuzzy + adversarial dataset generation** — adds content-level attacks and (optionally) neural prompt-to-prompt attacks.
4. **PII detection** — runs every configured detector over the dataset in batches, writing `datasets/predictions.csv`.
5. **Evaluation** — computes document-level and span-level metrics into `datasets/evaluations.csv`.

Detection is evaluated at the **document level**: a sample counts as blocked whenever a
detector emits any non-empty prediction. That is the unit the paper reports, and the only
one that compares span-emitting detectors against SLM guards (which return just safe/unsafe)
on equal terms.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Clone the project
    ```bash
    git clone https://github.com/RoeiArpaly/llms-pii-leakage.git
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

## Reproducing the paper figure

The orthogonality heatmap (each detector's five worst attack configurations and their
document-level block rate) is regenerated from a completed pipeline run:

```bash
uv run python main.py                                              # produces datasets/
uv run python -m evaluation.visualizations.orthogonality_heatmap   # then the figure
```

It reads `datasets/dataset.csv` and `datasets/evaluations.csv` and writes
`figures/orthogonality_heatmap.{pdf,png}`. Those inputs are **not** distributed — the
published dataset is the corpus, not the per-detector predictions — so run the pipeline
first to generate them. The script asserts its worst-case numbers against the published
table, so it fails loudly if the results drift.

## Testing

```bash
# Fast suite — everything except the model-downloading tests
uv run pytest --ignore=tests/test_detectors/test_integration.py \
              --ignore=tests/test_detectors/test_attack_effectiveness

# Full suite — downloads and runs every guard/SLM model (slow, GPU/RAM heavy)
uv run pytest

uv run flake8        # lint
```
