# Datasets

All data lives in three consolidated CSV files:

| File | Description |
|------|-------------|
| [`dataset.csv`](dataset.csv) | All samples: benign, hard negatives, and adversarial variants |
| [`predictions.csv`](predictions.csv) | Model predictions in long format (one row per uid × model) |
| [`evaluations.csv`](evaluations.csv) | Span-level evaluation scores (one row per uid × model) |

The interactive HTML report is generated on-the-fly: `python -m evaluation.report`

---

## Dataset (`dataset.csv`)

A consolidated benchmark dataset for evaluating PII detection under adversarial attacks.
Generated using GPT-4o-mini for text and Presidio for PII injection.

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `uid` | int | Unique row identifier |
| `input_id` | int or null | Reference to the original benign sample (null for benign/hard_negative rows) |
| `category` | str | Sample category (see below) |
| `pii_techniques` | JSON list or null | PII-level attack techniques applied |
| `content_techniques` | JSON list or null | Content-level attack techniques applied |
| `llm_input` | str | The input text, potentially containing PII |
| `pii_spans` | JSON list | Ground truth PII spans: `[{"value", "start", "end", "type"}]` |

### Categories

| Category | Description |
|----------|-------------|
| `benign` | Original text with real PII injected |
| `hard_negative` | Original text without PII (likely to cause false positives) |
| `pii_level` | PII obfuscated using fuzzy techniques (homoglyph, emojify, chunking, etc.) |
| `pii_and_content_level` | PII obfuscated + adversarial content added (supportive context, prompt injection, affixes) |

### PII Types

`credit_card_number`, `iban`, `ssn`, `phone_number`, `email`

---

## Predictions (`predictions.csv`)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `uid` | int | References `dataset.csv` uid |
| `model` | str | Detector model name (e.g., `presidio`, `gliner-defend`, `gpt-4o-mini`) |
| `prediction` | JSON list | Detected PII spans |
| `perplexity` | float or null | Perplexity score (LLM-based detectors only) |

---

## Evaluations (`evaluations.csv`)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `uid` | int | References `dataset.csv` uid |
| `model` | str | Detector model name |
| `prediction` | JSON list | Detected PII spans |
| `spans_score` | JSON dict | `{exact_match, true_positive, false_positive, false_negative, precision, recall, f1}` |

Aggregated views (by category, by technique, etc.) are computed on-the-fly by the report generator.
