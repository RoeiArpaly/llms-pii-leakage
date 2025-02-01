# LLMs PII Leakage

## Description
This project contains adversarial attacks against LLM PII detection Guardrails.

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
