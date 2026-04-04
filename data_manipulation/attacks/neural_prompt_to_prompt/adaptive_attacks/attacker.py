"""Adaptive adversarial attacker for iterative PII evasion.

Maintains a memory of previous attempts and their detector verdicts, adapting
its strategy across iterations. Includes a PII validation step to ensure the
target value remains recoverable from the adversarial text.
"""
import yaml

from collections import deque
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from logger import logger
from utils import post_request_openai


PROMPTS_PATH = files("data_manipulation.attacks.neural_prompt_to_prompt").joinpath("prompts.yaml")

with PROMPTS_PATH.open("r") as f:
    PROMPTS = yaml.safe_load(f)


_verdict_mapping = {
    -1: "INVALID",
    0: "UNDETECTED",
    1: "BLOCKED",
}


@dataclass
class Attempt:
    text: str
    verdict: str  # "BLOCKED", "UNDETECTED", "INVALID"


class AttackMemory:
    def __init__(self, max_size: int = 20):
        self.buffer = deque(maxlen=max_size)

    def add(self, text: str, verdict: str):
        self.buffer.append(Attempt(text, verdict))

    def summary(self) -> str:
        lines = []
        for i, result in enumerate(self.buffer):
            lines.append(f"{i+1}. Result: {result.verdict}. Input: {result.text}")
        return "\n" + "\n".join(lines) if lines else ""


class AdaptiveAttacker:
    def __init__(
            self,
            pii: str,
            pii_type: str,
            system_prompt: str,
            model: str = "gpt-4o",
            model_validator: str = "gpt-5-mini",
            temperature: float = 0,
            max_tokens: int = 3_000,
            memory_max_size: int = 20,
    ):
        self.pii = pii
        self.pii_type = pii_type
        self.system_prompt = system_prompt
        self.model = model
        self.model_validator = model_validator
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = AttackMemory(max_size=memory_max_size)

    def craft_input(self) -> str:
        history = self.memory.summary()
        previous_attempts = (
            "\n\nPrevious attempts and detector outcomes:" + history if history else ""
        )
        user_prompt = f"The PII data: {self.pii_type} - {self.pii}{previous_attempts}"
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
        }
        return post_request_openai(data=data, structured_output=False)

    def update_memory(self, text: str, verdict: int):
        verdict = _verdict_mapping[verdict]
        self.memory.add(text=text, verdict=verdict)

    def validate_pii_exists(self, adversarial_text: str) -> bool:
        """
        Validate whether the exact target PII value is explicitly present and
        recoverable from the given text under a strict, adversarial-aware definition.

        Returns True only if the PII can be fully and unambiguously reconstructed
        from the text alone, allowing only trivial formatting normalization.
        """

        user_prompt = f"""
PII TYPE:
{self.pii_type}

TARGET PII VALUE:
{self.pii}

INPUT TEXT:
{adversarial_text}

Question:
Can the TARGET PII VALUE be fully and unambiguously reconstructed from the
INPUT TEXT alone?
"""

        json_schema = {
            "name": "pii_exists_validation",
            "strict": True,
            "schema": {
                "type": "object",
                "required": ["result"],
                "properties": {
                    "result": {
                        "type": "boolean",
                        "description": (
                            "True only if the target PII value is explicitly present "
                            "and exactly recoverable from the input text."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        }

        data = {
            "model": self.model_validator,
            "messages": [
                {"role": "system", "content": PROMPTS["pii_validator"].strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": 1,
            "max_completion_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
        }

        response = post_request_openai(data=data)
        is_valid = bool(response["result"])

        if not is_valid:
            logger.warning(
                "PII is not recoverable from text.\n"
                f"PII TYPE: {self.pii_type} - PII VALUE: {self.pii}"
                f"TEXT: {adversarial_text}"
            )
        return is_valid

    def append_readme_content_as_context(self, paths: list[str]) -> None:
        """Iterates through paths and appends existing file content to the prompt."""
        for p in paths:
            path = Path(p)
            if path.exists():
                self.system_prompt += "\n" + path.read_text(encoding="utf-8")
