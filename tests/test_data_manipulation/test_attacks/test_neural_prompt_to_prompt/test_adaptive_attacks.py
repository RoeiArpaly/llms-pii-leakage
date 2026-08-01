from pathlib import Path

from data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks.attacker import (
    AdaptiveAttacker,
    AttackMemory,
)
from data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks.const import (
    ADVERSARIAL_ATTACK_README_PATHS,
)
from data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks.loop import (
    _init_attacker,
    run_attack,
)


def test_attack_memory():
    memory = AttackMemory(max_size=3)
    memory.add("text1", "BLOCKED")
    memory.add("text2", "UNDETECTED")
    summary = memory.summary()
    assert "BLOCKED" in summary
    assert "UNDETECTED" in summary
    assert "text1" in summary


def test_attack_memory_max_size():
    memory = AttackMemory(max_size=2)
    memory.add("a", "BLOCKED")
    memory.add("b", "BLOCKED")
    memory.add("c", "BLOCKED")
    assert len(memory.buffer) == 2
    assert memory.buffer[0].text == "b"
    assert memory.buffer[1].text == "c"


def test_adaptive_attacker_init():
    attacker = AdaptiveAttacker(
        pii="4111-1111-1111-1111",
        pii_type="credit_card_number",
        system_prompt="test prompt",
    )
    assert attacker.pii == "4111-1111-1111-1111"
    assert attacker.pii_type == "credit_card_number"
    assert attacker.memory.buffer.maxlen == 20


def test_adaptive_attacker_update_memory():
    attacker = AdaptiveAttacker(
        pii="test", pii_type="ssn", system_prompt="prompt",
    )
    attacker.update_memory("some text", verdict=1)
    attacker.update_memory("other text", verdict=0)
    attacker.update_memory("invalid text", verdict=-1)
    assert len(attacker.memory.buffer) == 3
    assert attacker.memory.buffer[0].verdict == "BLOCKED"
    assert attacker.memory.buffer[1].verdict == "UNDETECTED"
    assert attacker.memory.buffer[2].verdict == "INVALID"


def test_adaptive_attacker_craft_input(mocker):
    attacker = AdaptiveAttacker(
        pii="123-45-6789", pii_type="ssn", system_prompt="test",
    )
    mock_post = mocker.patch(
        "data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks"
        ".attacker.post_request_openai",
        return_value="Here is 123-45-6789 for the record.",
    )
    result = attacker.craft_input()
    assert result == "Here is 123-45-6789 for the record."
    assert mock_post.call_count == 1


def test_adaptive_attacker_validate_pii_exists(mocker):
    attacker = AdaptiveAttacker(
        pii="123-45-6789", pii_type="ssn", system_prompt="test",
    )
    mocker.patch(
        "data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks"
        ".attacker.post_request_openai",
        return_value={"result": True},
    )
    assert attacker.validate_pii_exists("My SSN is 123-45-6789") is True


def test_readme_paths_exist():
    for path in ADVERSARIAL_ATTACK_README_PATHS:
        assert Path(path).exists(), f"README not found: {path}"


def test_init_attacker_awareness_none(mocker):
    attacker = _init_attacker(pii="test", pii_type="ssn", attacker_awareness="none")
    assert isinstance(attacker, AdaptiveAttacker)


def test_init_attacker_awareness_attacks(mocker):
    attacker = _init_attacker(pii="test", pii_type="ssn", attacker_awareness="attacks")
    assert "rule-based" not in attacker.system_prompt


def test_run_attack_mocked(mocker):
    mocker.patch(
        "data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks"
        ".attacker.post_request_openai",
        side_effect=[
            "My SSN is 123-45-6789",       # craft_input
            {"result": True},               # validate_pii_exists
            "Another attempt 123-45-6789",  # craft_input
            {"result": True},               # validate_pii_exists
        ],
    )
    mocker.patch(
        "data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks"
        ".loop.presidio_pii_analyzer",
        return_value=[{"type": "ssn", "value": "123-45-6789"}],
    )
    trace = run_attack(
        attack_id=1, pii="123-45-6789", pii_type="ssn",
        max_iters=2, attacker_awareness="none",
    )
    assert len(trace) > 0
    assert trace[0]["attack_id"] == 1
    assert trace[0]["pii"] == "123-45-6789"
    assert "detector" in trace[0]
