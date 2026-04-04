"""Fuzzy-matching Presidio recognizers for adversarially modified PII.

Wraps standard Presidio recognizers (credit card, IBAN, SSN, email) with
regex fuzzy matching (substitutions, deletions) to detect PII values that
have been lightly obfuscated.
"""
import regex as re

from functools import lru_cache

from presidio_analyzer import (
    Pattern,
    PatternRecognizer,
)
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    UsSsnRecognizer,
)


def fuzzy_pii_recognizer(recognizers: list) -> list:
    """
    fuzzy_counts is (n_substitutions, n_insertions, n_deletes)

    {i<=3} permit at most 3 insertions, but no other types
    {d<=3} permit at most 3 deletions, but no other types
    {s<=3} permit at most 3 substitutions, but no other types
    {i<=1,s<=2} permit at most 1 insertion and at most 2 substitutions, but no deletions
    {e<=3} permit at most 3 errors
    {1<=e<=3} permit at least 1 and at most 3 errors
    {i<=2,d<=2,e<=3} permit at most 2 insertions, at most 2 deletions, at most 3 errors in total
    """

    fuzzy_recognizers = []
    for recognizer, error_types in recognizers:
        supported_entity = recognizer.supported_entities[0]
        name = recognizer.name
        context = recognizer.context
        deny_list = recognizer.deny_list

        fuzzy_patterns = []
        if not error_types:
            raise ValueError("No error types provided for fuzzy matching")
        for error_type in error_types:
            for pattern in recognizer.patterns:
                deletions = error_type.get("deletions", 0)
                substitutions = error_type.get("substitutions", 0)
                if "insertions" in error_type:  # Insertions are not allowed!
                    raise ValueError("Insertions are not allowed in fuzzy matching")
                max_total_distance = deletions + substitutions
                fuzzy_patterns.append(
                    Pattern(
                        name=f"Fuzzy(s={substitutions},d={deletions}){pattern.name}",
                        regex=f"({pattern.regex}){{s<={substitutions},d<={deletions}}}",
                        score=round(pattern.score / (max_total_distance + 1) ** 2, 2),
                    )
                )
        fuzzy_recognizer = PatternRecognizer(
            supported_entity=supported_entity,
            name=f"Fuzzy{name}",
            patterns=fuzzy_patterns,
            context=context,
            deny_list=deny_list,
            global_regex_flags=re.BESTMATCH | re.IGNORECASE,
        )
        # fuzzy_recognizer.validate_result = lambda x: True  # (Can not validate fuzzy matches)
        fuzzy_recognizers.append(fuzzy_recognizer)
    return fuzzy_recognizers


@lru_cache(maxsize=1)
def get_fuzzy_recognizers() -> tuple:
    recognizers = [
            (
                IbanRecognizer(), [
                    dict(deletions=1),
                    dict(substitutions=1),
                ],
            ),
            (
                UsSsnRecognizer(), [
                    dict(deletions=1),
                    dict(deletions=2),
                    dict(substitutions=1),
                    dict(substitutions=1, deletions=1),
                ],
            ),
            (
                EmailRecognizer(), [
                    dict(deletions=1),
                    dict(substitutions=1),
                ],
            ),
            (
                CreditCardRecognizer(), [
                    dict(deletions=1),
                    dict(deletions=3),
                    dict(substitutions=3),
                    dict(substitutions=4),
                    dict(deletions=2, substitutions=2),
                ],
            ),
        ]
    return fuzzy_pii_recognizer(recognizers=recognizers)
