"""Post-detection PII span validators.

Filters out structurally invalid PII detections (false positives) by
applying format and checksum checks per PII type. Reuses Luhn and
Mod-97 validators from data_generation.pii_validators.
"""
import re

from data_generation.pii_validators import is_valid_credit_card, is_valid_iban
from logger import logger

_SHA256_RE = re.compile(r'^[0-9a-fA-F]{64}$')
_SHA1_RE = re.compile(r'^[0-9a-fA-F]{40}$')
_MD5_RE = re.compile(r'^[0-9a-fA-F]{32}$')
_UUID_RE = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}'
    r'-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
)
_MAC_RE = re.compile(r'^([0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}$')
_NON_PII_PATTERNS = [_SHA256_RE, _SHA1_RE, _MD5_RE, _UUID_RE, _MAC_RE]

# Long hex strings (16+ chars) that contain at least one a-f letter.
# Pure-digit strings are excluded to avoid catching credit cards / IBANs.
_HEX_LONG_RE = re.compile(r'^[0-9a-fA-F]{16,}$')
_HAS_HEX_ALPHA = re.compile(r'[a-fA-F]')


def _is_known_non_pii(value: str) -> bool:
    """Check if value matches common non-PII technical patterns."""
    if any(pattern.match(value) for pattern in _NON_PII_PATTERNS):
        return True
    stripped = value.replace(" ", "").replace("-", "")
    return bool(_HEX_LONG_RE.match(stripped) and _HAS_HEX_ALPHA.search(stripped))


def _is_valid_ssn(value: str) -> bool:
    """Validate US SSN format and range per SSA rules."""
    digits = value.replace("-", "").replace(" ", "")
    if not digits.isdigit() or len(digits) != 9:
        return False
    area, group, serial = int(digits[:3]), int(digits[3:5]), int(digits[5:])
    if area == 0 or area == 666 or 900 <= area <= 999:
        return False
    if group == 0 or serial == 0:
        return False
    return True


_TYPE_VALIDATORS = {
    "credit_card_number": is_valid_credit_card,
    "ssn": _is_valid_ssn,
    "iban": is_valid_iban,
}


def validate_pii_spans(spans: list[dict]) -> list[dict]:
    """Filter out structurally invalid PII detections.

    When a type-specific validator exists (Luhn, Mod-97, SSN range), it
    takes priority — the span is kept only if it passes.  For types
    without a validator, the universal non-PII pattern filter applies.
    """
    validated = []
    for span in spans:
        value = span.get("value") or ""
        pii_type = span.get("type", "")

        validator = _TYPE_VALIDATORS.get(pii_type)
        if validator and value:
            if not validator(value):
                logger.debug(f"Rejected invalid {pii_type}: {value[:30]}")
            else:
                validated.append(span)
            continue

        if value and _is_known_non_pii(value):
            logger.debug(f"Rejected non-PII pattern: {pii_type}={value[:30]}")
            continue

        validated.append(span)
    return validated
