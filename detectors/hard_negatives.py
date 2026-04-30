"""Hard-negatives detector: recognise known non-PII formats.

Final precision-tightening tier in the PII Shield cascade. Suppresses
detections whose flagged span (or, for tiers without a span, whose
input text) matches a known non-PII format (UUID, MAC, IPv6, hashes,
ETags, hex colors, serial numbers, dashed invoice IDs).

Real PII formats are mutually exclusive with these shapes by
construction: a Luhn-valid credit-card number cannot also be a UUID;
a Mod-97-valid IBAN cannot be a SHA-256 hash; a real phone number
will not match the IPv6-with-hex-letter pattern. The filter therefore
recovers precision on lookalike-non-PII without trading off recall on
real PII --- assuming the deny-list enumerates the lookalike formats
the deployment actually sees. Novel formats not in the deny-list
remain a precision risk.
"""
import re

_HEX = r"[0-9a-fA-F]"

# Span-level patterns: anchored, the entire span must match. Hex-only
# patterns (sha*, md5, hex_long) additionally require at least one
# a-f / A-F letter so that pure-digit chunked-PII spans (e.g. "065488"
# from a colon-chunked phone) are not mis-filtered as hashes.
_PATTERNS_SPAN = [
    ("uuid",            re.compile(rf"^{_HEX}{{8}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{12}}$")),
    ("mac",             re.compile(rf"^(?:{_HEX}{{2}}[:-]){{5}}{_HEX}{{2}}$")),
    ("ipv6_full",       re.compile(rf"^(?:{_HEX}{{1,4}}:){{7}}{_HEX}{{1,4}}$")),
    ("ipv6_compressed", re.compile(r"^[0-9a-fA-F:]*::[0-9a-fA-F:]*$")),
    ("sha256",          re.compile(rf"^(?=.*[a-fA-F]){_HEX}{{64}}$")),
    ("sha1",            re.compile(rf"^(?=.*[a-fA-F]){_HEX}{{40}}$")),
    ("md5",             re.compile(rf"^(?=.*[a-fA-F]){_HEX}{{32}}$")),
    ("hex_color",       re.compile(rf"^#{_HEX}{{3,8}}$")),
    ("etag",            re.compile(r'^W/"[^"]+"$')),
    ("serial",          re.compile(r"^(?:SN|S/N|SER|MDL|PT)[\-_]?[A-Z0-9]{6,}$", re.IGNORECASE)),
    ("hex_long",        re.compile(rf"^(?=.*[a-fA-F]){_HEX}{{20,}}$")),
    ("invoice_dashed",  re.compile(r"^(?:INV[\-_])?\d{4}-\d{4}-\d{2,4}$")),
]

# Input-level patterns: unanchored, search anywhere in the text.
# IPv6 patterns require at least one hex letter (a-f / A-F) so that
# chunked-with-colons attacks like "7:6:8: :4:7:8:..." (pure-digit
# segments around colons) do not match.
_PATTERNS_INPUT = [
    ("uuid",            re.compile(rf"\b{_HEX}{{8}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{4}}-{_HEX}{{12}}\b")),
    ("mac",             re.compile(rf"\b(?:{_HEX}{{2}}[:-]){{5}}{_HEX}{{2}}\b")),
    ("ipv6_full",       re.compile(rf"\b(?:{_HEX}{{1,4}}:){{7}}{_HEX}{{1,4}}\b")),
    ("ipv6_compressed", re.compile(
        rf"\b{_HEX}{{0,4}}(?::{_HEX}{{0,4}}){{2,}}::{_HEX}{{0,4}}(?::{_HEX}{{0,4}})*\b(?=.*?[a-fA-F])"
        rf"|(?=.*?[a-fA-F])\b{_HEX}{{0,4}}::{_HEX}{{0,4}}(?::{_HEX}{{0,4}}){{0,6}}\b"
    )),
    ("sha256",          re.compile(rf"\b{_HEX}{{64}}\b")),
    ("sha1",            re.compile(rf"\b{_HEX}{{40}}\b")),
    ("md5",             re.compile(rf"\b{_HEX}{{32}}\b")),
    ("hex_color",       re.compile(rf"#{_HEX}{{3,8}}\b")),
    ("etag",            re.compile(r'W/"[^"]+"')),
    ("serial",          re.compile(r"\b(?:SN|S/N|SER|MDL|PT)[\-_]?[A-Z0-9]{6,}\b", re.IGNORECASE)),
    ("hex_long",        re.compile(rf"\b{_HEX}{{20,}}\b")),
    ("invoice_dashed",  re.compile(r"\b(?:INV[\-_])?\d{4}-\d{4}-\d{2,4}\b")),
]


def is_hard_negative_value(value: str) -> str | None:
    """Return matched pattern name if value is a known non-PII format."""
    if not isinstance(value, str) or not value:
        return None
    v = value.strip()
    for name, pat in _PATTERNS_SPAN:
        if pat.fullmatch(v):
            return name
    return None


def is_hard_negative_input(text: str) -> str | None:
    """Return matched pattern name if input contains a known non-PII format."""
    if not isinstance(text, str) or not text:
        return None
    for name, pat in _PATTERNS_INPUT:
        if pat.search(text):
            return name
    return None


def filter_hard_negative_spans(spans: list[dict]) -> list[dict]:
    """Drop spans whose value matches a known non-PII format."""
    if not spans:
        return spans
    return [s for s in spans if not is_hard_negative_value(s.get("value") or "")]
