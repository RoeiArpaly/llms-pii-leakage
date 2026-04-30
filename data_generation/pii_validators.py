"""Validators for generated PII values (Luhn checksum for credit cards,
Mod-97 for IBANs) to ensure synthetic data is structurally valid.
"""


def is_valid_credit_card(card_number: str) -> bool:
    """
    Compute the Luhn checksum for the provided string of digits. Note this
    assumes the check digit is in place.
    """
    # check if all characters are digits
    card_number = card_number.replace(" ", "").replace("-", "")
    if not card_number.isdigit():
        return False
    digits = list(map(int, card_number))
    odd_sum = sum(digits[-1::-2])
    even_sum = sum([sum(divmod(2 * d, 10)) for d in digits[-2::-2]])
    is_valid = (odd_sum + even_sum) % 10 == 0
    return is_valid


def is_valid_iban(iban: str) -> bool:
    """Validate IBAN using Mod 97 checksum."""
    # Remove spaces, make uppercase
    iban = iban.upper().replace(" ", "").replace("-", "")
    if any([not iban.isalnum(), len(iban) < 15, len(iban) > 34, not iban[2:4].isdigit()]):
        return False
    iban = iban[4:] + iban[:4]
    iban_numeric = ""
    for char in iban:
        if char.isalpha():
            iban_numeric += str(ord(char) - 55)
        else:
            iban_numeric += char
    iban_numeric = int(iban_numeric)
    return iban_numeric % 97 == 1
