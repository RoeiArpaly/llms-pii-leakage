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
    iban = iban.replace(" ", "").upper()
    if not 15 <= len(iban) <= 34:
        return False
    # Move the four initial characters to the end
    rearranged = iban[4:] + iban[:4]
    # Replace letters with numbers (A=10, B=11, ..., Z=35)
    numeric_iban = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric_iban += ch
        elif ch.isalpha():
            numeric_iban += str(ord(ch) - 55)
        else:
            return False  # invalid character
    # Perform mod-97
    try:
        return int(numeric_iban) % 97 == 1
    except ValueError:
        return False
