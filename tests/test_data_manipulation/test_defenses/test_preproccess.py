import pytest

from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    mappings,
    remove_placeholders,
    remove_separators,
    textual_number_to_numeric,
    textual_symbol_to_symbol,
    transform_homoglyphs_to_alphabets,
)


@pytest.mark.parametrize("key, mapping", mappings.items())
def test_mapping(key, mapping):
    for symbol, letter in mapping.items():
        result = transform_homoglyphs_to_alphabets(symbol)
        assert result["text"] == letter


@pytest.mark.parametrize("input_text, expected_output", [
    # Social Security Numbers (SSN)
    ("SSN: ①②③-④⑤-⑥⑦⑧⑨", "SSN: 123-45-6789"),
    ("My SSN is 𝟏𝟐𝟑-𝟒𝟓-𝟔𝟕𝟖𝟗", "My SSN is 123-45-6789"),
    ("SSN: 🆂🆂🅽 ９８７-６５-４３２１", "SSN: SSN 987-65-4321"),
    ("SSN: 🅂🅂🄽 ➈➇➆-➅➄-➃➂➁➀", "SSN: SSN 987-65-4321"),

    # IBAN Numbers
    ("IBAN: 🏦 🄳🄴①②③④⑤⑥⑦⑧⑨⓪", "IBAN: :bank: DE1234567890"),
    ("Bank IBAN: 💰 𝐅𝐑𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗", "Bank IBAN: :money_bag: FR123456789"),
    ("IBAN: 🅴🆂①②③④⑤⑥⑦⑧⑨", "IBAN: ES123456789"),
    ("IBAN: 𝐈𝐓𝟎𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗", "IBAN: IT00123456789"),
    ("My bank account number is 🕐🕑🕒", "My bank account number is 123"),

    # Email Addresses
    ("Email: 📧 user@🅖🅜🅐🅘🅛.com", "Email: :e-mail: user@GMAIL.com"),
    ("Contact me at 𝓉𝑒𝓈𝓉@𝑒𝓍𝒶𝓂𝓅𝓁𝑒.𝒸𝑜𝓂", "Contact me at test@example.com"),
    ("Work Email: ✉️ 𝘢𝘭𝘪𝘤𝘦@𝘤𝘰𝘮𝘱𝘢𝘯𝘺.𝘤𝘰𝘮", "Work Email: :envelope: alice@company.com"),
    ("📨 Send mail to 🆄🆂🅴🆁@🆈🅰🅷🅾🅾.🅲🅾🅼", ":incoming_envelope: Send mail to USER@YAHOO.COM"),
    ("My email is: ԛβаｒｒетт＠ехамрӏе．пет", "My email is: qbarrett@example.net"),

    # Phone Numbers
    ("Call me at ①②③-④⑤⑥-⑦⑧⑨⓪", "Call me at 123-456-7890"),
    ("📞 My phone is +➀➂➁➀➂➁➀➂➁➀", ":telephone_receiver: My phone is +1321321321"),
    ("☎️ Reach me at 🆃🅴🅻: 𝟕𝟎𝟕-𝟖𝟕𝟔-𝟓𝟓𝟒𝟑", ":telephone: Reach me at TEL: 707-876-5543"),
    ("📱 Number: 𝟗𝟎𝟎➈➈➈-➆➆➆➀", ":mobile_phone: Number: 900999-7771"),

    # Credit Card Numbers
    ("My 💳 is ４５６７-８９０１-２３４５-６７８９", "My :credit_card: is 4567-8901-2345-6789"),
    ("Visa: ➀➁➂➃ ➄➅➆➇ ➈⓪⓪⓪ ⓪⓪⓪⓪", "Visa: 1234 5678 9000 0000"),
    ("CC is 4️⃣5️⃣6️⃣7️⃣-8️⃣9️⃣0️⃣1️⃣-2️⃣3️⃣4️⃣5️⃣-6️⃣7️⃣8️⃣9️⃣", "CC is 4567-8901-2345-6789"),
    ("CC: 𝟺𝟺𝟺𝟺-𝟿𝟿𝟿𝟿-𝟺𝟺𝟺𝟺-𝟿𝟿𝟿𝟿", "CC: 4444-9999-4444-9999"),
    ("Mastercard: ①②③④ ⑤⑥⑦⑧ ⑨⓪⓪⓪ ⓪⓪⓪⓪", "Mastercard: 1234 5678 9000 0000"),

    # Other edge cases
    (":grinning:", ":grinning:"),  # Emojis that should not change
    ("Hello World!", "Hello World!"),  # No emojis, should remain unchanged
    ("", ""),  # Empty string
    ("   ", "   "),  # Spaces should remain unchanged
])
def test_transform_emoji_to_text(input_text, expected_output):
    result = transform_homoglyphs_to_alphabets(input_text)
    assert result["text"] == expected_output
    assert result["homoglyph_detected"] == (input_text != expected_output)


@pytest.mark.parametrize("input_text, expected_output", [
    (
        "My 💳 is 4️⃣5️⃣6️⃣7️⃣-8️⃣9️⃣0️⃣1️⃣-2️⃣3️⃣4️⃣5️⃣-6️⃣7️⃣8️⃣9️⃣.",
        "My Credit Card is 4567-8901-2345-6789.",
    ),
    (
        "The 🏦 IBAN is 🄳🄴①②③④⑤⑥⑦⑧⑨⓪.",
        "The Bank IBAN is DE1234567890.",
    ),
])
def test_defensive_preprocess(input_text, expected_output):
    result = defensive_preprocess(input_text, include_sandwich=False)
    assert result == expected_output


@pytest.mark.parametrize("input_text, expected_output", [
    # Basic removal tests
    ("This is some REDACTED text.", "This is some text."),
    ("The value is NULL and REdaCted.", "The value is and."),
    ("Data: N/A, Status: UNDEFINED.", "Data:, Status:."),
    ("No issues here.", "No issues here."),
    ("null and redacted are gone.", "and are gone."),

    # Tests with delimiters
    ("Customer ID: 12345 (REDACTED), Transaction: NULL.", "Customer ID: 12345, Transaction:."),
    ("Access denied to [REDACTED] information.", "Access denied to information."),
    ("Value is {NULL} or (UNDEFINED).", "Value is or."),
    ("User (REDACTED) created an entry (NULL).", "User created an entry."),

    # Tests with numbers and punctuation
    ("Values: 1NULL2, NULL.", "Values: 1 2,."),  # 1NULL2 becomes 12
    ("Hello!REDACTEDWorld.", "Hello!REDACTEDWorld."),  # Adjacent to punctuation
    ("Amount: $NULL.50", "Amount: $.50"),
    ("Code: AAREDACTEDAA", "Code: AAREDACTEDAA"),  # Should NOT remove if part of alphanumeric
    ("Value_NULL", "Value_"),  # Remove NULL is not part of alphanumeric

    # Tests for not removing partial words
    ("This is notannulled, nor aredactable.", "This is notannulled, nor aredactable."),
    ("Do not unREDACTED this data.", "Do not unREDACTED this data."),

    # Edge cases / Multiple occurrences
    ("Multiple REDACTED, NULL values.", "Multiple, values."),
    ("  leading and trailing NULL spaces   ", "leading and trailing spaces"),
    ("REDACTED REDACTED NULL", ""),  # All placeholders, leading/trailing space cleaned

    # Empty string
    ("", ""),
    ("   ", ""),  # Only whitespace
])
def test_remove_placeholders(input_text, expected_output):
    """
    Tests the remove_placeholders function for various scenarios
    including delimited, standalone, and partial word occurrences.
    """
    result = remove_placeholders(input_text)
    assert result == expected_output


@pytest.mark.parametrize("input_text, expected_output", [
    ("My Credit Card is 4567---8901---2345---6789", "My Credit Card is 4567-8901-2345-6789"),
    ("My email is user@@domain..com", "My email is user@domain.com"),
    ("My phone number is 123   456   7890", "My phone number is 123 456 7890"),
    ("My SSN is 123$$45$$6789", "My SSN is 123-45-6789"),
    ("My SSN is (123) 45 6789", "My SSN is (123) 45 6789"),
    (
        """
        My Credit Card is "413" + "771" + "193" + "548" + "2232"
        """,
        "My Credit Card is 413-771-193-548-2232"
    ),
    # Commas/colons before PII must become spaces, not dashes,
    # to preserve NER token boundaries.
    ("Your full name, 803-54-1242, and contact", "Your full name 803-54-1242 and contact"),
    ("Contact number: 310.580.2368.", "Contact number 310.580.2368."),
    ("Expenditures: 4533148643409967912.", "Expenditures 4533148643409967912."),
    # Plus sign in phone numbers must be preserved.
    ("Call me at +1-293-926-6036.", "Call me at +1-293-926-6036."),
])
def test_remove_separators(input_text, expected_output):
    result = remove_separators(input_text)
    assert result == expected_output


@pytest.mark.parametrize("input_text, expected_output", [
    ("Hello one-two-three", "Hello 1-2-3"),
    ("Hello one two three", "Hello 1 2 3"),
    ("Hello uno dos three", "Hello 1 2 3"),
    ("Hello onetwothree", "Hello onetwothree"),
    ("Hello this is sixtytwo and one", "Hello this is sixtytwo and 1"),
    ("Hello this is sixty two and one", "Hello this is sixty 2 and 1"),
])
def test_text_to_numeric(input_text, expected_output):
    result = textual_number_to_numeric(input_text)
    assert result == expected_output


@pytest.mark.parametrize("input_text, expected_output", [
    ("Hello 1(dash)2(dash)3", "Hello 1-2-3"),
    ("Hello 1dash2dash3", "Hello 1-2-3"),
    ("Hello 1dashboard2dash3", "Hello 1dashboard2-3"),
    ("Hello 1dashed2dash3", "Hello 1dashed2-3"),
    ("Hello 1dash!2dash3", "Hello 1-!2-3"),
    ("Hello 1dashdot2dash3", "Hello 1dashdot2-3"),
    ("My email is: johndotdoe@gmail.com", "My email is: johndotdoe@gmail.com"),
    ("My email is: john(dot)doe@gmail.com", "My email is: john.doe@gmail.com"),
])
def test_textual_symbol_to_symbol(input_text, expected_output):
    result = textual_symbol_to_symbol(input_text)
    assert result == expected_output


# ── @ variant recovery through full pipeline ────────────────────────

@pytest.mark.parametrize("at_char, label", [
    ("@", "standard"),
    ("\uFF20", "fullwidth"),
    ("\uAA5C", "homoglyph"),
    ("\U0001F300", "emoji"),
    ("\uFE6B", "small form"),
])
def test_at_sign_variants_recovered(at_char, label):
    """All known @ substitutions must recover to user@domain.com."""
    text = f"Send to user{at_char}domain.com please."
    defended = defensive_preprocess(text, include_sandwich=False)
    assert "user@domain.com" in defended, (
        f"@ variant '{label}' (U+{ord(at_char):04X}) not recovered: "
        f"{defended!r}"
    )


@pytest.mark.parametrize("wrapper", ["(at)", "[at]", "{at}"])
def test_at_word_forms_recovered(wrapper):
    """Word forms of @ in all delimiter styles must recover."""
    text = f"Send to user{wrapper}domain.com please."
    defended = defensive_preprocess(text, include_sandwich=False)
    assert "user@domain.com" in defended, (
        f"'{wrapper}' not recovered: {defended!r}"
    )
