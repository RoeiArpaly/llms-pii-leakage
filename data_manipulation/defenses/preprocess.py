import random
import regex as re
import string

import emoji

from data_manipulation.constants import (
    ALPHABET_EMOJI_MAP,
    NUMBER_EMOJI_MAP,
    NUMBER_WORD_MAP,
)


def mapping_helper(start_hex: int, count: int, mapper: iter) -> dict:
    return dict(zip([chr(start_hex + i) for i in range(count)], mapper))


# Digits 1-10
digits_1_10 = [str(i) for i in range(1, 11)]

# Homoglyphs
mappings = {  # Check hex of letter by using: hex(ord("Ⓐ"))
    # Letters
    "circle_letters": mapping_helper(0x24D0, 26, string.ascii_lowercase),  # ⓐ-ⓩ
    "circle_uppercase_letters": mapping_helper(0x24B6, 26, string.ascii_uppercase),  # Ⓐ-Ⓩ
    "square_letters": mapping_helper(0x1F130, 26, string.ascii_uppercase),  # 🄰-🅉
    "square_uppercase_letters": mapping_helper(0x1F110, 26, string.ascii_uppercase),  # 🄐-🄩
    "fullwidth_letters": mapping_helper(0xFF21, 26, string.ascii_uppercase),  # Ａ-Ｚ
    "parenthesized_letters": mapping_helper(0x249C, 26, string.ascii_lowercase),  # ⒜-⒵
    "math_bold_script_small": mapping_helper(0x1d4b6, 26, string.ascii_lowercase),  # 𝒜-𝒵
    "italic_letters": mapping_helper(0x1D44E, 26, string.ascii_lowercase),  # 𝑎-𝑧
    "cyrillic_letters": mapping_helper(0x0430, 32, string.ascii_lowercase),  # а-я
    "cyrillic_uppercase_letters": mapping_helper(0x0410, 32, string.ascii_uppercase),  # А-Я
    "greek_letters": mapping_helper(0x03B1, 25, string.ascii_lowercase),  # α-ω
    "greek_uppercase_letters": mapping_helper(0x0391, 25, string.ascii_uppercase),  # Α-Ω
    "emoji_letters_circle": mapping_helper(0x1F170, 26, string.ascii_uppercase),  # 🅐-🅩
    "emoji_letters_circle_2": mapping_helper(0x1F150, 26, string.ascii_uppercase),  # 🅐-🅩
    "emoji_letters_square": mapping_helper(0x1F130, 26, string.ascii_uppercase),  # 🄰-🅉
    "bold_letters": mapping_helper(0x1D400, 26, string.ascii_uppercase),  # 𝐀-𝐙
    "bold_italic_letters": mapping_helper(0x1D41A, 26, string.ascii_lowercase),  # 𝐚-𝐳
    "bold_script_letters": mapping_helper(0x1D4D0, 26, string.ascii_uppercase),  # 𝓐-𝓩
    "monospace_letters": mapping_helper(0x1D670, 26, string.ascii_uppercase),  # 𝙰-𝚉
    "sans_serif_letters": mapping_helper(0x1D5A0, 26, string.ascii_uppercase),  # 𝖠-𝖹
    "sans_serif_bold_letters": mapping_helper(0x1D5D4, 26, string.ascii_uppercase),  # 𝗔-𝗭
    "sans_serif_italic_bold_letters": mapping_helper(0x1D622, 26, string.ascii_lowercase),  # 𝘢-𝘺
    "sans_serif_italic_bold_letters_2": mapping_helper(0x1D492, 26, string.ascii_lowercase),  # 𝓐-𝓩

    # Digits
    "fullwidth_digits": mapping_helper(0xFF10, 10, string.digits),  # ０-９
    "math_bold_digits": mapping_helper(0x1D7CE, 10, string.digits),  # 𝟘-𝟡
    "math_bold_digits_2": mapping_helper(0x1D7F6, 10, string.digits),  # 𝟶-𝟿
    "bold_digits": mapping_helper(0x1D7D9, 10, digits_1_10[:-1] + ["0"]),  # 𝟙-𝟡
    "parenthesized_numbers": mapping_helper(0x2474, 10, digits_1_10),  # ⑴-⑾
    "misc_enclosed_numbers": mapping_helper(0x2460, 10, digits_1_10),  # ①-⑩
    "circled_number": {"⓪": "0"},
    "small_digits": mapping_helper(0x2080, 10, string.digits),  # ₀-₉
    "dotted_circled_numbers": mapping_helper(0x24F5, 10, digits_1_10),  # ⓵-⓾
    "dingbat_numbers": mapping_helper(0x2776, 10, digits_1_10),  # ❶-❿
    "dingbat_negative_circled_digits": mapping_helper(0x277F, 10, ["10"] + digits_1_10[:-1]),  # ➀-➈
    # Emojis
    "regional_indicator_letters": mapping_helper(0x1F1E6, 26, string.ascii_uppercase),  # 🇦-🇿
    "emoji_clock_faces": mapping_helper(0x1F550, 12, [str(i + 1) for i in range(12)]),  # 🕐-🕟
    "emoji_enclosed_alphabet": {v: k for k, v in ALPHABET_EMOJI_MAP.items()},
}
# Emojis that can't be mapped to a single character
replace_mapping = {
    "emoji_keycap_number": {v: k for k, v in NUMBER_EMOJI_MAP.items()},
}


def transform_homoglyphs_to_alphabets(text: str, delimiter=":") -> dict:
    """
    Converts homoglyphs in the text to their respective alphabets.
    Includes emojis and special characters.
    """
    new_text = []
    for letter in text:
        for key, mapping in mappings.items():
            if letter in mapping:
                new_text.append(mapping[letter])
                break
        else:
            new_text.append(letter)

    new_text = "".join(new_text)
    for key, mapping in replace_mapping.items():
        for emoji_ in mapping:
            new_text = new_text.replace(emoji_, mapping[emoji_])

    new_text = emoji.demojize(new_text, delimiters=(delimiter, delimiter))
    return {
        "text": new_text,
        "homoglyph_detected": text != new_text,
    }


def remove_separators(text: str) -> str:
    """
    Transform all unsupported separators to '-'.
    Replace consecutive identical separators with a single separator.
    """
    # Define supported separators. (Note: '-' is already escaped below)
    allowed_separators = "()\-@. "  # noqa: W605
    # Replace unsupported separators with '-'
    text = re.sub(pattern=f"[^\w{allowed_separators}]", repl="-", string=text)  # noqa: W605
    # Collapse consecutive *identical* separators.
    text = re.sub(pattern=r"([" + re.escape(allowed_separators) + r"])\1+", repl=r"\1", string=text)
    return text


def textual_number_to_numeric(text: str) -> str:
    """
    Convert textual numbers to numeric representation.
    Example: "Hello one-two-three" -> "Hello 1-2-3"
    """
    # Define mapping of textual numbers to numeric representation
    for language in NUMBER_WORD_MAP:
        mapping = {v: k for k, v in NUMBER_WORD_MAP[language].items()}
        # Replace textual numbers with numeric representation
        for word, number in mapping.items():
            text = re.sub(pattern=r"\b" + word + r"\b", repl=number, string=text)
    return text


def defensive_preprocess(text: str) -> str:
    """
    Defensive preprocessing to convert homoglyphs and emojis to alphabets.
    """
    rand_n = random.randint(10, 20)  # Random delimiter length to avoid delimiter attacks
    delimiter = "|" * rand_n
    esc_delimiter = re.escape(delimiter)
    result = transform_homoglyphs_to_alphabets(text=text, delimiter=delimiter)

    # Format the result by replacing underscores with spaces and title casing
    formatted_text = re.sub(
        pattern=rf"{esc_delimiter}([\w_]+){esc_delimiter}",
        repl=lambda m: m[1].replace("_", " ").title(),  # Convert emoji representation to text
        string=result["text"],
    )
    new_text = "".join(formatted_text.split(delimiter))
    new_text = textual_number_to_numeric(new_text)
    new_text = remove_separators(new_text)
    return new_text
