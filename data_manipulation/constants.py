CREDIT_CARD_VARIATIONS = [
    "credit card",
    "credit card number",
    "credit card no",
    "credit card num",
    "cc",
    "cc number",
    "cc no",
    "cc num",
    "credit",
    "card number",
    "card no",
    "card num",
]
SSN_VARIATIONS = [
    "ssn",
    "social security number",
    "social security",
    "social number",
    "ss",
    "social",
    "ssn number",
    "ss number",
    "ssn no",
    "ss no",
]
IBAN_VARIATIONS = [
    "iban",
    "iban code",
    "iban number",
    "iban no",
    "international bank account number",
    "international bank account",
    "international bank number",
    "international bank no",
    "bank account number",
    "bank account",
    "bank number",
    "bank no",
]
PHONE_VARIATIONS = [
    "phone number",
    "phone",
    "mobile number",
    "mobile",
    "smartphone",
    "cellphone",
    "cell phone",
    "cell",
    "contact number",
    "contact no",
    "tel",
    "telephone",
    "tele number",
    "handphone",
    "mobile phone",
    "work phone",
    "home phone",
    "office phone",
    "personal phone",
]
EMAIL_VARIATIONS = [
    "email",
    "email address",
    "email id",
    "email no",
    "email no.",
    "email id.",
    "mail",
    "mail address",
    "mail id",
    "mail no",
    "mail no.",
    "mail id.",
]
NUMBER_EMOJI_MAP = {
    "0": "0️⃣",
    "1": "1️⃣",
    "2": "2️⃣",
    "3": "3️⃣",
    "4": "4️⃣",
    "5": "5️⃣",
    "6": "6️⃣",
    "7": "7️⃣",
    "8": "8️⃣",
    "9": "9️⃣",
}
ALPHABET_EMOJI_MAP = {
    "A": "🅰",
    "B": "🅱",
    "C": "🅲",
    "D": "🅳",
    "E": "🅴",
    "F": "🅵",
    "G": "🅶",
    "H": "🅷",
    "I": "🅸",
    "J": "🅹",
    "K": "🅺",
    "L": "🅻",
    "M": "🅼",
    "N": "🅽",
    "O": "🅾",
    "P": "🅿",
    "Q": "🆀",
    "R": "🆁",
    "S": "🆂",
    "T": "🆃",
    "U": "🆄",
    "V": "🆅",
    "W": "🆆",
    "X": "🆇",
    "Y": "🆈",
    "Z": "🆉",
}
PII_EMOJI_MAP = {
    "CREDIT_CARD": "💳",
    "SSN": "🪪",
    "BANK_ACCOUNT": "🏦",
    "PHONE_NUMBER": "📞",
    "EMAIL": "📧",
}
PII_HOMOGLYPH_MAP = {
    "CREDIT_CARD": "𝓬𝓻𝓮𝓭𝓲𝓽 𝓬𝓪𝓻𝓭",
    "SSN": "𝓼𝓸𝓬𝓲𝓪𝓵 𝓼𝓮𝓬𝓾𝓻𝓲𝓽𝔂 𝓷𝓾𝓶𝓫𝓮𝓻",
    "BANK_ACCOUNT": "𝓫𝓪𝓷𝓴 𝓪𝓬𝓬𝓸𝓾𝓷𝓽",
    "PHONE_NUMBER": "𝓹𝓱𝓸𝓷𝓮 𝓷𝓾𝓶𝓫𝓮𝓻",
    "EMAIL": "𝓮𝓶𝓪𝓲𝓵",
}
NUMBER_WORD_MAP = {
    "english": {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    },
    "spanish": {
        "0": "cero",
        "1": "uno",
        "2": "dos",
        "3": "tres",
        "4": "cuatro",
        "5": "cinco",
        "6": "seis",
        "7": "siete",
        "8": "ocho",
        "9": "nueve",
    },
    "french": {
        "0": "zéro",
        "1": "un",
        "2": "deux",
        "3": "trois",
        "4": "quatre",
        "5": "cinq",
        "6": "six",
        "7": "sept",
        "8": "huit",
        "9": "neuf",
    },
}
WORD_SYMBOLS_MAP = {
    ".": "dot",
    ",": "comma",
    "-": "dash",
    "—": "long dash",
    "_": "underscore",
    "/": "slash",
    "@": "at",
    ":": "colon",
    ";": "semicolon",
    "\"": "quote",
    "'": "apostrophe",
    "(": "left parenthesis",
    ")": "right parenthesis",
    "[": "left bracket",
    "]": "right bracket",
    "{": "left brace",
    "}": "right brace",
    "<": "less than",
    ">": "greater than",
    "=": "equals",
    "+": "plus",
    "×": "multiply",
    "÷": "divide",
    "^": "caret",
    "!": "exclamation",
    "?": "question",
    "#": "hash",
    "$": "dollar",
    "%": "percent",
    "&": "and",
    "*": "asterisk",
}
HOMOGLYPH_MAP = {
    # Lowercase letters
    "a": "а",  # Cyrillic a (U+0430)
    "b": "β",  # Greek beta (U+03B2)
    "c": "с",  # Cyrillic es (U+0441)
    "d": "ԁ",  # Cyrillic small letter Komi de (U+0501)
    "e": "е",  # Cyrillic ie (U+0435)
    "f": "ƒ",  # Latin small letter f with hook (U+0192)
    "g": "ɡ",  # Latin small letter script g (U+0261)
    "h": "һ",  # Cyrillic shha (U+04BB)
    "i": "і",  # Cyrillic byelorussian-ukrainian i (U+0456)
    "j": "ј",  # Cyrillic je (U+0458)
    "k": "к",  # Cyrillic ka (U+043A)
    "l": "ӏ",  # Cyrillic el (U+04CF)
    "m": "м",  # Cyrillic em (U+043C)
    "n": "п",  # Cyrillic pe (U+043F)
    "o": "о",  # Cyrillic o (U+043E)
    "p": "р",  # Cyrillic er (U+0440)
    "q": "ԛ",  # Cyrillic small letter qa (U+051B)
    "r": "ｒ",  # Fullwidth r (U+FF52)
    "s": "ѕ",  # Cyrillic dze (U+0455)
    "t": "т",  # Cyrillic te (U+0442)
    "u": "ｕ",  # Fullwidth u (U+FF55)
    "v": "ν",  # Greek nu (U+03BD)
    "w": "ԝ",  # Cyrillic small letter we (U+051D)
    "x": "х",  # Cyrillic ha (U+0445)
    "y": "у",  # Cyrillic u (U+0443)
    "z": "ｚ",  # Fullwidth z (U+FF5A)

    # Uppercase letters
    "A": "А",  # Cyrillic A (U+0410)
    "B": "В",  # Cyrillic Ve (U+0412)
    "C": "С",  # Cyrillic Es (U+0421)
    "D": "Ｄ",  # Fullwidth D (U+FF24)
    "E": "Е",  # Cyrillic IE (U+0415)
    "F": "Ｆ",  # Fullwidth F (U+FF26)
    "G": "Ｇ",  # Fullwidth G (U+FF27)
    "H": "Н",  # Cyrillic En (U+041D)
    "I": "І",  # Ukrainian I (U+0406)
    "J": "Ј",  # Cyrillic Je (U+0408)
    "K": "К",  # Cyrillic Ka (U+041A)
    "L": "Ｌ",  # Fullwidth L (U+FF2C)
    "M": "М",  # Cyrillic Em (U+041C)
    "N": "Ｎ",  # Fullwidth N (U+FF2D)
    "O": "О",  # Cyrillic O (U+041E)
    "P": "Р",  # Cyrillic Er (U+0420)
    "Q": "Ԛ",  # Cyrillic capital letter Qa (U+051A)
    "R": "Ｒ",  # Fullwidth R (U+FF32)
    "S": "Ѕ",  # Cyrillic Dze (U+0405)
    "T": "Т",  # Cyrillic Te (U+0422)
    "U": "Ｕ",  # Fullwidth U (U+FF35)
    "V": "Ｖ",  # Fullwidth V (U+FF36)
    "W": "Ԝ",  # Cyrillic capital letter We (U+051C)
    "X": "Х",  # Cyrillic Ha (U+0425)
    "Y": "Υ",  # Greek Upsilon (U+03A5)
    "Z": "Ζ",  # Greek Zeta (U+0396)

    # Digits
    "0": "０",  # Fullwidth 0 (U+FF10)
    "1": "１",  # Fullwidth 1 (U+FF11)
    "2": "２",  # Fullwidth 2 (U+FF12)
    "3": "３",  # Fullwidth 3 (U+FF13)
    "4": "４",  # Fullwidth 4 (U+FF14)
    "5": "５",  # Fullwidth 5 (U+FF15)
    "6": "６",  # Fullwidth 6 (U+FF16)
    "7": "７",  # Fullwidth 7 (U+FF17)
    "8": "８",  # Fullwidth 8 (U+FF18)
    "9": "９",  # Fullwidth 9 (U+FF19)

    # Symbols
    "@": "＠",  # Fullwidth at sign (U+FF20)
    "#": "＃",  # Fullwidth number sign (U+FF03)
    "$": "＄",  # Fullwidth dollar sign (U+FF04)
    "%": "％",  # Fullwidth percent sign (U+FF05)
    "&": "＆",  # Fullwidth ampersand (U+FF06)
    "*": "＊",  # Fullwidth asterisk (U+FF0A)
    "-": "－",  # Fullwidth hyphen-minus (U+FF0D)
    "=": "＝",  # Fullwidth equals sign (U+FF1D)
    "+": "＋",  # Fullwidth plus sign (U+FF0B)
    "/": "／",  # Fullwidth solidus (U+FF0F)
    ":": "：",  # Fullwidth colon (U+FF1A)
    ";": "；",  # Fullwidth semicolon (U+FF1B)
    "<": "＜",  # Fullwidth less-than sign (U+FF1C)
    ">": "＞",  # Fullwidth greater-than sign (U+FF1E)
    "?": "？",  # Fullwidth question mark (U+FF1F)
    "!": "！",  # Fullwidth exclamation mark (U+FF01)
    ".": "．",  # Fullwidth full stop (U+FF0E)
    ",": "，",  # Fullwidth comma (U+FF0C)
    "'": "＇",  # Fullwidth apostrophe (U+FF07)
    '"': "＂",  # Fullwidth quotation mark (U+FF02)
    "(": "（",  # Fullwidth left parenthesis (U+FF08)
    ")": "）",  # Fullwidth right parenthesis (U+FF09)
    "[": "［",  # Fullwidth left square bracket (U+FF3B)
    "]": "］",  # Fullwidth right square bracket (U+FF3D)
    "{": "｛",  # Fullwidth left curly bracket (U+FF5B)
    "}": "｝",  # Fullwidth right curly bracket (U+FF5D)
}
SEPARATORS = ["-", "_", ".", " ", "/", ":", ";", ","]
PARENTHESES = ["(", ")", "[", "]", "{", "}"]
