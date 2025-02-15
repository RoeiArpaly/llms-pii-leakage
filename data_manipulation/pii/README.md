## Fuzzy PII Attacks

Fuzzy PII attacks are a type of adversarial attack that aims to make the detection of PIIs
more difficult by adding noise to the PIIs, but still keeping them recognizable by humans.

---
### 1. Emojify Attack

#### Description
An emojify attack is a type of adversarial attack that aims to make the detection of PIIs
more difficult by replacing characters in PIIs with emojis.

#### Example
| Original Content                                  | Adversarial Content                                                                 |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my credit card number is `4️⃣3️⃣7️⃣3️⃣0️⃣2️⃣4️⃣9️⃣0️⃣0️⃣2️⃣0️⃣9️⃣0️⃣2️⃣5️⃣`. |

---

### 2. Homoglyph Attack

#### Description
A homoglyph attack is a type of adversarial attack that aims to make the detection of PIIs
more difficult by replacing characters in PIIs with similar-looking characters (usually Cyrillic letters).

#### Example
| Original Content            | Adversarial Content           |
|-----------------------------|-------------------------------|
| Hello, my name is John Doe. | Hello, my name is `Јоһп Дое`. |

---

### 3. Number-to-Roman Attack

#### Description
Replaces numbers in PIIs with Roman numerals.

#### Example
| Original Content                          | Adversarial Content                                |
|-------------------------------------------|----------------------------------------------------|
| My social security number is 123-45-6789. | My social security number is `CXXIII-XLV-VIIVIII`. |

---

### 4. Number-to-Word Attack

#### Description
Replaces numbers in PIIs with their English word equivalents.

#### Example
| Original Content                 | Adversarial Content                                                    |
|----------------------------------|------------------------------------------------------------------------|
| My phone number is 555-123-4567. | My phone number is `five five five-one two three-four five six seven`. |

---

### 5. Reverse Attack

#### Description
Reverses the order of characters in PIIs.

#### Example
| Original Content                                  | Adversarial Content                                 |
|---------------------------------------------------|-----------------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my credit card number is `5029200204903734`. |

---

### 6. Separators Attack

#### Description
Adds random separators to PIIs to make them harder to detect.

#### Example
| Original Content                                  | Adversarial Content                                          |
|---------------------------------------------------|--------------------------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my credit card number is `4373;;;0249;;;0020;;;9025`. |

---
