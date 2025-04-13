## Fuzzy PII Attacks

Fuzzy PII attacks are a type of adversarial attack that aims to make the detection of PIIs
more difficult by adding noise to the PIIs, but still keeping them recognizable by humans.

---
### 1. Chunking Attack
#### Description
A chunking attack is a type of adversarial attacks that aims to degrade the performance of PII detection systems,<br>
by splitting PIIs into chunks of information.

#### Example
| Original Content                                  | Adversarial Content                                        |
|---------------------------------------------------|------------------------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my credit card number is `"43730249" + "00209025"`. |

---

### 2. Emojify Attack

#### Description
An emojify attack is a type of adversarial attack that aims to make the detection of PIIs
more difficult by replacing characters in PIIs with emojis.

#### Example
| Original Content                                  | Adversarial Content                                                                 |
|---------------------------------------------------|-------------------------------------------------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my credit card number is `4️⃣3️⃣7️⃣3️⃣0️⃣2️⃣4️⃣9️⃣0️⃣0️⃣2️⃣0️⃣9️⃣0️⃣2️⃣5️⃣`. |

---

### 3. Homoglyph Attack

#### Description
A homoglyph attack is a type of adversarial attack that aims to degrade PIIs detection performance.<br>
This is done by replacing characters in PIIs with similar-looking characters (homoglyphs).<br>
e.g. using Cyrillic characters that look like Latin characters.

#### Example
| Original Content            | Adversarial Content           |
|-----------------------------|-------------------------------|
| Hello, my name is John Doe. | Hello, my name is `Јоһп Ｄое`. |

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
Reverses the order of characters in PIIs. This attack is effective for certain types of PIIs such as email,
And less effective when human cannot recognize the reversed PII without knowing it is reversed.

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

### 7. Word Symbols Attack
#### Description
Replaces symbols in PIIs with their word equivalents.

#### Example
| Original Content                      | Adversarial Content                                 |
|---------------------------------------|-----------------------------------------------------|
| Hello, my email is john.doe@gmail.com | Hello, my email is `john(dot)doe(at)gmail(dot)com`. |

---
