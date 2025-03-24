## Defenses

---
### 1. Data Processing
#### Description
The data processing defense is a technique that processes the content to remove or replace PII.

---
#### 1) Homoglyph Detection and Transformation

##### Description
A homoglyph is a character that is visually identical or nearly identical to another character.
The homoglyph detection and transformation defense is a technique that detects and replaces homoglyphs and emojis in the content.

##### Example
| Original Content                                              | Transformed Content                   |
|---------------------------------------------------------------|---------------------------------------|
| The 🏦 IBAN is 🄳🄴①②③④⑤⑥⑦⑧⑨⓪.                                | The Bank IBAN is DE1234567890.        |
| My 💳 is 4️⃣5️⃣6️⃣7️⃣-8️⃣9️⃣0️⃣1️⃣-2️⃣3️⃣4️⃣5️⃣-6️⃣7️⃣8️⃣9️⃣. | My Credit Card is 4567-8901-2345-6789 |

---
#### 2) Separator Removal

##### Description
The separator removal defense is a technique that removes separators from the content.
Unsupported separators are replaced with '-' and consecutive separators are replaced with a single separator.

##### Example
| Original Content                        | Transformed Content              |
|-----------------------------------------|----------------------------------|
| My Email is john.doe@@example.com       | My Email is john.doe@example.com |
| My Phone Number is 123-----456-----7890 | My Phone Number is 123-456-7890  |
| My SSN is 123$45$6789                   | My SSN is 123-45-6789            |
| My SSN is 123$$$$$45$$$$$6789           | My SSN is 123-45-6789            |

---
### 2. Fuzzy Matching
#### Description
The fuzzy PII matching defense is a technique that detects PII in relaxed conditions.

---
#### 1) Regex Fuzzy Matching
##### Description
for Regex based approaches, each PII type pattern can be relaxed to allow substitutions and deletions.
Hence, detecting PII in a more relaxed manner based on Levenshtein distance (without insertions).

#### Example
| Original Content                    | Spans                                                                           |
|-------------------------------------|---------------------------------------------------------------------------------|
| My Email is john.doe@@example.com   | [{"value": "john.doe@@example.com", "start": 11, "end": 32, "type": "email"}]   |
| My Phone Number is 123 and 456-7890 | [{"value": "123 and 456-7890", "start": 20, "end": 36, "type": "phone_number"}] |

---
#### 2) Fuzzy Matching Learning
##### Description
For Machine Learning based approaches, the model can be trained on relaxed PII patterns (supervised learning).

---
