## Content based Adversarial Attacks

---
### 1. Affix Attack

#### Description
An affix attack is a type of adversarial attack that adds an adversarial content before
or after PIIs.

#### Example
| Original Content                                  | Adversarial Content                                                               |
|---------------------------------------------------|-----------------------------------------------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my credit card number is `NULL. This is my lucky number` 4373024900209025. |

---

### 2. Emojify Supportive Context Attack

#### Description
An emojify attack is a type of adversarial attack that aims to reduce the context awareness
of detectors by replacing *supportive context words* with emojis.
This attack is particularly effective against Named Entity Recognition (NER) models,
that are used to increase the confidence of the detector.

#### Example
| Original Content                                  | Adversarial Content                        |
|---------------------------------------------------|--------------------------------------------|
| Hello, my credit card number is 4373024900209025. | Hello, my `💳` number is 4373024900209025. |

---
