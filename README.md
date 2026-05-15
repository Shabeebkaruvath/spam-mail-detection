# 📧 Fuzzy Logic Spam Classifier

A rule-based spam detection system using fuzzy logic to classify emails as **SPAM** or **HAM** with confidence scoring.

---

## Table of Contents

1. [Training Phase](#1-training-phase)
2. [Feature Extraction](#2-feature-extraction)
3. [Threshold Learning](#3-threshold-learning)
4. [Fuzzy System Construction](#4-fuzzy-system-construction)
5. [Fuzzy Rules](#5-fuzzy-rules)
6. [Classification](#6-classification)
7. [Decision Rule](#7-decision-rule)

---

## 1. Training Phase

> `train()` — Reads the dataset (`spam.csv`) and learns spam patterns.

### Steps

1. **Load dataset** — Each row maps to a `(label, email_text)` pair where label is either `spam` or `ham`.
2. **Separate texts** — Splits emails into `spam_texts` and `ham_texts`.
3. **Word frequency calculation** — Uses `collections.Counter` to count word occurrences in each category.
4. **Find spam keywords** — A word is flagged as a spam keyword if it meets both conditions:
   - Appears **≥ 8 times** in spam emails
   - Has a **Spam Ratio ≥ 5**, calculated as:


### Example Keywords Learned

| Keyword  |
|----------|
| `free`   |
| `win`    |
| `prize`  |
| `offer`  |
| `click`  |
| `urgent` |

---

## 2. Feature Extraction

> `raw_features()` / `extract_features()` — Converts each email into numerical features normalized to the **0–1 range**.

### Features

| Feature               | Description                          |
|-----------------------|--------------------------------------|
| `spam_kw_score`       | Ratio of spam keywords in the message |
| `uppercase_ratio`     | Percentage of uppercase letters       |
| `exclamation_density` | Number of `!` marks per word          |
| `digit_ratio`         | Percentage of digit characters        |
| `url_score`           | Number of links detected              |

### Example

**Input email:**
```
WIN FREE PRIZE!!! Click http://offer.com now
```

**Output features:**
```
spam_kw_score       = 0.40
uppercase_ratio     = 0.35
exclamation_density = 0.30
digit_ratio         = 0.00
url_score           = 0.33
```

---

## 3. Threshold Learning

For each feature, percentile boundaries are calculated separately from ham and spam emails to define fuzzy decision zones.

| Source       | Percentiles Computed      |
|--------------|---------------------------|
| Ham emails   | `ham_p75`, `ham_p90`      |
| Spam emails  | `spam_p10`, `spam_p25`    |

### Example — `uppercase_ratio`

```
ham_p75  = 0.05
ham_p90  = 0.08
spam_p10 = 0.10
spam_p25 = 0.18
```

---

## 4. Fuzzy System Construction

> `build_fuzzy_system()` — Uses `skfuzzy` to model uncertainty in classification.

### Input Variables (Antecedents)

Each of the five features becomes a fuzzy variable with two membership sets:

| Membership Set | Shape        | Represents     |
|----------------|--------------|----------------|
| `LOW`          | Trapezoidal  | Ham-like values |
| `HIGH`         | Trapezoidal  | Spam-like values |

**Conceptual membership graph:**
```
LOW         HIGH
 |--\    /--|
 0  0.3 0.7  1
```

### Output Variable (Consequent)

**`spam_score`** — ranges from **0 to 100**

| Class  | Range   | Membership Function        |
|--------|---------|----------------------------|
| `HAM`  | 0 – 50  | `trimf(0, 25, 50)`         |
| `SPAM` | 50 – 100| `trimf(50, 75, 100)`       |

---

## 5. Fuzzy Rules

Rules encode human-like reasoning to infer spam probability.

### 🟢 HAM Rules

```
IF keyword_score=LOW  AND uppercase=LOW AND exclamation=LOW  → HAM
IF keyword_score=LOW  AND url=LOW                            → HAM
```

### 🔴 SPAM Rules

```
IF keyword_score=HIGH                                        → SPAM
IF keyword_score=HIGH AND url=HIGH                          → SPAM
IF uppercase=HIGH     AND exclamation=HIGH AND url=HIGH     → SPAM
```

---

## 6. Classification

> `classify()` — Runs a full fuzzy inference pipeline on a user-provided email.

### Pipeline

```
Email Input
    │
    ▼
① Extract Features
    │
    ▼
② Feed into Fuzzy System
   sim.input[feature] = value
    │
    ▼
③ Fuzzy Inference
   ┌─────────────────────────────┐
   │  Fuzzification              │
   │       → Rule Evaluation     │
   │           → Aggregation     │
   │               → Defuzz.     │
   └─────────────────────────────┘
    │
    ▼
④ Output: spam_score (0–100)
```

---

## 7. Decision Rule

### Classification Threshold

| Condition       | Result |
|-----------------|--------|
| `score >= 50`   | 🔴 SPAM |
| `score < 50`    | 🟢 HAM  |

### Confidence Levels

| Condition              | Confidence |
|------------------------|------------|
| `\|score - 50\| > 25`  | High       |
| `\|score - 50\| > 10`  | Medium     |
| Otherwise              | Low        |

### Example Output

```
RESULT     : SPAM
SPAM SCORE : 78.3 / 100
CONFIDENCE : High
```