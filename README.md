1. Training Phase (train())

Reads the dataset (spam.csv) and learns patterns.

Steps

Load dataset

Each row → (label, email_text)

Label = spam or ham.

Separate texts

spam_texts

ham_texts

Word frequency calculation

Uses collections.Counter.

Counts words appearing in spam and ham emails.

Find spam keywords

A word becomes a spam keyword if:

Appears ≥ 8 times in spam

Appears much more in spam than ham

Formula used:

𝑆
𝑝
𝑎
𝑚
𝑅
𝑎
𝑡
𝑖
𝑜
=
𝑐
/
𝑠
𝑝
𝑎
𝑚
_
𝑐
𝑜
𝑢
𝑛
𝑡
(
ℎ
𝑎
𝑚
_
𝑐
𝑜
𝑢
𝑛
𝑡
/
ℎ
𝑎
𝑚
_
𝑡
𝑜
𝑡
𝑎
𝑙
)
+
0.005
SpamRatio=
(ham_count/ham_total)+0.005
c/spam_count
	​


If ratio ≥ 5, the word is treated as spam keyword.

Example keywords learned:

free
win
prize
offer
click
urgent
2. Feature Extraction (raw_features() / extract_features())

Each email is converted into numerical features (0–1 range).

Features used:

Feature	Meaning
spam_kw_score	ratio of spam keywords in message
uppercase_ratio	% of uppercase letters
exclamation_density	number of ! per word
digit_ratio	% of digits
url_score	number of links

Example email:

WIN FREE PRIZE!!! Click http://offer.com now

Feature output example:

spam_kw_score = 0.4
uppercase_ratio = 0.35
exclamation_density = 0.3
digit_ratio = 0
url_score = 0.33
3. Threshold Learning

For each feature, the code calculates percentiles:

From ham emails

ham_p75
ham_p90

From spam emails

spam_p10
spam_p25

These define fuzzy boundaries.

Example:

uppercase_ratio
ham_p75 = 0.05
ham_p90 = 0.08
spam_p10 = 0.10
spam_p25 = 0.18
4. Fuzzy System Construction (build_fuzzy_system())

Uses skfuzzy to create fuzzy variables.

Input variables

Each feature becomes an Antecedent.

Example:

spam_kw_score
uppercase_ratio
exclamation_density
digit_ratio
url_score

Each has two fuzzy sets

LOW
HIGH

Membership functions:

LOW  = trapezoidal (ham-like values)
HIGH = trapezoidal (spam-like values)

Example graph concept:

LOW      HIGH
 |--------|
0   0.3  0.7   1
Output variable
spam_score (0–100)

Two fuzzy classes:

HAM  -> 0–50
SPAM -> 50–100

Membership functions:

HAM  = trimf(0,25,50)
SPAM = trimf(50,75,100)
5. Fuzzy Rules

Rules determine spam probability.

HAM rules

Examples:

IF keyword_score LOW AND uppercase LOW AND exclamation LOW
THEN HAM
IF keyword_score LOW AND url LOW
THEN HAM
SPAM rules

Examples:

IF keyword_score HIGH
THEN SPAM
IF keyword_score HIGH AND url HIGH
THEN SPAM
IF uppercase HIGH AND exclamation HIGH AND url HIGH
THEN SPAM

These rules mimic human reasoning.

6. Classification (classify())

When a user pastes an email:

Step 1

Extract features.

Step 2

Feed features into fuzzy system:

sim.input[feature] = value
Step 3

Fuzzy inference runs

Process:

Fuzzification
→ Rule evaluation
→ Aggregation
→ Defuzzification

Final output:

spam_score (0–100)
7. Decision Rule
score >= 50  → SPAM
score < 50   → HAM

Confidence:

|score-50| > 25 → High
|score-50| > 10 → Medium
else → Low

Example output:

RESULT     : SPAM
SPAM SCORE : 78.3 / 100
CONFIDENCE : High
 