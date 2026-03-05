import re
import sys
import csv
import collections
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TRAIN FROM CSV
#     Learns: spam keyword set + feature distribution thresholds
# ─────────────────────────────────────────────────────────────────────────────

def train(csv_path: str):
    """
    Read the CSV and return:
      - spam_keywords : set of words that strongly indicate spam
      - thresholds    : dict of learned percentile values per feature
    """
    texts, labels = [], []
    with open(csv_path, newline='', encoding='latin-1') as f:
        reader = csv.reader(f)
        next(reader)           # skip header
        for row in reader:
            if len(row) >= 2 and row[0] in ('spam', 'ham'):
                labels.append(row[0])
                texts.append(row[1])

    spam_texts = [t for t, l in zip(texts, labels) if l == 'spam']
    ham_texts  = [t for t, l in zip(texts, labels) if l == 'ham']

    # ── Learn spam keywords via spam/ham frequency ratio ──────────────────
    spam_wc = collections.Counter()
    ham_wc  = collections.Counter()
    for t in spam_texts:
        for w in re.findall(r'\b\w+\b', t.lower()):
            spam_wc[w] += 1
    for t in ham_texts:
        for w in re.findall(r'\b\w+\b', t.lower()):
            ham_wc[w] += 1

    sn, hn = len(spam_texts), len(ham_texts)
    spam_keywords = set()
    for w, cnt in spam_wc.items():
        if cnt >= 8:   # must appear in at least 8 spam messages
            ratio = (cnt / sn) / ((ham_wc.get(w, 0) / hn) + 0.005)
            if ratio >= 5.0:   # 5x more likely in spam than ham
                spam_keywords.add(w)

    # ── Learn feature thresholds from the data ────────────────────────────
    def raw_features(text, kws):
        tl    = text.lower()
        words = re.findall(r'\b\w+\b', tl)
        tc    = max(len(text), 1)
        tw    = max(len(words), 1)

        kw_hits  = sum(1 for w in words if w in kws)
        bigrams  = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
        kw_hits += sum(1 for b in bigrams if b in kws)

        urls = re.findall(r'http[s]?://\S+|www\.\S+', tl)
        return {
            'spam_kw_score':       min(kw_hits / tw, 1.0),
            'uppercase_ratio':     sum(1 for c in text if c.isupper()) / tc,
            'exclamation_density': min(text.count('!') / tw, 1.0),
            'digit_ratio':         sum(1 for c in text if c.isdigit()) / tc,
            'url_score':           min(len(urls) / 3.0, 1.0),
        }

    feat_keys = ['spam_kw_score', 'uppercase_ratio',
                 'exclamation_density', 'digit_ratio', 'url_score']

    spam_vals = {k: [] for k in feat_keys}
    ham_vals  = {k: [] for k in feat_keys}

    for t in spam_texts:
        f = raw_features(t, spam_keywords)
        for k in feat_keys: spam_vals[k].append(f[k])
    for t in ham_texts:
        f = raw_features(t, spam_keywords)
        for k in feat_keys: ham_vals[k].append(f[k])

  
    thresholds = {}
    for k in feat_keys:
        s = np.array(spam_vals[k])
        h = np.array(ham_vals[k])
        thresholds[k] = {
            'ham_p75':   float(np.percentile(h, 75)),
            'ham_p90':   float(np.percentile(h, 90)),
            'spam_p10':  float(np.percentile(s, 10)),
            'spam_p25':  float(np.percentile(s, 25)),
            'spam_p50':  float(np.percentile(s, 50)),
        }


    return spam_keywords, thresholds


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BUILD FUZZY SYSTEM using learned thresholds
# ─────────────────────────────────────────────────────────────────────────────

def build_fuzzy_system(thresholds: dict):
    """Create skfuzzy antecedents/consequent and rules from learned thresholds."""

    feat_keys = ['spam_kw_score', 'uppercase_ratio',
                 'exclamation_density', 'digit_ratio', 'url_score']

    antecedents = {}
    for k in feat_keys:
        antecedents[k] = ctrl.Antecedent(np.arange(0, 1.01, 0.01), k)

    spam_score = ctrl.Consequent(np.arange(0, 101, 1), 'spam_score')

    # ── Membership functions built from learned percentiles ───────────────
    for k, ant in antecedents.items():
        t = thresholds[k]

        # low  : flat 0 → ham_p75, falls to 0 at ham_p90
        low_a  = 0.0
        low_b  = 0.0
        low_c  = max(t['ham_p75'], 0.01)
        low_d  = max(t['ham_p90'], low_c + 0.01)

        # high : starts rising at spam_p10, flat from spam_p25 onward
        high_a = max(t['spam_p10'], low_c)
        high_b = max(t['spam_p25'], high_a + 0.01)
        high_c = 1.0
        high_d = 1.0

        # med  : peaks at midpoint between low_d and high_a
        med_peak = (low_d + high_b) / 2.0
        med_a    = low_c
        med_c    = min(high_d, 1.0)

        ant['low']  = fuzz.trapmf(ant.universe, [low_a,  low_b,  low_c,  low_d])
        ant['med']  = fuzz.trimf (ant.universe, [med_a,  med_peak, med_c])
        ant['high'] = fuzz.trapmf(ant.universe, [high_a, high_b, high_c, high_d])

    # Output regions (fixed — score meaning doesn't change)
    spam_score['ham']        = fuzz.trimf(spam_score.universe, [0,  15, 35])
    spam_score['borderline'] = fuzz.trimf(spam_score.universe, [30, 50, 70])
    spam_score['spam']       = fuzz.trimf(spam_score.universe, [65, 85, 100])

    kw = antecedents['spam_kw_score']
    up = antecedents['uppercase_ratio']
    ex = antecedents['exclamation_density']
    dg = antecedents['digit_ratio']
    ur = antecedents['url_score']

    rules = [
        # HAM
        ctrl.Rule(kw['low'] & up['low'] & ex['low'],            spam_score['ham']),
        ctrl.Rule(kw['low'] & ur['low'],                        spam_score['ham']),
        ctrl.Rule(kw['low'] & up['low'] & ex['low']
                  & dg['low'] & ur['low'],                      spam_score['ham']),
        ctrl.Rule(kw['low'] & up['med'],                        spam_score['ham']),
        # BORDERLINE
        ctrl.Rule(kw['med'] & up['low'],                        spam_score['borderline']),
        ctrl.Rule(kw['low'] & up['high'],                       spam_score['borderline']),
        ctrl.Rule(kw['med'] & ex['med'],                        spam_score['borderline']),
        ctrl.Rule(dg['high'] & ur['med'],                       spam_score['borderline']),
        ctrl.Rule(kw['med'] & ur['med'],                        spam_score['borderline']),
        # SPAM
        ctrl.Rule(kw['high'],                                   spam_score['spam']),
        ctrl.Rule(kw['high'] & up['high'],                      spam_score['spam']),
        ctrl.Rule(kw['high'] & ex['high'],                      spam_score['spam']),
        ctrl.Rule(kw['med']  & up['high'] & ex['high'],         spam_score['spam']),
        ctrl.Rule(up['high'] & ex['high'] & ur['high'],         spam_score['spam']),
        ctrl.Rule(kw['high'] & ur['high'],                      spam_score['spam']),
        ctrl.Rule(kw['med']  & dg['high'] & ur['high'],         spam_score['spam']),
        ctrl.Rule(kw['med']  & up['high'],                      spam_score['spam']),
    ]

    spam_ctrl = ctrl.ControlSystem(rules)
    sim       = ctrl.ControlSystemSimulation(spam_ctrl)
    return antecedents, spam_score, sim


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE EXTRACTION  (uses learned keyword set)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(text: str, spam_keywords: set) -> dict:
    tl    = text.lower()
    words = re.findall(r'\b\w+\b', tl)
    tc    = max(len(text), 1)
    tw    = max(len(words), 1)

    kw_hits  = sum(1 for w in words if w in spam_keywords)
    bigrams  = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    kw_hits += sum(1 for b in bigrams if b in spam_keywords)

    urls = re.findall(r'http[s]?://\S+|www\.\S+', tl)
    return {
        'spam_kw_score':       round(min(kw_hits / tw, 1.0), 4),
        'uppercase_ratio':     round(sum(1 for c in text if c.isupper()) / tc, 4),
        'exclamation_density': round(min(text.count('!') / tw, 1.0), 4),
        'digit_ratio':         round(sum(1 for c in text if c.isdigit()) / tc, 4),
        'url_score':           round(min(len(urls) / 3.0, 1.0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLASSIFY
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD = 50.0

def classify(text: str, spam_keywords: set, sim) -> None:
    features = extract_features(text, spam_keywords)

    sim.input['spam_kw_score']       = features['spam_kw_score']
    sim.input['uppercase_ratio']     = features['uppercase_ratio']
    sim.input['exclamation_density'] = features['exclamation_density']
    sim.input['digit_ratio']         = features['digit_ratio']
    sim.input['url_score']           = features['url_score']

    try:
        sim.compute()
        score = sim.output['spam_score']
    except Exception:
        score = 20.0   # no rules fired → lean HAM

    label      = "SPAM" if score >= THRESHOLD else "HAM"
    dist       = abs(score - THRESHOLD)
    confidence = "High" if dist > 25 else ("Medium" if dist > 10 else "Low")

    print("\n" + "=" * 55)
    print(f"  RESULT      : {' SPAM' if label == 'SPAM' else ' NOT SPAM'}")
    print(f"  SPAM SCORE  : {score:.1f} / 100")
    print(f"  CONFIDENCE  : {confidence}")
    print("-" * 55)
   


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "spam.csv"

    print("\n" + "=" * 55)
    print("  FUZZY LOGIC SPAM CLASSIFIER")
    print("=" * 55)
    print(f"\nTraining from: {csv_path}")

    spam_keywords, thresholds = train(csv_path)
    _, _, sim = build_fuzzy_system(thresholds)

    print("\nPaste your email below and press Enter twice.")
    print("-" * 55)

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)

    if lines:
        classify("\n".join(lines), spam_keywords, sim)