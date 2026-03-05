import re, sys, csv, collections
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def train(csv_path):
    texts, labels = [], []
    with open(csv_path, newline='', encoding='latin-1') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[0] in ('spam', 'ham'):
                labels.append(row[0])
                texts.append(row[1])

    spam_texts = [t for t, l in zip(texts, labels) if l == 'spam']
    ham_texts  = [t for t, l in zip(texts, labels) if l == 'ham']

    spam_wc = collections.Counter(w for t in spam_texts for w in re.findall(r'\b\w+\b', t.lower()))
    ham_wc  = collections.Counter(w for t in ham_texts  for w in re.findall(r'\b\w+\b', t.lower()))

    sn, hn = len(spam_texts), len(ham_texts)
    spam_keywords = {w for w, c in spam_wc.items()
                     if c >= 8 and (c/sn) / ((ham_wc[w]/hn) + 0.005) >= 5.0}

    def raw_features(text, kws):
        words = re.findall(r'\b\w+\b', text.lower())
        tc, tw = max(len(text), 1), max(len(words), 1)
        kh = sum(1 for w in words if w in kws)
        kh += sum(1 for b in [" ".join(words[i:i+2]) for i in range(len(words)-1)] if b in kws)
        urls = re.findall(r'http[s]?://\S+|www\.\S+', text.lower())
        return {
            'spam_kw_score':       min(kh / tw, 1.0),
            'uppercase_ratio':     sum(c.isupper() for c in text) / tc,
            'exclamation_density': min(text.count('!') / tw, 1.0),
            'digit_ratio':         sum(c.isdigit() for c in text) / tc,
            'url_score':           min(len(urls) / 3.0, 1.0),
        }

    thresholds = {}
    for feature in ['spam_kw_score', 'uppercase_ratio', 'exclamation_density', 'digit_ratio', 'url_score']:
        sv = [raw_features(t, spam_keywords)[feature] for t in spam_texts]
        hv = [raw_features(t, spam_keywords)[feature] for t in ham_texts]
        thresholds[feature] = {
            'ham_p75':  float(np.percentile(hv, 75)),
            'ham_p90':  float(np.percentile(hv, 90)),
            'spam_p10': float(np.percentile(sv, 10)),
            'spam_p25': float(np.percentile(sv, 25)),
        }

    print(f"  Learned {len(spam_keywords)} keywords from {sn} spam / {hn} ham emails.")
    return spam_keywords, thresholds

def build_fuzzy_system(thresholds):
    ants = {k: ctrl.Antecedent(np.arange(0, 1.01, 0.01), k) for k in thresholds}
    out  = ctrl.Consequent(np.arange(0, 101, 1), 'spam_score')

    for k, t in thresholds.items():
        lc = max(t['ham_p75'], 0.01)
        ld = max(t['ham_p90'], lc + 0.01)
        ha = max(t['spam_p10'], lc)
        hb = max(t['spam_p25'], ha + 0.01)
        ants[k]['low']  = fuzz.trapmf(ants[k].universe, [0, 0, lc, ld])
        ants[k]['high'] = fuzz.trapmf(ants[k].universe, [ha, hb, 1, 1])

    # Only two output regions — no borderline
    out['ham']  = fuzz.trimf(out.universe, [0,  25, 50])
    out['spam'] = fuzz.trimf(out.universe, [50, 75, 100])

    kw, up, ex, dg, ur = [ants[k] for k in thresholds]
    rules = [
        # HAMLESS rules
        ctrl.Rule(kw['low'] & up['low'] & ex['low'],                         out['ham']),
        ctrl.Rule(kw['low'] & ur['low'],                                      out['ham']),
        ctrl.Rule(kw['low'] & up['low'] & ex['low'] & dg['low'] & ur['low'], out['ham']),
        ctrl.Rule(kw['low'] & up['low'],                                      out['ham']),
        ctrl.Rule(kw['low'] & ex['low'],                                      out['ham']),
        # SPAM rules
        ctrl.Rule(kw['high'],                                                 out['spam']),
        ctrl.Rule(kw['high'] & up['high'],                                    out['spam']),
        ctrl.Rule(kw['high'] & ex['high'],                                    out['spam']),
        ctrl.Rule(kw['high'] & ur['high'],                                    out['spam']),
        ctrl.Rule(kw['high'] & dg['high'],                                    out['spam']),
        ctrl.Rule(up['high'] & ex['high'] & ur['high'],                       out['spam']),
        ctrl.Rule(kw['low']  & up['high'] & ex['high'],                       out['spam']),
        ctrl.Rule(kw['low']  & dg['high'] & ur['high'],                       out['spam']),
    ]

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def extract_features(text, spam_keywords):
    words = re.findall(r'\b\w+\b', text.lower())
    tc, tw = max(len(text), 1), max(len(words), 1)
    kh = sum(1 for w in words if w in spam_keywords)
    kh += sum(1 for b in [" ".join(words[i:i+2]) for i in range(len(words)-1)] if b in spam_keywords)
    urls = re.findall(r'http[s]?://\S+|www\.\S+', text.lower())
    return {
        'spam_kw_score':       round(min(kh / tw, 1.0), 4),
        'uppercase_ratio':     round(sum(c.isupper() for c in text) / tc, 4),
        'exclamation_density': round(min(text.count('!') / tw, 1.0), 4),
        'digit_ratio':         round(sum(c.isdigit() for c in text) / tc, 4),
        'url_score':           round(min(len(urls) / 3.0, 1.0), 4),
    }

def classify(text, spam_keywords, sim):
    features = extract_features(text, spam_keywords)
    for k, v in features.items(): sim.input[k] = v
    try:
        sim.compute()
        score = sim.output['spam_score']
    except Exception:
        score = 20.0

    label      = "SPAM" if score >= 50 else "HAM"
    confidence = "High" if abs(score-50) > 25 else "Medium" if abs(score-50) > 10 else "Low"

    print("\n" + "=" * 45)
    print(f"  RESULT     : {label}")
    print(f"  SPAM SCORE : {score:.1f} / 100")
    print(f"  CONFIDENCE : {confidence}")
    print("=" * 45)

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "spam.csv"
    print("\n" + "=" * 45)
    print("  FUZZY LOGIC SPAM CLASSIFIER")
    print("=" * 45)
    print(f"\nTraining from: {csv_path}")

    spam_keywords, thresholds = train(csv_path)
    sim = build_fuzzy_system(thresholds)

    print("\nPaste your email and press Enter twice.\n" + "-" * 45)
    lines = []
    while True:
        try:    line = input()
        except: break
        if not line and lines: break
        lines.append(line)

    if lines:
        classify("\n".join(lines), spam_keywords, sim)