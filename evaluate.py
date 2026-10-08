"""
evaluate.py - test model.joblib on a FRESH set that was not used in training.

    python evaluate.py --train-csv data/train.csv

Fresh set = the current OpenPhish feed (phishing) + random Tranco URLs (benign).
Anything whose exact URL OR registered domain appears in the training CSV is
removed, so a model that merely memorized training domains gets no credit.

For this to be meaningful, run it some days after training: the OpenPhish feed
only lists currently-active URLs, and right after fetch_data.py most of it
would be removed as duplicates.

Offline use: pass --phish-file / --benign-file (one URL per line).

What the numbers do and do not mean (also printed at the end):
  * Metrics cover the ML model alone, not the brand/domain-age rules in server.py.
  * Benign URLs from Tranco are bare homepages, which flatters the false-positive
    rate compared with real browsing.
  * Precision depends on the class mix. Real browsing is overwhelmingly benign,
    so real-world precision would be lower than shown on a balanced set.
"""
import argparse
import random

import joblib
import numpy as np
import pandas as pd

from datasets import fetch_openphish, fetch_tranco
from features import extract_features_vector, parse_url, registered_domain
from metrics import compute_metrics, print_metrics


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def dedupe_against_training(urls, train_urls, train_domains, label):
    seen, kept, dropped_exact, dropped_domain = set(), [], 0, 0
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        if u in train_urls:
            dropped_exact += 1
        elif registered_domain(parse_url(u)[2]) in train_domains:
            dropped_domain += 1
        else:
            kept.append(u)
    print(f"{label}: kept {len(kept)} | removed {dropped_exact} exact-URL and {dropped_domain} same-domain matches with training data")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--train-csv", required=True, help="the CSV the model was trained on (for deduplication)")
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--phish-file")
    ap.add_argument("--benign-file")
    ap.add_argument("--benign-n", type=int, default=0, help="benign count (default: same as phishing)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7],
                    help="0.7 matches the extension's block threshold (score >= 70)")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    bundle = joblib.load(args.model)
    model = bundle["model"]
    print(f"Model trained {bundle.get('trained_at', '?')} on {bundle.get('n_train', '?')} URLs")

    train_urls = set(pd.read_csv(args.train_csv)[args.url_col].astype(str).str.strip())
    train_domains = {registered_domain(parse_url(u)[2]) for u in train_urls}

    phish_raw = read_lines(args.phish_file) if args.phish_file else fetch_openphish()
    if args.benign_file:
        benign_raw = read_lines(args.benign_file)
    else:
        rows = fetch_tranco(100_000)
        benign_raw = [f"https://{d}/" for _, d in rows]
        rnd.shuffle(benign_raw)

    phish = dedupe_against_training(phish_raw, train_urls, train_domains, "Phishing")
    benign = dedupe_against_training(benign_raw, train_urls, train_domains, "Benign  ")
    if not phish:
        raise SystemExit("No fresh phishing URLs left after deduplication. Wait a few days and retry.")
    benign = benign[: args.benign_n or len(phish)]
    print(f"\nFresh evaluation set: {len(phish)} phishing + {len(benign)} benign")

    urls = phish + benign
    y = np.array([1] * len(phish) + [0] * len(benign))
    X = np.array([extract_features_vector(u) for u in urls])
    pos = list(model.classes_).index(1)
    proba = model.predict_proba(X)[:, pos]

    for t in args.thresholds:
        print_metrics(compute_metrics(y, (proba >= t).astype(int)), f"Fresh set @ threshold {t}")

    first = (proba >= args.thresholds[0]).astype(int)
    fp = [i for i in range(len(y)) if y[i] == 0 and first[i] == 1]
    fn = [i for i in range(len(y)) if y[i] == 1 and first[i] == 0]
    print(f"\nMisclassified @ {args.thresholds[0]}: {len(fp)} false positives, {len(fn)} false negatives")
    for name, idx in (("False positives (benign flagged as phishing)", fp),
                      ("False negatives (phishing that was missed)", fn)):
        print(f"\n{name} - showing up to {args.examples}:")
        for i in sorted(idx, key=lambda k: -abs(proba[k] - y[k]))[: args.examples]:
            print(f"  p(phish)={proba[i]:.2f}  {urls[i][:110]}")

    print(
        "\nCaveats: model-only metrics; benign set = bare homepages (optimistic FPR);\n"
        "precision here assumes a balanced mix, real traffic has far fewer phishing URLs."
    )


if __name__ == "__main__":
    main()
