"""
train.py - train the PhishGuard RandomForest on a labeled URL CSV.

Usage:
    python train.py --data data/train.csv

The CSV needs one URL column and one label column (default names: url, label;
override with --url-col / --label-col). Which label value means "phishing" is
set with --phishing-values (default: 1,phishing,bad,malicious,phish). The label
column must contain exactly two distinct values so nothing is silently
mislabeled.

The test set is held out BY REGISTERED DOMAIN: all URLs from one domain land on
the same side of the split. A plain random split would put sibling URLs from
the same phishing campaign in both train and test and inflate the scores.
"""
import argparse
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from features import FEATURE_NAMES, extract_features_vector, parse_url, registered_domain
from metrics import compute_metrics, print_metrics


def load_dataset(path, url_col, label_col, phishing_values):
    df = pd.read_csv(path)
    for col in (url_col, label_col):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found in {path}. Columns present: {list(df.columns)}")
    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label_raw"]
    df["url"] = df["url"].astype(str).str.strip()
    df["label_raw"] = df["label_raw"].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset="url")

    values = sorted(df["label_raw"].unique())
    if len(values) != 2:
        sys.exit(f"Expected exactly 2 distinct label values, found {values}")
    wanted = {v.strip().lower() for v in phishing_values.split(",")}
    phish_found = [v for v in values if v in wanted]
    if len(phish_found) != 1:
        sys.exit(f"Could not tell which of {values} means phishing. Use --phishing-values.")
    df["label"] = (df["label_raw"] == phish_found[0]).astype(int)
    print(f"Label mapping: '{phish_found[0]}' -> phishing(1), the other value -> benign(0)")
    return df[["url", "label"]].reset_index(drop=True)


def warn_about_path_shortcut(df):
    """Datasets built from bare domain lists (e.g. Tranco) for benign and full
    URLs for phishing teach the model 'has a path = phishing'. Flag it."""
    def has_path(u):
        p = parse_url(u)[1].path
        return len(p) > 1

    rates = df.assign(p=df["url"].map(has_path)).groupby("label")["p"].mean()
    if 0 in rates and 1 in rates and abs(rates[1] - rates[0]) > 0.4:
        print(
            f"\nWARNING: {rates[1]:.0%} of phishing URLs have a path vs {rates[0]:.0%} of benign URLs.\n"
            "  The model may learn 'has a path = phishing', which would produce many false\n"
            "  positives on normal deep links. Add benign URLs that contain paths.\n"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="labeled CSV")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--phishing-values", default="1,phishing,bad,malicious,phish")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_dataset(args.data, args.url_col, args.label_col, args.phishing_values)
    print(f"Loaded {len(df)} unique URLs: {int(df.label.sum())} phishing, {int((1 - df.label).sum())} benign")
    if df.label.nunique() < 2 or len(df) < 200:
        sys.exit("Need both classes and at least ~200 URLs to train something meaningful.")
    warn_about_path_shortcut(df)

    print("Extracting features...")
    X = np.array([extract_features_vector(u) for u in df["url"]])
    y = df["label"].to_numpy()
    groups = df["url"].map(lambda u: registered_domain(parse_url(u)[2])).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    print(f"Train: {len(train_idx)} URLs | Held-out test: {len(test_idx)} URLs (disjoint domains)")

    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=args.seed,
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    metrics = compute_metrics(y_te, pred)
    print_metrics(metrics, "Held-out test set (threshold 0.5)")

    print("\nTop feature importances:")
    for i in np.argsort(-model.feature_importances_)[:10]:
        print(f"  {FEATURE_NAMES[i]:<22} {model.feature_importances_[i]:.3f}")

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "holdout_metrics": metrics,
        },
        args.out,
    )
    print(f"\nSaved {args.out}")
    print("Next: run evaluate.py on fresh URLs for a more honest number than the hold-out above.")


if __name__ == "__main__":
    main()
