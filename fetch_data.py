"""
fetch_data.py - build a starter training CSV (url,label) from public feeds.

    python fetch_data.py --out data/train.csv

Phishing: today's OpenPhish feed (label 1).  Benign: a random sample of the
Tranco top-100k domains as https://<domain>/ (label 0).

This is a STARTER set. It is small (OpenPhish lists a few hundred to a few
thousand live URLs) and has the homepage-vs-deep-link bias described in the
README. For a better model, merge it with a larger labeled dataset that
contains benign URLs WITH paths, using --extra-csv.

Run evaluate.py a few days AFTER training so the live feed has new URLs.
"""
import argparse
import os
import random

import pandas as pd

from datasets import fetch_openphish, fetch_tranco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train.csv")
    ap.add_argument("--benign-n", type=int, default=0, help="benign count (default: same as phishing count)")
    ap.add_argument("--extra-csv", help="optional CSV with url,label columns (1=phishing,0=benign) to merge")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    phish = sorted(set(fetch_openphish()))
    print(f"OpenPhish: {len(phish)} URLs")
    n_benign = args.benign_n or len(phish)
    tranco = fetch_tranco(100_000)
    sample = rnd.sample(tranco, min(n_benign, len(tranco)))
    benign = [f"https://{domain}/" for _, domain in sample]
    print(f"Tranco: {len(benign)} benign URLs sampled from the top 100k")

    df = pd.DataFrame({"url": phish + benign, "label": [1] * len(phish) + [0] * len(benign)})
    if args.extra_csv:
        extra = pd.read_csv(args.extra_csv)[["url", "label"]]
        df = pd.concat([df, extra], ignore_index=True)
        print(f"Merged {len(extra)} rows from {args.extra_csv}")
    df = df.drop_duplicates(subset="url").sample(frac=1, random_state=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print("WARNING: benign rows are bare homepages; read 'Known data bias' in the README.")


if __name__ == "__main__":
    main()
