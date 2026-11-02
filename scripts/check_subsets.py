"""Check that your copy of MMBench reproduces the paper's question subsets.

Re-runs the exact sampling of saif.data.load_mmbench (perception filter, then
pandas .sample(n, random_state=seed)) without decoding any images, and compares
the selected `index` values with data/question_ids/. Also checks that the
CircularEval subset is the one extract_circular.py draws.

Usage:
  python scripts/check_subsets.py --tsv data/MMBench_DEV_EN.tsv
Requires: pandas numpy
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saif.data import PERCEPTION_L2, _norm  # noqa: E402


def read_ids(name):
    lines = (ROOT / "data" / "question_ids" / name).read_text().splitlines()
    return [int(x) for x in lines if x and not x.startswith("#")]


def mmbench_subset(tsv, n, seed=0):
    df = pd.read_csv(tsv, sep="\t", usecols=lambda c: c != "image")
    l2col = next(c for c in df.columns if _norm(c) in ("l2_category", "l2category"))
    l2n = df[l2col].map(_norm)
    df = df[l2n.isin({_norm(c) for c in PERCEPTION_L2}) | l2n.str.contains("perception", regex=False)]
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df["index"].astype(int).tolist()   # in sampling (file) order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="data/MMBench_DEV_EN.tsv")
    a = ap.parse_args()
    ok = True
    for n, fname in ((2000, "mmbench_2000.txt"), (1000, "mmbench_1000.txt")):
        got = mmbench_subset(a.tsv, n)
        match = sorted(got) == sorted(read_ids(fname))
        ok &= match
        print(f"MMBench n={n}: {'OK' if match else 'MISMATCH'} ({len(got)} questions vs {fname})")
        if n == 2000:
            # extract_circular.py: RandomState(0) permutation of the 2,000-question list, first 500
            order = np.random.RandomState(0).permutation(len(got))[:500]
            circ = sorted(got[i] for i in order)
            m = circ == sorted(read_ids("mmbench_circular_500.txt"))
            ok &= m
            print(f"CircularEval subset: {'OK' if m else 'MISMATCH'} (500 questions)")
    print("all subsets match" if ok else "MISMATCH -- check your MMBench TSV version")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
