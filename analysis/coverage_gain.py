"""coverage_gain.py -- Table 4: does an internal-signal combiner change attainable coverage?

A +0.0076 AUROC gain is only "practically meaningful" if it buys coverage at a
low target risk. For each model this compares CONFIDENCE-ONLY selection with a
CLEAN-TRAINED (confidence + best signal) combiner, and reports the coverage
attainable at target selective-risk tolerances on worst-severity held-out data.

Setup matches gain_beyond_confidence.py: RandomState(0), 50/50 split of clean-condition
question indices; both combiners are logistic, fit on the CLEAN calibration
half; evaluation is on the held-out half of severity-5 (pooled) corrupted
questions. Bootstrap CI on the coverage delta (2000 draws, seed 0).

Usage:
  python coverage_gain.py --root outputs --alphas 0.05 0.10
Requires: pandas numpy scikit-learn pyarrow
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 0
SIGNALS = [("lens_agree_frac", False), ("attn_img_entropy", True), ("attn_img_mass", False)]
CLUSTER = False  # set by --cluster: question-clustered bootstrap
DISP = {"lens_agree_frac": "logit-lens", "attn_img_entropy": "attn.ent", "attn_img_mass": "attn.mass"}


def coverage_at_risk(score, correct, alpha):
    order = np.argsort(-score)
    c = correct[order].astype(float)
    risk = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(risk <= alpha)[0]
    return (ok.max() + 1) / len(c) if len(ok) else 0.0


def fit_scores(fit, ev, sig, sgn):
    cf = LogisticRegression(max_iter=2000).fit(fit[["conf"]].values, fit["correct"].values)
    bf = LogisticRegression(max_iter=2000).fit(
        np.column_stack([fit["conf"].values, sgn * fit[sig].values]), fit["correct"].values)
    pc = cf.predict_proba(ev[["conf"]].values)[:, 1]
    pb = bf.predict_proba(np.column_stack([ev["conf"].values, sgn * ev[sig].values]))[:, 1]
    return pc, pb


def analyse(big, name, alphas):
    if "severity" not in big.columns or not (big.severity == 0).any():
        print(f"=== {name} ===  no clean baseline; skipped.\n"); return
    rng = np.random.RandomState(SEED)
    clean_idx = big[big.severity == 0]["index"].unique()
    calib = set(rng.permutation(clean_idx)[: len(clean_idx) // 2])
    big = big.copy(); big["calib"] = big["index"].isin(calib)
    maxsev = int(big.severity.max())
    fit = big[(big.severity == 0) & (big.calib)]           # clean calib
    ev = big[(big.severity == maxsev) & (~big.calib)]      # worst-sev held-out
    sigs = [(s, n) for (s, n) in SIGNALS if s in big.columns]

    print(f"=== {name} ===  (fit on clean calib, eval on sev{maxsev} held-out)")
    for a in alphas:
        base = None; best = (-9, None, None, None)
        for sig, neg in sigs:
            f = fit[np.isfinite(fit["conf"]) & np.isfinite(fit[sig]) & np.isfinite(fit["correct"])]
            e = ev[np.isfinite(ev["conf"]) & np.isfinite(ev[sig]) & np.isfinite(ev["correct"])].reset_index(drop=True)
            if len(f) < 30 or len(e) < 30:
                continue
            pc, pb = fit_scores(f, e, sig, -1.0 if neg else 1.0)
            y = e["correct"].values.astype(float)
            cov_c = coverage_at_risk(pc, y, a); cov_b = coverage_at_risk(pb, y, a)
            if base is None:
                base = cov_c
            if (cov_b - cov_c) > best[0]:
                # bootstrap CI on the coverage delta (question-clustered if --cluster)
                deltas = []
                if CLUSTER:
                    qid = e["index"].values
                    uniq = np.unique(qid); pos = {u: np.where(qid == u)[0] for u in uniq}
                    for _ in range(2000):
                        draw = rng.choice(uniq, size=len(uniq), replace=True)
                        idx = np.concatenate([pos[u] for u in draw]); yb = y[idx]
                        deltas.append(coverage_at_risk(pb[idx], yb, a) - coverage_at_risk(pc[idx], yb, a))
                else:
                    for _ in range(2000):
                        idx = rng.randint(0, len(e), len(e)); yb = y[idx]
                        deltas.append(coverage_at_risk(pb[idx], yb, a) - coverage_at_risk(pc[idx], yb, a))
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                best = (cov_b - cov_c, sig, cov_c, cov_b, lo, hi)
        if base is None:
            print(f"  risk<= {a:.0%}: insufficient data"); continue
        d, sig, cc, cb, lo, hi = best[0], best[1], best[2], best[3], best[4], best[5]
        print(f"  risk<= {a:4.0%}: conf-only coverage {cc:5.1%} | +{DISP.get(sig,sig):9s} coverage {cb:5.1%}"
              f" | delta {d:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
    print()


def find_dirs(args):
    if args.dirs:
        return [d for d in args.dirs if os.path.isdir(d)]
    roots = [args.root] if args.root else [".", "outputs", "./outputs"]
    seen, out = set(), []
    for r in roots:
        for d in glob.glob(os.path.join(r, "features_*")):
            rp = os.path.realpath(d)
            if os.path.isdir(d) and rp not in seen and "fake" not in d.lower():
                seen.add(rp); out.append(d)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None); ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.10])
    ap.add_argument("--cluster", action="store_true", help="question-clustered bootstrap")
    args = ap.parse_args()
    global CLUSTER; CLUSTER = args.cluster
    dirs = find_dirs(args)
    if not dirs:
        print("No feature dirs found."); sys.exit(1)
    print("Coverage at target selective-risk: confidence-only vs best clean-trained")
    print("(confidence + signal) combiner, on worst-severity held-out questions.\n")
    for d in dirs:
        pqs = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not pqs: continue
        big = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        analyse(big, os.path.basename(d).replace("features_", ""), args.alphas)


if __name__ == "__main__":
    main()
