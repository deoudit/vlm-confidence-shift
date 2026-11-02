"""gain_beyond_confidence.py -- Tables 5-7: best AUROC gain beyond confidence.

Adds, versus compute_ci.py:
  (#2) corrected one-sided p-value p = (1 + #{M_b >= obs}) / (nperm + 1), and the
       raw exceedance count, so p is never reported as 0 (Phipson & Smyth 2010).
  (#2) --nbins to sweep the confidence-bin count (run 5 / 10 / 20 to check the
       InternVL3 result is robust).
  (#8) single-condition dirs (DisasterVQA) now get a bootstrap CI on the best
       gain via a 50/50 question split -- fills the missing Table 6 uncertainty.

Everything else (seed 0, 50/50 clean-index split, clean-trained combiner,
within-confidence-bin permutation, max-statistic null) is unchanged.

Usage:
  python gain_beyond_confidence.py --root outputs --nperm 2000 --nbins 10
  python gain_beyond_confidence.py --dirs outputs/features_internvl3_2b --nperm 2000 --nbins 5
  python gain_beyond_confidence.py --dirs outputs/features_internvl3_2b --nperm 2000 --nbins 20
Requires: pandas numpy scikit-learn pyarrow
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

SIGNALS = [("lens_agree_frac", False), ("attn_img_entropy", True), ("attn_img_mass", False)]
DISP = {"lens_agree_frac": "logit-lens", "attn_img_entropy": "attn. entropy", "attn_img_mass": "attn. mass"}
SCALE = False  # set by --scale: standardize [conf, signal] before the logistic fit


def _lr():
    lr = LogisticRegression(max_iter=2000)
    return make_pipeline(StandardScaler(), lr) if SCALE else lr


def _clean(df, sig):
    m = (np.isfinite(df["conf"].values) & np.isfinite(df[sig].values) & np.isfinite(df["correct"].values))
    return df[m]


def _delta(fit, ev, sig, sgn):
    if len(fit) < 30 or len(ev) < 30:
        return np.nan
    yv = ev["correct"].values
    if yv.min() == yv.max():
        return np.nan
    cf = _lr().fit(fit[["conf"]].values, fit["correct"].values)
    bf = _lr().fit(
        np.column_stack([fit["conf"].values, sgn * fit[sig].values]), fit["correct"].values)
    ac = roc_auc_score(yv, cf.predict_proba(ev[["conf"]].values)[:, 1])
    ab = roc_auc_score(yv, bf.predict_proba(np.column_stack([ev["conf"].values, sgn * ev[sig].values]))[:, 1])
    return ab - ac


def _perm_within_confbins(vals, conf, rng, nbins):
    out = vals.copy(); order = np.argsort(conf)
    edges = np.linspace(0, len(conf), nbins + 1).astype(int)
    for i in range(nbins):
        idx = order[edges[i]:edges[i + 1]]
        if len(idx) > 1:
            out[idx] = rng.permutation(vals[idx])
    return out


def _boot_ci(fit, ev, sig, sgn, nboot, rng):
    ev = ev.reset_index(drop=True)
    cf = _lr().fit(fit[["conf"]].values, fit["correct"].values)
    bf = _lr().fit(
        np.column_stack([fit["conf"].values, sgn * fit[sig].values]), fit["correct"].values)
    pc = cf.predict_proba(ev[["conf"]].values)[:, 1]
    pb = bf.predict_proba(np.column_stack([ev["conf"].values, sgn * ev[sig].values]))[:, 1]
    y = ev["correct"].values; boot = []
    for _ in range(nboot):
        idx = rng.randint(0, len(ev), len(ev)); yb = y[idx]
        if yb.min() != yb.max():
            boot.append(roc_auc_score(yb, pb[idx]) - roc_auc_score(yb, pc[idx]))
    return np.percentile(boot, [2.5, 97.5])


def analyse_grid(big, name, margin, nperm, nboot, nbins):
    signals = [(s, n) for (s, n) in SIGNALS if s in big.columns]
    if not signals:
        print(f"=== {name} ===  no internal-signal columns; skipping.\n"); return
    rng = np.random.RandomState(0)
    clean_idx = big[big.severity == 0]["index"].unique()
    calib = set(rng.permutation(clean_idx)[: len(clean_idx) // 2])
    big = big.copy(); big["calib"] = big["index"].isin(calib)
    groups = [(c, s, d) for (c, s), d in big[big.severity > 0].groupby(["corruption", "severity"])]

    best = (-np.inf, None, None, None)
    for c, s, d in groups:
        for sig, neg in signals:
            sgn = -1.0 if neg else 1.0
            dd = _delta(_clean(d[d.calib], sig), _clean(d[~d.calib], sig), sig, sgn)
            if np.isfinite(dd) and dd > best[0]:
                best = (dd, sig, neg, (c, s))
    obs, bsig, bneg, bcond = best

    Mb = []
    for _ in range(nperm):
        best_b = -np.inf
        for c, s, d in groups:
            for sig, neg in signals:
                sgn = -1.0 if neg else 1.0
                f, e = _clean(d[d.calib], sig).copy(), _clean(d[~d.calib], sig).copy()
                if len(f) < 30 or len(e) < 30:
                    continue
                f[sig] = _perm_within_confbins(f[sig].values, f["conf"].values, rng, nbins)
                e[sig] = _perm_within_confbins(e[sig].values, e["conf"].values, rng, nbins)
                dd = _delta(f, e, sig, sgn)
                if np.isfinite(dd) and dd > best_b:
                    best_b = dd
        if np.isfinite(best_b):
            Mb.append(best_b)
    Mb = np.array(Mb)
    count = int((Mb >= obs).sum())
    p_corr = (1 + count) / (len(Mb) + 1) if len(Mb) else np.nan
    floor = np.percentile(Mb, 95) if len(Mb) else np.nan

    c, s = bcond
    d = big[(big.corruption == c) & (big.severity == s)]
    lo, hi = _boot_ci(_clean(d[d.calib], bsig), _clean(d[~d.calib], bsig), bsig, -1.0 if bneg else 1.0, nboot, rng)

    print(f"=== {name} ===  (nperm={nperm}, nbins={nbins})")
    print(f"  best gain: {obs:+.4f}  ({DISP.get(bsig,bsig)} at {c}@{int(s)})")
    print(f"  max-null 95th pctile: {floor:.4f}")
    print(f"  exceedances #(M_b >= obs): {count} / {len(Mb)}")
    print(f"  corrected p = (1+{count})/({len(Mb)}+1) = {p_corr:.4f}")
    print(f"  bootstrap 95% CI on selected gain: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  upper < margin {margin:.3f}? {'YES' if hi < margin else 'NO'}\n")


def analyse_single(big, name, nboot):
    """DisasterVQA / single-condition: 50/50 question split, bootstrap CI on best gain."""
    signals = [(s, n) for (s, n) in SIGNALS if s in big.columns]
    if not signals:
        print(f"=== {name} ===  no internal-signal columns; skipping.\n"); return
    rng = np.random.RandomState(0)
    idx = big["index"].unique()
    calib = set(rng.permutation(idx)[: len(idx) // 2])
    big = big.copy(); big["calib"] = big["index"].isin(calib)
    best = (-np.inf, None, None)
    for sig, neg in signals:
        sgn = -1.0 if neg else 1.0
        dd = _delta(_clean(big[big.calib], sig), _clean(big[~big.calib], sig), sig, sgn)
        if np.isfinite(dd) and dd > best[0]:
            best = (dd, sig, neg)
    obs, bsig, bneg = best
    lo, hi = _boot_ci(_clean(big[big.calib], bsig), _clean(big[~big.calib], bsig), bsig, -1.0 if bneg else 1.0, nboot, rng)
    print(f"=== {name} ===  (single condition; 50/50 question split)")
    print(f"  best gain: {obs:+.4f}  ({DISP.get(bsig,bsig)})")
    print(f"  bootstrap 95% CI on best gain: [{lo:+.4f}, {hi:+.4f}]  (for Table 6)\n")


def analyse(big, name, margin, nperm, nboot, nbins):
    if "severity" in big.columns and big.severity.max() > 0:
        analyse_grid(big, name, margin, nperm, nboot, nbins)
    else:
        analyse_single(big, name, nboot)


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
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--scale", action="store_true", help="standardize [conf,signal] before logistic fit (sensitivity check)")
    args = ap.parse_args()
    global SCALE; SCALE = args.scale
    dirs = find_dirs(args)
    if not dirs:
        print("No feature dirs found."); sys.exit(1)
    print(f"nperm={args.nperm}, nbins={args.nbins}, margin={args.margin}, scale={args.scale}\n")
    for d in dirs:
        pqs = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not pqs: continue
        big = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        analyse(big, os.path.basename(d).replace("features_", ""), args.margin, args.nperm, args.nboot, args.nbins)


if __name__ == "__main__":
    main()
