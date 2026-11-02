"""risk_transfer_table.py -- Table 3: clean -> corrupt operating-point transfer.

Produces, per model, the clean->corrupt operating-point-transfer numbers with
proper train/eval hygiene and bootstrap CIs:

  * Threshold is SELECTED on the calibration half of CLEAN questions.
  * Clean risk & coverage are EVALUATED on the held-out half of CLEAN questions
    (not the same examples used to pick the threshold).
  * Shifted risk & coverage are EVALUATED on the held-out half of the
    WORST-SEVERITY corrupted questions.
  * "Worst severity" = severity 5 POOLED over the four corruption types (primary);
    we also report the single worst corruption condition at severity 5.
  * 95% bootstrap CIs (resampling held-out questions, seed 0, 2000 draws).

Split matches gain_beyond_confidence.py: RandomState(0), 50/50 on the clean-condition
question indices, so a question is in the same half across all conditions.

Usage:
  python risk_transfer_table.py --root outputs --alphas 0.05 0.10 --nboot 2000
Requires: pandas numpy pyarrow
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

SEED = 0
CLUSTER = False  # set by --cluster: question-clustered bootstrap for pooled results


def thr_for_risk(conf, correct, alpha):
    """Confidence threshold achieving selective risk <= alpha at max coverage."""
    order = np.argsort(-conf)
    c = correct[order].astype(float)
    risk = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.where(risk <= alpha)[0]
    return conf[order][ok.max()] if len(ok) else np.inf


def risk_cov_at_thr(conf, correct, thr):
    sel = conf >= thr
    if sel.sum() == 0:
        return np.nan, 0.0
    return 1.0 - correct[sel].mean(), sel.mean()


def boot_ci(conf, correct, thr, nboot, rng, qid=None, cluster=False):
    rs, cs = [], []
    if cluster and qid is not None:
        uniq = np.unique(qid)
        pos = {u: np.where(qid == u)[0] for u in uniq}
        for _ in range(nboot):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([pos[u] for u in draw])
            r, c = risk_cov_at_thr(conf[idx], correct[idx], thr)
            if np.isfinite(r):
                rs.append(r); cs.append(c)
    else:
        n = len(conf)
        for _ in range(nboot):
            b = rng.randint(0, n, n)
            r, c = risk_cov_at_thr(conf[b], correct[b], thr)
            if np.isfinite(r):
                rs.append(r); cs.append(c)
    f = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5)) if a else (np.nan, np.nan)
    return f(rs), f(cs)


def split_calib(big):
    rng = np.random.RandomState(SEED)
    clean_idx = big[big.severity == 0]["index"].unique()
    calib = set(rng.permutation(clean_idx)[: len(clean_idx) // 2])
    return big["index"].isin(calib).values


def analyse(big, name, alphas, nboot):
    if "severity" not in big.columns or not (big.severity == 0).any():
        print(f"=== {name} ===  no clean baseline; skipped (natural-shift dir).\n"); return
    rng = np.random.RandomState(SEED)
    calib_mask = split_calib(big)
    big = big.copy(); big["calib"] = calib_mask
    maxsev = int(big.severity.max())

    clean = big[big.severity == 0]
    cc_fit = clean[clean.calib]; cc_ev = clean[~clean.calib]
    sev5_ev = big[(big.severity == maxsev) & (~big.calib)]

    print(f"=== {name} ===  (worst severity = {maxsev}, pooled over corruption types)")
    print(f"  {'target':>7} | {'clean-heldout risk / cov':>34} | {'shift(sev{}) risk / cov':>36}".format(maxsev))
    for a in alphas:
        thr = thr_for_risk(cc_fit["conf"].values, cc_fit["correct"].values.astype(float), a)
        if not np.isfinite(thr):
            print(f"  {a:6.0%}  | target unattainable on clean calib split"); continue
        cr, cv = risk_cov_at_thr(cc_ev["conf"].values, cc_ev["correct"].values.astype(float), thr)
        (crl, crh), (cvl, cvh) = boot_ci(cc_ev["conf"].values, cc_ev["correct"].values.astype(float), thr, nboot, rng, qid=cc_ev["index"].values, cluster=CLUSTER)
        sr, sv = risk_cov_at_thr(sev5_ev["conf"].values, sev5_ev["correct"].values.astype(float), thr)
        (srl, srh), (svl, svh) = boot_ci(sev5_ev["conf"].values, sev5_ev["correct"].values.astype(float), thr, nboot, rng, qid=sev5_ev["index"].values, cluster=CLUSTER)
        # worst single corruption condition at sev5 (realised risk)
        worst_r = -1; worst_c = None
        for cty, d in sev5_ev.groupby("corruption"):
            r, _ = risk_cov_at_thr(d["conf"].values, d["correct"].values.astype(float), thr)
            if np.isfinite(r) and r > worst_r:
                worst_r, worst_c = r, cty
        print(f"  {a:6.0%}  | risk {cr:5.1%} [{crl:.1%},{crh:.1%}]  cov {cv:5.1%}"
              f" | risk {sr:5.1%} [{srl:.1%},{srh:.1%}]  cov {sv:5.1%}"
              f"  (thr={thr:.3f}; worst single: {worst_c} {worst_r:.1%})")
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
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--cluster", action="store_true", help="question-clustered bootstrap (resample questions, carry their condition-rows)")
    args = ap.parse_args()
    global CLUSTER; CLUSTER = args.cluster
    dirs = find_dirs(args)
    if not dirs:
        print("No feature dirs found."); sys.exit(1)
    print("Threshold SELECTED on clean calib half; clean risk EVALUATED on clean")
    print("held-out half; shifted risk on worst-severity held-out half. CIs = 95% "
          f"bootstrap (nboot={args.nboot}, seed {SEED}, cluster={args.cluster}).\n")
    for d in dirs:
        pqs = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not pqs: continue
        big = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        analyse(big, os.path.basename(d).replace("features_", ""), args.alphas, args.nboot)


if __name__ == "__main__":
    main()
