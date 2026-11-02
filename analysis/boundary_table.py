"""boundary_table.py -- reproduce every calibration-boundary number in the paper.

For each model directory outputs/features_<name>/ (each a set of per-condition
*.parquet files), prints:
  - n, clean ECE, worst ECE (+which condition), delta-ECE, verdict (thr 0.03)
  - clean / worst confidence->correctness AUROC
  - r(ECE, conf AUROC) across corrupted conditions  (the mechanism correlation)
  - confidence advantage over random @50% and @80% coverage, worst severity
    per corruption family  (the inversion / no-inversion evidence)
  - threshold-sensitivity verdicts at 0.02 / 0.03 / 0.04 / 0.05

Natural-shift dirs with a single 'clean' condition (e.g. DisasterVQA) are
handled specially: no severity axis, so it reports ECE, AUROC and the
selective-prediction advantage only, and says so.

Conventions match the paper: 15-bin ECE, boundary threshold 0.03,
selective advantage = selective-accuracy(top-tau by confidence) - base accuracy.

Usage:
  python boundary_table.py                      # auto-discover outputs/features_*
  python boundary_table.py --root outputs
  python boundary_table.py --dirs outputs/features_smolvlm_instruct ...

Requires: pandas numpy scikit-learn pyarrow
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

THR = 0.03
FAMILIES = ["fog", "snow", "defocus_blur", "brightness"]


def ece(conf, correct, n_bins=15):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def auroc(score, correct):
    score = np.asarray(score, float); y = np.asarray(correct)
    ok = np.isfinite(score)
    if ok.sum() == 0 or y[ok].min() == y[ok].max():
        return float("nan")
    return roc_auc_score(y[ok], score[ok])


def sel_advantage(conf, correct, cov):
    conf = np.asarray(conf, float); correct = np.asarray(correct, float)
    k = int(np.ceil(cov * len(conf)))
    order = np.argsort(-conf)
    return correct[order[:k]].mean() - correct.mean()


def load_dir(d):
    pqs = sorted(glob.glob(os.path.join(d, "*.parquet")))
    if not pqs:
        return None
    return pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)


def analyse(big, name):
    conds = big.groupby(["corruption", "severity"])
    rows = []
    for (c, s), d in conds:
        y = d["correct"].values
        rows.append(dict(corruption=c, severity=s, n=len(d),
                         acc=y.mean(), ece=ece(d["conf"].values, y),
                         auc=auroc(d["conf"].values, y)))
    rep = pd.DataFrame(rows)
    has_clean = (rep.corruption == "clean").any()
    corrupted = rep[rep.corruption != "clean"]

    print("=" * 66)
    print(f"MODEL: {name}   (n={len(big)}, {len(rep)} condition(s))")
    print("=" * 66)

    # NATURAL-SHIFT single-condition case (no severity grid)
    if len(corrupted) == 0:
        r = rep.iloc[0]
        print("  [single condition -- natural-shift benchmark, no severity axis]")
        print(f"  ECE                 : {r.ece:.4f}")
        print(f"  conf->correct AUROC : {r.auc:.4f}")
        for cov in (0.5, 0.8):
            adv = sel_advantage(big["conf"].values, big["correct"].values, cov)
            print(f"  sel. advantage @{int(cov*100)}%: {adv:+.4f}")
        print("  (report as consistency check; no delta-ECE without a clean baseline)")
        print()
        return

    clean_ece = rep[rep.corruption == "clean"].ece.iloc[0] if has_clean else corrupted.ece.min()
    worst = rep.loc[rep.ece.idxmax()]
    dece = worst.ece - clean_ece
    r = np.corrcoef(corrupted.ece.values, corrupted.auc.values)[0, 1] if len(corrupted) > 2 else float("nan")

    print(f"  clean ECE           : {clean_ece:.4f}")
    print(f"  worst ECE           : {worst.ece:.4f}   ({worst.corruption}@{int(worst.severity)})")
    print(f"  delta-ECE           : {dece:+.4f}")
    print(f"  VERDICT (thr {THR})   : {'DECALIBRATING' if dece > THR else 'CALIBRATED'}")
    if has_clean:
        print(f"  clean conf AUROC    : {rep[rep.corruption=='clean'].auc.iloc[0]:.4f}")
    print(f"  worst conf AUROC    : {corrupted.auc.min():.4f}")
    print(f"  r(ECE, conf AUROC)  : {r:+.4f}")

    # inversion: advantage at worst severity per family
    print("  inversion check (confidence advantage over random):")
    maxsev = int(big.severity.max())
    for fam in FAMILIES:
        d = big[(big.corruption == fam) & (big.severity == maxsev)]
        if len(d) == 0:
            continue
        a50 = sel_advantage(d["conf"].values, d["correct"].values, 0.5)
        a80 = sel_advantage(d["conf"].values, d["correct"].values, 0.8)
        print(f"    {fam}@{maxsev}: @50%={a50:+.4f}  @80%={a80:+.4f}")

    # threshold sensitivity
    verdicts = {t: ("decal." if dece > t else "cal.") for t in (0.02, 0.03, 0.04, 0.05)}
    print(f"  threshold sensitivity: " + "  ".join(f"{t}->{v}" for t, v in verdicts.items()))
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
    ap.add_argument("--root", default=None)
    ap.add_argument("--dirs", nargs="*", default=None)
    args = ap.parse_args()
    dirs = find_dirs(args)
    if not dirs:
        print("No feature dirs found. Run from project root, or pass --dirs/--root.")
        sys.exit(1)
    print(f"Found {len(dirs)} model dir(s): {[os.path.basename(d) for d in dirs]}\n")
    for d in dirs:
        big = load_dir(d)
        if big is None or big.empty:
            print(f"skip {d}: empty"); continue
        analyse(big, os.path.basename(d).replace("features_", ""))


if __name__ == "__main__":
    main()
