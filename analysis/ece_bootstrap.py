"""ece_bootstrap.py -- question-clustered bootstrap CIs for clean ECE, worst-case
corrupted ECE and Delta-ECE (Tables 1 and 6), plus a debiased check.

Reads the same per-question parquet files as make_figures_pro.py:
    <root>/features_<name>/*.parquet
with columns: index (question id), corruption ("clean" at severity 0),
severity, conf, correct.

Bootstrap: resample QUESTIONS with replacement (all 21 condition rows of a
question travel together), then recompute every condition's ECE and take the
worst-case maximum inside each replicate. The CI therefore includes the
selection of the worst condition, which is the honest interval for the
statistic the paper reports.

Also reports
  * P(Delta-ECE > 0.03): fraction of replicates on the decalibrating side;
  * Delta-ECE at the observed worst condition (fixed, no re-selection);
  * a debiased ECE (Kumar et al. 2019 / Roelofs et al. 2022 style:
    per-bin squared-gap minus its sampling variance, clipped at 0, then
    square-rooted), as a check that the verdicts are not binning bias.

Run:
  python ece_bootstrap.py --root outputs                 # all features_* dirs
  python ece_bootstrap.py --root outputs --dirs features_qwen2vl_2b features_qwen2vl_2b_aokvqa
Requires: pandas numpy pyarrow
"""
import argparse, glob, os
import numpy as np, pandas as pd

NBINS, THR = 15, 0.03


_EDGES = np.linspace(0, 1, NBINS + 1)


def _bin_ids(conf):
    # (lo, hi] bins on [0, 1], identical to make_figures_pro.ece;
    # conf == 0 gets -1 and is dropped (but still counts in N), as there.
    return np.searchsorted(_EDGES, conf, side="left") - 1


def ece_cols(C, Y):
    """ECE for every column of C/Y (n_questions x n_conditions); NaN-safe."""
    out = np.empty(C.shape[1])
    for j in range(C.shape[1]):
        c, y = C[:, j], Y[:, j]
        ok = np.isfinite(c) & np.isfinite(y)
        c, y = c[ok], y[ok]
        if len(c) == 0:
            out[j] = np.nan; continue
        b = _bin_ids(c); n = len(c); k = (b >= 0) & (b < NBINS)
        cnt = np.bincount(b[k], minlength=NBINS)
        sc = np.bincount(b[k], weights=c[k], minlength=NBINS)
        sy = np.bincount(b[k], weights=y[k], minlength=NBINS)
        m = cnt > 0
        out[j] = np.sum(np.abs(sy[m] - sc[m]) / n)
    return out


def debiased_ece(c, y):
    ok = np.isfinite(c) & np.isfinite(y); c, y = c[ok], y[ok]
    b = _bin_ids(c); n = len(c); tot = 0.0
    for k in range(NBINS):
        m = b == k; nk = m.sum()
        if nk < 2: continue
        acc, cf = y[m].mean(), c[m].mean()
        tot += nk / n * max((acc - cf) ** 2 - acc * (1 - acc) / (nk - 1), 0.0)
    return float(np.sqrt(tot))


def to_matrices(big):
    big = big.copy()
    big["cond"] = np.where(big.severity == 0, "clean",
                           big.corruption.astype(str) + "@" + big.severity.astype(int).astype(str))
    conds = ["clean"] + sorted(c for c in big.cond.unique() if c != "clean")
    C = big.pivot_table(index="index", columns="cond", values="conf", aggfunc="first")[conds]
    Y = big.pivot_table(index="index", columns="cond", values="correct", aggfunc="first")[conds]
    return C.values.astype(float), Y.values.astype(float), conds


def analyse(name, big, nboot, seed):
    C, Y, conds = to_matrices(big)
    nq = C.shape[0]
    e = ece_cols(C, Y)
    clean, worst_j = e[0], 1 + int(np.nanargmax(e[1:]))
    obs = dict(clean=clean, worst=e[worst_j], delta=e[worst_j] - clean)
    deb_clean = debiased_ece(C[:, 0], Y[:, 0])
    deb_worst = max(debiased_ece(C[:, j], Y[:, j]) for j in range(1, C.shape[1]))

    rng = np.random.RandomState(seed)
    B = np.empty((nboot, 4))
    for b in range(nboot):
        idx = rng.randint(0, nq, nq)
        eb = ece_cols(C[idx], Y[idx])
        B[b] = (eb[0], np.nanmax(eb[1:]), np.nanmax(eb[1:]) - eb[0],
                eb[worst_j] - eb[0])
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    return dict(
        model=name, n_questions=nq, worst_condition=conds[worst_j],
        clean_ece=obs["clean"], clean_lo=lo[0], clean_hi=hi[0],
        worst_ece=obs["worst"], worst_lo=lo[1], worst_hi=hi[1],
        delta_ece=obs["delta"], delta_lo=lo[2], delta_hi=hi[2],
        delta_fixed_lo=lo[3], delta_fixed_hi=hi[3],
        p_decal=float(np.mean(B[:, 2] > THR)),
        debiased_clean=deb_clean, debiased_worst=deb_worst,
        debiased_delta=deb_worst - deb_clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="feature dir names under --root (default: all features_*)")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ece_bootstrap.csv")
    a = ap.parse_args()
    dirs = ([os.path.join(a.root, d) for d in a.dirs] if a.dirs
            else sorted(glob.glob(os.path.join(a.root, "features_*"))))
    rows = []
    for d in dirs:
        pqs = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not pqs: continue
        big = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        if "severity" not in big or big.severity.max() == 0:
            print(f"skip {d}: no corrupted conditions"); continue
        name = os.path.basename(d).replace("features_", "")
        print(f"bootstrapping {name} ({big['index'].nunique()} questions) ...")
        rows.append(analyse(name, big, a.nboot, a.seed))
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)

    f = lambda v: f"{v:.3f}"
    s = lambda v: f"{v:+.3f}"
    print("\n=== Delta-ECE with question-clustered 95% CIs ===")
    for r in rows:
        print(f"{r['model']:<28} clean {f(r['clean_ece'])} [{f(r['clean_lo'])},{f(r['clean_hi'])}]"
              f"  worst {f(r['worst_ece'])} [{f(r['worst_lo'])},{f(r['worst_hi'])}] ({r['worst_condition']})"
              f"  dECE {s(r['delta_ece'])} [{s(r['delta_lo'])},{s(r['delta_hi'])}]"
              f"  P(dECE>{THR})={r['p_decal']:.2f}  debiased dECE {s(r['debiased_delta'])}")
    print("\n=== LaTeX rows (Model & Clean ECE & Worst ECE & dECE [95% CI] & P(>0.03)) ===")
    for r in rows:
        print(f"{r['model']} & {f(r['clean_ece'])} & {f(r['worst_ece'])} & "
              f"${s(r['delta_ece'])}$ $[{s(r['delta_lo'])},{s(r['delta_hi'])}]$ & "
              f"{r['p_decal']:.2f} \\\\")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
