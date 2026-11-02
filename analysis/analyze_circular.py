"""circular_analyze.py -- option-order robustness analysis for CircularEval runs.

Reads <features_dir>_circular/*.parquet written by scripts/extract_circular.py
(columns o0..o3 = option probabilities already mapped back to the ORIGINAL option
order, NaN for absent options; n_opt; rot; answer_orig) and reports, per model:
  * position-bias rate: fraction of questions whose chosen OPTION changes across the
    4 rotations (clean and severity 5);
  * CircularEval accuracy (correct under all 4 rotations) vs single-order accuracy;
  * the paper's three findings recomputed with (a) rotation-0 confidence (original
    order, = paper setting on this subset) and (b) rotation-averaged ("debiased")
    confidence, which removes letter-position bias:
      - Delta-ECE (worst corrupted condition - clean), 15 bins, question-clustered
        bootstrap 95% CI;
      - confidence -> correctness AUROC at severity 5;
      - clean -> severity-5 threshold transfer at the 5% risk target
        (threshold set on a clean calibration half, realized risk on held-out
        severity-5 data, question-clustered bootstrap CI).
  * each rotation k analysed as a separate single-order run (rotation k exists
    for every item with more than k options; rotation 3 covers the 4-option
    items only): Delta-ECE, AUROC at severity 5, held-out clean and severity-5
    realized risk of the clean 5%-risk threshold, clean accuracy. Table C1
    reports the ranges of these over the four rotations.
Prints the Table C1 numbers and saves circular_results.csv and
circular_per_rotation.csv.

Usage:  python analysis/analyze_circular.py --dirs outputs/features_qwen25vl_3b_circular outputs/features_internvl3_2b_circular
Requires: numpy pandas pyarrow scikit-learn
"""
import argparse, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

NB, EDGES = 15, np.linspace(0, 1, 16)


def ece(conf, y):
    b = np.searchsorted(EDGES, conf, side="left") - 1
    k = (b >= 0) & (b < NB); n = len(conf)
    cnt = np.bincount(b[k], minlength=NB)
    sc = np.bincount(b[k], weights=conf[k], minlength=NB)
    sy = np.bincount(b[k], weights=y[k], minlength=NB)
    m = cnt > 0
    return float(np.sum(np.abs(sy[m] - sc[m])) / n)


O = ["o0", "o1", "o2", "o3"]


def per_question(df):
    """One row per (index, condition) with rotation-0 and rotation-averaged predictions."""
    df = df.copy()
    opt = df[O].values.astype(float)                       # original option order
    df["pred_opt"] = np.nanargmax(opt, axis=1)
    df["correct_rot"] = (df.pred_opt == df.answer_orig).astype(float)
    key = ["index", "corruption", "severity"]
    g = df.groupby(key)
    avg = g[O].mean()                                      # NaN stays NaN for absent options
    out = pd.DataFrame(index=avg.index)
    out["conf_avg"] = np.nanmax(avg.values, axis=1)
    out["pred_avg"] = np.nanargmax(avg.values, axis=1)
    out["correct_avg"] = (out.pred_avg == g.answer_orig.first()).astype(float)
    r0 = df[df.rot == 0].set_index(key).reindex(out.index)      # align to `out` rows
    out["conf_r0"] = np.nanmax(r0[O].values, axis=1)
    out["correct_r0"] = r0.correct_rot.values
    out["circ_correct"] = g.correct_rot.min()              # correct under every rotation
    out["flips"] = (g.pred_opt.nunique() > 1).astype(float)
    out["n_opt"] = g.n_opt.first()
    return out.reset_index()


def delta_ece(q, conf, corr, idx):
    """Worst corrupted ECE - clean ECE on the questions in idx."""
    sub = q[q["index"].isin(idx)] if idx is not None else q
    e = {c: ece(d[conf].values, d[corr].values)
         for c, d in sub.groupby(["corruption", "severity"])}
    clean = e[("clean", 0)]
    return max(v for k, v in e.items() if k != ("clean", 0)) - clean


def threshold_transfer(q, conf, corr, calib, held, target=0.05):
    c = q[(q.severity == 0) & q["index"].isin(calib)].sort_values(conf, ascending=False)
    risk = 1 - c[corr].expanding().mean().values
    ok = np.where(risk <= target)[0]
    if len(ok) == 0:
        return np.nan, np.nan
    thr = c[conf].values[ok.max()]
    h = q[(q.severity == 5) & q["index"].isin(held) & (q[conf] >= thr)]
    return thr, (1 - h[corr].mean()) if len(h) else np.nan


def analyse(model, q, nboot, seed):
    rng = np.random.RandomState(seed)
    qids = np.sort(q["index"].unique())
    perm = np.random.RandomState(0).permutation(qids)
    calib, held = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
    s5, cl = q[q.severity == 5], q[q.severity == 0]
    res = dict(model=model, n_questions=len(qids),
               flip_clean=cl.flips.mean(), flip_sev5=s5.flips.mean(),
               acc_single_clean=cl.correct_r0.mean(), acc_circ_clean=cl.circ_correct.mean(),
               acc_single_sev5=s5.correct_r0.mean(), acc_circ_sev5=s5.circ_correct.mean())
    for tag, conf, corr in (("r0", "conf_r0", "correct_r0"), ("avg", "conf_avg", "correct_avg")):
        res[f"dece_{tag}"] = delta_ece(q, conf, corr, None)
        res[f"auroc5_{tag}"] = roc_auc_score(s5[corr], s5[conf])
        res[f"risk5_{tag}"] = threshold_transfer(q, conf, corr, calib, held)[1]
        thr = threshold_transfer(q, conf, corr, calib, held)[0]
        bd, br = [], []
        for _ in range(nboot):
            bq = rng.choice(qids, len(qids), replace=True)
            cnt = pd.Series(bq).value_counts()
            bqq = q[q["index"].isin(cnt.index)].copy()
            bqq = bqq.loc[bqq.index.repeat(bqq["index"].map(cnt).values)]
            bd.append(delta_ece(bqq, conf, corr, None))
            hb = bqq[(bqq.severity == 5) & bqq["index"].isin(held)]
            hb = hb[hb[conf] >= thr]
            br.append(1 - hb[corr].mean() if len(hb) else np.nan)
        res[f"dece_{tag}_lo"], res[f"dece_{tag}_hi"] = np.nanpercentile(bd, [2.5, 97.5])
        res[f"risk5_{tag}_lo"], res[f"risk5_{tag}_hi"] = np.nanpercentile(br, [2.5, 97.5])
    return res


def per_rotation(model, df):
    """Each rotation as a single-order run (rows with rot == k)."""
    df = df.copy()
    opt = df[O].values.astype(float)
    df["conf"] = np.nanmax(opt, axis=1)
    df["correct"] = (np.nanargmax(opt, axis=1) == df.answer_orig).astype(float)
    qids = np.sort(df["index"].unique())
    perm = np.random.RandomState(0).permutation(qids)
    calib, held = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
    rows = []
    for k in sorted(df.rot.unique()):
        sel = df[df.rot == k]
        e = {c: ece(g.conf.values, g.correct.values) for c, g in sel.groupby(["corruption", "severity"])}
        dece = max(v for c, v in e.items() if c != ("clean", 0)) - e[("clean", 0)]
        s5 = sel[sel.severity == 5]
        c = sel[(sel.severity == 0) & sel["index"].isin(calib)].sort_values("conf", ascending=False)
        risk = 1 - c.correct.expanding().mean().values
        thr = c.conf.values[np.where(risk <= 0.05)[0].max()]
        hc = sel[(sel.severity == 0) & sel["index"].isin(held) & (sel.conf >= thr)]
        h5 = sel[(sel.severity == 5) & sel["index"].isin(held) & (sel.conf >= thr)]
        rows.append(dict(model=model, rot=int(k), n_questions=sel["index"].nunique(), dece=dece,
                         auroc5=roc_auc_score(s5.correct, s5.conf),
                         risk_clean=1 - hc.correct.mean(), risk5=1 - h5.correct.mean(),
                         acc_clean=sel[sel.severity == 0].correct.mean()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="one or more <features_dir>_circular directories")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="circular_results.csv")
    a = ap.parse_args()
    rows, rot_rows = [], []
    for d in a.dirs:
        fs = sorted(glob.glob(os.path.join(d, "*.parquet"))) or sorted(glob.glob(os.path.join(d, "*.pkl")))
        if not fs:
            print(f"skip {d}: no tables"); continue
        df = pd.concat([pd.read_parquet(f) if f.endswith(".parquet") else pd.read_pickle(f)
                        for f in fs], ignore_index=True)
        name = os.path.basename(os.path.normpath(d)).replace("features_", "").replace("_circular", "") or "model"
        print(f"analysing {name} ({df['index'].nunique()} questions, "
              f"{df.groupby(['corruption','severity']).ngroups} conditions) ...")
        rows.append(analyse(name, per_question(df), a.nboot, a.seed))
        rot_rows += per_rotation(name, df)
    r = pd.DataFrame(rows); r.to_csv(a.out, index=False)

    pct = lambda v: f"{100 * v:.1f}\\%"
    print("\n=== Option-order robustness ===")
    for x in rows:
        print(f"\n{x['model']} ({x['n_questions']} questions)")
        print(f"  chosen option changes across rotations: clean {x['flip_clean']:.1%}, sev-5 {x['flip_sev5']:.1%}")
        print(f"  accuracy single-order / CircularEval: clean {x['acc_single_clean']:.3f} / {x['acc_circ_clean']:.3f}, "
              f"sev-5 {x['acc_single_sev5']:.3f} / {x['acc_circ_sev5']:.3f}")
        for tag, name in (("r0", "original order"), ("avg", "rotation-averaged")):
            print(f"  [{name}] dECE {x[f'dece_{tag}']:+.3f} [{x[f'dece_{tag}_lo']:+.3f},{x[f'dece_{tag}_hi']:+.3f}]"
                  f"  AUROC@sev5 {x[f'auroc5_{tag}']:.3f}"
                  f"  realized risk@sev5 (5% target) {x[f'risk5_{tag}']:.1%} "
                  f"[{x[f'risk5_{tag}_lo']:.1%},{x[f'risk5_{tag}_hi']:.1%}]")
    print("\n=== LaTeX rows: Model & Confidence & dECE [CI] & AUROC@5 & Risk@5 [CI] ===")
    for x in rows:
        for tag, name in (("r0", "original"), ("avg", "rot.-avg.")):
            print(f"{x['model']} & {name} & ${x[f'dece_{tag}']:+.3f}$ $[{x[f'dece_{tag}_lo']:+.3f},"
                  f"{x[f'dece_{tag}_hi']:+.3f}]$ & {x[f'auroc5_{tag}']:.3f} & "
                  f"{pct(x[f'risk5_{tag}'])} $[{100*x[f'risk5_{tag}_lo']:.1f},{100*x[f'risk5_{tag}_hi']:.1f}]$ \\\\")
    rr = pd.DataFrame(rot_rows)
    rr.to_csv(os.path.splitext(a.out)[0] + "_per_rotation.csv", index=False)
    print("\n=== Table C1: range over the single-order rotation runs ===")
    for m, g in rr.groupby("model", sort=False):
        print(f"{m}: dECE {g.dece.min():+.3f} to {g.dece.max():+.3f} | AUROC sev-5 "
              f"{g.auroc5.min():.3f} to {g.auroc5.max():.3f} | realized risk clean "
              f"{100*g.risk_clean.min():.1f} to {100*g.risk_clean.max():.1f}% | sev-5 "
              f"{100*g.risk5.min():.1f} to {100*g.risk5.max():.1f}% | clean accuracy "
              f"{g.acc_clean.min():.3f} to {g.acc_clean.max():.3f} | runs with sev-5 risk > 5%: "
              f"{int((g.risk5 > 0.05).sum())}/{len(g)}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
