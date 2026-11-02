"""make_figures.py -- Figures 2-6 of the paper (Fig. 1 is a TikZ diagram).

Output files keep their internal names: fig1_ece_by_severity = Fig. 2,
fig2_boundary_bars = Fig. 3, fig3_mechanism_scatter = Fig. 4,
fig4_headline_auroc = Fig. 5, fig5_delta_beyond_conf = Fig. 6.

Self-contained (reads outputs/features_<model>/*.parquet directly; no saif
dependency). Produces vector PDF (journal-ready) + PNG preview for all five
figures, with a consistent colourblind-safe palette (Okabe-Ito), serif fonts to
match a LaTeX body, recessive grids, and direct value labels.

Fig. 6: bars = best gain beyond confidence, whiskers = bootstrap 95% CI,
dashed line = prespecified practical margin (0.020).

Sizing: Springer sn-basic text width is ~5.0 in, so every figure is drawn at
the size it is printed (fonts stay >= 7.5 pt in the PDF). Include the .pdf
versions in LaTeX (vector).

Run:
  python analysis/make_figures.py --root outputs --out figures
Requires: pandas numpy scikit-learn matplotlib pyarrow
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---- publication style -----------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "grid.alpha": 0.9, "axes.axisbelow": True,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc", "legend.borderpad": 0.4,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "lines.linewidth": 1.6, "lines.markersize": 5,
})

# Okabe-Ito, fixed per entity + distinct markers (secondary encoding for CVD)
MODEL_COLOR = {"qwen25vl_7b": "#56B4E9", "qwen25vl_3b": "#0072B2", "qwen2vl_2b": "#E69F00",
               "smolvlm_instruct": "#009E73", "internvl3_2b": "#CC79A7"}
MODEL_MARK = {"qwen25vl_7b": "v", "qwen25vl_3b": "o", "qwen2vl_2b": "s",
              "smolvlm_instruct": "^", "internvl3_2b": "D"}
DISPLAY = {"qwen25vl_7b": "Qwen2.5-VL-7B", "qwen25vl_3b": "Qwen2.5-VL-3B", "qwen2vl_2b": "Qwen2-VL-2B",
           "smolvlm_instruct": "SmolVLM-2.2B", "internvl3_2b": "InternVL3-2B"}
ORDER = ["qwen25vl_7b", "qwen25vl_3b", "qwen2vl_2b", "internvl3_2b", "smolvlm_instruct"]   # 7B is skipped if its features dir is absent
CORR = {"fog": ("#0072B2", "o"), "snow": ("#E69F00", "s"),
        "defocus_blur": ("#009E73", "^"), "brightness": ("#D55E00", "D")}
CAL, DECAL, BORDER = "#0072B2", "#D55E00", "#8C8C8C"
# Models whose bootstrap Delta-ECE interval includes the cutoff (paper Table 1)
BORDERLINE = {"qwen25vl_7b"}
THR, MARGIN = 0.03, 0.020
TEXTW = 5.0   # inches; sn-basic \textwidth


def ece(conf, y, n=15):
    b = np.linspace(0, 1, n + 1); e = 0.0; N = len(conf)
    for lo, hi in zip(b[:-1], b[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum(): e += m.sum() / N * abs(y[m].mean() - conf[m].mean())
    return float(e)


def auroc(s, y):
    s = np.asarray(s, float); ok = np.isfinite(s)
    if ok.sum() == 0 or y[ok].min() == y[ok].max(): return np.nan
    return roc_auc_score(y[ok], s[ok])


def load(root):
    data = {}
    for d in sorted(glob.glob(os.path.join(root, "features_*"))):
        short = os.path.basename(d).replace("features_", "")
        if short not in MODEL_COLOR:  # only the five MMBench models
            continue
        pqs = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if not pqs: continue
        big = pd.concat([pd.read_parquet(p) for p in pqs], ignore_index=True)
        if "severity" not in big or big.severity.max() == 0:
            continue
        rows = []
        for (c, s), g in big.groupby(["corruption", "severity"]):
            y = g["correct"].values.astype(float)
            rows.append(dict(corruption=c, severity=int(s),
                             ece=ece(g["conf"].values, y),
                             auc=auroc(g["conf"].values, y)))
        data[short] = (pd.DataFrame(rows), big)
    return {k: data[k] for k in ORDER if k in data}


def savefig(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig); print(f"  saved {name}.pdf / .png")


def fig1(data, out):
    n = len(data)
    ncol = 2 if n > 1 else 1; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTW, 2.05 * nrow + 0.3),
                             sharey=True, sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]: ax.set_visible(False)
    for ax, k in zip(axes, data):
        rep, _ = data[k]; clean = rep[rep.corruption == "clean"].ece.iloc[0]
        for c, (col, mk) in CORR.items():
            d = rep[rep.corruption == c].sort_values("severity")
            if d.empty: continue
            ax.plot([0] + list(d.severity), [clean] + list(d.ece),
                    color=col, marker=mk, markersize=4, linewidth=1.4,
                    label=c.replace("_", " "), markeredgecolor="white",
                    markeredgewidth=0.4)
        ax.axhline(clean + THR, color="#555555", ls=(0, (4, 3)), lw=0.9)
        ax.set_title(DISPLAY[k]); ax.set_xticks(range(6))
        ax.margins(x=0.03)
    for i, ax in enumerate(axes[:n]):
        # x labels on the lowest VISIBLE panel of each column (handles an odd
        # number of models, where the bottom-right panel is hidden)
        if i + ncol >= n:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("corruption severity")
        if i % ncol == 0: ax.set_ylabel("ECE")
    axes[0].legend(loc="upper left", handlelength=1.4, borderaxespad=0.3)
    fig.tight_layout(); savefig(fig, out, "fig1_ece_by_severity")


def fig2(data, out):
    labels, deltas, cols, hatches = [], [], [], []
    for k in data:
        rep, _ = data[k]; clean = rep[rep.corruption == "clean"].ece.iloc[0]
        dece = rep.ece.max() - clean
        labels.append(DISPLAY[k]); deltas.append(dece)
        if k in BORDERLINE:
            cols.append(BORDER); hatches.append("..")
        else:
            decal = dece > THR
            cols.append(DECAL if decal else CAL); hatches.append("///" if decal else "")
    fig, ax = plt.subplots(figsize=(0.65 * TEXTW, 2.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, width=0.62, color=cols, edgecolor="white",
                  linewidth=0.8)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    ax.axhline(THR, color="#222222", ls=(0, (5, 3)), lw=1.0,
               label=f"cutoff (+{THR:g})")
    for xi, d in zip(x, deltas):
        ax.annotate(f"+{d:.3f}", (xi, d), ha="center", va="bottom",
                    fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(r"$\Delta$ECE (worst corruption $-$ clean)")
    ax.set_ylim(0, max(deltas) * 1.22)
    from matplotlib.patches import Patch
    leg = [Patch(fc=CAL, ec="white", label="calibrated"),
           Patch(fc=DECAL, ec="white", hatch="///", label="decalibrating"),
           Patch(fc=BORDER, ec="white", hatch="..", label="borderline"),
           plt.Line2D([0], [0], color="#222222", ls=(0, (5, 3)), lw=1.0,
                      label=f"cutoff (+{THR:g})")]
    ax.legend(handles=leg, loc="upper left")
    fig.tight_layout(); savefig(fig, out, "fig2_boundary_bars")


def fig3(data, out):
    fig, ax = plt.subplots(figsize=(0.9 * TEXTW, 3.3))
    handles = []
    for k in data:
        rep, _ = data[k]; corr = rep[rep.corruption != "clean"]
        col, mk = MODEL_COLOR[k], MODEL_MARK[k]
        ax.scatter(corr.ece, corr.auc, color=col, marker=mk, s=26, alpha=0.85,
                   edgecolor="white", linewidth=0.4, zorder=3)
        if len(corr) > 2:
            r = np.corrcoef(corr.ece, corr.auc)[0, 1]
            m, b = np.polyfit(corr.ece, corr.auc, 1)
            xs = np.linspace(corr.ece.min(), corr.ece.max(), 20)
            ax.plot(xs, m * xs + b, color=col, lw=1.2, alpha=0.7, zorder=2)
            handles.append(plt.Line2D([0], [0], color=col, marker=mk, lw=1.2,
                           markeredgecolor="white", markeredgewidth=0.4,
                           label=f"{DISPLAY[k]}  ($r={r:+.2f}$)"))
    ax.set_xlabel("Expected Calibration Error (per condition)")
    ax.set_ylabel(r"confidence $\rightarrow$ correctness AUROC")
    ax.legend(handles=handles, loc="lower left", handlelength=1.6)
    fig.tight_layout(); savefig(fig, out, "fig3_mechanism_scatter")


def fig4(data, out):
    fig, ax = plt.subplots(figsize=(0.9 * TEXTW, 2.9))
    for k in data:
        rep, _ = data[k]
        by = rep.groupby("severity").auc.mean().reset_index()
        ax.plot(by.severity, by.auc, color=MODEL_COLOR[k], marker=MODEL_MARK[k],
                markersize=5, linewidth=1.7, markeredgecolor="white",
                markeredgewidth=0.5, label=DISPLAY[k])
    ax.axhline(0.5, color="#999999", ls=":", lw=1.0, label="chance")
    ax.set_xlabel("corruption severity")
    ax.set_ylabel(r"confidence $\rightarrow$ correctness AUROC")
    ax.set_xticks(range(6)); ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower left", ncol=1, handlelength=1.6)
    fig.tight_layout(); savefig(fig, out, "fig4_headline_auroc")


def _best_gain_ci(big, nboot=2000, seed=0):
    rng = np.random.RandomState(seed)
    SIG = [("lens_agree_frac", False), ("attn_img_entropy", True),
           ("attn_img_mass", False)]
    SIG = [(s, n) for s, n in SIG if s in big.columns]
    clean_idx = big[big.severity == 0]["index"].unique()
    calib = set(rng.permutation(clean_idx)[: len(clean_idx) // 2])
    big = big.copy(); big["calib"] = big["index"].isin(calib)

    def delta(fit, ev, sig, sgn):
        m = np.isfinite(fit["conf"]) & np.isfinite(fit[sig]) & np.isfinite(fit["correct"])
        me = np.isfinite(ev["conf"]) & np.isfinite(ev[sig]) & np.isfinite(ev["correct"])
        fit, ev = fit[m], ev[me]
        if len(fit) < 30 or len(ev) < 30: return np.nan
        yv = ev["correct"].values
        if yv.min() == yv.max(): return np.nan
        cf = LogisticRegression(max_iter=2000).fit(fit[["conf"]].values, fit["correct"].values)
        bf = LogisticRegression(max_iter=2000).fit(
            np.column_stack([fit["conf"].values, sgn * fit[sig].values]), fit["correct"].values)
        return (roc_auc_score(yv, bf.predict_proba(np.column_stack([ev["conf"].values, sgn * ev[sig].values]))[:, 1])
                - roc_auc_score(yv, cf.predict_proba(ev[["conf"]].values)[:, 1]))
    best = (-9, None, None, None)
    for (c, s), d in big[big.severity > 0].groupby(["corruption", "severity"]):
        for sig, neg in SIG:
            dd = delta(d[d.calib], d[~d.calib], sig, -1.0 if neg else 1.0)
            if np.isfinite(dd) and dd > best[0]: best = (dd, sig, neg, (c, s))
    bd, bsig, bneg, bc = best
    c, s = bc; d = big[(big.corruption == c) & (big.severity == s)]
    sgn = -1.0 if bneg else 1.0
    m = np.isfinite(d["conf"]) & np.isfinite(d[bsig]) & np.isfinite(d["correct"])
    d = d[m]; fit, ev = d[d.calib], d[~d.calib].reset_index(drop=True)
    cf = LogisticRegression(max_iter=2000).fit(fit[["conf"]].values, fit["correct"].values)
    bf = LogisticRegression(max_iter=2000).fit(
        np.column_stack([fit["conf"].values, sgn * fit[bsig].values]), fit["correct"].values)
    pc = cf.predict_proba(ev[["conf"]].values)[:, 1]
    pb = bf.predict_proba(np.column_stack([ev["conf"].values, sgn * ev[bsig].values]))[:, 1]
    y = ev["correct"].values; boot = []
    for _ in range(nboot):
        idx = rng.randint(0, len(ev), len(ev)); yb = y[idx]
        if yb.min() != yb.max():
            boot.append(roc_auc_score(yb, pb[idx]) - roc_auc_score(yb, pc[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return bd, lo, hi


def fig5(data, out):
    labels, gains, los, his, cols = [], [], [], [], []
    for k in data:
        _, big = data[k]
        bd, lo, hi = _best_gain_ci(big)
        labels.append(DISPLAY[k]); gains.append(bd); los.append(lo); his.append(hi)
        cols.append(MODEL_COLOR[k])
    fig, ax = plt.subplots(figsize=(0.9 * TEXTW, 2.9))
    x = np.arange(len(labels))
    err = np.array([[g - l for g, l in zip(gains, los)],
                    [h - g for g, h in zip(gains, his)]])
    ax.bar(x, gains, width=0.6, color=cols, edgecolor="white", linewidth=0.8, zorder=2)
    ax.errorbar(x, gains, yerr=err, fmt="none", ecolor="#333333",
                elinewidth=1.0, capsize=3, capthick=1.0, zorder=3)
    ax.axhline(MARGIN, color="#222222", ls=(0, (5, 3)), lw=1.0,
               label=f"practical margin (+{MARGIN:g})")
    ax.axhline(0, color="#999999", lw=0.8)
    for xi, g, h in zip(x, gains, his):
        ax.annotate(f"{g:+.4f}", (xi, h), ha="center", va="bottom", fontsize=7.5,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(r"best $\Delta$AUROC beyond confidence")
    ax.set_ylim(min(0, min(los)) - 0.002, max(MARGIN, max(his)) * 1.2)
    ax.legend(loc="upper left")
    fig.tight_layout(); savefig(fig, out, "fig5_delta_beyond_conf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="*", default=None, help="paper figure numbers, e.g. 2 6")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load(args.root)
    print(f"Loaded {len(data)} models: {[DISPLAY[k] for k in data]}")
    if not data:
        print("No MMBench model dirs found under", args.root); return
    figs = {"2": fig1, "3": fig2, "4": fig3, "5": fig4, "6": fig5}   # keys = figure numbers in the paper
    for num, fn in figs.items():
        if args.only and num not in args.only: continue
        try:
            fn(data, args.out)
        except Exception as e:
            print(f"  fig{num} FAILED: {e}")
    print(f"Done. Vector PDFs + PNGs in {args.out}/")


if __name__ == "__main__":
    main()