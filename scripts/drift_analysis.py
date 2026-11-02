"""Hidden-state drift analysis (paper Sec. 4.4, Eq. A6).

Needs the <tag>_hidden.npz files written by extract.py (not distributed with the
release because of their size; re-run the extraction to regenerate them).

Reads the parquet + hidden-state files produced by extract.py, computes the
drift feature against the clean reference, and prints per-condition:
  accuracy, ECE, AUROC of each signal, and gain-beyond-confidence Delta
  for each internal signal (fit on one half of questions, eval on the other).

Usage:
  python scripts/drift_analysis.py --config configs/qwen25vl_3b.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saif.drift import DriftReference
from saif.io import save_table, load_table, condition_tables
from saif.metrics import auroc, ece, gain_beyond_confidence

SIGNALS = ["conf", "margin", "entropy",
           "lens_agree_frac", "attn_img_entropy", "attn_img_mass",
           "drift_answer_pos", "drift_img_mean", "drift_img_mean_mid"]
# signals where LOWER should mean MORE reliable -> negate before AUROC
NEGATE = {"entropy", "attn_img_entropy",
          "drift_answer_pos", "drift_img_mean", "drift_img_mean_mid"}


DRIFT_VARIANTS = ["answer_pos", "img_mean", "img_mean_mid"]


def _load_hidden(features_dir: Path, tag: str) -> dict | None:
    """Load hidden states for one condition. Prefers the new .npz (three
    variants), falls back to the old .npy (answer-pos only) so existing
    extractions still work — the old-format columns will simply be NaN for
    the newer variants. Returns None if the hidden-state file is missing;
    the caller decides whether to skip the condition or abort."""
    npz = features_dir / f"{tag}_hidden.npz"
    npy = features_dir / f"{tag}_hidden.npy"
    if npz.exists():
        z = np.load(npz)
        return {k: z[k] for k in z.files}
    if npy.exists():
        return {"answer_pos": np.load(npy)}
    return None


def load_all(features_dir: Path) -> pd.DataFrame:
    dfs, skipped = [], []
    for pq in condition_tables(features_dir):
        H = _load_hidden(features_dir, pq.stem)
        if H is None:
            skipped.append(pq.stem)
            continue
        df = load_table(pq)
        df["_hidden_row"] = range(len(df))
        df["_tag"] = pq.stem
        dfs.append((df, H))
    if skipped:
        print(f"[load] WARNING: skipped {len(skipped)} condition(s) missing "
              f"hidden-state file:")
        for tag in skipped:
            print(f"       tag={tag!r}  looked for: "
                  f"{tag}_hidden.npz  or  {tag}_hidden.npy")
        # dump directory contents so filename mismatches are obvious
        actual = sorted(p.name for p in Path(features_dir).iterdir())
        print(f"[load] {len(actual)} files in {features_dir}: "
              f"{actual if len(actual) < 30 else actual[:15] + ['...'] + actual[-10:]}")
    if not dfs:
        raise RuntimeError(
            f"no complete conditions found in {features_dir}. "
            f"Expected pairs of <tag>.pkl/.parquet + <tag>_hidden.npy/.npz")
    big = pd.concat([d for d, _ in dfs], ignore_index=True)

    # calibration split: half the clean questions define the reference manifold,
    # the other half plus all corrupted rows are the held-out audit
    clean_idx = big[(big.severity == 0)]["index"].unique()
    rng = np.random.RandomState(0)
    calib_q = set(rng.permutation(clean_idx)[: len(clean_idx) // 2])
    big["calib_half"] = big["index"].isin(calib_q)

    hid = {tag: H for (d, H), tag in
           zip(dfs, [d["_tag"].iloc[0] for d, _ in dfs])}

    # compute drift per variant so we can compare placements directly
    def gather(variant, rows):
        arrs = []
        for t, r in zip(rows._tag, rows._hidden_row):
            store = hid[t]
            arrs.append(store[variant][r] if variant in store else None)
        return arrs

    clean = big[(big.severity == 0) & big.calib_half]
    for variant in DRIFT_VARIANTS:
        clean_h = gather(variant, clean)
        if any(x is None for x in clean_h):
            print(f"  [drift] variant '{variant}' not present in extractions; "
                  f"skipping (re-run extract.py to populate)")
            big[f"drift_{variant}"] = np.nan
            continue
        ref = DriftReference()
        ref.fit(np.stack(clean_h))
        all_h = gather(variant, big)
        big[f"drift_{variant}"] = ref.distance(np.stack(all_h))

    # legacy alias so downstream code that reads "drift" still works: uses the
    # answer-pos variant.
    big["drift"] = big["drift_answer_pos"]

    # Rescue NaN/inf from numerical edges (e.g. n_img==1 attention entropy).
    # Legacy-format extractions leave drift_img_mean* as NaN by design; those
    # columns are excluded from cleaning so a real gap doesn't get median-filled.
    signal_cols = (["conf", "margin", "entropy", "lens_agree_frac",
                    "attn_img_entropy", "attn_img_mass"]
                   + [f"drift_{v}" for v in DRIFT_VARIANTS]
                   + ["drift"])
    for c in signal_cols:
        if c not in big.columns or big[c].isna().all():
            continue
        n_bad = (~np.isfinite(big[c])).sum()
        if n_bad and n_bad < len(big):
            print(f"  [clean] {c}: {n_bad}/{len(big)} non-finite values "
                  f"replaced with column median")
            med = np.nanmedian(big.loc[np.isfinite(big[c]), c])
            big.loc[~np.isfinite(big[c]), c] = med
    return big.drop(columns=["_hidden_row", "_tag"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    big = load_all(Path(cfg["paths"]["features_dir"]))
    save_table(big, Path(cfg["paths"]["features_dir"]) / "_merged_with_drift")

    print("\n=== per-condition diagnostics ===")
    rows = []
    for (corr, sev), d in big.groupby(["corruption", "severity"]):
        y = d["correct"].values
        r = {"corruption": corr, "severity": sev, "n": len(d),
             "acc": y.mean(), "ece": ece(d["conf"].values, y)}
        for sig in SIGNALS:
            s = d[sig].values * (-1 if sig in NEGATE else 1)
            r[f"auroc_{sig}"] = auroc(s, y)
        # gain beyond confidence: fit combiners on calib half, eval on test half
        fit_d, ev_d = d[d.calib_half], d[~d.calib_half]
        internal_feats = ["lens_agree_frac", "attn_img_entropy", "attn_img_mass",
                          "drift_answer_pos", "drift_img_mean", "drift_img_mean_mid"]
        if len(fit_d) > 30 and len(ev_d) > 30:
            for sig in internal_feats:
                if sig not in d.columns or d[sig].isna().all():
                    r[f"delta_{sig}"] = float("nan")
                    continue
                sgn = -1 if sig in NEGATE else 1
                _, cf, bf = gain_beyond_confidence(
                    fit_d["conf"].values, sgn * fit_d[sig].values, fit_d["correct"].values)
                delta, _, _ = gain_beyond_confidence(
                    ev_d["conf"].values, sgn * ev_d[sig].values, ev_d["correct"].values,
                    conf_fit=cf, both_fit=bf)
                r[f"delta_{sig}"] = delta
        rows.append(r)
    rep = pd.DataFrame(rows).sort_values(["corruption", "severity"])
    pd.set_option("display.width", 240)
    print(rep.round(3).to_string(index=False))

    out = Path(cfg["paths"]["results_dir"]); out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "drift_analysis.csv", index=False)

    deltas = rep[[c for c in rep.columns if c.startswith("delta_")]]
    corrupted = rep[rep.severity > 0]
    if len(corrupted):
        # per-signal max Δ under corruption -- helps pinpoint WHICH signal
        # (if any) is doing the work
        print("\nMax Δ under corruption, per signal:")
        for c in deltas.columns:
            vals = deltas.loc[corrupted.index, c].dropna()
            if len(vals):
                print(f"  {c:32s} {vals.max():+.4f}  (at severity "
                      f"{int(corrupted.loc[vals.idxmax(), 'severity'])}, "
                      f"{corrupted.loc[vals.idxmax(), 'corruption']})")
        best = deltas.loc[corrupted.index].max().max()
        print(f"\nbest gain-beyond-confidence under corruption = {best:+.4f}")
    else:
        print("\nno corrupted conditions found in features.")


if __name__ == "__main__":
    main()
