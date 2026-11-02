"""CircularEval extraction for the option-order robustness check (paper appendix).

Re-asks each question N times (N = number of options) with the options rotated,
using the SAME saif VLMWrapper / build_mcq_prompt / corruption code as the main
grid, so prompt, resizing and letter pooling are identical by construction.

Rotation k shows original option (j + k) % N at letter j. For each forward pass
we store the probabilities mapped BACK to the original option order (o0..o3), so
the analysis never has to undo rotations.

Grid (default): clean + severity {3,5} x {fog, snow, defocus_blur, brightness}
= 9 conditions, on a fixed 500-question subset of the paper's 2,000 questions.
Light forward pass (no hidden states / attentions).

Usage (from the repo root, same configs as extract.py):
  # 1) consistency check: rotation 0 must reproduce the main-grid clean `conf`
  python scripts/extract_circular.py --config configs/qwen25vl_3b.yaml --subset 50 \
      --types --check
  # 2) full run (restartable: finished conditions are skipped)
  python scripts/extract_circular.py --config configs/qwen25vl_3b.yaml
  python scripts/extract_circular.py --config configs/internvl3_2b.yaml
Outputs: <features_dir>_circular/<corruption>_s<severity>.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saif.corruptions import corrupt_image, condition_grid
from saif.data import load_mmbench, Sample, LETTERS
from saif.io import save_table, condition_done, condition_tables, load_table
from saif.extractor import VLMWrapper


def rotated_sample(s: Sample, k: int) -> Sample:
    """Rotate the options by k: letter j shows original option (j + k) % N."""
    letters = [L for L in LETTERS if L in s.options]
    texts = [s.options[L] for L in letters]
    n = len(texts)
    rot = texts[k:] + texts[:k]
    a = letters.index(s.answer)
    return Sample(index=s.index, image=s.image, question=s.question, hint=s.hint,
                  options={L: t for L, t in zip(letters, rot)},
                  answer=letters[(a - k) % n], category=s.category)


@torch.no_grad()
def letter_probs(vlm: VLMWrapper, s: Sample, image) -> np.ndarray:
    """Same answer-position letter softmax as extract_signals (Eq. A1), light pass."""
    inputs = vlm.prepare_inputs(s, image)
    out = vlm.model(**inputs)
    pos = inputs["input_ids"].shape[1] - 1
    logits = out.logits[0, pos].float()
    letters = [L for L in LETTERS if L in s.options]
    ll = torch.stack([torch.logsumexp(logits[vlm.letter_ids[L]], dim=0) for L in letters])
    return F.softmax(ll, dim=0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset", type=int, default=500, help="questions (subset of the main 2,000)")
    ap.add_argument("--types", nargs="*", default=None,
                    help="corruption types (default: config types); pass with no values for clean only")
    ap.add_argument("--severities", nargs="*", type=int, default=[3, 5])
    ap.add_argument("--check", action="store_true",
                    help="compare rotation-0 clean conf with the main-grid features")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if cfg.get("DISABLED"):
        raise SystemExit(f"config {args.config} is DISABLED")
    types = cfg["corruptions"]["types"] if args.types is None else args.types
    main_dir = Path(cfg["paths"]["features_dir"])
    out_dir = Path(str(main_dir) + "_circular")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same 2,000-question sample as the main grid, then a fixed 500-question subset.
    samples = load_mmbench(cfg["data"]["mmbench_tsv"],
                           perception_only=cfg["data"]["perception_only"],
                           n_questions=cfg["data"]["n_questions"], seed=cfg["data"]["seed"])
    order = np.random.RandomState(0).permutation(len(samples))[: args.subset]
    samples = [samples[i] for i in sorted(order)]
    n_opts = pd.Series([len(s.options) for s in samples]).value_counts().sort_index()
    print(f"{len(samples)} questions; options per question: "
          + ", ".join(f"{k}-option: {v}" for k, v in n_opts.items()))

    vlm = VLMWrapper(cfg["model"]["name"], device=cfg["model"]["device"],
                     dtype=cfg["model"]["dtype"],
                     attn_implementation=cfg["model"]["attn_implementation"],
                     trust_remote_code=cfg["model"].get("trust_remote_code", False),
                     answer_prefix=cfg["model"].get("answer_prefix", ""),
                     max_patches=cfg["model"].get("max_patches", None),
                     max_pixels=cfg["model"].get("max_pixels", None))

    grid = [("clean", 0)] + [(t, s) for t in types for s in args.severities]
    for corr, sev in grid:
        tag = f"{corr}_s{sev}"
        if condition_done(out_dir, tag) and not args.check:
            print(f"[skip] {tag}"); continue
        rows, t0 = [], time.time()
        for i, s in enumerate(samples):
            img = corrupt_image(s.image, corr, sev) if sev > 0 else s.image   # once per question
            letters = [L for L in LETTERS if L in s.options]
            n, a = len(letters), letters.index(s.answer)
            for k in range(n):
                try:
                    p = letter_probs(vlm, rotated_sample(s, k), img)
                except Exception as e:
                    print(f"  [error] idx={s.index} rot={k}: {e}"); continue
                o = np.full(4, np.nan)
                for j in range(n):                     # letter j shows option (j+k)%n
                    o[(j + k) % n] = p[j]
                rows.append(dict(index=s.index, corruption=corr, severity=sev, rot=k,
                                 n_opt=n, answer_orig=a, o0=o[0], o1=o[1], o2=o[2], o3=o[3],
                                 pred_letter=letters[int(p.argmax())],
                                 conf_letter=float(p.max())))
            if (i + 1) % 50 == 0:
                print(f"  {tag}: {i+1}/{len(samples)} ({(time.time()-t0)/(i+1):.2f}s/question)")
        df = pd.DataFrame(rows)
        save_table(df, out_dir / tag)
        print(f"[done] {tag}: {len(df)} rows, {(time.time()-t0)/60:.1f} min")

        if args.check and corr == "clean":
            main_tabs = [p for p in condition_tables(main_dir) if p.stem == "clean_s0"]
            if not main_tabs:
                print(f"[check] no clean_s0 table in {main_dir}; cannot compare"); continue
            ref = load_table(main_tabs[0])[["index", "conf", "pred_letter"]]
            r0 = df[df.rot == 0].merge(ref, on="index", suffixes=("_circ", ""))
            corr_ = np.corrcoef(r0.conf, r0.conf_letter)[0, 1]
            mad = float(np.abs(r0.conf - r0.conf_letter).max())
            agree = float((r0.pred_letter == r0.pred_letter_circ).mean())
            print(f"\n[check] {len(r0)} questions: corr(conf) = {corr_:.5f}, "
                  f"max |diff| = {mad:.5f}, same predicted letter = {agree:.1%}")
            print("[check] OK -- rotation 0 reproduces the main grid." if corr_ > 0.999 and agree > 0.99
                  else "[check] MISMATCH -- investigate before the full run.")


if __name__ == "__main__":
    main()
