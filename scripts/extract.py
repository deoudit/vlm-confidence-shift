"""Run the VLM once per (question, condition) and save all signals.
Restartable: skips conditions whose table already exists.

Full 21-condition grid for one model:
  python scripts/extract.py --config configs/qwen25vl_3b.yaml

Quick check (clean + one corruption, 50 questions):
  python scripts/extract.py --config configs/qwen25vl_3b.yaml --types fog --severities 3 --n 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saif.corruptions import corrupt_image, condition_grid
from saif.data import load_mmbench
from saif.data_aokvqa import load_aokvqa
from saif.io import save_table, condition_done
from saif.extractor import VLMWrapper, extract_signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--types", nargs="*", default=None, help="override corruption types")
    ap.add_argument("--severities", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=None, help="override n_questions")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if cfg.get("DISABLED"):
        raise SystemExit(f"config {args.config} is marked DISABLED; see its "
                         f"header for the reason. Use a different config.")
    types = args.types or cfg["corruptions"]["types"]
    sevs = args.severities or cfg["corruptions"]["severities"]
    n_q = args.n or cfg["data"]["n_questions"]

    out_dir = Path(cfg["paths"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading benchmark...")
    bench = cfg["data"].get("benchmark", "mmbench")
    if bench == "disastervqa":
        from saif.data_disastervqa import load_disastervqa
        samples = load_disastervqa(n_questions=n_q, seed=cfg["data"]["seed"])
    elif bench == "aokvqa":
        samples = load_aokvqa(split=cfg["data"].get("aokvqa_split", "validation"),
                              n_questions=n_q, seed=cfg["data"]["seed"])
    else:
        samples = load_mmbench(cfg["data"]["mmbench_tsv"],
                               perception_only=cfg["data"]["perception_only"],
                               n_questions=n_q, seed=cfg["data"]["seed"])
    assert samples, "0 samples loaded — check the TSV path and the perception filter"
    print(f"{len(samples)} questions")

    print(f"loading {cfg['model']['name']} ...")
    vlm = VLMWrapper(cfg["model"]["name"], device=cfg["model"]["device"],
                     dtype=cfg["model"]["dtype"],
                     attn_implementation=cfg["model"]["attn_implementation"],
                     trust_remote_code=cfg["model"].get("trust_remote_code", False),
                     answer_prefix=cfg["model"].get("answer_prefix", ""),
                     max_patches=cfg["model"].get("max_patches", None),
                     max_pixels=cfg["model"].get("max_pixels", None))
    last_L = cfg["signals"]["logit_lens_last_L"]
    attn_layers = tuple(cfg["signals"]["attention_layers"])

    for corr, sev in condition_grid(types, sevs):
        tag = f"{corr}_s{sev}"
        if condition_done(out_dir, tag):
            print(f"[skip] {tag}")
            continue
        rows = []
        hidden_stacks = {"answer_pos": [], "img_mean": [], "img_mean_mid": []}
        t0 = time.time()
        for i, s in enumerate(samples):
            img = corrupt_image(s.image, corr, sev) if sev > 0 else s.image
            try:
                rec = extract_signals(vlm, s, img, corr, sev,
                                      last_L=last_L, attention_layers=attn_layers)
            except Exception as e:               # log and move on; audit later
                print(f"  [error] idx={s.index}: {e}")
                continue
            rows.append(rec.row())
            for k, v in rec.hiddens.items():
                hidden_stacks[k].append(v)
            if (i + 1) % 100 == 0:
                dt = (time.time() - t0) / (i + 1)
                print(f"  {tag}: {i+1}/{len(samples)}  ({dt:.2f}s/pair)")
        # Write hidden states FIRST, then the table -- so an interruption
        # between them leaves an orphan .npz (which condition_done ignores)
        # rather than an orphan table (which the gate would try to load).
        np.savez(out_dir / f"{tag}_hidden.npz",
                 **{k: np.stack(v).astype(np.float32) for k, v in hidden_stacks.items()})
        save_table(pd.DataFrame(rows), out_dir / tag)
        print(f"[done] {tag}: {len(rows)} rows, "
              f"acc={pd.DataFrame(rows)['correct'].mean():.3f}, "
              f"{(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
