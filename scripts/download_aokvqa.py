"""Explicitly download and verify A-OKVQA, so the source is visible and the
data is cached locally BEFORE any extraction run (mirrors how MMBench is a
local file you fetch once).

WHERE THE DATA COMES FROM:
    Hugging Face Hub dataset "HuggingFaceM4/A-OKVQA"
    https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA
    (public, ungated, bundles the COCO images -- no separate COCO download).

WHAT THIS DOES:
    1. Downloads the validation split via the `datasets` library.
    2. Prints the fields so you can confirm the schema matches the loader.
    3. Saves it to a local directory you control (default: ./data/aokvqa_val),
       so the extraction box can read it OFFLINE afterwards.

RUN THIS ONCE on a machine with network access:
    pip install datasets
    python scripts/download_aokvqa.py

If your GPU box is OFFLINE, run this on a connected machine, then copy the
saved folder (./data/aokvqa_val) to the same path on the GPU box. The loader
will read the local copy -- no network needed at extraction time.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", default="HuggingFaceM4/A-OKVQA")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--save_dir", default="data/aokvqa_val",
                    help="local dir to save to (read offline later)")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run:  pip install datasets")
        sys.exit(1)

    print(f"Source: Hugging Face Hub -> {args.hf_name} (split={args.split})")
    print("Downloading (first time only; cached afterwards)...")
    try:
        ds = load_dataset(args.hf_name, split=args.split)
    except Exception as e:
        print(f"\nDOWNLOAD FAILED: {e}")
        print("\nLikely causes:")
        print("  * this machine cannot reach huggingface.co (restricted egress)")
        print("  * -> run this script on a connected machine, then copy")
        print(f"       '{args.save_dir}' to the GPU box.")
        sys.exit(1)

    print(f"\nDownloaded {len(ds)} examples.")
    print("Fields present:", list(ds[0].keys()))

    # verify the fields the loader needs
    ex = ds[0]
    needed = ["question", "choices", "correct_choice_idx", "image"]
    missing = [k for k in needed if k not in ex]
    if missing:
        print(f"\nWARNING: expected fields missing: {missing}")
        print("The loader may need adjustment -- paste the field list above.")
    else:
        print("\nSchema OK. Sample:")
        print("  question           :", ex["question"])
        print("  choices            :", ex["choices"])
        print("  correct_choice_idx :", ex["correct_choice_idx"])
        print("  image              :", type(ex["image"]).__name__,
              getattr(ex["image"], "size", ""))

    # count labelled examples (validation should be fully labelled)
    labelled = sum(1 for i in range(len(ds))
                   if ds[i].get("correct_choice_idx", -1) is not None
                   and ds[i].get("correct_choice_idx", -1) >= 0)
    print(f"\nLabelled examples: {labelled}/{len(ds)}")

    # save locally for offline reuse
    import os
    os.makedirs(os.path.dirname(args.save_dir) or ".", exist_ok=True)
    ds.save_to_disk(args.save_dir)
    print(f"\nSaved to: {args.save_dir}")
    print("The loader can now read this offline via load_from_disk (see note).")


if __name__ == "__main__":
    main()
