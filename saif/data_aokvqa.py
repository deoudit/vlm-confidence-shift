"""A-OKVQA multiple-choice loader, producing the SAME `Sample` objects as the
MMBench loader so the rest of the pipeline (prompt builder, corruptions,
extractor) works unchanged.

A-OKVQA schema (HuggingFaceM4/A-OKVQA, which bundles the COCO images):
    question            : str
    choices             : list[str]   (exactly 4 for the MC task)
    correct_choice_idx  : int in 0..3
    image               : PIL image (already decoded by `datasets`)

We map choices -> letters A/B/C/D in order, and the correct index -> its letter.
This mirrors MMBench's Sample so `build_mcq_prompt`, corruptions, and the
extractor need no changes.

Usage (mirrors load_mmbench):
    from saif.data_aokvqa import load_aokvqa
    samples = load_aokvqa(split="validation", n_questions=2000, seed=0)

Requires: `datasets` (pip install datasets). The HF dataset download needs
network access the first time; it is then cached.
"""
from __future__ import annotations

import random

from .data import Sample  # reuse the exact same dataclass

_LETTERS = ["A", "B", "C", "D", "E", "F"]


def load_aokvqa(split: str = "validation",
                n_questions: int | None = None,
                seed: int = 0,
                hf_name: str = "HuggingFaceM4/A-OKVQA",
                local_dir: str | None = "data/aokvqa_val") -> list[Sample]:
    """Load A-OKVQA MC task as a list of `Sample`, matching load_mmbench().

    Data source, in priority order:
      1. A local copy saved by scripts/download_aokvqa.py at `local_dir`
         (read OFFLINE via load_from_disk) -- preferred, mirrors MMBench's
         local-file pattern and needs no network at extraction time.
      2. Otherwise, download from the Hugging Face Hub `hf_name` (needs network;
         cached after first call).

    split: "validation" (~1.1k, recommended) or "train" (17k). Not "test"
           (labels withheld).
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as e:
        raise ImportError(
            "A-OKVQA loading needs the `datasets` library: "
            "pip install datasets") from e

    import os
    if local_dir is not None and os.path.isdir(local_dir):
        print(f"[data] loading A-OKVQA from local copy: {local_dir}")
        ds = load_from_disk(local_dir)
    else:
        print(f"[data] local copy not found; downloading {hf_name} "
              f"(split={split}) from Hugging Face Hub")
        ds = load_dataset(hf_name, split=split)

    # A-OKVQA test split has correct_choice_idx = -1 (withheld); guard against it.
    def _has_label(ex):
        idx = ex.get("correct_choice_idx", -1)
        return idx is not None and idx >= 0 and len(ex.get("choices", [])) >= 2

    indices = [i for i in range(len(ds)) if _has_label(ds[i])]
    if not indices:
        raise ValueError(
            f"No labelled examples in A-OKVQA split '{split}'. "
            f"Use 'validation' or 'train' (not 'test').")

    # deterministic subsample to n_questions
    if n_questions is not None and len(indices) > n_questions:
        rng = random.Random(seed)
        indices = rng.sample(indices, n_questions)
        indices.sort()

    samples: list[Sample] = []
    for i in indices:
        ex = ds[i]
        choices = list(ex["choices"])
        n_opt = len(choices)
        if n_opt > len(_LETTERS):
            choices = choices[:len(_LETTERS)]
            n_opt = len(_LETTERS)
        letters = _LETTERS[:n_opt]
        options = {L: str(txt) for L, txt in zip(letters, choices)}

        gt_idx = int(ex["correct_choice_idx"])
        if gt_idx >= n_opt:            # safety: skip malformed
            continue
        answer = letters[gt_idx]

        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        samples.append(Sample(
            index=int(i),
            image=img,
            question=str(ex["question"]),
            hint="",                  # A-OKVQA has no hint field
            options=options,
            answer=answer,
            category="aokvqa",        # single flat category
        ))

    if not samples:
        raise ValueError("A-OKVQA produced 0 usable samples after filtering.")
    return samples
