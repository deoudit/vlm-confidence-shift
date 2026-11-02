"""DisasterVQA loader (natural-shift benchmark), written against the ACTUAL
schema confirmed by inspection:

    columns: question_id, image_id, image (PIL, bundled), question,
             question_type (Binary | Multiple-Choice | Open-Ended),
             groundtruth_answer (list[str]),
             choices (dict {letter: text} | None),
             disaster_type, dataset_source, ...

Key facts established from the data:
  * MCQ rows have `choices` as a DICT keyed by letter, e.g.
        {'A': 'Bridge', 'B': 'Road', 'C': 'Building', 'D': 'Dam'}
  * `groundtruth_answer` is a LIST OF LETTERS, e.g. ['B'] (single) or
    ['A','B'] ("select all that apply", multi-answer).
  * Of 1693 MCQ rows: 1434 single-answer, 189 two-, 61 three-, 9 four-answer.

We keep ONLY single-answer MCQ rows (len(groundtruth_answer)==1), because the
whole pipeline (single-letter confidence, one-correct-answer ECE and selective
prediction) assumes exactly one correct answer. Multi-answer "select all that
apply" items are a different (multi-label) task and are excluded. Some retained
questions keep "select all" phrasing but have a single ground-truth letter.

NATURAL SHIFT, NO SEVERITY AXIS:
    DisasterVQA images are already naturally degraded (real disaster imagery).
    Run at severity 0 only (no synthetic ImageNet-C on top) -- this is the
    clean-vs-natural-shift comparison, not a severity grid.
"""
from __future__ import annotations

import random

from .data import Sample

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def load_disastervqa(split: str = "train",
                     n_questions: int | None = None,
                     seed: int = 0,
                     hf_name: str = "QCRI/DisasterVQA",
                     local_dir: str | None = "data/disastervqa") -> list[Sample]:
    """Load single-answer DisasterVQA Multiple-Choice items as `Sample` objects."""
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as e:
        raise ImportError("needs `datasets`: pip install datasets") from e

    import os
    if local_dir is not None and os.path.isdir(local_dir):
        print(f"[data] loading DisasterVQA from local copy: {local_dir}")
        ds = load_from_disk(local_dir)
        if hasattr(ds, "keys"):
            ds = ds[list(ds.keys())[0]]
    else:
        print(f"[data] downloading {hf_name} (split={split}) from HF Hub")
        ds = load_dataset(hf_name, split=split)

    kept = []
    n_mcq = n_multi = n_badchoices = n_badans = 0
    for i in range(len(ds)):
        ex = ds[i]
        if str(ex.get("question_type", "")).strip().lower() not in (
                "multiple-choice", "multiple choice", "mcq"):
            continue
        n_mcq += 1

        gt = ex.get("groundtruth_answer")
        if not isinstance(gt, (list, tuple)) or len(gt) == 0:
            n_badans += 1
            continue
        if len(gt) > 1:                      # multi-answer -> exclude
            n_multi += 1
            continue

        choices = ex.get("choices")
        if not isinstance(choices, dict) or len(choices) < 2:
            n_badchoices += 1
            continue

        gt_letter = str(gt[0]).strip().upper()
        norm = {}
        for k, v in choices.items():
            kk = str(k).strip().upper()
            if kk in _LETTERS:
                norm[kk] = str(v)
        if gt_letter not in norm:
            n_badans += 1
            continue

        kept.append((i, norm, gt_letter))

    print(f"[data] DisasterVQA MCQ rows: {n_mcq} total; kept {len(kept)} "
          f"single-answer; excluded {n_multi} multi-answer, "
          f"{n_badchoices} bad-choices, {n_badans} answer-unmatched")

    if not kept:
        raise ValueError("No usable single-answer MCQ items in DisasterVQA.")

    if n_questions is not None and len(kept) > n_questions:
        rng = random.Random(seed)
        kept = rng.sample(kept, n_questions)
        kept.sort()

    samples: list[Sample] = []
    for i, norm, gt_letter in kept:
        ex = ds[i]
        letters = [L for L in _LETTERS if L in norm]
        options = {L: norm[L] for L in letters}
        img = ex["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        samples.append(Sample(
            index=int(i),
            image=img,
            question=str(ex["question"]),
            hint="",
            options=options,
            answer=gt_letter,
            category=str(ex.get("disaster_type", "disastervqa")),
        ))
    return samples
