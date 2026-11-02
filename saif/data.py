"""MMBench dev loading + MCQ prompt construction.

MMBench dev ships as a TSV with base64-encoded images and columns:
index, question, hint, A, B, C, D, answer, category, l2-category, image, ...
Perception subset = rows whose L2 category belongs to the perception taxonomy.

Why MMBench MCQ: the answer is a single letter, so correctness labels are
clean and the output-family signals (max-softmax over the letter tokens,
top-2 margin, entropy) are exactly defined — no free-form answer matching.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import pandas as pd
from PIL import Image

# MMBench taxonomy: L1 = {perception, reasoning}; the perception branch has
# three L2 abilities (coarse_perception, finegrained_perception instance-level
# and cross-instance), which fan out into leaf categories (image_style, ocr,
# ...). Different TSV releases store either level in the "l2-category" column,
# so the filter accepts both: exact leaf names OR any L2 value containing
# "perception" (which precisely selects the perception branch and excludes
# attribute/logic/relation_reasoning).
PERCEPTION_L2 = {
    # leaf-level names (older/legacy style)
    "image_style", "image_scene", "image_emotion", "image_quality",
    "image_topic", "object_localization", "attribute_recognition",
    "celebrity_recognition", "ocr", "action_recognition",
    "attribute_comparison", "spatial_relationship",
    # mid-level L2 ability names (VLMEvalKit-style dev TSV)
    "coarse_perception",
    "finegrained_perception (instance-level)",
    "finegrained_perception (cross-instance)",
}

LETTERS = ["A", "B", "C", "D"]


@dataclass
class Sample:
    index: int
    image: Image.Image
    question: str
    hint: str
    options: dict[str, str]      # letter -> option text (only letters present)
    answer: str                  # ground-truth letter
    category: str


def _norm(x: str) -> str:
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def load_mmbench(tsv_path: str, perception_only: bool = True,
                 n_questions: int | None = None, seed: int = 0) -> list[Sample]:
    df = pd.read_csv(tsv_path, sep="\t")
    l2col = next((c for c in df.columns
                  if _norm(c) in ("l2_category", "l2category")), None)
    # VLMEvalKit-format TSVs deduplicate images: some rows' `image` field holds
    # an integer index pointing at the row that carries the actual base64
    # (Legacy-format TSVs store base64 in every row). Resolve references so
    # both formats load identically.
    img_by_index = dict(zip(df["index"], df["image"]))

    def resolve_image(val):
        seen = set()
        while isinstance(val, (int, float)) or (
                isinstance(val, str) and len(val) < 64 and val.strip().isdigit()):
            key = int(val)
            if key in seen:
                raise ValueError(f"circular image reference at index {key}")
            seen.add(key)
            val = img_by_index[key]
        return val

    if perception_only and l2col is not None:
        wanted = {_norm(c) for c in PERCEPTION_L2}
        l2n = df[l2col].map(_norm)
        mask = l2n.isin(wanted) | l2n.str.contains("perception", regex=False)
        if mask.sum() == 0:
            raise ValueError(
                "Perception filter matched 0 rows. The TSV's L2 categories are:\n  "
                + ", ".join(sorted(df[l2col].map(str).unique()))
                + "\nUpdate PERCEPTION_L2 in saif/data.py to match (verify against"
                " the MMBench taxonomy), or set data.perception_only: false in the"
                " config to run on all categories.")
        df = df[mask]
    elif perception_only:
        print(f"[data] WARNING: no l2-category column found in {tsv_path}; "
              f"columns = {list(df.columns)}. Loading ALL rows.")
    if n_questions is not None and len(df) > n_questions:
        df = df.sample(n=n_questions, random_state=seed)
    samples = []
    for _, row in df.iterrows():
        opts = {L: str(row[L]) for L in LETTERS
                if L in row and pd.notna(row[L]) and str(row[L]) != "nan"}
        img = Image.open(io.BytesIO(base64.b64decode(resolve_image(row["image"])))).convert("RGB")
        samples.append(Sample(
            index=int(row["index"]),
            image=img,
            question=str(row["question"]),
            hint=str(row["hint"]) if "hint" in row and pd.notna(row["hint"]) else "",
            options=opts,
            answer=str(row["answer"]).strip(),
            category=str(row.get(l2col, "")) if l2col else "",
        ))
    return samples


def build_mcq_prompt(s: Sample) -> str:
    """Single-letter-answer MCQ prompt. The model's next token after the chat
    template's generation prompt should be the answer letter, which is what
    the whole signal-extraction pipeline conditions on."""
    parts = []
    if s.hint:
        parts.append(f"Hint: {s.hint}")
    parts.append(f"Question: {s.question}")
    parts.append("Options:")
    for L, text in s.options.items():
        parts.append(f"{L}. {text}")
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(parts)
