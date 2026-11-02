"""Table I/O with graceful fallback: parquet when pyarrow is installed
(recommended on the lab machine), pickle otherwise. All scripts read/write
through this module so the pipeline never hard-fails on a storage backend.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    EXT = ".parquet"
except ImportError:      # sandbox / minimal envs
    EXT = ".pkl"


def save_table(df: pd.DataFrame, path_stem: Path) -> Path:
    """Save df as <stem>.parquet or <stem>.pkl depending on availability."""
    p = Path(path_stem).with_suffix(EXT)
    if EXT == ".parquet":
        df.to_parquet(p)
    else:
        df.to_pickle(p)
    return p


def load_table(p: Path) -> pd.DataFrame:
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_pickle(p)


def condition_done(features_dir: Path, tag: str) -> bool:
    d = Path(features_dir)
    return (d / f"{tag}.parquet").exists() or (d / f"{tag}.pkl").exists()


def condition_tables(features_dir: Path) -> list[Path]:
    """All per-condition tables, one per stem (excluding merged artifacts
    starting with '_'). If both .parquet and .pkl exist for the same tag,
    prefers .parquet."""
    d = Path(features_dir)
    files = sorted([*d.glob("*.parquet"), *d.glob("*.pkl")])
    by_stem = {}
    for f in files:
        if f.stem.startswith("_"):
            continue
        # first-seen wins; sorted() puts .parquet before .pkl alphabetically
        by_stem.setdefault(f.stem, f)
    return sorted(by_stem.values())
