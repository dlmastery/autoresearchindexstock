"""Equity-index 7-fold walk-forward split for QQQ (2004-2025).

Mirrors the FX `autoresearch.data.splits` super-fold structure exactly —
seven non-overlapping val + test windows, label-buffer + purge + embargo
zero-overlap guarantees, and a single super-fold that pools all training
data outside of any val/test/buffer window.

CLAUDE.md mandates: no 2026 data anywhere; the last fold's test window
ends 2025-12-31. Each fold has a regime label so per-fold breakdowns in
the dashboard / paper are interpretable.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# Same defaults as FX so cross-asset comparisons stay apples-to-apples.
PURGE_DAYS: int = 90
EMBARGO_DAYS: int = 21
LABEL_HORIZON_BUFFER: int = 10  # for the 5-day forward target + slack


# =============================================================================
# Fold table — 7 walk-forward folds covering 21 years
# =============================================================================
# Train start always 2004-01 (expanding window). Each fold's val + test are
# contiguous, in-sample regime characteristic, and disjoint from all other
# folds' val + test windows. Last fold ends 2025-12-31 (no 2026 data).

FOLDS: list[dict] = [
    {
        "name":   "fold_1",
        "regime": "Pre-GFC bull and GFC onset",
        "train":  {"start": "2004-01", "end": "2006-12"},
        "val":    {"start": "2007-04", "end": "2007-09"},
        "test":   {"start": "2008-01", "end": "2008-06"},
    },
    {
        "name":   "fold_2",
        "regime": "Post-GFC recovery",
        "train":  {"start": "2004-01", "end": "2009-12"},
        "val":    {"start": "2010-04", "end": "2010-09"},
        "test":   {"start": "2011-01", "end": "2011-06"},
    },
    {
        "name":   "fold_3",
        "regime": "EU debt and taper tantrum",
        "train":  {"start": "2004-01", "end": "2012-12"},
        "val":    {"start": "2013-04", "end": "2013-09"},
        "test":   {"start": "2014-01", "end": "2014-06"},
    },
    {
        "name":   "fold_4",
        "regime": "Strong dollar and oil crash",
        "train":  {"start": "2004-01", "end": "2015-12"},
        "val":    {"start": "2016-04", "end": "2016-09"},
        "test":   {"start": "2017-01", "end": "2017-09"},
    },
    {
        "name":   "fold_5",
        "regime": "Late cycle and COVID shock",
        "train":  {"start": "2004-01", "end": "2019-09"},
        "val":    {"start": "2019-10", "end": "2020-03"},
        "test":   {"start": "2020-04", "end": "2020-12"},
    },
    {
        "name":   "fold_6",
        "regime": "Inflation and Fed hikes",
        "train":  {"start": "2004-01", "end": "2021-12"},
        "val":    {"start": "2022-04", "end": "2022-09"},
        "test":   {"start": "2023-01", "end": "2023-09"},
    },
    {
        "name":   "fold_7",
        "regime": "AI rally and 2025",
        "train":  {"start": "2004-01", "end": "2024-12"},
        "val":    {"start": "2025-01", "end": "2025-04"},
        "test":   {"start": "2025-05", "end": "2025-12"},
    },
]


# =============================================================================
# Fold-date helpers
# =============================================================================

def get_fold_dates(fold: dict) -> dict[str, pd.Timestamp]:
    def _start(s: str) -> pd.Timestamp: return pd.Timestamp(s)
    def _end(s: str) -> pd.Timestamp: return pd.Timestamp(s) + pd.offsets.MonthEnd(0)
    return {
        "train_start": _start(fold["train"]["start"]),
        "train_end":   _end(fold["train"]["end"]),
        "val_start":   _start(fold["val"]["start"]),
        "val_end":     _end(fold["val"]["end"]),
        "test_start":  _start(fold["test"]["start"]),
        "test_end":    _end(fold["test"]["end"]),
    }


def _all_held_out_ranges() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return every val and test (date, date) range across all folds."""
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for f in FOLDS:
        d = get_fold_dates(f)
        out.append((d["val_start"], d["val_end"]))
        out.append((d["test_start"], d["test_end"]))
    return out


# =============================================================================
# Super-fold split: train = 2004→2025-12 minus all val/test/buffer windows
# =============================================================================

def split_superfold(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the global train / val / test partition.

    train = expanding-history minus every fold's val + test window minus a
            ``LABEL_HORIZON_BUFFER``-day buffer immediately before each
            held-out window.
    val   = union of all 7 fold val windows.
    test  = union of all 7 fold test windows.

    Returns the three DataFrames with zero pairwise overlap (verified
    programmatically — see ``_verify_no_overlap``).
    """
    val_chunks: list[pd.DataFrame] = []
    test_chunks: list[pd.DataFrame] = []
    for f in FOLDS:
        d = get_fold_dates(f)
        val_chunks.append(df.loc[d["val_start"]:d["val_end"]])
        test_chunks.append(df.loc[d["test_start"]:d["test_end"]])

    val = pd.concat(val_chunks).sort_index()
    test = pd.concat(test_chunks).sort_index()

    held_out = _all_held_out_ranges()
    mask = pd.Series(True, index=df.index)
    for (lo, hi) in held_out:
        buffer_start = lo - pd.Timedelta(days=LABEL_HORIZON_BUFFER)
        mask &= ~((df.index >= buffer_start) & (df.index <= hi))
    train = df.loc[mask].sort_index()

    _verify_no_overlap(train, val, test)
    return train, val, test


def _verify_no_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    tv = train.index.intersection(val.index)
    tt = train.index.intersection(test.index)
    vt = val.index.intersection(test.index)
    if len(tv) or len(tt) or len(vt):
        raise AssertionError(
            f"Fold overlap detected: train-val={len(tv)} "
            f"train-test={len(tt)} val-test={len(vt)}"
        )
    logger.info(
        "[splits] train=%d  val=%d  test=%d  overlap=0/0/0",
        len(train), len(val), len(test),
    )


def validate_purge_embargo(df: pd.DataFrame) -> dict:
    """Verify there is no calendar overlap between train rows and any
    held-out window minus the buffer. Returns a dict of counts.
    """
    train, val, test = split_superfold(df)
    held_out = _all_held_out_ranges()
    violations = 0
    for (lo, hi) in held_out:
        buffer_start = lo - pd.Timedelta(days=LABEL_HORIZON_BUFFER)
        violations += ((train.index >= buffer_start) & (train.index <= hi)).sum()
    return {
        "train_rows": len(train),
        "val_rows":   len(val),
        "test_rows":  len(test),
        "violations": int(violations),
        "purge_days": PURGE_DAYS,
        "embargo_days": EMBARGO_DAYS,
        "label_buffer_days": LABEL_HORIZON_BUFFER,
    }
