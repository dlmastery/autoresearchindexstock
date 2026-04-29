"""Evaluation metrics for the equity-index variant.

Re-exports the proven FX metric functions (Sharpe, Sortino, PSR, IC, hit
rate, classification metrics) and adds two equity-specific metrics:

* ``buy_and_hold_metrics(returns)`` — long-only baseline.
* ``excess_sharpe(strategy_returns, market_returns)`` — strategy minus
  buy-and-hold Sharpe; the fair-comparison metric for a trending index.

The composite KEEP/DISCARD score is computed on the *primary* target
(1-day forward return per CLAUDE.md), but the runner additionally tracks
the four target variants A / B / C / D side-by-side.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# Re-export everything the runner needs from the FX evaluation module so we
# do not maintain two implementations of Sharpe / PSR / IC / classification.
from autoresearch.evaluation.metrics import (  # noqa: F401
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    information_coefficient,
    classification_metrics,
    trading_report,
    max_drawdown,
)


def sortino_ratio(returns) -> float:
    """Downside-risk-only Sharpe variant (Sortino 1994).

    The FX metrics module reports sortino inside ``trading_report`` but does
    not expose a top-level helper, so we implement the common annualised
    daily form here.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0: return 0.0
    downside = r[r < 0]
    if downside.size == 0: return 0.0
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return float(np.mean(r) / dd * np.sqrt(252)) if dd > 0 else 0.0


def buy_and_hold_metrics(market_returns: np.ndarray) -> Dict[str, float]:
    """Compute baseline (always-long) Sharpe / return for the same window.

    Parameters
    ----------
    market_returns : array-like, the realised per-period returns of the
        underlying instrument (NOT scaled by any prediction).
    """
    rets = np.asarray(market_returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return {"bh_sharpe": 0.0, "bh_return_pct": 0.0, "bh_n": 0}
    sh = float(sharpe_ratio(rets))
    cum = float(np.exp(np.log1p(rets).sum()) - 1.0) * 100.0
    return {"bh_sharpe": sh, "bh_return_pct": cum, "bh_n": int(rets.size)}


def excess_sharpe(strategy_returns: np.ndarray,
                  market_returns: np.ndarray) -> float:
    """Strategy Sharpe minus buy-and-hold Sharpe over the same window.

    Sharpe of a *long-only* baseline is the right benchmark for a trending
    equity index per Kahn 2018 ("Pure Alpha vs. Total Return") — we do not
    care about absolute Sharpe inflated by the trend.
    """
    s = np.asarray(strategy_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    n = min(len(s), len(m))
    s = s[:n]; m = m[:n]
    s = s[np.isfinite(s)]; m = m[np.isfinite(m)]
    if s.size == 0 or m.size == 0:
        return 0.0
    return float(sharpe_ratio(s)) - float(sharpe_ratio(m))


def evaluate_target_variant(predictions: np.ndarray,
                            actuals: np.ndarray,
                            label: str = "A") -> Dict[str, float]:
    """Compute strategy metrics for one target variant.

    Strategy: take the sign of the prediction, multiply by realised return.
    Returns a flat dict with prefix ``f"{label}_"`` so the runner can merge
    A/B/C/D into a single JSONL entry.
    """
    pred = np.asarray(predictions, dtype=float)
    act = np.asarray(actuals, dtype=float)
    n = min(len(pred), len(act))
    pred = pred[:n]; act = act[:n]
    mask = np.isfinite(pred) & np.isfinite(act)
    pred = pred[mask]; act = act[mask]
    if pred.size == 0:
        return {f"{label}_sharpe": 0.0, f"{label}_excess_sharpe": 0.0,
                f"{label}_return_pct": 0.0, f"{label}_hit_rate": 0.0,
                f"{label}_n": 0, f"{label}_bh_sharpe": 0.0}

    direction = np.sign(pred)
    strat = direction * act
    # Safety clip: trading_report computes cumulative log1p which goes
    # complex if any return <= -1. Real per-period returns on liquid
    # equities are always > -1; this clip is defensive against bad target
    # construction (e.g. vol-adjusted returns scaled outside (-1,1)).
    strat = np.clip(strat, -0.99, np.inf)
    try:
        rpt = trading_report(strat)
    except ZeroDivisionError:
        # All-positive downside or constant-loss series can produce
        # downside_std = 0 inside trading_report. Fall back to defaults
        # the runner expects so the pipeline does not crash on the rare
        # zero-vol-loss-bucket case.
        rpt = {
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "max_drawdown_pct": 0.0,
        }
    bh = buy_and_hold_metrics(act)
    excess = float(sharpe_ratio(strat)) - bh["bh_sharpe"]
    return {
        f"{label}_sharpe":        round(float(sharpe_ratio(strat)), 4),
        f"{label}_sortino":       round(float(sortino_ratio(strat)), 4),
        f"{label}_excess_sharpe": round(excess, 4),
        f"{label}_return_pct":    round(rpt["total_return_pct"], 2),
        f"{label}_hit_rate":      round(rpt["win_rate"], 2),
        f"{label}_n":             int(strat.size),
        f"{label}_bh_sharpe":     round(bh["bh_sharpe"], 4),
        f"{label}_bh_return_pct": round(bh["bh_return_pct"], 2),
        f"{label}_psr":           round(float(probabilistic_sharpe_ratio(strat)), 4),
        f"{label}_ic":            round(float(information_coefficient(pred, act)["ic_spearman"]), 4),
    }


def composite_score(test_sharpe: float, val_sharpe: float, n_neg_folds: int) -> float:
    """Same composite as the FX project — KEEP / DISCARD driver.

    composite = min(test_sharpe, val_sharpe) - 0.1 * n_negative_folds
    """
    return float(min(test_sharpe, val_sharpe) - 0.1 * n_neg_folds)
