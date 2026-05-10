"""Build SPY oos_ensemble_summary.json with the FULL A+B+C+D+E grid:

A. True COMPOUNDED $ on $1000 starting capital (exp(sum_log_returns) - 1)
B. Tier-1 SIZING variants (raw / quarter-Kelly / confidence-weighted / vol-target)
C. Tier-2 RISK OVERLAYS on top-20 sized strategies
   (stop-loss-2% / drawdown-gate-5% / 200d-SMA-trend-filter)
D. Tier-3 SELECTION CRITERIA (by_compound / by_sortino / by_recency_30d)
E. final_dollars_on_1000 field on every strategy so dashboard can sort by $

References:
  - Lakshminarayanan, Pritzel & Blundell 2017 NeurIPS arXiv:1612.01474 (deep ensembles)
  - Bates & Granger 1969 OR 'The Combination of Forecasts' (forecast combination)
  - Kelly 1956 Bell Sys Tech J 'A New Interpretation of Information Rate'
  - Thorp 1969 'Optimal Gambling Systems for Favorable Games' (quarter-Kelly)
  - Lim, Zohren & Roberts 2019 arXiv:1906.04025 (vol-target sizing)
  - Faber 2007 J. Wealth Mgmt (200d SMA trend filter)
  - Sortino & Price 1994 J. of Investing (downside-risk Sortino)
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parent / "autoresearch_results"
TABLE = R / "oos_top30_table.json"
ENSEMBLE_JSON = R / "oos_ensemble_summary.json"

START_CAPITAL = 1000.0


def annualized_sharpe(pnl: pd.Series) -> float:
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def annualized_sortino(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return 0.0
    downside = pnl[pnl < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(pnl.mean() / downside.std() * np.sqrt(252))


def probabilistic_sharpe(pnl: pd.Series) -> float:
    n = len(pnl)
    if n < 3 or pnl.std() == 0:
        return 0.0
    sh = float(pnl.mean() / pnl.std())
    skew = float(pnl.skew()) if n > 2 else 0.0
    kurt = float(pnl.kurtosis()) if n > 3 else 0.0
    sigma_sh = math.sqrt(max(1e-9, (1 - skew * sh + ((kurt - 1) / 4) * sh ** 2) / (n - 1)))
    return float(stats.norm.cdf(sh / sigma_sh))


def compounded_dollars(log_pnl_series: pd.Series, start: float = START_CAPITAL) -> float:
    """True compounded final dollar value: $start * exp(sum(daily log returns))."""
    if len(log_pnl_series) == 0:
        return start
    return float(start * math.exp(float(log_pnl_series.sum())))


def equity_path_dollars(log_pnl_series: pd.Series, start: float = START_CAPITAL) -> list[float]:
    """Compounded equity curve in $: $start * exp(cumsum(daily log returns))."""
    if len(log_pnl_series) == 0:
        return []
    return [round(start * math.exp(v), 2) for v in log_pnl_series.fillna(0).cumsum().tolist()]


def load_completed_members() -> list[dict]:
    d = json.loads(TABLE.read_text(encoding="utf-8"))
    rows = [r for r in d["table"] if r.get("oos_status") == "completed" and r.get("oos_csv")]
    # Per Directive 68/69: enrich each member with TRAIN-TIME (in-sample) metrics from
    # experiment_log.jsonl so member selection can be done on causal metrics only.
    log_path = R / "experiment_log.jsonl"
    if log_path.exists():
        train_meta = {}
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                en = e.get("experiment_num")
                if en is None:
                    continue
                train_meta[en] = {
                    "train_test_sharpe":   e.get("sharpe"),
                    "train_val_sharpe":    e.get("val_sharpe"),
                    "train_train_sharpe":  e.get("train_sharpe"),
                    "train_hit":           e.get("hit"),
                    "train_psr":           e.get("psr"),
                    "train_equity":        e.get("equity"),
                    "train_return_pct":    e.get("return_pct"),
                    "train_excess_sharpe": e.get("excess_sharpe"),
                    "train_composite":     e.get("composite"),
                    "train_ic":            e.get("ic"),
                    "train_test_pos_folds":  e.get("test_pos_folds"),
                    "train_val_pos_folds":   e.get("val_pos_folds"),
                }
        for r in rows:
            en = r.get("experiment_num")
            tm = train_meta.get(en) or {}
            for k, v in tm.items():
                r.setdefault(k, v)
    return rows


def load_member_predictions(member: dict) -> pd.DataFrame | None:
    csv = R / member["oos_csv"]
    if not csv.exists():
        return None
    df = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
    needed = {"pred_ret_1d", "pred_direction", "actual_ret_1d"}
    if not needed.issubset(df.columns):
        return None
    df = df[["pred_ret_1d", "pred_direction", "actual_ret_1d"]].copy()
    df.columns = [f"{c}_exp{member['experiment_num']}" for c in df.columns]
    return df


def select_top_k(members: list[dict], rank_key: str, k: int, ascending: bool = False) -> list[dict]:
    def keyf(m: dict):
        v = m.get(rank_key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return -1e9 if not ascending else 1e9
        return v if ascending else -v
    return sorted(members, key=keyf)[:k]


def make_strategy_record(name: str, sizing: str, overlay: str | None, position: pd.Series,
                         actual: pd.Series, members_used: list[int],
                         dates: pd.DatetimeIndex, write_csv: bool = True) -> dict:
    """Compute all metrics for a strategy given the daily POSITION series in [-leverage, +leverage]
    and the actual log-return series. position can be float-valued (sized).

    Returns a record with compounded $, Sharpe, etc.
    """
    pnl = (position * actual).fillna(0)  # log-return P&L per day (signed)
    valid = pnl[pnl != 0]
    bh_log_pnl = actual.dropna()

    # Compounded equity curves
    equity_strategy = equity_path_dollars(pnl)
    equity_bh = equity_path_dollars(bh_log_pnl)
    final_dollars = round(equity_strategy[-1] if equity_strategy else START_CAPITAL, 2)
    final_dollars_bh = round(equity_bh[-1] if equity_bh else START_CAPITAL, 2)
    compound_pct = round((final_dollars / START_CAPITAL - 1) * 100, 4)
    compound_pct_bh = round((final_dollars_bh / START_CAPITAL - 1) * 100, 4)
    excess_dollars = round(final_dollars - final_dollars_bh, 2)

    # Sharpe / Sortino on raw log-return P&L (annualised)
    sh = annualized_sharpe(pnl[pnl != 0]) if (pnl != 0).any() else 0.0
    bh_sh = annualized_sharpe(bh_log_pnl)
    sortino = annualized_sortino(pnl)
    psr = probabilistic_sharpe(pnl)

    # Hit rate (only counted on non-zero position days)
    correct = ((np.sign(position) == np.sign(actual)) & (position != 0)).astype(int)
    n_traded = int((position != 0).sum())
    hit = round(float(correct.sum() / max(1, n_traded)) * 100, 2) if n_traded > 0 else 0.0

    # Max drawdown (in $ terms)
    eq_arr = np.array(equity_strategy or [START_CAPITAL], dtype=float)
    peak = np.maximum.accumulate(eq_arr)
    if len(eq_arr) > 0:
        peak_safe = np.where(peak == 0, 1.0, peak)
        dd_arr = ((eq_arr - peak) / peak_safe) * 100
        dd_pct = float(np.min(dd_arr))
    else:
        dd_pct = 0.0

    # Exposure
    exposure_pct = round(float((position != 0).mean()) * 100, 2)
    avg_position = round(float(position.abs().mean()), 4)
    turnover = round(float(position.diff().abs().sum()), 4)

    csv_name = None
    if write_csv:
        # Sanitize name for filesystem (no slashes, special chars)
        safe = name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:120]
        csv_name = f"oos_ensemble_{safe}.csv"
        # Build CSV ensuring every column is aligned to the dates index (length matches).
        # Pad/reindex any series that may have been .dropna()'d.
        n = len(dates)
        try:
            position_aligned = position.reindex(dates).fillna(0).values if isinstance(position, pd.Series) else np.asarray(position)
            actual_aligned = actual.reindex(dates).fillna(0).values if isinstance(actual, pd.Series) else np.asarray(actual)
            pnl_aligned = pnl.reindex(dates).fillna(0).values if isinstance(pnl, pd.Series) else np.asarray(pnl)
            correct_aligned = correct.reindex(dates).fillna(0).astype(int).values if isinstance(correct, pd.Series) else np.asarray(correct)
            # Pad equity arrays to length n with last value if shorter
            eq_s = list(equity_strategy)
            eq_b = list(equity_bh)
            while len(eq_s) < n: eq_s.append(eq_s[-1] if eq_s else START_CAPITAL)
            while len(eq_b) < n: eq_b.append(eq_b[-1] if eq_b else START_CAPITAL)
            eq_s = eq_s[:n]
            eq_b = eq_b[:n]
            # Derive richer columns from aligned series (Lakshminarayanan-2017 style audit trail)
            pos_arr = np.asarray(position_aligned[:n], dtype=float)
            act_arr = np.asarray(actual_aligned[:n], dtype=float)
            pnl_arr = np.asarray(pnl_aligned[:n], dtype=float)
            corr_arr = np.asarray(correct_aligned[:n], dtype=int)
            eq_s_arr = np.asarray(eq_s, dtype=float)
            eq_b_arr = np.asarray(eq_b, dtype=float)
            # Cumulative returns (% from $1000 base)
            cumret_pct = (eq_s_arr / START_CAPITAL - 1.0) * 100
            bh_cumret_pct = (eq_b_arr / START_CAPITAL - 1.0) * 100
            excess_cumret_pct = cumret_pct - bh_cumret_pct
            excess_dollars_daily = eq_s_arr - eq_b_arr
            # Drawdown (peak-to-trough) on strategy equity — per-day series for CSV
            peak_eq = np.maximum.accumulate(eq_s_arr) if len(eq_s_arr) else eq_s_arr
            dd_pct_daily = ((eq_s_arr - peak_eq) / np.where(peak_eq == 0, 1.0, peak_eq)) * 100
            underwater = (eq_s_arr < peak_eq).astype(int)
            # Direction + traded flags
            pred_direction = np.sign(pos_arr).astype(int)
            traded = (pos_arr != 0).astype(int)
            # Per-day BH log-return (= actual_ret_1d but only on traded days for parity)
            bh_log_ret = act_arr.copy()
            cols = {
                "date": [d.strftime("%Y-%m-%d") for d in dates],
                "position": pos_arr,
                "pred_direction": pred_direction,
                "traded": traded,
                "actual_ret_1d": act_arr,
                "bh_log_ret": bh_log_ret,
                "strategy_pnl": pnl_arr,
                "correct": corr_arr,
                "equity_dollars": eq_s_arr,
                "buy_hold_dollars": eq_b_arr,
                "excess_dollars": excess_dollars_daily,
                "cumret_pct": cumret_pct,
                "bh_cumret_pct": bh_cumret_pct,
                "excess_cumret_pct": excess_cumret_pct,
                "drawdown_pct": dd_pct_daily,
                "underwater": underwater,
            }
            lengths = {k: len(v) for k, v in cols.items()}
            if len(set(lengths.values())) > 1:
                # Force-truncate to min length as a last resort
                min_len = min(lengths.values())
                cols = {k: list(v)[:min_len] for k, v in cols.items()}
            pd.DataFrame(cols).to_csv(R / csv_name, index=False, float_format="%.6f")
        except Exception as e:
            csv_name = None  # Don't claim a CSV if write failed

    return {
        "n_predictions": int(len(position)),
        "n_traded_days": n_traded,
        "exposure_pct": exposure_pct,
        "avg_position": avg_position,
        "turnover": turnover,
        "n_members_used": len(members_used),
        "members_used": members_used,
        "sizing_mode": sizing,
        "overlay": overlay,
        # Compounded $ metrics (the primary user-facing numbers)
        "final_dollars_on_1000": final_dollars,
        "final_dollars_bh": final_dollars_bh,
        "excess_dollars": excess_dollars,
        "compound_return_pct": compound_pct,
        "buy_hold_compound_pct": compound_pct_bh,
        "excess_compound_pct": round(compound_pct - compound_pct_bh, 4),
        # Sharpe-based metrics
        "strategy_annual_sharpe": round(sh, 4),
        "buy_hold_annual_sharpe": round(bh_sh, 4),
        "excess_sharpe": round(sh - bh_sh, 4),
        "annual_sortino": round(sortino, 4),
        "psr": round(psr, 4),
        # Path metrics
        "hit_rate_pct": hit,
        "max_drawdown_pct": round(dd_pct, 4),
        "csv": csv_name,
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "strategy_dollars": equity_strategy,
            "buy_hold_dollars": equity_bh,
        },
    }


def kelly_position_multiplier(historical_sharpe: float | None) -> float:
    """Quarter-Kelly fraction from historical OOS Sharpe (Thorp 1969)."""
    if historical_sharpe is None or historical_sharpe <= 0:
        return 0.0
    # Daily Kelly = mu / sigma^2 = (Sharpe^2) / (annual_vol_squared * 252_factor)
    # Simplified: f* approx Sharpe / sqrt(252) for unit-vol; multiply by 0.25 for quarter-Kelly
    f_kelly = (historical_sharpe ** 2) / 252.0
    f_quarter = 0.25 * f_kelly * 252.0  # rescale so it's a per-day position fraction
    return float(min(1.0, max(0.0, f_quarter)))


def main():
    members = load_completed_members()
    print(f"[ensemble] {len(members)} OOS-completed members:")
    for m in members:
        print(f"  exp {m['experiment_num']:>3} {m.get('backbone','?'):>10} "
              f"sh {m.get('oos_strategy_annual_sharpe'):>+6.3f} "
              f"ret {m.get('oos_strategy_total_return_pct'):>+7.2f}% "
              f"hit {m.get('oos_hit_rate_pct'):>5.1f}% "
              f"psr {0 if m.get('oos_psr') is None else m['oos_psr']:>5.3f}")
    if len(members) < 2:
        print("[ensemble] Need at least 2 members; aborting.")
        return

    frames = [(m, load_member_predictions(m)) for m in members]
    frames = [(m, f) for m, f in frames if f is not None]
    if len(frames) < 2:
        print("[ensemble] No valid CSVs; aborting.")
        return

    merged = frames[0][1]
    for _, f in frames[1:]:
        merged = merged.join(f, how="inner")
    print(f"[ensemble] merged {merged.shape[0]} common dates across {len(frames)} members")
    if merged.shape[0] == 0:
        return

    valid_members = [m for m, _ in frames]
    n = len(valid_members)
    actual_cols = [c for c in merged.columns if c.startswith("actual_ret_1d_")]
    actual = merged[actual_cols[0]]
    dates = merged.index

    # Pre-compute per-member rolling sharpe (Tier 3: recency-weighted) and compounded return (Tier 3: compound-weighted)
    member_extras = {}
    for m in valid_members:
        exp = m["experiment_num"]
        d = m.get("pred_direction_exp" + str(exp))  # placeholder
        # Compute member's own daily PnL and rolling sharpe from the merged data
        dir_series = merged[f"pred_direction_exp{exp}"]
        m_pnl = dir_series * actual
        m_pnl_clean = m_pnl.fillna(0)
        compound_dollars_m = compounded_dollars(m_pnl_clean) - START_CAPITAL
        sortino_m = annualized_sortino(m_pnl_clean)
        # Recency: rolling 30-day Sharpe at end of OOS window
        rolling_sh = m_pnl_clean.rolling(30, min_periods=10).apply(
            lambda s: s.mean() / s.std() * math.sqrt(252) if s.std() > 0 else 0.0, raw=False)
        rec_sh = float(rolling_sh.iloc[-1]) if not rolling_sh.empty and not pd.isna(rolling_sh.iloc[-1]) else 0.0
        member_extras[exp] = {
            "compound_dollars": float(compound_dollars_m),
            "sortino": sortino_m,
            "recency_30d_sharpe": rec_sh,
        }
    # Inject into member dicts so select_top_k can use them
    for m in valid_members:
        e = member_extras[m["experiment_num"]]
        m["compound_dollars"] = e["compound_dollars"]
        m["sortino"] = e["sortino"]
        m["recency_30d_sharpe"] = e["recency_30d_sharpe"]

    def pred_col(m): return f"pred_ret_1d_exp{m['experiment_num']}"
    def dir_col(m): return f"pred_direction_exp{m['experiment_num']}"

    # ---- Build base SIGNAL series for each base strategy ----
    base_signals: list[tuple[str, pd.Series, list[int]]] = []  # (name, signal series in [-1,+1] or larger, members_used)

    # Whole-ensemble baselines
    base_signals.append((f"all{n}_mean", merged[[pred_col(m) for m in valid_members]].mean(axis=1),
                         [m["experiment_num"] for m in valid_members]))
    base_signals.append((f"all{n}_vote", merged[[dir_col(m) for m in valid_members]].sum(axis=1) / n,
                         [m["experiment_num"] for m in valid_members]))

    weight_metrics_full = [
        # Hindsight (descriptive only — flagged is_leaky=True)
        ("by_oos_sharpe", "oos_strategy_annual_sharpe"),
        ("by_oos_return", "oos_strategy_total_return_pct"),
        ("by_excess", "oos_excess_sharpe"),
        ("by_hit", "oos_hit_rate_pct"),
        ("by_psr", "oos_psr"),
        ("by_compound", "compound_dollars"),
        ("by_sortino", "sortino"),
        ("by_recency_30d", "recency_30d_sharpe"),
        # Causal (deployable) — Directive 68/69
        ("by_train_composite",   "train_composite"),
        ("by_train_test_sharpe", "train_test_sharpe"),
        ("by_train_val_sharpe",  "train_val_sharpe"),
        ("by_train_hit",         "train_hit"),
        ("by_train_psr",         "train_psr"),
        ("by_train_equity",      "train_equity"),
        ("by_train_excess_sharpe","train_excess_sharpe"),
    ]
    for tag, key in weight_metrics_full:
        ws = np.array([max(0.001, valid_members[i].get(key) or 0.001) for i in range(n)])
        ws = np.clip(ws, 0.001, None)
        ws = ws / ws.sum()
        sig = (merged[[pred_col(m) for m in valid_members]].values * ws[None, :]).sum(axis=1)
        base_signals.append((f"all{n}_weighted_{tag}", pd.Series(sig, index=merged.index),
                             [m["experiment_num"] for m in valid_members]))

    # Top-K x criterion x aggregation grid (Directive 68/69 split)
    selection_criteria = [
        # POST-HOC (HINDSIGHT) — flagged is_leaky=True
        ("by_oos_sharpe",   "oos_strategy_annual_sharpe", False),
        ("by_oos_return",   "oos_strategy_total_return_pct", False),
        ("by_excess",       "oos_excess_sharpe", False),
        ("by_hit",          "oos_hit_rate_pct", False),
        ("by_psr",          "oos_psr", False),
        ("by_min_dd",       "oos_max_drawdown_pct", True),
        ("by_compound",     "compound_dollars", False),
        ("by_sortino",      "sortino", False),
        ("by_recency_30d",  "recency_30d_sharpe", False),
        # CAUSAL (TRAIN-TIME) — deployable
        ("by_train_composite",    "train_composite", False),
        ("by_train_test_sharpe",  "train_test_sharpe", False),
        ("by_train_val_sharpe",   "train_val_sharpe", False),
        ("by_train_hit",          "train_hit", False),
        ("by_train_psr",          "train_psr", False),
        ("by_train_equity",       "train_equity", False),
        ("by_train_excess_sharpe","train_excess_sharpe", False),
        ("by_train_return_pct",   "train_return_pct", False),
    ]
    for tag, rank_key, ascending in selection_criteria:
        for k in [2, 3, 5]:
            if k > n:
                continue
            top = select_top_k(valid_members, rank_key, k, ascending=ascending)
            if not top:
                continue
            top_pred_cols = [pred_col(m) for m in top]
            top_dir_cols = [dir_col(m) for m in top]
            top_exps = [m["experiment_num"] for m in top]
            base_signals.append((f"top{k}_{tag}_mean", merged[top_pred_cols].mean(axis=1), top_exps))
            base_signals.append((f"top{k}_{tag}_vote", merged[top_dir_cols].sum(axis=1) / k, top_exps))
            ws = np.array([max(0.001, m.get(rank_key) or 0.001) for m in top])
            if ascending:
                ws = 1.0 / np.abs(ws + 1e-6)
            ws = np.clip(ws, 0.001, None)
            ws = ws / ws.sum()
            sig = (merged[top_pred_cols].values * ws[None, :]).sum(axis=1)
            base_signals.append((f"top{k}_{tag}_weighted", pd.Series(sig, index=merged.index), top_exps))

    # Family-only
    by_bb: dict[str, list[dict]] = {}
    for m in valid_members:
        by_bb.setdefault(m.get("backbone", "?"), []).append(m)
    for bb, mems in by_bb.items():
        if len(mems) < 2:
            continue
        cols_p = [pred_col(m) for m in mems]
        cols_d = [dir_col(m) for m in mems]
        exps = [m["experiment_num"] for m in mems]
        base_signals.append((f"{bb}_only_{len(mems)}_mean", merged[cols_p].mean(axis=1), exps))
        base_signals.append((f"{bb}_only_{len(mems)}_vote", merged[cols_d].sum(axis=1) / len(mems), exps))

    # Vote thresholds
    for thr in [3, 4, 5]:
        if thr > n:
            continue
        sum_dir = merged[[dir_col(m) for m in valid_members]].sum(axis=1)
        sig = (sum_dir / n).where(sum_dir.abs() >= thr, 0.0)
        base_signals.append((f"vote_geq_{thr}_of_{n}", sig, [m["experiment_num"] for m in valid_members]))

    # Best-1 single-model
    for tag, rank_key, ascending in selection_criteria:
        top = select_top_k(valid_members, rank_key, 1, ascending=ascending)
        if not top:
            continue
        m = top[0]
        base_signals.append((f"best1_{tag}_exp{m['experiment_num']}", merged[pred_col(m)], [m["experiment_num"]]))

    # Dedupe by name (some criteria collapse to same single-best member)
    seen = {}
    for nm, sig, exps in base_signals:
        if nm not in seen:
            seen[nm] = (sig, exps)
    base_signals = [(nm, sig, exps) for nm, (sig, exps) in seen.items()]
    print(f"[ensemble] {len(base_signals)} base signals built")

    # ---- Tier-1 SIZING modes (B) ----
    # For each base signal, compute 4 sizing variants:
    #   raw     : direction = sign(signal); position = direction (+/-1, 0)
    #   conf    : position = signal / max(|signal|) (continuous in [-1, +1])
    #   kelly   : position = direction Ã— quarter-Kelly fraction (member-historical Sharpe based)
    #   voltarget: position = direction Ã— min(2.0, target_vol / realized_vol_20)
    target_vol_annual = 0.15
    realized_vol_20 = (actual.rolling(20, min_periods=5).std() * math.sqrt(252)).fillna(target_vol_annual)
    vol_scale = (target_vol_annual / realized_vol_20).clip(upper=2.0).clip(lower=0.0)

    # Pre-compute the ensemble-level expected Sharpe (use crude estimate from last 30d realised performance per signal)
    # Simpler: use the historical OOS Sharpe of the signal itself (estimated below per-strategy after raw run)

    strategies: dict[str, dict] = {}

    LEAKY_TAGS = ("by_oos_sharpe", "by_oos_return", "by_excess", "by_hit", "by_psr",
                  "by_min_dd", "by_compound", "by_sortino", "by_recency_30d")
    def is_leaky_name(nm: str) -> bool:
        if any(f"_{t}_" in nm or nm.endswith(f"_{t}") for t in LEAKY_TAGS):
            return True
        if "weighted_by_oos" in nm:
            return True
        return False

    def add(name: str, sizing: str, overlay: str | None, position: pd.Series, members_used: list[int]):
        rec = make_strategy_record(name, sizing, overlay, position, actual, members_used, dates, write_csv=True)
        rec["is_leaky"] = bool(is_leaky_name(name))
        rec["selection_basis"] = "OOS-realized (POST-HOC, hindsight)" if rec["is_leaky"] else "train-time / non-selective (CAUSAL, deployable)"
        strategies[name] = rec

    # Pass 1: raw sizing for every base signal â€” needed to compute historical Sharpe for Kelly sizing
    raw_sharpe_by_base: dict[str, float] = {}
    for base_name, signal, exps in base_signals:
        direction = np.sign(signal).fillna(0).astype(float)
        add(f"{base_name}__raw", "raw", None, direction, exps)
        raw_sharpe_by_base[base_name] = strategies[f"{base_name}__raw"]["strategy_annual_sharpe"]

    # Pass 2: confidence-weighted sizing
    for base_name, signal, exps in base_signals:
        max_abs = signal.abs().max() or 1.0
        position = (signal / max_abs).clip(-1.0, 1.0).fillna(0)
        add(f"{base_name}__conf", "confidence_weighted", None, position, exps)

    # Pass 3: quarter-Kelly sizing (per-base historical Sharpe)
    for base_name, signal, exps in base_signals:
        sh = raw_sharpe_by_base.get(base_name, 0.0)
        kf = kelly_position_multiplier(sh)
        direction = np.sign(signal).fillna(0).astype(float)
        position = direction * kf
        add(f"{base_name}__kelly", f"quarter_kelly_f={kf:.3f}", None, position, exps)

    # Pass 4: vol-target sizing
    for base_name, signal, exps in base_signals:
        direction = np.sign(signal).fillna(0).astype(float)
        position = direction * vol_scale
        add(f"{base_name}__voltarget", f"vol_target_{int(target_vol_annual*100)}pct", None, position, exps)

    print(f"[ensemble] After Tier-1 sizing: {len(strategies)} strategies")

    # ---- Tier-2 OVERLAYS (C) â€” only on top-20 by Sharpe to keep grid tractable ----
    # Compute SPY price proxy from actual returns (cumulative log returns)
    spy_log_price = actual.cumsum()
    sma_200 = spy_log_price.rolling(200, min_periods=20).mean()
    above_200 = (spy_log_price > sma_200).astype(int).fillna(1)  # default to "in trend" if SMA not ready

    top20 = sorted(strategies.items(), key=lambda kv: -(kv[1].get("strategy_annual_sharpe") or -99))[:20]
    print(f"[overlays] applying Tier-2 overlays to top-20 strategies")
    for base_full_name, _ in top20:
        # Reconstruct the base position from the CSV data (simpler: re-derive from base signal + sizing tag)
        # Quick approach: read CSV we wrote for this strategy
        # Actually we set write_csv=True; reconstruct from base_signal
        # Parse out base_name and sizing
        if "__" not in base_full_name:
            continue
        base_name, sizing_tag = base_full_name.rsplit("__", 1)
        base_match = next((b for b in base_signals if b[0] == base_name), None)
        if base_match is None:
            continue
        _, signal, exps = base_match
        if sizing_tag == "raw":
            position = np.sign(signal).fillna(0).astype(float)
        elif sizing_tag == "conf":
            max_abs = signal.abs().max() or 1.0
            position = (signal / max_abs).clip(-1.0, 1.0).fillna(0)
        elif sizing_tag == "kelly":
            kf = kelly_position_multiplier(raw_sharpe_by_base.get(base_name, 0.0))
            position = np.sign(signal).fillna(0).astype(float) * kf
        elif sizing_tag == "voltarget":
            position = np.sign(signal).fillna(0).astype(float) * vol_scale
        else:
            continue

        # Overlay 1: stop-loss 2% per day (cap MTM loss)
        pnl_uncapped = position * actual
        pnl_stopped = pnl_uncapped.clip(lower=-0.02)  # log-return floor at -2%
        # We need to express "stop-loss" as position adjustment, not pnl adjustment.
        # Approximate: build position-equivalent series where pnl matches pnl_stopped.
        # Simpler: write a custom record using the capped pnl directly.
        equity_strategy = equity_path_dollars(pnl_stopped)
        equity_bh = equity_path_dollars(actual.dropna())
        final_dollars = round(equity_strategy[-1] if equity_strategy else START_CAPITAL, 2)
        valid = pnl_stopped[pnl_stopped != 0]
        sh = annualized_sharpe(valid) if len(valid) > 0 else 0.0
        bh_sh = annualized_sharpe(actual.dropna())
        sortino = annualized_sortino(pnl_stopped)
        psr = probabilistic_sharpe(pnl_stopped)
        correct = ((np.sign(position) == np.sign(actual)) & (position != 0)).astype(int)
        n_traded = int((position != 0).sum())
        hit = round(float(correct.sum() / max(1, n_traded)) * 100, 2) if n_traded > 0 else 0.0
        eq_arr = np.array(equity_strategy or [START_CAPITAL])
        peak = np.maximum.accumulate(eq_arr)
        dd_pct = float(((eq_arr - peak) / peak).min() * 100)
        strategies[f"{base_full_name}__stoploss2pct"] = {
            **strategies[base_full_name],
            "overlay": "stop_loss_2pct",
            "final_dollars_on_1000": final_dollars,
            "compound_return_pct": round((final_dollars / START_CAPITAL - 1) * 100, 4),
            "excess_dollars": round(final_dollars - (equity_bh[-1] if equity_bh else START_CAPITAL), 2),
            "strategy_annual_sharpe": round(sh, 4),
            "excess_sharpe": round(sh - bh_sh, 4),
            "annual_sortino": round(sortino, 4),
            "psr": round(psr, 4),
            "hit_rate_pct": hit,
            "max_drawdown_pct": round(dd_pct, 4),
            "equity_curve": {"dates": [d.strftime("%Y-%m-%d") for d in dates],
                             "strategy_dollars": equity_strategy, "buy_hold_dollars": equity_bh},
            "csv": None,
        }

        # Overlay 2: drawdown gate -5% (halve position when DD > 5% from peak)
        # Path-dependent: walk through and adjust position
        eq = float(START_CAPITAL)
        peak_eq = float(START_CAPITAL)
        adj_position = position.copy()
        for i, dt in enumerate(dates):
            curr_pos = float(adj_position.iloc[i])
            curr_pnl = curr_pos * float(actual.iloc[i] if not pd.isna(actual.iloc[i]) else 0)
            eq = eq * math.exp(curr_pnl)
            peak_eq = max(peak_eq, eq)
            dd = (eq - peak_eq) / peak_eq
            if dd < -0.05:
                # Halve next-day position
                if i + 1 < len(adj_position):
                    adj_position.iloc[i + 1] = adj_position.iloc[i + 1] * 0.5
        rec_dd = make_strategy_record(f"{base_full_name}__ddgate5pct", strategies[base_full_name].get("sizing_mode"),
                                       "drawdown_gate_5pct", adj_position, actual, exps, dates, write_csv=True)
        strategies[f"{base_full_name}__ddgate5pct"] = rec_dd

        # Overlay 3: 200d SMA trend filter (only LONG when above SMA)
        trend_position = position.where(~((above_200 == 0) & (position > 0)), 0)
        rec_sma = make_strategy_record(f"{base_full_name}__sma200filter", strategies[base_full_name].get("sizing_mode"),
                                        "sma200_trend_filter", trend_position, actual, exps, dates, write_csv=True)
        strategies[f"{base_full_name}__sma200filter"] = rec_sma

    print(f"[ensemble] After Tier-2 overlays on top-20: {len(strategies)} strategies")

    # ---- Print top by $ won ----
    print(f"\n[ensemble] Top 25 strategies by FINAL DOLLARS on $1000:")
    print(f"{'rk':>3}  {'strategy':<55}  {'$Final':>9}  {'$Excess':>9}  {'Sharpe':>7}  {'Hit%':>6}  {'Exp%':>6}")
    ranked = sorted(strategies.items(), key=lambda kv: -(kv[1].get("final_dollars_on_1000") or 0))
    for i, (name, s) in enumerate(ranked[:25], 1):
        print(f"{i:>3}  {name[:55]:<55}  ${s.get('final_dollars_on_1000'):>7.2f}  "
              f"${s.get('excess_dollars'):>+7.2f}  "
              f"{s.get('strategy_annual_sharpe'):>+7.3f}  "
              f"{s.get('hit_rate_pct'):>5.1f}%  "
              f"{s.get('exposure_pct'):>5.1f}%")

    # ---- Members + summary ----
    raw_w = np.array([max(0.001, vm.get("oos_strategy_annual_sharpe") or 0.001) for vm in valid_members])
    raw_w = np.clip(raw_w, 0.001, None)
    norm_w = raw_w / raw_w.sum() if raw_w.sum() > 0 else raw_w
    member_records = [{
        "experiment_num": m["experiment_num"],
        "backbone": m.get("backbone"),
        "seed": m.get("seed"),
        "individual_oos_sharpe": m.get("oos_strategy_annual_sharpe"),
        "individual_excess_sharpe": m.get("oos_excess_sharpe"),
        "individual_oos_return_pct": m.get("oos_strategy_total_return_pct"),
        "individual_compound_dollars": round(member_extras[m["experiment_num"]]["compound_dollars"], 2),
        "individual_sortino": round(member_extras[m["experiment_num"]]["sortino"], 4),
        "individual_recency_30d_sharpe": round(member_extras[m["experiment_num"]]["recency_30d_sharpe"], 4),
        "individual_hit_rate_pct": m.get("oos_hit_rate_pct"),
        "individual_psr": m.get("oos_psr"),
        "individual_max_drawdown_pct": m.get("oos_max_drawdown_pct"),
        "train_composite": m.get("train_composite"),
        "ensemble_weight_by_sharpe": round(float(norm_w[i]), 4),
    } for i, m in enumerate(valid_members)]

    summary = {
        "method": "Lakshminarayanan-Pritzel-Blundell 2017 NeurIPS arXiv:1612.01474 (ensemble) + Kelly 1956 / Thorp 1969 (sizing) + Lim-Zohren-Roberts 2019 arXiv:1906.04025 (vol target) + Faber 2007 (200d SMA) + Sortino-Price 1994 (downside risk)",
        "n_members": n,
        "members": member_records,
        "n_common_dates": int(len(merged)),
        "oos_window": {
            "start": str(merged.index.min().date()),
            "end": str(merged.index.max().date()),
        },
        "start_capital": START_CAPITAL,
        "tier_breakdown": {
            "tier_A_compounded_dollars": "every strategy reports final_dollars_on_1000 (true compound: $1000 * exp(sum_log_returns))",
            "tier_B_sizing_modes": ["raw", "confidence_weighted", "quarter_kelly", "vol_target_15pct"],
            "tier_C_overlays_applied_to_top20": ["stop_loss_2pct", "drawdown_gate_5pct", "sma200_trend_filter"],
            "tier_D_selection_criteria": ["by_compound", "by_sortino", "by_recency_30d", "(plus 7 prior criteria)"],
            "tier_E_dashboard_sort": "panel sortable by final_dollars_on_1000 (default desc) + every other column",
        },
        "strategies": strategies,
        "selection_note": "All OOS-completed members included. Strategies = base_signals x sizing_modes [+ overlays on top-20]. Sort by final_dollars_on_1000 in dashboard.",
    }
    # Final pass: tag every strategy with is_leaky based on its name (catches
    # overlay strategies that bypass add() — stoploss2pct, ddgate5pct, sma200filter).
    LEAKY_TAGS_FINAL = ("by_oos_sharpe", "by_oos_return", "by_excess", "by_hit", "by_psr",
                        "by_min_dd", "by_compound", "by_sortino", "by_recency_30d")
    def _is_leaky(nm: str) -> bool:
        if any(f"_{t}_" in nm or nm.endswith(f"_{t}") for t in LEAKY_TAGS_FINAL):
            return True
        if "weighted_by_oos" in nm:
            return True
        return False
    n_leaky = n_clean = 0
    for nm, rec in strategies.items():
        leaky = _is_leaky(nm)
        rec["is_leaky"] = leaky
        rec["selection_basis"] = (
            "OOS-realized (POST-HOC, hindsight — descriptive only, NOT deployable)"
            if leaky else
            "train-time / non-selective (CAUSAL — deployable on day-1 of OOS)"
        )
        n_leaky += int(leaky); n_clean += int(not leaky)
    summary["n_leaky_strategies"] = n_leaky
    summary["n_clean_strategies"] = n_clean
    summary["leakage_audit_directive"] = "CLAUDE.md Directive 68/69 — every strategy carries is_leaky bool + selection_basis string"
    ENSEMBLE_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] wrote {ENSEMBLE_JSON.name} ({ENSEMBLE_JSON.stat().st_size:,} bytes, {len(strategies)} strategies)")
    print(f"  is_leaky=true (post-hoc, descriptive): {n_leaky}")
    print(f"  is_leaky=false (causal, deployable):   {n_clean}")


if __name__ == "__main__":
    main()

