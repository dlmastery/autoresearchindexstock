"""QQQ OOS Top-30 inference â€” adapted from SPY run_oos_top30.py.

For each top-30-by-composite QQQ experiment with an archived checkpoint:
1) Load checkpoint (auto-detect mamba d_state/expand from state_dict shapes)
2) Run sliding-window OOS inference over the OOS window
3) Write 16-column CSV (CLAUDE.md Directive 64 schema)
4) Add equity_curve to the row dict (so dashboard sparklines render)

Output: autoresearch_results/oos_top30_table.json + per-exp CSVs.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SPY_ROOT = Path(r"C:\Users\evija\autoresearchindexspy\autoresearchspy")
if SPY_ROOT.exists():
    sys.path.insert(0, str(SPY_ROOT))

try:
    from autoresearchspy.run_autoresearch import _pin_to_safe_cores
    _pin_to_safe_cores()
except Exception as e:
    print(f"[warn] could not pin cores: {e}")

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from data.features import compute_qqq_features, compute_qqq_targets  # noqa
from data.download import ALL_SIGNALS  # noqa
from autoresearchspy.model.backbone import create_model  # noqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_top30_qqq")

RESULTS = ROOT / "autoresearch_results"
WINNERS = RESULTS / "winners"
JSONL = RESULTS / "experiment_log.jsonl"
START_CAPITAL = 1000.0


def _download_no_cap(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz else pd.to_datetime(df.index)
    return df


def download_for_oos(start: str, end: str) -> dict:
    out = {}
    fetched = 0
    for group_name, group in ALL_SIGNALS.items():
        for ticker in group:
            try:
                df = _download_no_cap(ticker, start, end)
                if df is not None and not df.empty:
                    out[ticker] = df
                    fetched += 1
            except Exception as e:
                log.warning("Skip %s: %s", ticker, e)
    log.info("[download] %d tickers fetched (%s -> %s)", fetched, start, end)
    return out


def find_checkpoint_for_exp(exp_num: int) -> Path | None:
    """Search winners/ for any dir matching exp{N}."""
    if not WINNERS.exists():
        return None
    for d in WINNERS.iterdir():
        if not d.is_dir():
            continue
        if f"exp{exp_num}_" in d.name or f"_exp{exp_num}_" in d.name or d.name.endswith(f"exp{exp_num}"):
            for fn in ("model_checkpoint.pt", "model_checkpoint.pkl"):
                p = d / fn
                if p.exists():
                    return p
    # Try root best_model.pt as fallback only for the global champion
    bm = RESULTS / "best_model.pt"
    if bm.exists():
        try:
            ck = torch.load(str(bm), map_location="cpu", weights_only=False)
            if int(ck.get("experiment_num", -1)) == int(exp_num):
                return bm
        except Exception:
            pass
    return None


def detect_arch_from_state_dict(sd: dict, backbone: str) -> dict:
    """Extract architecture HPs from state_dict shapes when config dict is incomplete."""
    arch = {}
    if backbone == "mamba":
        # Detect variant: dmamba has trend_mlp.* keys; samba/hybrid have other markers
        has_trend = any(k.startswith("trend_mlp.") for k in sd)
        has_chronos = any("chronos" in k for k in sd)
        if has_trend:
            arch["mamba_variant"] = "dmamba"
        elif has_chronos:
            arch["mamba_variant"] = "samba"
        for key in sd:
            if key.endswith(".A_log"):
                arch["d_state"] = sd[key].shape[1]
                break
        # x_proj output dim = 2 * d_state + dt_rank; with default dt_rank=1 we have 2*d_state+1
        for key in sd:
            if key.endswith(".x_proj.weight"):
                out_dim = sd[key].shape[0]
                # If d_state known, derive dt_rank
                ds = arch.get("d_state", 16)
                dt_rank = max(1, out_dim - 2 * ds)
                arch["dt_rank"] = dt_rank
                # expand: blocks.X.in_proj.weight shape is (2 * d_inner, d_model)
                # d_inner = expand * d_model, default d_model=256
                break
        for key in sd:
            if key.endswith(".in_proj.weight"):
                d_inner_x2 = sd[key].shape[0]  # 2 * d_inner
                d_model = sd[key].shape[1]
                arch["expand"] = (d_inner_x2 // 2) // d_model
                break
    return arch


def run_oos_for_checkpoint(ckpt_path: Path, raw: dict, exp_num: int,
                           oos_start: str, oos_end: str) -> dict:
    log.info("Loading %s ...", ckpt_path.relative_to(ROOT))
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    feature_cols = ck.get("feature_columns") or []
    sm = ck.get("scaler_mean")
    sc = ck.get("scaler_scale")
    scaler_mean = np.asarray(sm if sm is not None else [], dtype=float)
    scaler_scale = np.asarray(sc if sc is not None else [], dtype=float)
    backbone = ck.get("backbone", cfg.get("backbone", "mlp"))
    seq_len = int(cfg.get("seq_len", 10))
    n_features = int(ck.get("n_features", len(feature_cols) or 1))

    # Auto-detect arch HPs from state_dict (handles missing d_state/expand in cfg)
    detected_arch = detect_arch_from_state_dict(ck["model_state_dict"], backbone)
    arch_keys = {"hidden_size", "num_layers", "head_dropout", "seq_len",
                 "mamba_d_state", "mamba_expand", "mamba_variant"}
    arch_cfg = {k: v for k, v in cfg.items() if k in arch_keys and v is not None}
    # Map detected d_state/expand to SPY backbone signature names
    if "d_state" in detected_arch:
        arch_cfg["mamba_d_state"] = detected_arch["d_state"]
    if "expand" in detected_arch:
        arch_cfg["mamba_expand"] = detected_arch["expand"]
    if "mamba_variant" in detected_arch:
        arch_cfg["mamba_variant"] = detected_arch["mamba_variant"]
    arch_cfg.setdefault("seq_len", seq_len)

    log.info("  backbone=%s seq_len=%d n_features=%d arch_cfg=%s",
             backbone, seq_len, n_features, arch_cfg)
    try:
        model = create_model(backbone, n_features, **arch_cfg)
        model.load_state_dict(ck["model_state_dict"])
    except Exception as e:
        return {"experiment_num": exp_num, "error": f"load_failed: {str(e)[:200]}",
                "n_predictions": 0}
    model.eval()

    # Compute features over OOS window
    feats = compute_qqq_features(raw)
    tgts = compute_qqq_targets(raw)
    df_idx = feats.join(tgts, how="inner")
    if feature_cols:
        missing = [c for c in feature_cols if c not in df_idx.columns]
        for m in missing:
            df_idx[m] = 0.0
        X_df = df_idx[feature_cols].copy()
    else:
        X_df = df_idx.iloc[:, :n_features].copy()
    X_df = X_df.fillna(0.0)  # Per Directive 64: don't dropna(); fill missing features with 0 instead
    X = X_df.values
    log.info("  X_df shape after fillna: %s, date range %s -> %s",
             X.shape, X_df.index.min(), X_df.index.max())
    if scaler_mean.size == X.shape[1]:
        X = (X - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
    actual_1d_full = df_idx.get("fwd_ret_1d", pd.Series(0.0, index=df_idx.index))

    inf_start = pd.Timestamp(oos_start)
    inf_end = pd.Timestamp(oos_end)
    rows = []
    with torch.no_grad():
        for i in range(seq_len, len(X_df)):
            predict_date = X_df.index[i]
            if predict_date < inf_start or predict_date > inf_end:
                continue
            window = X[i - seq_len: i]
            x = torch.from_numpy(window).float().unsqueeze(0)
            try:
                out = model(x)
            except Exception:
                continue
            if isinstance(out, dict):
                mu = float(out.get("ret_1d", torch.zeros(1, 1))[:, 0].numpy()[0])
            elif isinstance(out, tuple):
                mu = float(out[0].numpy().ravel()[0])
            else:
                mu = float(out.numpy().ravel()[0])
            actual = float(actual_1d_full.get(predict_date, np.nan))
            direction = int(np.sign(mu)) if mu != 0 else 0
            traded = int(direction != 0)
            pnl = direction * actual if not np.isnan(actual) else 0.0
            correct = int((np.sign(mu) == np.sign(actual)) and traded) if not np.isnan(actual) else 0
            rows.append({
                "date": predict_date.strftime("%Y-%m-%d"),
                "position": float(direction),
                "pred_direction": direction,
                "traded": traded,
                "actual_ret_1d": actual if not np.isnan(actual) else 0.0,
                "bh_log_ret": actual if not np.isnan(actual) else 0.0,
                "strategy_pnl": pnl,
                "correct": correct,
                "pred_ret_1d": mu,
            })

    if not rows:
        return {"experiment_num": exp_num, "n_predictions": 0,
                "error": "no rows in OOS window"}

    df = pd.DataFrame(rows)
    df["equity_dollars"] = (START_CAPITAL * np.exp(df["strategy_pnl"].cumsum())).round(4)
    df["buy_hold_dollars"] = (START_CAPITAL * np.exp(df["bh_log_ret"].cumsum())).round(4)
    df["excess_dollars"] = (df["equity_dollars"] - df["buy_hold_dollars"]).round(4)
    df["cumret_pct"] = ((df["equity_dollars"] / START_CAPITAL - 1) * 100).round(4)
    df["bh_cumret_pct"] = ((df["buy_hold_dollars"] / START_CAPITAL - 1) * 100).round(4)
    df["excess_cumret_pct"] = (df["cumret_pct"] - df["bh_cumret_pct"]).round(4)
    peak = df["equity_dollars"].cummax()
    df["drawdown_pct"] = ((df["equity_dollars"] - peak) / peak.replace(0, 1) * 100).round(4)
    df["underwater"] = (df["equity_dollars"] < peak).astype(int)
    canonical = ["date", "position", "pred_direction", "traded", "actual_ret_1d", "bh_log_ret",
                 "strategy_pnl", "correct", "equity_dollars", "buy_hold_dollars", "excess_dollars",
                 "cumret_pct", "bh_cumret_pct", "excess_cumret_pct", "drawdown_pct", "underwater",
                 "pred_ret_1d"]
    df = df[[c for c in canonical if c in df.columns]]

    csv_name = f"oos_exp{exp_num}.csv"
    csv_path = RESULTS / csv_name
    df.to_csv(csv_path, index=False, float_format="%.6f")

    valid = df[df["actual_ret_1d"] != 0]
    n_traded = int(df["traded"].sum())
    n_correct = int(df["correct"].sum())
    metrics = {
        "experiment_num": exp_num,
        "n_predictions": len(df),
        "n_with_actuals": len(valid),
        "n_traded_days": n_traded,
        "csv": csv_name,
        "checkpoint_source": str(ckpt_path.relative_to(RESULTS)),
        "hit_rate_pct": round((n_correct / max(1, n_traded)) * 100, 2) if n_traded else 0,
    }
    if len(valid) > 1 and valid["strategy_pnl"].std() > 0:
        metrics["strategy_annual_sharpe"] = round(
            (valid["strategy_pnl"].mean() / valid["strategy_pnl"].std()) * np.sqrt(252), 4)
    if len(valid) > 1 and valid["actual_ret_1d"].std() > 0:
        metrics["buy_hold_annual_sharpe"] = round(
            (valid["actual_ret_1d"].mean() / valid["actual_ret_1d"].std()) * np.sqrt(252), 4)
    metrics["excess_sharpe"] = round(
        metrics.get("strategy_annual_sharpe", 0) - metrics.get("buy_hold_annual_sharpe", 0), 4)
    metrics["strategy_total_return_pct"] = round(float(df["cumret_pct"].iloc[-1]), 4)
    metrics["buy_hold_total_return_pct"] = round(float(df["bh_cumret_pct"].iloc[-1]), 4)
    metrics["excess_return_pct"] = round(float(df["excess_cumret_pct"].iloc[-1]), 4)
    metrics["max_drawdown_pct"] = round(float(df["drawdown_pct"].min()), 4)
    metrics["equity_curve"] = {
        "dates": df["date"].tolist(),
        "strategy_dollars": [round(float(v), 2) for v in df["equity_dollars"].tolist()],
        "buy_hold_dollars": [round(float(v), 2) for v in df["buy_hold_dollars"].tolist()],
    }
    metrics["equity_curve"]["strategy_pct"] = [round((v / START_CAPITAL - 1) * 100, 4) for v in metrics["equity_curve"]["strategy_dollars"]]
    metrics["equity_curve"]["buy_hold_pct"] = [round((v / START_CAPITAL - 1) * 100, 4) for v in metrics["equity_curve"]["buy_hold_dollars"]]
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")  # 2-year download for SMA200 warmup
    p.add_argument("--end", default="2026-04-30")
    p.add_argument("--oos-start", default="2025-12-01")
    p.add_argument("--oos-end", default="2026-04-30")
    args = p.parse_args()

    with open(JSONL, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    top30 = sorted(lines, key=lambda e: -(e.get("composite") or -99))[:30]

    log.info("Downloading data %s -> %s ...", args.start, args.end)
    raw = download_for_oos(args.start, args.end)

    table = []
    for rank, e in enumerate(top30, 1):
        exp_num = e.get("experiment_num")
        ckpt_path = find_checkpoint_for_exp(exp_num)
        row = {
            "rank": rank,
            "experiment_num": exp_num,
            "backbone": e.get("backbone"),
            "seed": e.get("seed", e.get("config", {}).get("seed")),
            "description": (e.get("description") or "")[:80],
            "train_composite": round(e.get("composite") or 0, 4),
            "checkpoint_status": "available" if ckpt_path else "missing",
        }
        if ckpt_path is None:
            row["oos_status"] = "skipped â€” checkpoint not archived"
        else:
            log.info("[rank %d] exp%s â€” running OOS", rank, exp_num)
            try:
                oos = run_oos_for_checkpoint(ckpt_path, raw, exp_num, args.oos_start, args.oos_end)
                row.update({f"oos_{k}": v for k, v in oos.items() if k != "experiment_num"})
                row["oos_status"] = "completed" if oos.get("n_with_actuals", 0) > 0 else f"failed: {oos.get('error', 'unknown')}"
            except Exception as ex:
                row["oos_status"] = f"error: {str(ex)[:200]}"
                log.error("[rank %d] exp%s FAILED: %s", rank, exp_num, ex)
        table.append(row)

    out = {
        "oos_run_at": pd.Timestamp.utcnow().isoformat(),
        "oos_window": {"start": args.oos_start, "end": args.oos_end},
        "n_total_top30": len(table),
        "n_with_checkpoint": sum(1 for r in table if r["checkpoint_status"] == "available"),
        "n_completed": sum(1 for r in table if r.get("oos_status") == "completed"),
        "table": table,
    }
    out_path = RESULTS / "oos_top30_table.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    log.info("[done] %d completed / %d top30 -> %s",
             out["n_completed"], out["n_total_top30"], out_path.name)


if __name__ == "__main__":
    main()

