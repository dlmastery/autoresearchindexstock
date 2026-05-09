"""QQQ OOS inference — adapted from SPY's run_oos_inference.py.

Loads the QQQ champion checkpoint (autoresearch_results/best_model.pt),
re-fetches QQQ + features over an OOS window (Nov 2025 - Apr 2026 by
default), and writes daily prediction CSV using QQQ's local
data/features.py (compute_qqq_features) + the SPY-side autoresearchspy
package's model creation utilities (architecture-compatible).
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so `from data.features import ...` works
# Add SPY project so autoresearchspy.* (model.backbone, _pin_to_safe_cores) is importable
SPY_ROOT = Path(r"C:\Users\evija\autoresearchindexspy\autoresearchspy")
if SPY_ROOT.exists():
    sys.path.insert(0, str(SPY_ROOT))

# Pin CPU before torch import
try:
    from autoresearchspy.run_autoresearch import _pin_to_safe_cores  # noqa
    _pin_to_safe_cores()
except Exception as e:
    print(f"[warn] could not pin CPU cores: {e}")

import numpy as np
import pandas as pd
import torch
import yfinance as yf

# QQQ-local feature pipeline
from data.features import compute_qqq_features, compute_qqq_targets  # noqa
from data.download import ALL_SIGNALS  # noqa
from data.splits import FOLDS, get_fold_dates, LABEL_HORIZON_BUFFER, EMBARGO_DAYS, PURGE_DAYS  # noqa

# Architecture from the SPY-side package (same backbone classes as FX origins)
from autoresearchspy.model.backbone import create_model  # noqa

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_qqq")


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
    total = sum(len(g) for g in ALL_SIGNALS.values())
    fetched = 0
    for group_name, group in ALL_SIGNALS.items():
        for ticker in group:
            try:
                df = _download_no_cap(ticker, start, end)
            except Exception as e:
                log.warning("Skip %s (%s): %s", ticker, group_name, e)
                continue
            if df is None or df.empty:
                log.warning("Empty %s (%s)", ticker, group_name)
                continue
            out[ticker] = df
            fetched += 1
    log.info("[download oos] %d/%d tickers fetched (%s -> %s, NO 2026 cap)",
             fetched, total, start, end)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-11-01")
    p.add_argument("--end", default="2026-04-30")
    p.add_argument("--checkpoint", default=str(ROOT / "autoresearch_results" / "best_model.pt"))
    p.add_argument("--out", default=str(ROOT / "autoresearch_results" / "oos_predictions_qqq.csv"))
    args = p.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.exists():
        log.error("Checkpoint not found at %s", ck_path)
        sys.exit(1)

    log.info("Loading checkpoint %s ...", ck_path)
    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    feature_cols = ck.get("feature_columns") or []
    sm = ck.get("scaler_mean")
    sc = ck.get("scaler_scale")
    scaler_mean = np.asarray(sm if sm is not None else [], dtype=float)
    scaler_scale = np.asarray(sc if sc is not None else [], dtype=float)
    backbone = ck.get("backbone", cfg.get("backbone", "mlp"))
    seq_len = int(cfg.get("seq_len", 10))
    n_features = int(ck.get("n_features", len(feature_cols) or 1))
    log.info("Checkpoint backbone=%s seq_len=%d n_features=%d (composite=%s)",
             backbone, seq_len, n_features, ck.get("composite"))

    log.info("Building model architecture %s ...", backbone)
    # Only pass architecture HPs, not training HPs
    arch_keys = {"hidden_size", "num_layers", "head_dropout", "seq_len", "d_state",
                 "expand", "mamba_variant", "n_heads", "patch_len", "stride", "kernel_size",
                 "dropout", "max_depth", "num_leaves", "n_estimators"}
    arch_cfg = {k: v for k, v in cfg.items() if k in arch_keys and v is not None}
    arch_cfg.setdefault("seq_len", seq_len)
    try:
        model = create_model(backbone, n_features, **arch_cfg)
    except Exception as e:
        log.warning("create_model failed (%s), retrying without arch_cfg ...", e)
        model = create_model(backbone, n_features, seq_len)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()

    log.info("Downloading QQQ + cross-asset signals %s -> %s (no 2026 cap)", args.start, args.end)
    raw = download_for_oos(args.start, args.end)
    if "QQQ" not in raw and "qqq" not in raw:
        log.error("QQQ ticker not present in downloaded data. Aborting.")
        sys.exit(2)

    log.info("Computing QQQ features ...")
    feats = compute_qqq_features(raw)
    tgts = compute_qqq_targets(raw)
    df = feats.join(tgts, how="inner").dropna(subset=feature_cols if feature_cols else None)
    log.info("Feature matrix: %d rows x %d cols (asked for %d cols)",
             len(df), df.shape[1], len(feature_cols))

    # Reindex to checkpoint feature ordering, fill any missing with 0 (and warn loudly)
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            log.warning("Missing %d / %d expected feature cols (filling 0): %s ...",
                        len(missing), len(feature_cols), missing[:5])
            for m in missing:
                df[m] = 0.0
        X_df = df[feature_cols].copy()
    else:
        X_df = df.iloc[:, : n_features].copy()
    X = X_df.values
    if scaler_mean.size == X.shape[1]:
        X = (X - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)

    actual_1d = df.get("fwd_ret_1d", pd.Series(0.0, index=df.index)).values

    # Sliding-window inference
    rows = []
    for i in range(seq_len, len(X)):
        win = torch.from_numpy(X[i - seq_len : i]).float().unsqueeze(0)
        with torch.no_grad():
            try:
                out = model(win)
            except Exception:
                out = model(win.squeeze(0))
        if isinstance(out, tuple):
            out = out[0]
        out = out.detach().cpu().numpy().ravel()
        mu = float(out[0]) if len(out) > 0 else 0.0
        date = df.index[i].strftime("%Y-%m-%d")
        actual = float(actual_1d[i])
        direction = int(np.sign(mu))
        traded = int(direction != 0)
        pnl = direction * actual
        correct = int((np.sign(mu) == np.sign(actual)) and traded)
        rows.append({
            "date": date,
            "prediction": mu,
            "pred_direction": direction,
            "traded": traded,
            "actual_ret_1d": actual,
            "bh_log_ret": actual,
            "strategy_pnl": pnl,
            "correct": correct,
        })
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        # Compounded equity (CLAUDE.md Directive 64 schema)
        START = 1000.0
        out_df["equity_dollars"] = (START * np.exp(out_df["strategy_pnl"].cumsum())).round(4)
        out_df["buy_hold_dollars"] = (START * np.exp(out_df["bh_log_ret"].cumsum())).round(4)
        out_df["excess_dollars"] = (out_df["equity_dollars"] - out_df["buy_hold_dollars"]).round(4)
        out_df["cumret_pct"] = ((out_df["equity_dollars"] / START - 1) * 100).round(4)
        out_df["bh_cumret_pct"] = ((out_df["buy_hold_dollars"] / START - 1) * 100).round(4)
        out_df["excess_cumret_pct"] = (out_df["cumret_pct"] - out_df["bh_cumret_pct"]).round(4)
        peak = out_df["equity_dollars"].cummax()
        out_df["drawdown_pct"] = ((out_df["equity_dollars"] - peak) / peak.replace(0, 1) * 100).round(4)
        out_df["underwater"] = (out_df["equity_dollars"] < peak).astype(int)
        # Position (mu used for sizing — surface the prediction itself)
        out_df["position"] = out_df["pred_direction"].astype(float)
        # Reorder to canonical 16-col schema + prediction extra
        canonical = ["date","position","pred_direction","traded","actual_ret_1d","bh_log_ret",
                     "strategy_pnl","correct","equity_dollars","buy_hold_dollars","excess_dollars",
                     "cumret_pct","bh_cumret_pct","excess_cumret_pct","drawdown_pct","underwater",
                     "prediction"]
        out_df = out_df[[c for c in canonical if c in out_df.columns]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    log.info("Wrote %d predictions -> %s", len(out_df), out_path)

    if not out_df.empty:
        n_traded = int(out_df["traded"].sum())
        n_correct = int(out_df["correct"].sum())
        hit = (n_correct / max(1, n_traded)) * 100
        sh_pnl = out_df["strategy_pnl"]
        sh = float(sh_pnl.mean() / sh_pnl.std() * np.sqrt(252)) if sh_pnl.std() > 0 else 0.0
        bh_pnl = out_df["bh_log_ret"]
        bh_sh = float(bh_pnl.mean() / bh_pnl.std() * np.sqrt(252)) if bh_pnl.std() > 0 else 0.0
        log.info("OOS summary: n=%d traded=%d hit=%.2f%% strategy_sharpe=%.3f bh_sharpe=%.3f $final=%.2f",
                 len(out_df), n_traded, hit, sh, bh_sh, out_df['equity_dollars'].iloc[-1])


if __name__ == "__main__":
    main()
