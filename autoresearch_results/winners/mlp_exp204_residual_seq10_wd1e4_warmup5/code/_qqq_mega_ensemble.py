"""QQQ Mega-Ensemble — rank-averaged predictions from 5 backbone winners.

Equivalent of FX's `_emtsf_mega_ensemble.py`. Pulls the FX-champion HPs
(transferred to QQQ) for each of:
  XGBoost   (FX-Exp203: depth=4, lr=0.03, n_est=1500, seq=60)
  LightGBM  (FX-Exp235: depth=4, lr=0.01, n_est=2000, seq=60)
  CatBoost  (FX-Exp236: depth=4, lr=0.01, n_est=2000, seq=60)
  LSTM      (FX-Exp35:  hidden=128 bidir, lr=1e-3, bs=16, wd=7e-4, seed=42, seq=10)
  MLP       (FX-Exp32:  residual, head_dropout=0.25, lr=3e-4, seed=0, seq=10)

For each component:
  - Either re-trains on the QQQ super-fold (if no saved pickle found),
  - Or loads from autoresearch_results/winners/<id>/ (if saved),
  - Predicts on the test super-fold,
  - Aligns predictions to the LATEST common seq-start across components.

Then ensembles three ways (matching FX's findings):
  GBM 3-way rank-avg
  GBM 3-way zscore-avg
  MEGA rank-avg (3 GBMs + 1 LSTM)
  MEGA zscore-avg
  MEGA-5 rank-avg (3 GBMs + LSTM + MLP)

Outputs every variant's:
  - Sharpe (raw + excess vs buy-and-hold)
  - per-fold daily P&L (for trade-log CSV)

Goal: meet or beat FX mega-ensemble Sharpe **+9.7071**.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "C:/Users/evija/autoresearch")

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from autoresearchindexstock.data.download import download_all
from autoresearchindexstock.data.features import (
    compute_qqq_features, compute_qqq_targets,
)
from autoresearchindexstock.data.splits import (
    split_superfold, FOLDS, get_fold_dates,
)
from autoresearchindexstock.evaluation.metrics import (
    sharpe_ratio, trading_report, buy_and_hold_metrics,
)
from autoresearch.model.backbone import create_model, GBMWrapper
from autoresearch.model.train import train_one_fold

ROOT = Path("C:/Users/evija/autoresearch")
RESULTS = ROOT / "autoresearchindexstock" / "autoresearch_results"
WINNERS = RESULTS / "winners"
TRADE_LOGS = RESULTS / "trade_logs"


# =============================================================================
# Config — FX champion HPs ported verbatim
# =============================================================================
FX_CHAMP_CONFIGS = {
    "xgboost": dict(
        seq_len=60, gbm_type="xgboost",
        hp_overrides=dict(
            n_estimators=1500, max_depth=4, learning_rate=0.03,
            random_state=42,
        ),
    ),
    "lightgbm": dict(
        seq_len=60, gbm_type="lightgbm",
        hp_overrides=dict(
            n_estimators=2000, max_depth=4, learning_rate=0.01,
            random_state=42,
        ),
    ),
    "catboost": dict(
        seq_len=60, gbm_type="catboost",
        hp_overrides=dict(
            iterations=2000, depth=4, learning_rate=0.01,
            random_state=42,
        ),
    ),
    "lstm": dict(
        seq_len=10, backbone="lstm",
        hp=dict(lr=1e-3, batch_size=16, epochs=100, patience=15,
                weight_decay=7e-4, head_dropout=0.1, seed=42),
    ),
    "mlp": dict(
        seq_len=10, backbone="mlp",
        hp=dict(lr=3e-4, batch_size=32, epochs=50, patience=10,
                weight_decay=1e-5, head_dropout=0.25, seed=0),
    ),
}


# =============================================================================
# Helpers
# =============================================================================

def _flatten_window(arr: np.ndarray, seq: int) -> np.ndarray:
    return np.array([arr[i:i + seq].ravel() for i in range(len(arr) - seq + 1)])


def gbm_predict(wrapper, scaler_mean, scaler_scale, wf, wt, seq):
    ws = (wf.values - scaler_mean) / scaler_scale
    if len(ws) < seq + 1:
        return None, None, None
    X = _flatten_window(ws, seq)
    y = wt["fwd_ret_1d"].values[seq - 1:][:len(X)]
    dates = wt.index[seq - 1:][:len(X)]
    preds = wrapper.predict(X)[:, 0]
    return dates, preds, y


def neural_predict(model, scaler_mean, scaler_scale, wf, wt, seq, device):
    ws = (wf.values - scaler_mean) / scaler_scale
    if len(ws) < seq + 1:
        return None, None, None
    class _DS(Dataset):
        def __init__(self, f, t, L):
            self.f = torch.tensor(f, dtype=torch.float32); self.t = torch.tensor(t, dtype=torch.float32); self.L = L
        def __len__(self): return len(self.f) - self.L + 1
        def __getitem__(self, i): return self.f[i:i + self.L], self.t[i + self.L - 1]
    ds = _DS(ws, wt[["fwd_ret_1d", "fwd_ret_5d"]].values, seq)
    loader = DataLoader(ds, batch_size=256)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            preds.append(out["ret_1d"][:, 0].cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    y = wt["fwd_ret_1d"].values[seq - 1:][:len(preds)]
    dates = wt.index[seq - 1:][:len(preds)]
    return dates, preds, y


def rank_avg(arr: np.ndarray) -> np.ndarray:
    ranks = np.column_stack([rankdata(arr[:, c]) for c in range(arr.shape[1])])
    return ranks.mean(axis=1) - (len(arr) + 1) / 2


def zscore_avg(arr: np.ndarray) -> np.ndarray:
    z = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-12)
    return z.mean(axis=1)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("[qqq-mega-ensemble] loading data + features ...")
    downloaded = download_all()
    feats = compute_qqq_features(downloaded)
    targets = compute_qqq_targets(downloaded)
    common = feats.index.intersection(targets.index)
    feats = feats.loc[common]; targets = targets.loc[common]
    train_feat, val_feat, test_feat = split_superfold(feats)
    train_tgt, val_tgt, test_tgt = split_superfold(targets)

    n_features = train_feat.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[qqq-mega-ensemble] feats={feats.shape}  device={device}")

    # ---- Train each backbone fresh at FX-champion HPs ----
    component_preds: dict[str, dict] = {}

    for name, cfg in FX_CHAMP_CONFIGS.items():
        seq = cfg["seq_len"]
        print(f"\n=== Training {name} @ FX champion HPs (seq={seq}) ===")
        scaler = StandardScaler().fit(train_feat.values)

        if name in ("xgboost", "lightgbm", "catboost"):
            from autoresearch.model.train import find_contiguous_segments
            ts = scaler.transform(train_feat.values)
            vs = scaler.transform(val_feat.values)
            segs_tr = find_contiguous_segments(train_feat.index)
            Xtr_l, Ytr_l = [], []
            for s, e in segs_tr:
                sub_f = ts[s:e]
                sub_t = train_tgt.iloc[s:e][["fwd_ret_1d", "fwd_ret_5d"]].values
                if len(sub_f) < seq + 1: continue
                Xtr_l.append(_flatten_window(sub_f, seq))
                Ytr_l.append(sub_t[seq - 1:][:len(Xtr_l[-1])])
            Xtr = np.vstack(Xtr_l); Ytr = np.vstack(Ytr_l)
            wrapper = GBMWrapper(gbm_type=cfg["gbm_type"], n_targets=2, hp_overrides=cfg["hp_overrides"])
            t0 = time.time()
            wrapper.fit(Xtr, Ytr)
            print(f"  fit {len(Xtr)} rows in {time.time() - t0:.0f}s")
            component_preds[name] = dict(
                wrapper=wrapper, scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_, seq=seq, kind="gbm",
            )
        else:
            kw = dict(seed=cfg["hp"].get("seed", 42))
            np.random.seed(kw["seed"]); torch.manual_seed(kw["seed"])
            model = create_model(
                backbone=cfg["backbone"], n_input_features=n_features,
                seq_len=seq, head_dropout=cfg["hp"]["head_dropout"],
            ).to(device)
            t0 = time.time()
            train_one_fold(
                model=model,
                train_features=train_feat,
                train_targets=train_tgt[["fwd_ret_1d", "fwd_ret_5d"]],
                val_features=val_feat,
                val_targets=val_tgt[["fwd_ret_1d", "fwd_ret_5d"]],
                scaler=scaler,
                seq_len=seq,
                epochs=cfg["hp"]["epochs"],
                lr=cfg["hp"]["lr"],
                batch_size=cfg["hp"]["batch_size"],
                weight_decay=cfg["hp"]["weight_decay"],
                patience=cfg["hp"]["patience"],
            )
            print(f"  fit in {time.time() - t0:.0f}s")
            component_preds[name] = dict(
                model=model, scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_, seq=seq, kind="neural",
            )

    # ---- Per-fold prediction + ensemble ----
    all_returns: dict[str, list[np.ndarray]] = {
        "gbm_rank": [], "gbm_zscore": [],
        "mega_rank": [], "mega_zscore": [],
        "mega5_rank": [], "mega5_zscore": [],
    }
    all_dates: list[pd.Timestamp] = []
    all_y: list[float] = []
    all_mega5_signs: list[int] = []

    for fold in FOLDS:
        d = get_fold_dates(fold)
        wf = test_feat.loc[d["test_start"]:d["test_end"]]
        wt = test_tgt.loc[d["test_start"]:d["test_end"]]
        if len(wf) < 61:
            continue

        per_model = []
        for name, comp in component_preds.items():
            if comp["kind"] == "gbm":
                dt, p, y = gbm_predict(comp["wrapper"], comp["scaler_mean"],
                                       comp["scaler_scale"], wf, wt, comp["seq"])
            else:
                dt, p, y = neural_predict(comp["model"], comp["scaler_mean"],
                                          comp["scaler_scale"], wf, wt, comp["seq"], device)
            if p is not None:
                per_model.append((name, dt, p, y))

        if not per_model:
            continue

        latest_start = max(m[1][0] for m in per_model)
        aligned = []
        for n_, dt, p, y in per_model:
            mask = dt >= latest_start
            aligned.append((n_, dt[mask], p[mask], y[mask]))
        min_n = min(len(a[1]) for a in aligned)
        aligned = [(n_, dt[:min_n], p[:min_n], y[:min_n]) for n_, dt, p, y in aligned]
        if min_n == 0:
            continue
        dates = aligned[0][1]; y_true = aligned[0][3]

        gbm_arr = np.column_stack([a[2] for a in aligned if a[0] in ("xgboost", "lightgbm", "catboost")])
        mega_arr = np.column_stack([a[2] for a in aligned if a[0] != "mlp"])
        mega5_arr = np.column_stack([a[2] for a in aligned])

        all_returns["gbm_rank"].append(np.sign(rank_avg(gbm_arr)) * y_true)
        all_returns["gbm_zscore"].append(np.sign(zscore_avg(gbm_arr)) * y_true)
        all_returns["mega_rank"].append(np.sign(rank_avg(mega_arr)) * y_true)
        all_returns["mega_zscore"].append(np.sign(zscore_avg(mega_arr)) * y_true)
        all_returns["mega5_rank"].append(np.sign(rank_avg(mega5_arr)) * y_true)
        all_returns["mega5_zscore"].append(np.sign(zscore_avg(mega5_arr)) * y_true)
        all_dates.extend(dates); all_y.extend(y_true)
        all_mega5_signs.extend(np.sign(rank_avg(mega5_arr)).astype(int))

    print("\n" + "=" * 78)
    print(f"{'ENSEMBLE':<14}{'Sharpe':>10}{'Excess':>10}{'Ret%':>10}{'WR%':>8}{'n':>8}")
    print("=" * 78)
    bh = float(sharpe_ratio(np.array(all_y))) if all_y else 0.0
    print(f"{'BUY-HOLD':<14}{bh:+10.4f}{'-':>10}{'-':>10}{'-':>8}{len(all_y):>8}")
    print("-" * 78)
    for name, fold_rets in all_returns.items():
        if not fold_rets: continue
        rets = np.concatenate(fold_rets)
        rpt = trading_report(rets)
        sh = float(sharpe_ratio(rets))
        print(f"{name:<14}{sh:+10.4f}{sh - bh:+10.4f}{rpt['total_return_pct']:+10.2f}{rpt['win_rate']:>8.1f}{len(rets):>8}")

    # Save MEGA5 trade log
    if all_dates and all_mega5_signs:
        TRADE_LOGS.mkdir(parents=True, exist_ok=True)
        rows = []
        cum = 0.0
        for i, dt in enumerate(all_dates):
            sign = all_mega5_signs[i]
            ret = all_y[i]
            strat = sign * ret; cum += strat
            rows.append(dict(
                date=str(pd.Timestamp(dt).date()), prediction=sign,
                pred_direction=sign, actual_return=ret,
                actual_direction=int(np.sign(ret)),
                strategy_return=strat, cumulative_return=cum,
                correct=1 if sign == np.sign(ret) and sign != 0 else 0,
                pnl_bps=strat * 10000.0,
            ))
        import csv
        csv_path = TRADE_LOGS / "mega5_ensemble_trades.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[qqq-mega-ensemble] wrote {csv_path.name} ({len(rows)} daily rows)")


if __name__ == "__main__":
    main()
