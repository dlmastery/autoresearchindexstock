"""Single-experiment runner for the QQQ equity-index variant.

Mirrors the FX ``autoresearch.run_autoresearch`` API surface so the
research workflow (Claude as the loop, this script executes one
experiment + logs the result) is identical. Differences:

* Loads QQQ + 30 cross-asset signals via ``data.download.download_all``.
* Builds the equity-native ~120-feature matrix.
* Splits across the 7-fold equity calendar (last test 2025-12, no 2026).
* Optimises on the **A** target (1d forward log return) — primary
  composite — but **always logs A / B / C / D side by side** so the
  dashboard can plot every variant per CLAUDE.md.
* Each JSONL row also carries the buy-and-hold baseline Sharpe and the
  excess-Sharpe = strategy − BH for the same window.

Usage::

    cd C:/Users/evija/autoresearch
    "C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock.run_autoresearch \
        --backbone xgboost --description "xgboost: SOTA baseline seq=60"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Hardware safety: pin to P-cores (parent CLAUDE.md mandate).
sys.path.insert(0, "C:/Users/evija/autoresearch")
from autoresearch.run_autoresearch import _pin_to_safe_cores  # noqa: E402

_pin_to_safe_cores()

from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .data.download import download_all, DEFAULT_END  # noqa: E402
from .data.features import compute_qqq_features, compute_qqq_targets  # noqa: E402
from .data.splits import (  # noqa: E402
    FOLDS, get_fold_dates, split_superfold, validate_purge_embargo,
)
from .evaluation.metrics import (  # noqa: E402
    composite_score, evaluate_target_variant,
)
from autoresearch.model.backbone import (  # noqa: E402
    create_model, get_seq_len, GBMWrapper,
)
from autoresearch.model.train import (  # noqa: E402
    create_contiguous_datasets, train_one_fold,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).resolve().parent / "autoresearch_results"
LOG_PATH = RESULTS_DIR / "experiment_log.jsonl"
BEST_PATH = RESULTS_DIR / "best_config.json"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pt"
TRADE_LOGS_DIR = RESULTS_DIR / "trade_logs"
RUNNING_PATH = RESULTS_DIR / "running.json"


def _is_gbm(backbone: str) -> bool:
    return backbone in {"xgboost", "lightgbm", "catboost"}


def _next_experiment_num() -> int:
    if not LOG_PATH.exists():
        return 1
    n = 0
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            n = max(n, int(json.loads(line).get("experiment_num", 0)))
        except Exception:
            continue
    return n + 1


def _flatten_window(features: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Stack each (seq_len, n_feat) window into a single row vector."""
    arr = features.values
    return np.array([arr[i:i + seq_len].ravel() for i in range(len(arr) - seq_len + 1)])


def _evaluate_per_window(
    *,
    model,
    scaler: StandardScaler,
    feats: pd.DataFrame,
    targets: pd.DataFrame,
    seq_len: int,
    device: torch.device,
    is_gbm: bool,
    window_type: str,
) -> dict:
    """Evaluate per fold, return a dict with per-fold breakdown for the
    PRIMARY target A (1d return) plus aggregates for A / B / C / D and
    a flat list of trade rows used to write the daily CSV.
    """
    rows: list[dict] = []
    trade_rows: list[dict] = []
    pred_a_all, pred_b_all, pred_d_all = [], [], []
    actual_a_all, actual_b_all, actual_d_all = [], [], []

    for fold in FOLDS:
        d = get_fold_dates(fold)
        if window_type == "val":
            ws, we = d["val_start"], d["val_end"]
        else:
            ws, we = d["test_start"], d["test_end"]
        wf = feats.loc[ws:we]
        wt = targets.loc[ws:we]
        if len(wf) < seq_len + 1:
            rows.append({"fold": fold["name"], "regime": fold["regime"],
                         "n": 0, "A_sharpe": 0.0, "skipped": True})
            continue

        # Scale features
        ws_arr = scaler.transform(wf.values)

        if is_gbm:
            X = np.array([ws_arr[i:i + seq_len].ravel()
                          for i in range(len(ws_arr) - seq_len + 1)])
            preds_all_targets = model.predict(X)  # shape (n, n_targets)
            # Map columns to A, B, D
            pred_a = preds_all_targets[:, 0]
            pred_b = preds_all_targets[:, 1] if preds_all_targets.shape[1] > 1 else pred_a
            pred_d = preds_all_targets[:, 1] if preds_all_targets.shape[1] > 1 else pred_a
            dates = wt.index[seq_len - 1:][:len(pred_a)]
            actuals_a = wt["fwd_ret_1d"].values[seq_len - 1:][:len(pred_a)]
            actuals_b = wt["fwd_ret_5d"].values[seq_len - 1:][:len(pred_a)]
            actuals_d = wt["fwd_voladj_ret_1d"].values[seq_len - 1:][:len(pred_a)]
        else:
            ws_df = pd.DataFrame(ws_arr, index=wf.index, columns=wf.columns)
            ds = create_contiguous_datasets(ws_df, wt, seq_len)
            if len(ds) == 0:
                rows.append({"fold": fold["name"], "regime": fold["regime"],
                             "n": 0, "A_sharpe": 0.0, "skipped": True})
                continue
            loader = DataLoader(ds, batch_size=256, shuffle=False)
            model.eval()
            preds_a, preds_b, actuals_a, actuals_b = [], [], [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    out = model(x)
                    preds_a.append(out["ret_1d"][:, 0].cpu().numpy())
                    if "ret_5d" in out:
                        preds_b.append(out["ret_5d"][:, 0].cpu().numpy())
                    actuals_a.append(y[:, 0].numpy())
                    if y.shape[1] > 1:
                        actuals_b.append(y[:, 1].numpy())
            pred_a = np.concatenate(preds_a)
            pred_b = np.concatenate(preds_b) if preds_b else pred_a
            actuals_a = np.concatenate(actuals_a)
            actuals_b = np.concatenate(actuals_b) if actuals_b else actuals_a
            # D for neural: divide A pred by 20d realised vol of training-scale features
            # (we already produce a vol-adjusted target column in compute_qqq_targets)
            dates = wt.index[seq_len - 1:][:len(pred_a)]
            actuals_d = wt["fwd_voladj_ret_1d"].values[seq_len - 1:][:len(pred_a)]
            pred_d = pred_a  # for neural, no separate D head; reuse A

        if len(pred_a) == 0:
            rows.append({"fold": fold["name"], "regime": fold["regime"],
                         "n": 0, "A_sharpe": 0.0, "skipped": True})
            continue

        # Per-fold metrics on PRIMARY target A
        per_a = evaluate_target_variant(pred_a, actuals_a, label="A")
        per_b = evaluate_target_variant(pred_b, actuals_b, label="B")
        per_d = evaluate_target_variant(pred_d, actuals_d, label="D")

        row = {
            "fold":   fold["name"],
            "regime": fold["regime"],
            "n":      int(per_a["A_n"]),
            **per_a, **per_b, **per_d,
        }
        rows.append(row)

        # Per-day trade log rows (target A primary)
        cum = 0.0
        for i, dt in enumerate(dates):
            pred_dir = int(np.sign(pred_a[i]))
            act_dir = int(np.sign(actuals_a[i]))
            strat = pred_dir * float(actuals_a[i])
            cum += strat
            correct = 1 if pred_dir == act_dir and pred_dir != 0 else 0
            trade_rows.append({
                "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "fold": fold["name"],
                "regime": fold["regime"],
                "prediction": float(pred_a[i]),
                "pred_direction": pred_dir,
                "actual_return": float(actuals_a[i]),
                "actual_direction": act_dir,
                "strategy_return": strat,
                "cumulative_return": cum,
                "confidence": "",
                "aleatoric": "",
                "epistemic": "",
                "correct": correct,
                "pnl_bps": strat * 10000.0,
                # Extra columns: target B / D for downstream plotting
                "B_pred": float(pred_b[i]) if i < len(pred_b) else "",
                "B_actual": float(actuals_b[i]) if i < len(actuals_b) else "",
                "D_pred": float(pred_d[i]) if i < len(pred_d) else "",
                "D_actual": float(actuals_d[i]) if i < len(actuals_d) else "",
            })

        pred_a_all.append(pred_a); actual_a_all.append(actuals_a)
        pred_b_all.append(pred_b); actual_b_all.append(actuals_b)
        pred_d_all.append(pred_d); actual_d_all.append(actuals_d)

    # Aggregate over the union of fold windows
    pa = np.concatenate(pred_a_all) if pred_a_all else np.array([])
    aa = np.concatenate(actual_a_all) if actual_a_all else np.array([])
    pb = np.concatenate(pred_b_all) if pred_b_all else np.array([])
    ab = np.concatenate(actual_b_all) if actual_b_all else np.array([])
    pd_ = np.concatenate(pred_d_all) if pred_d_all else np.array([])
    ad = np.concatenate(actual_d_all) if actual_d_all else np.array([])
    agg_a = evaluate_target_variant(pa, aa, label="A")
    agg_b = evaluate_target_variant(pb, ab, label="B")
    agg_d = evaluate_target_variant(pd_, ad, label="D")
    n_neg = sum(1 for r in rows if r.get("A_sharpe", 0.0) < 0 and not r.get("skipped"))

    return {
        "per_window": rows,
        "trade_rows": trade_rows,
        "n_negative_folds": n_neg,
        **agg_a,
        **agg_b,
        **agg_d,
    }


def _write_trade_csv(exp_num: int, trade_rows: list[dict], summary: dict) -> None:
    import csv
    TRADE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date", "fold", "regime", "prediction", "pred_direction",
        "actual_return", "actual_direction", "strategy_return",
        "cumulative_return", "confidence", "aleatoric", "epistemic",
        "correct", "pnl_bps", "B_pred", "B_actual", "D_pred", "D_actual",
    ]
    csv_path = TRADE_LOGS_DIR / f"exp{exp_num}_trades.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trade_rows:
            w.writerow(r)
    sum_path = TRADE_LOGS_DIR / f"exp{exp_num}_trade_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bs", "--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None,
                        help="GBM only — tree depth")
    parser.add_argument("--gbm-lr", type=float, default=None,
                        help="GBM only — boosting learning rate")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    is_gbm = _is_gbm(args.backbone)
    seq_len = args.seq_len if args.seq_len is not None else get_seq_len(args.backbone)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRADE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    exp_num = _next_experiment_num()

    # Heart-beat for the dashboard
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    RUNNING_PATH.write_text(json.dumps({
        "experiment_num": exp_num,
        "backbone": args.backbone,
        "description": args.description,
        "config": vars(args),
        "started": started,
    }, indent=2), encoding="utf-8")

    t0 = time.time()
    logger.info("[run] exp %d  backbone=%s  seq_len=%d  end=%s",
                exp_num, args.backbone, seq_len, DEFAULT_END)

    # 1. Data ---------------------------------------------------------------
    downloaded = download_all()
    feats = compute_qqq_features(downloaded)
    targets = compute_qqq_targets(downloaded)
    common = feats.index.intersection(targets.index)
    feats = feats.loc[common]; targets = targets.loc[common]
    logger.info("[data] feats=%s  targets=%s  range=%s..%s",
                feats.shape, targets.shape,
                str(feats.index[0])[:10], str(feats.index[-1])[:10])

    # 2. Splits -------------------------------------------------------------
    train_feat, val_feat, test_feat = split_superfold(feats)
    train_tgt, val_tgt, test_tgt = split_superfold(targets)
    diag = validate_purge_embargo(feats)
    logger.info("[splits] %s", diag)

    n_features = train_feat.shape[1]
    n_targets = 2  # we train on (fwd_ret_1d, fwd_ret_5d) — primary + auxiliary

    # 3. Model --------------------------------------------------------------
    if is_gbm:
        # GBM path — flatten windows on training set
        scaler = StandardScaler().fit(train_feat.values)
        ts = scaler.transform(train_feat.values)
        vs = scaler.transform(val_feat.values)

        # We need contiguous segments for windowing (training has holes).
        from autoresearch.model.train import find_contiguous_segments  # noqa: E402
        segs_tr = find_contiguous_segments(train_feat.index)
        Xtr_list, Ytr_list = [], []
        for s, e in segs_tr:
            sub_f = ts[s:e]
            sub_t = train_tgt.iloc[s:e][["fwd_ret_1d", "fwd_ret_5d"]].values
            if len(sub_f) < seq_len + 1: continue
            Xtr_list.append(np.array([sub_f[i:i + seq_len].ravel()
                                      for i in range(len(sub_f) - seq_len + 1)]))
            Ytr_list.append(sub_t[seq_len - 1:][:len(Xtr_list[-1])])
        Xtr = np.vstack(Xtr_list); Ytr = np.vstack(Ytr_list)

        segs_v = find_contiguous_segments(val_feat.index)
        Xv_list, Yv_list = [], []
        for s, e in segs_v:
            sub_f = vs[s:e]
            sub_t = val_tgt.iloc[s:e][["fwd_ret_1d", "fwd_ret_5d"]].values
            if len(sub_f) < seq_len + 1: continue
            Xv_list.append(np.array([sub_f[i:i + seq_len].ravel()
                                     for i in range(len(sub_f) - seq_len + 1)]))
            Yv_list.append(sub_t[seq_len - 1:][:len(Xv_list[-1])])
        Xv = np.vstack(Xv_list); Yv = np.vstack(Yv_list)

        hp = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.gbm_lr,
            "random_state": args.seed,
        }
        wrapper = GBMWrapper(
            gbm_type=args.backbone,
            n_targets=2,
            hp_overrides={k: v for k, v in hp.items() if v is not None},
        )
        wrapper.fit(Xtr, Ytr)
        model = wrapper
        device = torch.device("cpu")
        val_loss = float("nan")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kw = {}
        if args.hidden_size is not None: kw["hidden_size"] = args.hidden_size
        if args.num_layers is not None: kw["num_layers"] = args.num_layers
        model = create_model(
            backbone=args.backbone,
            n_input_features=n_features,
            seq_len=seq_len,
            head_dropout=args.head_dropout,
            **kw,
        ).to(device)

        scaler = StandardScaler().fit(train_feat.values)
        result = train_one_fold(
            model=model,
            train_features=train_feat,
            train_targets=train_tgt[["fwd_ret_1d", "fwd_ret_5d"]],
            val_features=val_feat,
            val_targets=val_tgt[["fwd_ret_1d", "fwd_ret_5d"]],
            scaler=scaler,
            seq_len=seq_len,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            huber_delta=args.huber_delta,
            patience=args.patience,
            grad_clip=args.grad_clip,
        )
        val_loss = float(result.get("best_val_loss", float("nan")))

    # 4. Evaluate per-window on val + test ----------------------------------
    val_eval = _evaluate_per_window(
        model=model, scaler=scaler, feats=val_feat, targets=val_tgt,
        seq_len=seq_len, device=device, is_gbm=is_gbm, window_type="val",
    )
    test_eval = _evaluate_per_window(
        model=model, scaler=scaler, feats=test_feat, targets=test_tgt,
        seq_len=seq_len, device=device, is_gbm=is_gbm, window_type="test",
    )

    # 5. Composite + KEEP/DISCARD on PRIMARY target (A) ---------------------
    test_sharpe_a = test_eval["A_sharpe"]
    val_sharpe_a = val_eval["A_sharpe"]
    composite = composite_score(
        test_sharpe_a, val_sharpe_a, test_eval["n_negative_folds"]
    )

    # 6. Trade log + summary ------------------------------------------------
    fold_summary: dict[str, dict] = {}
    for r in test_eval["trade_rows"]:
        f_ = r["fold"]
        d = fold_summary.setdefault(f_, {"n": 0, "wins": 0, "losses": 0,
                                          "win_pnls": [], "loss_pnls": []})
        d["n"] += 1
        if r["correct"] == 1:
            d["wins"] += 1; d["win_pnls"].append(r["pnl_bps"])
        else:
            d["losses"] += 1; d["loss_pnls"].append(r["pnl_bps"])
    for f_, d in fold_summary.items():
        d["avg_win_bps"] = round(float(np.mean(d["win_pnls"])), 2) if d["win_pnls"] else 0.0
        d["avg_loss_bps"] = round(float(np.mean(d["loss_pnls"])), 2) if d["loss_pnls"] else 0.0
        d["max_win_bps"] = round(float(np.max(d["win_pnls"])), 2) if d["win_pnls"] else 0.0
        d["max_loss_bps"] = round(float(np.min(d["loss_pnls"])), 2) if d["loss_pnls"] else 0.0
        d["win_rate"] = round(100 * d["wins"] / d["n"], 2) if d["n"] else 0.0
        d.pop("win_pnls"); d.pop("loss_pnls")
    summary = {
        "exp": exp_num,
        "total_trades": len(test_eval["trade_rows"]),
        "wins": sum(1 for r in test_eval["trade_rows"] if r["correct"] == 1),
        "losses": sum(1 for r in test_eval["trade_rows"] if r["correct"] == 0),
        "total_pnl_bps": float(sum(r["pnl_bps"] for r in test_eval["trade_rows"])),
        "test_sharpe_A": test_sharpe_a,
        "test_excess_sharpe_A": test_eval["A_excess_sharpe"],
        "per_fold": fold_summary,
    }
    _write_trade_csv(exp_num, test_eval["trade_rows"], summary)

    # 7. JSONL entry --------------------------------------------------------
    elapsed = round(time.time() - t0, 1)
    entry = {
        "experiment_num": exp_num,
        "backbone": args.backbone,
        "description": args.description,
        "config": {
            "seq_len": seq_len, "lr": args.lr, "batch_size": args.batch_size,
            "epochs": args.epochs, "weight_decay": args.weight_decay,
            "patience": args.patience, "warmup_epochs": args.warmup_epochs,
            "head_dropout": args.head_dropout, "huber_delta": args.huber_delta,
            "grad_clip": args.grad_clip, "seed": args.seed,
            "hidden_size": args.hidden_size, "num_layers": args.num_layers,
            "max_depth": args.max_depth, "gbm_lr": args.gbm_lr,
            "n_estimators": args.n_estimators,
        },
        "composite": round(composite, 4),
        "test_pos_folds": sum(1 for r in test_eval["per_window"]
                              if r.get("A_sharpe", 0.0) > 0 and not r.get("skipped")),
        "val_pos_folds": sum(1 for r in val_eval["per_window"]
                             if r.get("A_sharpe", 0.0) > 0 and not r.get("skipped")),
        "val_loss": val_loss,
        "elapsed_sec": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # PRIMARY (target A — 1d return)
        "sharpe":     test_sharpe_a,
        "val_sharpe": val_sharpe_a,
        "ic":         test_eval["A_ic"],
        "hit":        test_eval["A_hit_rate"],
        "psr":        test_eval["A_psr"],
        "equity":     1000.0 * (1 + test_eval["A_return_pct"] / 100.0),
        "return_pct": test_eval["A_return_pct"],
        "bh_sharpe":      test_eval["A_bh_sharpe"],
        "bh_return_pct":  test_eval["A_bh_return_pct"],
        "excess_sharpe":  test_eval["A_excess_sharpe"],
        # SECONDARY (B — 5d, D — vol-adjusted)
        "B_sharpe":         test_eval["B_sharpe"],
        "B_excess_sharpe":  test_eval["B_excess_sharpe"],
        "B_return_pct":     test_eval["B_return_pct"],
        "D_sharpe":         test_eval["D_sharpe"],
        "D_excess_sharpe":  test_eval["D_excess_sharpe"],
        "D_return_pct":     test_eval["D_return_pct"],
        "per_window":       test_eval["per_window"],
        "per_window_val":   val_eval["per_window"],
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    # 8. Champion check ----------------------------------------------------
    is_champion = False
    if BEST_PATH.exists():
        prev = json.loads(BEST_PATH.read_text(encoding="utf-8"))
        if composite > prev.get("composite", -1e9):
            is_champion = True
    else:
        is_champion = True
    if is_champion:
        BEST_PATH.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        if not is_gbm:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": entry["config"],
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "feature_columns": list(train_feat.columns),
                "target_columns": ["fwd_ret_1d", "fwd_ret_5d"],
                "n_features": n_features,
                "composite": composite,
                "description": args.description,
                "backbone": args.backbone,
                "experiment_num": exp_num,
            }, BEST_MODEL_PATH)

    # Heart-beat off
    try: RUNNING_PATH.unlink()
    except FileNotFoundError: pass

    print(f"[exp {exp_num}] composite={composite:+.4f}  "
          f"A_sharpe={test_sharpe_a:+.4f}  excess={test_eval['A_excess_sharpe']:+.4f}  "
          f"B={test_eval['B_sharpe']:+.4f}  D={test_eval['D_sharpe']:+.4f}  "
          f"BH={test_eval['A_bh_sharpe']:+.4f}  elapsed={elapsed}s  "
          f"{'CHAMPION' if is_champion else ''}")


if __name__ == "__main__":
    main()
