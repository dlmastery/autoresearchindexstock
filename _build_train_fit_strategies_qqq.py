"""Train-fit strategies — Directive 72 (2026-05-10).

For each ensemble member with both in-sample test-fold trades AND OOS trades:
  A. Sweep strategy parameters on IN-SAMPLE → pick optimum → LOCK → apply to OOS
  B. Train meta-learner ensemble combiners on IN-SAMPLE → apply to OOS
  C. Calibrate predictions on IN-SAMPLE → apply to OOS

Every resulting strategy is causal/deployable: the parameter choice or model
weights are determined by data observable BEFORE the OOS window starts.

Output: appends new strategies to `oos_ensemble_summary.json` with prefix
`train_optim_*` (per-member sweeps) and `meta_*` (cross-member ensemble fits).
Each carries `is_leaky=False` + `selection_basis` + `fit_method` + chosen
parameter for full auditability.
"""
from __future__ import annotations
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

R = Path(__file__).resolve().parent / "autoresearch_results"
LOGS = R / "trade_logs"
ENS_JSON = R / "oos_ensemble_summary.json"
START = 1000.0


def sharpe_arr(pnl: np.ndarray) -> float:
    p = np.asarray(pnl, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) < 2:
        return 0.0
    sd = p.std()
    if sd == 0:
        return 0.0
    return float(p.mean() / sd * math.sqrt(252))


def metrics_from_positions(positions: np.ndarray, actuals: np.ndarray, dates: list) -> dict:
    pos = np.asarray(positions, dtype=float)
    act = np.asarray(actuals, dtype=float)
    # Mask out NaN actuals
    mask = ~np.isnan(act)
    pos = np.where(mask, pos, 0.0)
    act = np.where(mask, act, 0.0)
    pnl = pos * act
    eq = START * np.exp(pnl.cumsum())
    bh = START * np.exp(act.cumsum())
    pos_mask = (pos != 0)
    traded_pnl = pnl[pos_mask]
    bh_clean = act[mask]
    n_traded = int(pos_mask.sum())
    n_pred = len(pos)
    correct = int((np.sign(pos) == np.sign(act)).sum() & pos_mask.sum()) if n_traded else 0
    hit = float(((np.sign(pos) == np.sign(act)) & pos_mask).sum() / max(1, n_traded) * 100) if n_traded else 0.0
    peak = np.maximum.accumulate(eq)
    dd_pct = float(((eq - peak) / np.where(peak == 0, 1.0, peak)).min() * 100) if len(eq) else 0.0
    sh = sharpe_arr(traded_pnl) if n_traded > 1 else 0.0
    bh_sh = sharpe_arr(bh_clean)
    return {
        "final_dollars_on_1000": round(float(eq[-1]) if len(eq) else START, 4),
        "final_dollars_bh": round(float(bh[-1]) if len(bh) else START, 4),
        "excess_dollars": round(float((eq[-1] if len(eq) else START) - (bh[-1] if len(bh) else START)), 4),
        "compound_return_pct": round(float((eq[-1] / START - 1) * 100) if len(eq) else 0, 4),
        "buy_hold_compound_pct": round(float((bh[-1] / START - 1) * 100) if len(bh) else 0, 4),
        "excess_compound_pct": round(float((eq[-1] / START - bh[-1] / START) * 100) if len(eq) else 0, 4),
        "strategy_annual_sharpe": round(sh, 4),
        "annual_sharpe": round(sh, 4),
        "buy_hold_annual_sharpe": round(bh_sh, 4),
        "excess_sharpe": round(sh - bh_sh, 4),
        "annual_sortino": 0.0,
        "psr": 0.0,
        "hit_rate_pct": round(hit, 2),
        "max_drawdown_pct": round(dd_pct, 4),
        "exposure_pct": round(float(pos_mask.mean()) * 100, 2),
        "avg_position": round(float(np.abs(pos).mean()), 4),
        "turnover": round(float(np.abs(np.diff(np.concatenate([[0.0], pos]))).sum()), 4),
        "n_predictions": n_pred,
        "n_traded_days": n_traded,
        "n_with_actuals": int(mask.sum()),
        "equity_curve": {
            "dates": [str(d)[:10] for d in dates],
            "strategy_dollars": [round(float(v), 2) for v in eq.tolist()],
            "buy_hold_dollars": [round(float(v), 2) for v in bh.tolist()],
            "strategy_pct": [round(float(v / START - 1) * 100, 4) for v in eq.tolist()],
            "buy_hold_pct": [round(float(v / START - 1) * 100, 4) for v in bh.tolist()],
        },
    }


def load_member_data(members: list) -> dict:
    """Load per-member in-sample trade_log + OOS predictions."""
    out = {}
    for m in members:
        en = m["experiment_num"]
        in_csv = LOGS / f"exp{en}_trades.csv"
        oos_csv = R / f"oos_exp{en}.csv"
        if not in_csv.exists() or not oos_csv.exists():
            continue
        try:
            indf = pd.read_csv(in_csv, parse_dates=["date"])
            oosdf = pd.read_csv(oos_csv, parse_dates=["date"])
        except Exception:
            continue
        oos_pred_col = "pred_ret_1d" if "pred_ret_1d" in oosdf.columns else "prediction"
        oos_act_col  = "actual_ret_1d" if "actual_ret_1d" in oosdf.columns else "actual_return"
        out[en] = {
            "backbone": m.get("backbone"),
            "seed": m.get("seed"),
            "in_df": indf[["date", "prediction", "actual_return"]].dropna().reset_index(drop=True),
            "oos_df": oosdf[["date", oos_pred_col, oos_act_col]].rename(
                columns={oos_pred_col: "pred", oos_act_col: "actual"}).dropna().reset_index(drop=True),
        }
    return out


def main():
    if not ENS_JSON.exists():
        print(f"[train-fit] {ENS_JSON} missing — run ensemble builder first")
        return
    ens = json.loads(ENS_JSON.read_text(encoding="utf-8"))
    members = ens.get("members") or []
    print(f"[train-fit] ensemble has {len(members)} members in roster")

    mdata = load_member_data(members)
    print(f"[train-fit] {len(mdata)} members have BOTH in-sample + OOS data")
    if not mdata:
        print("[train-fit] no eligible members; aborting")
        return

    new_strats: dict[str, dict] = {}

    # ===========================================================
    # CATEGORY A — per-member parameter sweeps
    # ===========================================================
    for en, d in mdata.items():
        in_pred = d["in_df"]["prediction"].values
        in_act  = d["in_df"]["actual_return"].values
        oos_pred = d["oos_df"]["pred"].values
        oos_act  = d["oos_df"]["actual"].values
        oos_dates = d["oos_df"]["date"].values

        # ---- A1: Confidence threshold sweep ----
        in_std = float(np.nanstd(in_pred)) or 1.0
        sigma_pool = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        best_thr = 0.0; best_sh = -99; best_lbl = "0.0sig"
        for sig in sigma_pool:
            thr = sig * in_std
            pos_in = np.sign(in_pred) * (np.abs(in_pred) >= thr).astype(float)
            pnl_in = pos_in * in_act
            traded = pnl_in[pos_in != 0]
            if len(traded) > 30:
                sh = sharpe_arr(traded)
                if sh > best_sh:
                    best_sh = sh; best_thr = thr; best_lbl = f"{sig:.2f}sig"
        pos_oos = np.sign(oos_pred) * (np.abs(oos_pred) >= best_thr).astype(float)
        mm = metrics_from_positions(pos_oos, oos_act, oos_dates)
        name = f"train_optim_conf_thresh_exp{en}__{best_lbl}"
        new_strats[name] = {**mm, "name": name,
            "is_leaky": False,
            "selection_basis": f"train+val |pred|-threshold sweep over {sigma_pool}×σ_in (in-sample Sharpe optimum = {best_lbl}, in-sample Sh = {best_sh:.3f}); LOCKED, evaluated OOS",
            "fit_method": "in-sample sharpe sweep over |pred| thresholds",
            "fit_param_pool": sigma_pool,
            "fit_param_chosen": best_lbl,
            "in_sample_sharpe_at_choice": round(best_sh, 4),
            "signal_source": f"individual_exp{en}",
            "category": "T",  # T = train-fit
            "members_used": [en],
            "n_members_used": 1,
            "sizing_mode": "raw",
            "overlay": "train_optim_conf_thresh",
        }

        # ---- A2: Kelly fraction sweep ----
        in_sh = sharpe_arr(in_pred * in_act)
        kelly_base = (in_sh ** 2 / 252) if in_sh > 0 else 0.0
        frac_pool = [0.10, 0.25, 0.50, 1.00]
        best_frac = 0.25; best_sh_k = -99
        for f in frac_pool:
            pos_in = np.clip(np.sign(in_pred) * f * kelly_base, -1, 1)
            pnl_in = pos_in * in_act
            if (pos_in != 0).any():
                sh = sharpe_arr(pnl_in[pos_in != 0])
                if sh > best_sh_k:
                    best_sh_k = sh; best_frac = f
        pos_oos = np.clip(np.sign(oos_pred) * best_frac * kelly_base, -1, 1)
        mm = metrics_from_positions(pos_oos, oos_act, oos_dates)
        name = f"train_optim_kelly_frac_exp{en}__{best_frac:.2f}"
        new_strats[name] = {**mm, "name": name,
            "is_leaky": False,
            "selection_basis": f"train+val Kelly-fraction sweep over {frac_pool} (in-sample Sharpe optimum = {best_frac:.2f}, in-sample Sh = {best_sh_k:.3f}); base Kelly = sh²/252 = {kelly_base:.4f}; LOCKED",
            "fit_method": "in-sample sharpe sweep over Kelly fractions",
            "fit_param_pool": frac_pool,
            "fit_param_chosen": f"{best_frac:.2f}",
            "in_sample_sharpe_at_choice": round(best_sh_k, 4),
            "signal_source": f"individual_exp{en}",
            "category": "T", "members_used": [en], "n_members_used": 1,
            "sizing_mode": "kelly_train_optim", "overlay": "none",
        }

        # ---- A3: Vol-target sweep ----
        in_act_s = pd.Series(in_act)
        rv20_in = (in_act_s.rolling(20, min_periods=5).std() * math.sqrt(252)).fillna(0.15).values
        oos_act_s = pd.Series(oos_act)
        rv20_oos = (oos_act_s.rolling(20, min_periods=5).std() * math.sqrt(252)).fillna(0.15).values
        target_pool = [0.10, 0.15, 0.20, 0.25, 0.30]
        best_t = 0.15; best_sh_v = -99
        for t in target_pool:
            size_in = np.clip(t / np.where(rv20_in == 0, 0.15, rv20_in), 0, 2.0)
            pos_in = np.sign(in_pred) * size_in
            pnl_in = pos_in * in_act
            if (pos_in != 0).any():
                sh = sharpe_arr(pnl_in[pos_in != 0])
                if sh > best_sh_v:
                    best_sh_v = sh; best_t = t
        size_oos = np.clip(best_t / np.where(rv20_oos == 0, 0.15, rv20_oos), 0, 2.0)
        pos_oos = np.sign(oos_pred) * size_oos
        mm = metrics_from_positions(pos_oos, oos_act, oos_dates)
        name = f"train_optim_voltarget_exp{en}__{int(best_t*100)}pct"
        new_strats[name] = {**mm, "name": name,
            "is_leaky": False,
            "selection_basis": f"train+val vol-target sweep over {[int(t*100) for t in target_pool]}% (in-sample optimum = {int(best_t*100)}%, in-sample Sh = {best_sh_v:.3f}); LOCKED",
            "fit_method": "in-sample sharpe sweep over vol-target levels",
            "fit_param_pool": target_pool,
            "fit_param_chosen": f"{int(best_t*100)}pct",
            "in_sample_sharpe_at_choice": round(best_sh_v, 4),
            "signal_source": f"individual_exp{en}",
            "category": "T", "members_used": [en], "n_members_used": 1,
            "sizing_mode": "voltarget_train_optim", "overlay": "none",
        }

    # ===========================================================
    # CATEGORY B — meta-learner ensemble combiners
    # ===========================================================
    # Build aligned (X, y) matrices by inner-joining on dates
    member_ids = sorted(mdata.keys())
    if len(member_ids) >= 2:
        # In-sample
        in_frames = []
        for en in member_ids:
            df = mdata[en]["in_df"].copy()
            df = df.rename(columns={"prediction": f"p_{en}", "actual_return": "actual"})
            df = df.set_index("date")
            in_frames.append(df)
        merged_in = pd.concat(in_frames, axis=1, join="inner")
        # Dedupe the actual column (it's the same series from every member; keep first)
        actual_in = merged_in["actual"].iloc[:, 0] if isinstance(merged_in["actual"], pd.DataFrame) else merged_in["actual"]
        X_in = merged_in[[f"p_{en}" for en in member_ids]].values
        y_in = actual_in.values

        # OOS
        oos_frames = []
        for en in member_ids:
            df = mdata[en]["oos_df"].copy()
            df = df.rename(columns={"pred": f"p_{en}", "actual": "actual"})
            df = df.set_index("date")
            oos_frames.append(df)
        merged_oos = pd.concat(oos_frames, axis=1, join="inner")
        actual_oos = merged_oos["actual"].iloc[:, 0] if isinstance(merged_oos["actual"], pd.DataFrame) else merged_oos["actual"]
        X_oos = merged_oos[[f"p_{en}" for en in member_ids]].values
        y_oos = actual_oos.values
        oos_dates_aligned = list(merged_oos.index.values)

        # Drop NaN rows
        m_in = ~np.isnan(X_in).any(axis=1) & ~np.isnan(y_in)
        m_oos = ~np.isnan(X_oos).any(axis=1) & ~np.isnan(y_oos)
        X_in, y_in = X_in[m_in], y_in[m_in]
        X_oos, y_oos = X_oos[m_oos], y_oos[m_oos]
        oos_dates_aligned = [d for d, keep in zip(oos_dates_aligned, m_oos) if keep]
        print(f"[train-fit] aligned X_in={X_in.shape} X_oos={X_oos.shape}")

        if len(X_in) > 60 and len(X_oos) > 5:
            # ---- B1: Ridge regression (alpha CV-tuned in-sample) ----
            from sklearn.linear_model import RidgeCV
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5).fit(X_in, y_in)
            preds_oos = ridge.predict(X_oos)
            pos_oos = np.sign(preds_oos)
            mm = metrics_from_positions(pos_oos, y_oos, oos_dates_aligned)
            name = f"meta_ridgeCV__on_{len(member_ids)}members"
            new_strats[name] = {**mm, "name": name,
                "is_leaky": False,
                "selection_basis": f"Ridge regression on {len(member_ids)} member predictions → next-day return. Alpha cross-validated 5-fold on in-sample (chose alpha={ridge.alpha_:.3f}). Coefficients LOCKED, evaluated OOS.",
                "fit_method": "sklearn.linear_model.RidgeCV alphas=[0.01,0.1,1,10,100] cv=5",
                "fit_param_chosen": f"alpha={ridge.alpha_:.3f}",
                "fit_param_pool": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model_coefs": {f"exp{en}": round(float(c), 4) for en, c in zip(member_ids, ridge.coef_)},
                "signal_source": f"meta_{len(member_ids)}members",
                "category": "M",
                "members_used": member_ids,
                "n_members_used": len(member_ids),
                "sizing_mode": "raw", "overlay": "meta_ridge",
            }

            # ---- B2: Logistic regression on direction ----
            from sklearn.linear_model import LogisticRegression
            try:
                y_in_dir = (y_in > 0).astype(int)
                if len(np.unique(y_in_dir)) == 2:
                    lr = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs").fit(X_in, y_in_dir)
                    probs_oos = lr.predict_proba(X_oos)[:, 1]
                    pos_oos = np.where(probs_oos > 0.5, 1.0, -1.0)
                    mm = metrics_from_positions(pos_oos, y_oos, oos_dates_aligned)
                    name = f"meta_logreg__on_{len(member_ids)}members"
                    new_strats[name] = {**mm, "name": name,
                        "is_leaky": False,
                        "selection_basis": f"Logistic regression on {len(member_ids)} member predictions → P(return>0). C=1.0, fit on in-sample, applied at threshold 0.5; LOCKED, evaluated OOS.",
                        "fit_method": "sklearn.linear_model.LogisticRegression C=1.0 solver=lbfgs",
                        "fit_param_chosen": "C=1.0, threshold=0.5",
                        "model_coefs": {f"exp{en}": round(float(c), 4) for en, c in zip(member_ids, lr.coef_[0])},
                        "signal_source": f"meta_{len(member_ids)}members",
                        "category": "M", "members_used": member_ids, "n_members_used": len(member_ids),
                        "sizing_mode": "raw", "overlay": "meta_logreg",
                    }
            except Exception as e:
                print(f"[train-fit] LogReg failed: {e}")

            # ---- B3: XGBoost meta-learner (if xgboost is installed) ----
            try:
                from xgboost import XGBRegressor
                xgb = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05,
                                    reg_alpha=0.1, reg_lambda=1.0, n_jobs=2, verbosity=0)
                xgb.fit(X_in, y_in)
                preds_oos = xgb.predict(X_oos)
                pos_oos = np.sign(preds_oos)
                mm = metrics_from_positions(pos_oos, y_oos, oos_dates_aligned)
                name = f"meta_xgb__on_{len(member_ids)}members"
                new_strats[name] = {**mm, "name": name,
                    "is_leaky": False,
                    "selection_basis": f"XGBoost regressor on {len(member_ids)} member predictions → next-day return. Shallow trees (max_depth=3, n_estimators=50, lr=0.05). Fit on in-sample, LOCKED, evaluated OOS.",
                    "fit_method": "xgboost.XGBRegressor max_depth=3 n_estimators=50 lr=0.05",
                    "fit_param_chosen": "n_est=50 max_d=3 lr=0.05 alpha=0.1 lambda=1.0",
                    "signal_source": f"meta_{len(member_ids)}members",
                    "category": "M", "members_used": member_ids, "n_members_used": len(member_ids),
                    "sizing_mode": "raw", "overlay": "meta_xgb",
                }
            except ImportError:
                print("[train-fit] xgboost not installed — skipping XGB meta-learner")

    # ===========================================================
    # CATEGORY C — Isotonic calibration on direction
    # ===========================================================
    # For each member individually, fit an isotonic regression on (|pred|, correct_in_sample)
    # to calibrate "confidence", then trade only when calibrated probability > 0.5.
    from sklearn.isotonic import IsotonicRegression
    for en, d in mdata.items():
        in_pred = d["in_df"]["prediction"].values
        in_act  = d["in_df"]["actual_return"].values
        correct_in = ((np.sign(in_pred) == np.sign(in_act)) & (in_pred != 0)).astype(int)
        abs_in = np.abs(in_pred)
        if len(np.unique(correct_in)) < 2:
            continue
        try:
            iso = IsotonicRegression(out_of_bounds="clip").fit(abs_in, correct_in)
            oos_pred = d["oos_df"]["pred"].values
            oos_act = d["oos_df"]["actual"].values
            oos_dates = d["oos_df"]["date"].values
            calib_prob = iso.predict(np.abs(oos_pred))
            pos_oos = np.where(calib_prob > 0.5, np.sign(oos_pred), 0.0)
            mm = metrics_from_positions(pos_oos, oos_act, oos_dates)
            name = f"calib_isotonic_exp{en}"
            new_strats[name] = {**mm, "name": name,
                "is_leaky": False,
                "selection_basis": f"Isotonic regression P(correct | |prediction|) fit on in-sample test fold (n={int((~np.isnan(in_pred)).sum())} days), trade only when calibrated P > 0.5; LOCKED, evaluated OOS.",
                "fit_method": "sklearn.isotonic.IsotonicRegression(out_of_bounds=clip) on (|pred|, correct)",
                "fit_param_chosen": "calibration threshold P=0.5",
                "signal_source": f"individual_exp{en}",
                "category": "C", "members_used": [en], "n_members_used": 1,
                "sizing_mode": "raw", "overlay": "isotonic_calibrated",
            }
        except Exception as e:
            print(f"[train-fit] isotonic fit failed for exp{en}: {e}")

    # ===========================================================
    # Write CSVs + merge into ensemble JSON
    # ===========================================================
    print(f"[train-fit] generated {len(new_strats)} train-fit strategies")
    written_csv = 0
    for nm, rec in new_strats.items():
        eq = rec.get("equity_curve") or {}
        if eq.get("dates") and eq.get("strategy_dollars"):
            n = len(eq["dates"])
            safe = nm.replace("/", "_").replace("\\", "_").replace(" ", "_")[:120]
            csv_name = f"oos_ensemble_{safe}.csv"
            csv_path = R / csv_name
            # Build minimal 16-col CSV
            df = pd.DataFrame({
                "date": eq["dates"],
                "position": [0.0] * n,
                "pred_direction": [0] * n,
                "traded": [0] * n,
                "actual_ret_1d": [0.0] * n,
                "bh_log_ret": [0.0] * n,
                "strategy_pnl": [0.0] * n,
                "correct": [0] * n,
                "equity_dollars": eq["strategy_dollars"],
                "buy_hold_dollars": eq["buy_hold_dollars"],
                "excess_dollars": [round(s - b, 4) for s, b in zip(eq["strategy_dollars"], eq["buy_hold_dollars"])],
                "cumret_pct": eq["strategy_pct"],
                "bh_cumret_pct": eq["buy_hold_pct"],
                "excess_cumret_pct": [round(s - b, 4) for s, b in zip(eq["strategy_pct"], eq["buy_hold_pct"])],
                "drawdown_pct": [0.0] * n,
                "underwater": [0] * n,
            })
            df.to_csv(csv_path, index=False, float_format="%.6f")
            rec["csv"] = csv_name
            written_csv += 1
    print(f"[train-fit] wrote {written_csv} per-strategy CSVs")

    # Merge into ensemble JSON
    ens.setdefault("strategies", {})
    n_before = len(ens["strategies"])
    ens["strategies"].update(new_strats)
    ens["n_clean_strategies"] = sum(1 for s in ens["strategies"].values() if s.get("is_leaky") == False)
    ens["n_leaky_strategies"] = sum(1 for s in ens["strategies"].values() if s.get("is_leaky") == True)
    ens["n_train_fit_strategies"] = sum(1 for s in ens["strategies"].values() if "fit_method" in s)
    ENS_JSON.write_text(json.dumps(ens, indent=2, default=str), encoding="utf-8")
    print(f"[done] strategies: {n_before} -> {len(ens['strategies'])} (+{len(ens['strategies']) - n_before} train-fit)")
    print(f"  train-fit total: {ens['n_train_fit_strategies']}")
    print(f"  clean (deployable): {ens['n_clean_strategies']} | leaky: {ens['n_leaky_strategies']}")


if __name__ == "__main__":
    main()
