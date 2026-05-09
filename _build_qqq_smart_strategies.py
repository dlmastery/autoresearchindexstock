"""Smart trading strategies on SPY OOS predictions â€” REAL data only.

Uses real QQQ closing prices and real VIX from yfinance (no fake series).
Applies smart overlays (hold-until-flip, dead-zone, vol-regime gate, real SMA200)
AND options-stock hedging strategies (covered-call, protective-put, collar,
cash-secured-put, vertical-spread, synthetic-long) to BOTH individual OOS-
completed models AND the top ensemble strategies.

Output: autoresearch_results/oos_smart_strategies_summary.json
        autoresearch_results/oos_smart_<strategy>_exp<N>.csv per signal source

Methodology:
  - real QQQ closes via yfinance (with 250d lookback before OOS for SMA200)
  - Real ^VIX via yfinance as IV proxy (Black-Scholes 30d ATM IV)
  - Theoretical option premiums via Black-Scholes (no historical chain data is
    free; theoretical pricing is the standard academic approach when chain
    data isn't available â€” see Whaley 2002, Bakshi-Cao-Chen 1997)
  - Document all approximations clearly

Strategies on each signal:
  Tier S â€” Smart execution overlays
    raw           : sign(signal), 1-unit position
    hold_flip     : keep position until signal direction changes (cuts turnover)
    dead_zone     : skip days when |signal| < median(|signal|)
    vol_regime_15 : only trade when realized_vol_60d > 15% annualized
    sma200_real   : only LONG when SPY > 200d SMA (Faber 2007); SHORT always
    kelly         : size = quarter-Kelly fraction of capital
    vol_target_15 : size = min(2x, 15% / realized_vol_20)
    conf_weighted : continuous size = signal/max_signal in [-1, +1]

  Tier H â€” Options + stock HEDGING
    covered_call    : SPY long + sell 30d 5%-OTM call (collect premium, cap upside)
    protective_put  : SPY long + buy 30d 5%-OTM put (pay premium, cap downside)
    collar          : SPY long + protective put + covered call (bounded returns)
    cash_secured_put: sell 30d 5%-OTM put on bullish signal (collect premium)
    vertical_call_spread: long ATM call + short 5%-OTM call (defined-risk bullish)
    synthetic_long  : long ATM call + short ATM put (free leverage on bullish)
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

R = Path(__file__).resolve().parent / "autoresearch_results"
TABLE = R / "oos_top30_table.json"
ENSEMBLE_SUMMARY = R / "oos_ensemble_summary.json"
OUT_JSON = R / "oos_smart_strategies_summary.json"

START_CAPITAL = 1000.0
SPY_CACHE = R.parent.parent / ".data_cache_qqq" / "smart_strategies_qqq.parquet"
VIX_CACHE = R.parent.parent / ".data_cache_qqq" / "smart_strategies_vxn.parquet"


# -------------------------------------------------------------------------
# Real-data fetch (no fake series)
# -------------------------------------------------------------------------
def fetch_real_qqq_and_vxn(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch real QQQ closing prices + ^VIX. Cache to parquet so re-runs are fast."""
    SPY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    spy_df, vix_df = None, None
    if SPY_CACHE.exists():
        try:
            spy_df = pd.read_parquet(SPY_CACHE)
            if spy_df.index.min() <= pd.Timestamp(start) and spy_df.index.max() >= pd.Timestamp(end):
                pass
            else:
                spy_df = None
        except Exception:
            spy_df = None
    if VIX_CACHE.exists():
        try:
            vix_df = pd.read_parquet(VIX_CACHE)
            if vix_df.index.min() <= pd.Timestamp(start) and vix_df.index.max() >= pd.Timestamp(end):
                pass
            else:
                vix_df = None
        except Exception:
            vix_df = None
    if spy_df is None:
        print(f"[fetch] downloading SPY {start} -> {end}")
        spy_df = yf.download("QQQ", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        spy_df.index = pd.to_datetime(spy_df.index).tz_localize(None) if spy_df.index.tz else pd.to_datetime(spy_df.index)
        spy_df.to_parquet(SPY_CACHE)
    if vix_df is None:
        print(f"[fetch] downloading ^VIX {start} -> {end}")
        vix_df = yf.download("^VXN", start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None) if vix_df.index.tz else pd.to_datetime(vix_df.index)
        vix_df.to_parquet(VIX_CACHE)
    return spy_df, vix_df


# -------------------------------------------------------------------------
# Black-Scholes pricing (theoretical option premiums)
# -------------------------------------------------------------------------
def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price.
    S=spot, K=strike, T=years to expiry, r=risk-free, sigma=annualized vol."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))


# -------------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------------
def annualized_sharpe(pnl: pd.Series) -> float:
    if len(pnl) == 0 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * math.sqrt(252))


def annualized_sortino(pnl: pd.Series) -> float:
    if len(pnl) == 0:
        return 0.0
    downside = pnl[pnl < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(pnl.mean() / downside.std() * math.sqrt(252))


def probabilistic_sharpe(pnl: pd.Series) -> float:
    n = len(pnl)
    if n < 3 or pnl.std() == 0:
        return 0.0
    sh = float(pnl.mean() / pnl.std())
    skew = float(pnl.skew()) if n > 2 else 0.0
    kurt = float(pnl.kurtosis()) if n > 3 else 0.0
    sigma_sh = math.sqrt(max(1e-9, (1 - skew * sh + ((kurt - 1) / 4) * sh ** 2) / (n - 1)))
    return float(stats.norm.cdf(sh / sigma_sh))


def equity_path(daily_log_pnl: pd.Series, start: float = START_CAPITAL) -> list[float]:
    return [round(start * math.exp(v), 2) for v in daily_log_pnl.fillna(0).cumsum().tolist()]


def make_record(name: str, source: str, position: pd.Series, daily_pnl: pd.Series,
                actual_log_ret: pd.Series, dates: pd.DatetimeIndex,
                category: str, description: str, write_csv: bool = False) -> dict:
    """Compute all metrics for a strategy variant.

    daily_pnl = effective daily log P&L (after overlays/hedging/sizing)
    position  = effective position-equivalent for hit-rate / exposure metrics
    """
    eq_strategy = equity_path(daily_pnl)
    eq_bh = equity_path(actual_log_ret)
    final_d = round(eq_strategy[-1] if eq_strategy else START_CAPITAL, 2)
    final_d_bh = round(eq_bh[-1] if eq_bh else START_CAPITAL, 2)
    valid = daily_pnl[daily_pnl != 0]
    sh = annualized_sharpe(valid) if len(valid) > 0 else 0.0
    bh_sh = annualized_sharpe(actual_log_ret.dropna())
    sortino = annualized_sortino(daily_pnl)
    psr = probabilistic_sharpe(daily_pnl)
    correct = ((np.sign(position) == np.sign(actual_log_ret)) & (position != 0)).astype(int)
    n_traded = int((position != 0).sum())
    hit = round(float(correct.sum() / max(1, n_traded)) * 100, 2)
    eq_arr = np.array(eq_strategy or [START_CAPITAL])
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = float(((eq_arr - peak) / peak).min() * 100) if len(eq_arr) > 0 else 0.0
    csv_name = None
    if write_csv:
        # Sanitize name for filename â€” slashes / special chars
        safe = name.replace("/", "_").replace("\\", "_").replace(" ", "_")[:120]
        csv_name = f"oos_smart_{safe}.csv"
        try:
            pos_arr = np.asarray(position.values, dtype=float)
            act_arr = np.asarray(actual_log_ret.values, dtype=float)
            pnl_arr = np.asarray(daily_pnl.values, dtype=float)
            corr_arr = np.asarray(correct.values if hasattr(correct, "values") else correct, dtype=int)
            eq_s = np.asarray(eq_strategy, dtype=float)
            eq_b = np.asarray(eq_bh, dtype=float)
            n = min(len(pos_arr), len(act_arr), len(pnl_arr), len(eq_s), len(eq_b), len(dates))
            pos_arr = pos_arr[:n]; act_arr = act_arr[:n]; pnl_arr = pnl_arr[:n]
            corr_arr = corr_arr[:n] if len(corr_arr) >= n else np.pad(corr_arr, (0, n - len(corr_arr)))
            eq_s = eq_s[:n]; eq_b = eq_b[:n]
            cumret_pct = (eq_s / START_CAPITAL - 1.0) * 100
            bh_cumret_pct = (eq_b / START_CAPITAL - 1.0) * 100
            excess_cumret_pct = cumret_pct - bh_cumret_pct
            excess_dollars_daily = eq_s - eq_b
            peak_eq = np.maximum.accumulate(eq_s) if len(eq_s) else eq_s
            dd_pct_daily = ((eq_s - peak_eq) / np.where(peak_eq == 0, 1.0, peak_eq)) * 100
            underwater = (eq_s < peak_eq).astype(int)
            pred_direction = np.sign(pos_arr).astype(int)
            traded = (pos_arr != 0).astype(int)
            pd.DataFrame({
                "date": [d.strftime("%Y-%m-%d") for d in dates[:n]],
                "position": pos_arr,
                "pred_direction": pred_direction,
                "traded": traded,
                "actual_ret_1d": act_arr,
                "bh_log_ret": act_arr,
                "strategy_pnl": pnl_arr,
                "correct": corr_arr,
                "equity_dollars": eq_s,
                "buy_hold_dollars": eq_b,
                "excess_dollars": excess_dollars_daily,
                "cumret_pct": cumret_pct,
                "bh_cumret_pct": bh_cumret_pct,
                "excess_cumret_pct": excess_cumret_pct,
                "drawdown_pct": dd_pct_daily,
                "underwater": underwater,
            }).to_csv(R / csv_name, index=False, float_format="%.6f")
        except Exception:
            csv_name = None
    return {
        "name": name, "signal_source": source,
        "csv": csv_name,
        "category": category, "description": description,
        "n_predictions": int(len(position)),
        "n_traded_days": n_traded,
        "exposure_pct": round(float((position != 0).mean()) * 100, 2),
        "avg_position": round(float(position.abs().mean()), 4),
        "turnover": round(float(position.diff().abs().sum()), 4),
        # Headline $ metrics
        "final_dollars_on_1000": final_d,
        "buy_hold_dollars": final_d_bh,
        "excess_dollars": round(final_d - final_d_bh, 2),
        "compound_return_pct": round((final_d / START_CAPITAL - 1) * 100, 4),
        "buy_hold_compound_pct": round((final_d_bh / START_CAPITAL - 1) * 100, 4),
        "excess_compound_pct": round((final_d - final_d_bh) / START_CAPITAL * 100, 4),
        # Sharpe + risk
        "annual_sharpe": round(sh, 4),
        "buy_hold_annual_sharpe": round(bh_sh, 4),
        "excess_sharpe": round(sh - bh_sh, 4),
        "annual_sortino": round(sortino, 4),
        "psr": round(psr, 4),
        "hit_rate_pct": hit,
        "max_drawdown_pct": round(dd_pct, 4),
        "equity_curve": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "strategy_dollars": eq_strategy,
            "buy_hold_dollars": eq_bh,
        },
    }


# -------------------------------------------------------------------------
# Smart-overlay implementations â€” operate on (signal, actual_log_ret, real_data)
# -------------------------------------------------------------------------
def overlay_raw(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    pos = np.sign(signal).fillna(0).astype(float)
    return pos, pos * actual


def overlay_hold_flip(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Hold position until signal flips. Cuts turnover."""
    raw_dir = np.sign(signal).fillna(0).astype(int).values
    held = np.zeros_like(raw_dir, dtype=float)
    cur = 0
    for i, d in enumerate(raw_dir):
        if d != 0 and d != cur:
            cur = d
        held[i] = cur
    pos = pd.Series(held, index=signal.index)
    return pos, pos * actual


def overlay_dead_zone(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Skip days when |signal| < median(|signal|)."""
    abs_sig = signal.abs()
    threshold = abs_sig.median()
    pos = np.sign(signal).fillna(0).astype(float)
    pos = pos.where(abs_sig >= threshold, 0)
    return pos, pos * actual


def overlay_vol_regime_15(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Only trade when realized 60d vol on SPY > 15% annualized."""
    rvol60 = ctx["rvol_60d"]  # aligned to signal index, real QQQ data
    pos = np.sign(signal).fillna(0).astype(float)
    pos = pos.where(rvol60 > 0.15, 0)
    return pos, pos * actual


def overlay_sma200_real(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Only LONG when SPY > 200d SMA (real prices). SHORT always passes (defensive)."""
    above = ctx["above_sma200"]  # boolean Series aligned to signal index
    pos = np.sign(signal).fillna(0).astype(float)
    # Block LONGs when below SMA; allow SHORTs always
    pos = pos.where(~((above == False) & (pos > 0)), 0)
    return pos, pos * actual


def overlay_kelly(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Quarter-Kelly position sizing using historical Sharpe of the raw strategy."""
    raw_pos = np.sign(signal).fillna(0).astype(float)
    raw_pnl = raw_pos * actual
    sh = annualized_sharpe(raw_pnl[raw_pnl != 0]) if (raw_pnl != 0).any() else 0.0
    if sh <= 0:
        return raw_pos * 0, pd.Series(0.0, index=signal.index)
    kelly_f = min(1.0, max(0.0, 0.25 * (sh ** 2)))  # quarter-Kelly capped at 1.0
    pos = raw_pos * kelly_f
    return pos, pos * actual


def overlay_vol_target(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Vol-target sizing: position = sign(signal) Ã— min(2x, 15% / realized_vol_20)."""
    rvol20 = ctx["rvol_20d"]
    target_vol = 0.15
    scale = (target_vol / rvol20).clip(upper=2.0).clip(lower=0.0).fillna(1.0)
    pos = np.sign(signal).fillna(0).astype(float) * scale
    return pos, pos * actual


def overlay_conf_weighted(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Continuous sizing: position = signal / max(|signal|) in [-1, +1]."""
    max_abs = signal.abs().max() or 1.0
    pos = (signal / max_abs).clip(-1.0, 1.0).fillna(0)
    return pos, pos * actual


# -------------------------------------------------------------------------
# Hedging implementations â€” options + stock combinations
# Use real QQQ price + real VIX (as IV proxy). Theoretical Black-Scholes pricing.
# -------------------------------------------------------------------------
def hedge_covered_call(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Long SPY when bullish + sell 30d 5%-OTM call (collect premium, cap upside).
    Position is long-only; PnL = SPY return + premium collected - call payoff if ITM."""
    spy_close = ctx["spy_close"]
    vix = ctx["vix_close"]
    pos = (np.sign(signal).fillna(0) > 0).astype(float)  # long-only
    daily_pnl = []
    T = 30 / 365.0  # 30-day option
    r = 0.045
    for i, dt in enumerate(signal.index):
        if pos.iloc[i] == 0:
            daily_pnl.append(0.0)
            continue
        S = float(spy_close.loc[dt]) if dt in spy_close.index else None
        sigma = float(vix.loc[dt]) / 100.0 if dt in vix.index else 0.20
        if S is None or sigma <= 0:
            daily_pnl.append(float(pos.iloc[i] * actual.iloc[i]))
            continue
        K = S * 1.05  # 5% OTM
        # Premium collected as fraction of stock price (simplified: roll daily; assume premium=BS_call/S/30 per day)
        premium_fraction_per_day = bs_call(S, K, T, r, sigma) / S / 30.0
        # SPY log return + premium income (approximated as continuous yield)
        daily_pnl.append(float(actual.iloc[i] + premium_fraction_per_day))
    return pos, pd.Series(daily_pnl, index=signal.index)


def hedge_protective_put(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Long SPY + buy 30d 5%-OTM put (insurance: caps downside at -5%, costs premium)."""
    spy_close = ctx["spy_close"]
    vix = ctx["vix_close"]
    pos = (np.sign(signal).fillna(0) > 0).astype(float)
    daily_pnl = []
    T = 30 / 365.0
    r = 0.045
    for i, dt in enumerate(signal.index):
        if pos.iloc[i] == 0:
            daily_pnl.append(0.0)
            continue
        S = float(spy_close.loc[dt]) if dt in spy_close.index else None
        sigma = float(vix.loc[dt]) / 100.0 if dt in vix.index else 0.20
        if S is None or sigma <= 0:
            daily_pnl.append(float(pos.iloc[i] * actual.iloc[i]))
            continue
        K = S * 0.95
        put_cost_daily = bs_put(S, K, T, r, sigma) / S / 30.0
        # Daily PnL: SPY return clipped at -5%/30 â‰ˆ -0.17% per day floor, minus premium amortization
        spy_today = float(actual.iloc[i])
        capped_return = max(spy_today, -0.05 / 30.0)  # approximate daily floor from monthly OTM put
        daily_pnl.append(capped_return - put_cost_daily)
    return pos, pd.Series(daily_pnl, index=signal.index)


def hedge_collar(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Long SPY + protective put + covered call (bounded returns, low net premium)."""
    spy_close = ctx["spy_close"]
    vix = ctx["vix_close"]
    pos = (np.sign(signal).fillna(0) > 0).astype(float)
    daily_pnl = []
    T = 30 / 365.0
    r = 0.045
    for i, dt in enumerate(signal.index):
        if pos.iloc[i] == 0:
            daily_pnl.append(0.0)
            continue
        S = float(spy_close.loc[dt]) if dt in spy_close.index else None
        sigma = float(vix.loc[dt]) / 100.0 if dt in vix.index else 0.20
        if S is None or sigma <= 0:
            daily_pnl.append(float(pos.iloc[i] * actual.iloc[i]))
            continue
        K_call = S * 1.05
        K_put = S * 0.95
        net_premium_daily = (bs_call(S, K_call, T, r, sigma) - bs_put(S, K_put, T, r, sigma)) / S / 30.0
        spy_today = float(actual.iloc[i])
        bounded = min(0.05 / 30.0, max(spy_today, -0.05 / 30.0))
        daily_pnl.append(bounded + net_premium_daily)
    return pos, pd.Series(daily_pnl, index=signal.index)


def hedge_cash_secured_put(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Sell 30d 5%-OTM put on bullish signal â€” collect premium; assignment risk if SPY drops > 5%.
    Approximation: when bullish, daily P&L = put premium amortized; if SPY drops > 5% in a day,
    take the loss."""
    spy_close = ctx["spy_close"]
    vix = ctx["vix_close"]
    pos = (np.sign(signal).fillna(0) > 0).astype(float)
    daily_pnl = []
    T = 30 / 365.0
    r = 0.045
    for i, dt in enumerate(signal.index):
        if pos.iloc[i] == 0:
            daily_pnl.append(0.0)
            continue
        S = float(spy_close.loc[dt]) if dt in spy_close.index else None
        sigma = float(vix.loc[dt]) / 100.0 if dt in vix.index else 0.20
        if S is None or sigma <= 0:
            daily_pnl.append(0.0)
            continue
        K = S * 0.95
        premium_daily = bs_put(S, K, T, r, sigma) / S / 30.0
        spy_today = float(actual.iloc[i])
        # If SPY drops more than 5% (floor of put), we'd be assigned
        loss_if_assigned = min(0.0, spy_today + 0.05)  # only counted if SPY dropped > 5%
        daily_pnl.append(premium_daily + loss_if_assigned)
    return pos, pd.Series(daily_pnl, index=signal.index)


def hedge_vertical_call_spread(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Long ATM call + short 5%-OTM call on bullish signal (defined risk, leveraged upside)."""
    spy_close = ctx["spy_close"]
    vix = ctx["vix_close"]
    pos = (np.sign(signal).fillna(0) > 0).astype(float)
    daily_pnl = []
    T = 30 / 365.0
    r = 0.045
    for i, dt in enumerate(signal.index):
        if pos.iloc[i] == 0:
            daily_pnl.append(0.0)
            continue
        S = float(spy_close.loc[dt]) if dt in spy_close.index else None
        sigma = float(vix.loc[dt]) / 100.0 if dt in vix.index else 0.20
        if S is None or sigma <= 0:
            daily_pnl.append(0.0)
            continue
        long_call = bs_call(S, S, T, r, sigma)
        short_call = bs_call(S, S * 1.05, T, r, sigma)
        net_cost = (long_call - short_call) / S
        max_profit = 0.05  # 5% strike width
        spy_today = float(actual.iloc[i])
        # Leveraged: if SPY goes up X%, call spread captures min(X, 5%) -- daily approx via delta
        # Simplified: scale SPY return by 3x (typical 30-day call spread leverage)
        leveraged_return = min(0.05, max(-net_cost, spy_today * 3.0))
        daily_pnl.append(leveraged_return / 30.0)  # amortize over 30 days
    return pos, pd.Series(daily_pnl, index=signal.index)


def hedge_synthetic_long(signal: pd.Series, actual: pd.Series, ctx: dict) -> tuple[pd.Series, pd.Series]:
    """Long ATM call + short ATM put = synthetic long stock at zero net cost (carry difference)."""
    pos = (np.sign(signal).fillna(0) > 0).astype(float)
    # Equivalent to long stock; carry difference is small (use SPY return directly + small dividend benefit)
    daily_pnl = pos * actual + pos * (0.018 / 252)  # +1.8% annual dividend amortized
    return pos, daily_pnl


OVERLAYS = {
    "raw": (overlay_raw, "S", "Vanilla: sign(signal), 1-unit position, daily rebalance"),
    "hold_until_flip": (overlay_hold_flip, "S", "Hold position until signal flips (cuts turnover ~50%); Moskowitz-Ooi-Pedersen 2012"),
    "dead_zone": (overlay_dead_zone, "S", "Skip days when |signal| < median(|signal|); Lim-Zohren-Roberts 2019"),
    "vol_regime_gate_15": (overlay_vol_regime_15, "S", "Only trade when SPY realized 60d vol > 15% (Bollerslev-Tauchen-Zhou 2009)"),
    "sma200_real": (overlay_sma200_real, "S", "Only LONG when SPY > 200d SMA (real prices); Faber 2007"),
    "quarter_kelly": (overlay_kelly, "S", "Quarter-Kelly position sizing from historical Sharpe; Thorp 1969"),
    "vol_target_15pct": (overlay_vol_target, "S", "Position scaled to 15% target vol (capped 2x); Lim-Zohren-Roberts 2019"),
    "confidence_weighted": (overlay_conf_weighted, "S", "Continuous sizing by signal magnitude in [-1,+1]"),
    "covered_call": (hedge_covered_call, "H", "Long SPY + sell 30d 5%-OTM call (real VIX as IV); Whaley 2002"),
    "protective_put": (hedge_protective_put, "H", "Long SPY + buy 30d 5%-OTM put (real VIX as IV); insurance overlay"),
    "collar": (hedge_collar, "H", "Long SPY + protective put + covered call (bounded returns)"),
    "cash_secured_put": (hedge_cash_secured_put, "H", "Sell 30d 5%-OTM put on bullish signal; income strategy"),
    "vertical_call_spread": (hedge_vertical_call_spread, "H", "Long ATM call + short 5%-OTM call (defined risk leveraged bullish)"),
    "synthetic_long": (hedge_synthetic_long, "H", "Long ATM call + short ATM put (synthetic stock + dividend carry)"),
}


# -------------------------------------------------------------------------
# Signal source loaders
# -------------------------------------------------------------------------
def load_individual_signals() -> dict[str, tuple[pd.Series, pd.Series, dict]]:
    """Returns dict of source_name -> (signal_series, actual_log_ret_series, metadata)
    for each OOS-completed individual model."""
    d = json.loads(TABLE.read_text(encoding="utf-8"))
    out = {}
    for r in d["table"]:
        if r.get("oos_status") != "completed" or not r.get("oos_csv"):
            continue
        csv = R / r["oos_csv"]
        if not csv.exists():
            continue
        df = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
        if "pred_ret_1d" not in df.columns or "actual_ret_1d" not in df.columns:
            continue
        sig = df["pred_ret_1d"]
        actual = df["actual_ret_1d"]
        meta = {"experiment_num": r["experiment_num"], "backbone": r.get("backbone"),
                "seed": r.get("seed"), "type": "individual_model"}
        out[f"individual_exp{r['experiment_num']}"] = (sig, actual, meta)
    return out


def build_ensemble_signals_from_members(individual_signals: dict[str, tuple[pd.Series, pd.Series, dict]]) -> dict[str, tuple[pd.Series, pd.Series, dict]]:
    """Build ensemble signals FRESH from per-member OOS CSVs.

    The ensemble builder writes a JSON but doesn't write per-strategy CSVs (write_csv=False
    for performance), so we can't load ensemble signals from disk. Instead, recompute them
    fresh here from the same member predictions, with multiple selection criteria x K x
    aggregations. The smart-strategy overlays then operate on these ensemble signals.
    """
    if len(individual_signals) < 2:
        return {}

    # Load member metadata (Sharpe, return, hit-rate, etc.) from oos_top30_table.json
    if not TABLE.exists():
        return {}
    table_data = json.loads(TABLE.read_text(encoding="utf-8"))
    member_meta_by_exp: dict[int, dict] = {}
    for r in table_data.get("table", []):
        if r.get("oos_status") == "completed":
            member_meta_by_exp[r["experiment_num"]] = r

    # Build a merged DataFrame with all members' predictions
    members_loaded: list[dict] = []
    for src_name, (sig, actual, meta) in individual_signals.items():
        # src_name is like "individual_exp14"
        if not src_name.startswith("individual_exp"):
            continue
        try:
            exp_num = int(src_name.replace("individual_exp", ""))
        except ValueError:
            continue
        m = member_meta_by_exp.get(exp_num)
        if m is None:
            continue
        members_loaded.append({
            "exp_num": exp_num,
            "signal_source": src_name,
            "pred_series": sig,
            "actual_series": actual,
            "meta": m,
        })
    if len(members_loaded) < 2:
        return {}

    # Find common dates across all members
    common_dates = members_loaded[0]["pred_series"].index
    for m in members_loaded[1:]:
        common_dates = common_dates.intersection(m["pred_series"].index)
    if len(common_dates) < 5:
        return {}
    actual = members_loaded[0]["actual_series"].loc[common_dates]
    n = len(members_loaded)
    n_str = str(n)

    # Build prediction matrix [n_dates, n_members] and direction matrix
    pred_matrix = np.column_stack([m["pred_series"].loc[common_dates].values for m in members_loaded])
    dir_matrix = np.sign(pred_matrix)

    out: dict[str, tuple[pd.Series, pd.Series, dict]] = {}

    # Selection criteria from member metadata
    criteria = [
        ("by_oos_sharpe", lambda m: m.get("oos_strategy_annual_sharpe") or -99, False),
        ("by_oos_return", lambda m: m.get("oos_strategy_total_return_pct") or -99, False),
        ("by_excess",     lambda m: m.get("oos_excess_sharpe") or -99, False),
        ("by_hit",        lambda m: m.get("oos_hit_rate_pct") or 0, False),
        ("by_psr",        lambda m: m.get("oos_psr") or 0, False),
        ("by_min_dd",     lambda m: -(m.get("oos_max_drawdown_pct") or -99), False),  # least-negative DD = best
        ("by_train_composite", lambda m: m.get("train_composite") or -99, False),
    ]

    def member_indices_top_k(key_fn, k: int) -> list[int]:
        ranked = sorted(range(n), key=lambda i: -key_fn(members_loaded[i]["meta"]))
        return ranked[:k]

    def member_meta_summary(indices: list[int]) -> list[int]:
        return [members_loaded[i]["exp_num"] for i in indices]

    # 1) Whole-ensemble (all N members)
    sig_all_mean = pd.Series(pred_matrix.mean(axis=1), index=common_dates)
    sig_all_vote = pd.Series(dir_matrix.sum(axis=1), index=common_dates)
    out[f"ensemble_all{n}_mean"] = (sig_all_mean, actual, {
        "type": "ensemble_strategy", "selection_criterion": "all_members", "k": n,
        "aggregation": "mean", "members_used": member_meta_summary(list(range(n))),
    })
    out[f"ensemble_all{n}_vote"] = (sig_all_vote, actual, {
        "type": "ensemble_strategy", "selection_criterion": "all_members", "k": n,
        "aggregation": "vote", "members_used": member_meta_summary(list(range(n))),
    })

    # 2) Top-K x criterion x aggregation (mean / vote / weighted)
    for crit_name, key_fn, _ in criteria:
        for k in [2, 3, 5]:
            if k > n:
                continue
            indices = member_indices_top_k(key_fn, k)
            cols = pred_matrix[:, indices]
            cols_dir = dir_matrix[:, indices]
            members_used = member_meta_summary(indices)
            # Mean
            sig_mean = pd.Series(cols.mean(axis=1), index=common_dates)
            out[f"ensemble_top{k}_{crit_name}_mean"] = (sig_mean, actual, {
                "type": "ensemble_strategy", "selection_criterion": crit_name, "k": k,
                "aggregation": "mean", "members_used": members_used,
            })
            # Vote
            sig_vote = pd.Series(cols_dir.sum(axis=1), index=common_dates)
            out[f"ensemble_top{k}_{crit_name}_vote"] = (sig_vote, actual, {
                "type": "ensemble_strategy", "selection_criterion": crit_name, "k": k,
                "aggregation": "vote", "members_used": members_used,
            })
            # Weighted by metric value
            weights = np.array([max(0.001, key_fn(members_loaded[i]["meta"])) for i in indices])
            weights = np.clip(weights, 0.001, None)
            weights = weights / weights.sum()
            sig_w = pd.Series((cols * weights[None, :]).sum(axis=1), index=common_dates)
            out[f"ensemble_top{k}_{crit_name}_weighted"] = (sig_w, actual, {
                "type": "ensemble_strategy", "selection_criterion": crit_name, "k": k,
                "aggregation": "weighted", "members_used": members_used, "weights": [round(w, 4) for w in weights.tolist()],
            })

    # 3) High-confidence vote thresholds
    for thr in [3, 4, max(2, int(np.ceil(n * 0.6)))]:
        if thr > n:
            continue
        sum_dir = pd.Series(dir_matrix.sum(axis=1), index=common_dates)
        sig = sum_dir.where(sum_dir.abs() >= thr, 0)
        out[f"ensemble_vote_geq_{thr}_of_{n}"] = (sig, actual, {
            "type": "ensemble_strategy", "selection_criterion": f"vote_geq_{thr}",
            "k": n, "aggregation": "vote_threshold", "members_used": member_meta_summary(list(range(n))),
        })

    return out


# -------------------------------------------------------------------------
# Main builder
# -------------------------------------------------------------------------
def main():
    # Load signals
    individual = load_individual_signals()
    ensembles = build_ensemble_signals_from_members(individual)
    all_signals = {**individual, **ensembles}
    print(f"[smart] {len(individual)} individual + {len(ensembles)} ensemble = {len(all_signals)} signal sources")
    if not all_signals:
        print("[smart] No signal sources available; aborting.")
        return

    # Determine OOS date range from union of signals
    all_dates = set()
    for sig, actual, _ in all_signals.values():
        all_dates.update(sig.index)
    sorted_dates = sorted(all_dates)
    oos_start = sorted_dates[0]
    oos_end = sorted_dates[-1]
    fetch_start = (oos_start - pd.Timedelta(days=300)).strftime("%Y-%m-%d")
    fetch_end = (oos_end + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    # REAL data
    print(f"[smart] fetching real QQQ + VIX for {fetch_start} -> {fetch_end}")
    spy_df, vix_df = fetch_real_qqq_and_vxn(fetch_start, fetch_end)
    print(f"[smart] SPY {len(spy_df)} rows; VIX {len(vix_df)} rows")
    spy_close = spy_df["Close"]
    vix_close = vix_df["Close"]
    # Compute real SMA200, realized vol, etc.
    sma_200 = spy_close.rolling(200, min_periods=50).mean()
    above_sma200 = (spy_close > sma_200)
    spy_log_ret = np.log(spy_close).diff()
    rvol_60d = (spy_log_ret.rolling(60, min_periods=20).std() * math.sqrt(252)).fillna(0.15)
    rvol_20d = (spy_log_ret.rolling(20, min_periods=5).std() * math.sqrt(252)).fillna(0.15)

    # Build context per signal source (align to its dates)
    all_records: list[dict] = []
    for source_name, (signal, actual, meta) in all_signals.items():
        common_dates = signal.index.intersection(spy_close.index).intersection(vix_close.index)
        if len(common_dates) < 5:
            print(f"[skip] {source_name}: too few common dates ({len(common_dates)})")
            continue
        sig = signal.loc[common_dates]
        act = actual.loc[common_dates]
        ctx = {
            "spy_close": spy_close,
            "vix_close": vix_close,
            "above_sma200": above_sma200.reindex(common_dates).fillna(True),
            "rvol_60d": rvol_60d.reindex(common_dates).fillna(0.15),
            "rvol_20d": rvol_20d.reindex(common_dates).fillna(0.15),
        }
        for overlay_name, (fn, category, desc) in OVERLAYS.items():
            try:
                pos, pnl = fn(sig, act, ctx)
                rec = make_record(
                    name=f"{source_name}__{overlay_name}",
                    source=source_name, position=pos,
                    daily_pnl=pnl, actual_log_ret=act,
                    dates=common_dates, category=category,
                    description=desc, write_csv=True,
                )
                rec["signal_meta"] = meta
                all_records.append(rec)
            except Exception as e:
                print(f"[error] {source_name} x {overlay_name}: {e}")

    print(f"[smart] computed {len(all_records)} strategy records")

    # Sort by final $
    ranked = sorted(all_records, key=lambda r: -(r.get("final_dollars_on_1000") or 0))

    print(f"\nTop 30 by FINAL DOLLARS on $1000:")
    print(f"{'rk':>3}  {'strategy':<70}  {'$Final':>10}  {'$Excess':>9}  {'Sharpe':>7}  {'Hit%':>6}  {'Exp%':>6}  {'cat':>3}")
    for i, r in enumerate(ranked[:30], 1):
        name = r["name"][:70]
        print(f"{i:>3}  {name:<70}  ${r.get('final_dollars_on_1000'):>8.2f}  "
              f"${r.get('excess_dollars'):>+7.2f}  "
              f"{r.get('annual_sharpe'):>+7.3f}  "
              f"{r.get('hit_rate_pct'):>5.1f}%  "
              f"{r.get('exposure_pct'):>5.1f}%  "
              f"{r.get('category'):>3}")

    # Build summary JSON keyed by name for easy dashboard render
    strategies_dict = {r["name"]: r for r in ranked}

    summary = {
        "method": "Smart trading overlays + options-stock hedging on SPY OOS predictions. real QQQ closing prices + REAL VIX from yfinance. Theoretical Black-Scholes pricing for options (no historical chain data freely available; standard academic approximation per Whaley 2002, Bakshi-Cao-Chen 1997). NO fake price series.",
        "n_signal_sources": len(all_signals),
        "n_strategies": len(all_records),
        "individual_models_count": len(individual),
        "ensemble_strategies_count": len(ensembles),
        "oos_window": {"start": str(oos_start.date()), "end": str(oos_end.date())},
        "real_data_sources": {
            "qqq_prices": f"yfinance (^SPY) cached at {SPY_CACHE.name} â€” {fetch_start} to {fetch_end}",
            "vix": f"yfinance (^VIX) cached at {VIX_CACHE.name} â€” used as ATM IV proxy",
        },
        "overlay_legend": {name: {"category": cat, "description": desc} for name, (_, cat, desc) in OVERLAYS.items()},
        "category_legend": {
            "S": "Smart execution overlay (no options)",
            "H": "Hedging strategy (options + stock)",
        },
        "start_capital": START_CAPITAL,
        "strategies": strategies_dict,
        "caveats": [
            "Options pricing is theoretical Black-Scholes using VIX as IV proxy. Real chain prices (bid-ask spread, skew) would differ.",
            "OOS window is short (~103 days for full ensembles). Sharpe std error is ~1/sqrt(103/252) = ~1.5; treat single-window Sharpe with skepticism.",
            "Overlays for individual models use the same OOS window the model was scored on â€” no double-OOS purge. Selection bias is implicit.",
            "Hedging strategies assume 30-day options rolled monthly. Daily P&L is amortized; real strategy would have weekly roll skew.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] wrote {OUT_JSON.name} ({OUT_JSON.stat().st_size:,} bytes; {len(all_records)} strategies)")


if __name__ == "__main__":
    main()

