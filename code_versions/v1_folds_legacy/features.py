"""Equity-index feature engineering for QQQ.

~120 features grouped by economic intuition; every group cites the seminal
paper that motivated it. Defensive against missing tickers (some yfinance
symbols are unreliable) — features computed from absent tickers are simply
omitted, never NaN-padded.

Public API:

* ``compute_qqq_features(downloaded)`` — feature matrix indexed by date.
* ``compute_qqq_targets(downloaded)`` — multi-target frame with the four
  variants A / B / C / D defined in CLAUDE.md.

Both consume the dict returned by ``download.download_all()``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WARMUP = 252  # one trading year of warmup so longest rolling windows fill


# =============================================================================
# 1.  Generic helpers — pure pandas, no third-party TA libs
# =============================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 1978 — Relative Strength Index."""
    delta = close.diff()
    up = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    dn = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 1978 — Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Appel 2005 — Moving Average Convergence Divergence."""
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def _bollinger_z(close: pd.Series, period: int = 20) -> pd.Series:
    """Bollinger 1980s — z-score of close relative to 20d MA / std."""
    m = close.rolling(period, min_periods=period).mean()
    s = close.rolling(period, min_periods=period).std()
    return (close - m) / s.replace(0, np.nan)


def _donchian_position(close: pd.Series, period: int) -> pd.Series:
    hi = close.rolling(period, min_periods=period).max()
    lo = close.rolling(period, min_periods=period).min()
    rng = (hi - lo).replace(0, np.nan)
    return (close - lo) / rng


def _parkinson_vol(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    """Parkinson 1980 — high-low realised vol estimator."""
    log_hl_sq = (np.log(high / low) ** 2)
    return np.sqrt(log_hl_sq.rolling(period, min_periods=period).mean() / (4 * np.log(2)))


def _garman_klass_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    """Garman & Klass 1980 — efficient vol estimator using OHLC."""
    rs = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(rs.rolling(period, min_periods=period).mean())


def _yang_zhang_vol(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    """Yang & Zhang 2000 — drift-independent realised vol; recommended over GK."""
    log_oc = np.log(o / c.shift(1))
    log_co = np.log(c / o)
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    sigma_o = log_oc.rolling(period, min_periods=period).var()
    sigma_c = log_co.rolling(period, min_periods=period).var()
    sigma_rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(period, min_periods=period).mean()
    k = 0.34 / (1.34 + (period + 1) / (period - 1))
    return np.sqrt(sigma_o + k * sigma_c + (1 - k) * sigma_rs)


def _winsorise(s: pd.Series, q: float = 0.005) -> pd.Series:
    lo = s.quantile(q)
    hi = s.quantile(1 - q)
    return s.clip(lo, hi)


# =============================================================================
# 2.  Per-asset OHLCV features (used for QQQ + sector ETFs)
# =============================================================================

def _ohlcv_features(df: pd.DataFrame, prefix: str, *, full: bool = True) -> pd.DataFrame:
    """Compute price/volume/vol features for one OHLCV frame.

    ``full=False`` gives a compact set (returns + 5/20d MA + 20d Z) for
    benchmark/sector tickers where we only need a few summary stats.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    v = df.get("Volume")
    out: Dict[str, pd.Series] = {}

    # --- Returns at multiple horizons (Lo & MacKinlay 1988 — variance ratio test)
    log_c = np.log(c)
    out[f"{prefix}_logret_1d"] = log_c.diff(1)
    out[f"{prefix}_logret_5d"] = log_c.diff(5)
    out[f"{prefix}_logret_20d"] = log_c.diff(20)
    if full:
        out[f"{prefix}_logret_60d"] = log_c.diff(60)
        out[f"{prefix}_logret_120d"] = log_c.diff(120)
        out[f"{prefix}_logret_252d"] = log_c.diff(252)

    if not full:
        out[f"{prefix}_ma20_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
        return pd.DataFrame(out)

    # --- Moving-average crossings (Brock, Lakonishok & LeBaron 1992 JF)
    for n in (5, 10, 20, 50, 100, 200):
        out[f"{prefix}_close_to_sma{n}"] = c / c.rolling(n, min_periods=n).mean() - 1.0

    # --- Oscillators
    out[f"{prefix}_rsi14"] = _rsi(c, 14)
    out[f"{prefix}_rsi28"] = _rsi(c, 28)

    macd_line, macd_sig, macd_hist = _macd(c)
    out[f"{prefix}_macd_norm"] = macd_line / c
    out[f"{prefix}_macd_hist_norm"] = macd_hist / c

    out[f"{prefix}_bb_z20"] = _bollinger_z(c, 20)

    # %K stochastic
    lo14 = l.rolling(14, min_periods=14).min()
    hi14 = h.rolling(14, min_periods=14).max()
    out[f"{prefix}_stoch_k14"] = (c - lo14) / (hi14 - lo14).replace(0, np.nan) * 100

    # Williams %R
    out[f"{prefix}_williams_r14"] = (hi14 - c) / (hi14 - lo14).replace(0, np.nan) * -100

    # ATR / N-ATR
    atr14 = _atr(h, l, c, 14)
    out[f"{prefix}_natr14"] = atr14 / c

    # --- Realised vol estimators
    out[f"{prefix}_park_vol_5d"] = _parkinson_vol(h, l, 5)
    out[f"{prefix}_park_vol_20d"] = _parkinson_vol(h, l, 20)
    out[f"{prefix}_gk_vol_20d"] = _garman_klass_vol(o, h, l, c, 20)
    out[f"{prefix}_yz_vol_20d"] = _yang_zhang_vol(o, h, l, c, 20)
    out[f"{prefix}_yz_vol_60d"] = _yang_zhang_vol(o, h, l, c, 60)

    # --- Donchian channel position
    out[f"{prefix}_donchian_pos20"] = _donchian_position(c, 20)
    out[f"{prefix}_donchian_pos50"] = _donchian_position(c, 50)
    out[f"{prefix}_donchian_pos252"] = _donchian_position(c, 252)

    # --- 52w high / low distance (George & Hwang 2004 JF)
    hi252 = c.rolling(252, min_periods=252).max()
    lo252 = c.rolling(252, min_periods=252).min()
    out[f"{prefix}_dist_52w_high"] = c / hi252 - 1.0
    out[f"{prefix}_dist_52w_low"] = c / lo252 - 1.0

    # --- Momentum + reversal (Jegadeesh & Titman 1993 JF; Lehmann 1990)
    out[f"{prefix}_mom_3m"] = c.pct_change(63)
    out[f"{prefix}_mom_6m"] = c.pct_change(126)
    out[f"{prefix}_mom_9m"] = c.pct_change(189)
    out[f"{prefix}_mom_12m"] = c.pct_change(252)
    # 12-1 and 12-2 skipped-month momentum (Asness, Moskowitz & Pedersen 2013)
    out[f"{prefix}_mom_12_1"] = c.shift(21).pct_change(231)
    out[f"{prefix}_mom_12_2"] = c.shift(42).pct_change(210)
    # Short-term reversal (1w, 1m)
    out[f"{prefix}_rev_1w"] = c.pct_change(5)
    out[f"{prefix}_rev_1m"] = c.pct_change(21)
    # MAX feature (Bali, Cakici & Whitelaw 2011 JFE)
    out[f"{prefix}_max_1m_ret"] = c.pct_change(1).rolling(21).max()

    # --- Volume / microstructure (where available)
    if v is not None and not v.isna().all():
        v_log = np.log1p(v.replace(0, np.nan))
        out[f"{prefix}_vol_z20"] = (v_log - v_log.rolling(20).mean()) / v_log.rolling(20).std().replace(0, np.nan)
        out[f"{prefix}_vol_z60"] = (v_log - v_log.rolling(60).mean()) / v_log.rolling(60).std().replace(0, np.nan)
        # Amihud 2002 illiquidity
        amihud = (c.pct_change().abs() / v.replace(0, np.nan)).rolling(20).mean()
        out[f"{prefix}_amihud_20d"] = np.log1p(amihud * 1e9)
        # On-balance volume Z
        sign = np.sign(c.diff())
        obv = (sign * v).cumsum()
        out[f"{prefix}_obv_z60"] = (obv - obv.rolling(60).mean()) / obv.rolling(60).std().replace(0, np.nan)

    return pd.DataFrame(out)


# =============================================================================
# 3.  Vol-regime / VIX-family features
# =============================================================================

def _vol_regime_features(downloaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """VIX, term structure, VVIX, SKEW, variance risk premium.

    Bollerslev, Tauchen & Zhou 2009 — variance risk premium (VRP) predicts
    future equity returns. Whaley 2009 — VIX as fear gauge.
    """
    out: Dict[str, pd.Series] = {}

    def _close(t: str) -> Optional[pd.Series]:
        return downloaded[t]["Close"] if t in downloaded else None

    vix = _close("^VIX")
    vix9d = _close("^VIX9D")
    vix3m = _close("^VIX3M")
    vix6m = _close("^VIX6M")
    vvix = _close("^VVIX")
    skew = _close("^SKEW")
    ovx = _close("^OVX")
    gvz = _close("^GVZ")

    if vix is not None:
        out["vix"] = vix
        out["vix_logret_1d"] = np.log(vix).diff(1)
        out["vix_logret_5d"] = np.log(vix).diff(5)
        out["vix_z60"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std().replace(0, np.nan)

    if vix is not None and vix9d is not None:
        out["vix9d_over_vix"] = vix9d / vix  # < 1 means contango (calm), > 1 backwardation (stress)
    if vix is not None and vix3m is not None:
        out["vix_over_vix3m"] = vix / vix3m
    if vix is not None and vix6m is not None:
        out["vix_over_vix6m"] = vix / vix6m
    if vix is not None and vvix is not None:
        out["vvix_over_vix"] = vvix / vix
    if skew is not None:
        out["skew"] = skew
        out["skew_change_5d"] = skew.diff(5)
    if ovx is not None and vix is not None:
        out["ovx_over_vix"] = ovx / vix
    if gvz is not None and vix is not None:
        out["gvz_over_vix"] = gvz / vix

    # Variance risk premium proxy: VIX^2 (annualised) − realised QQQ var (annualised)
    qqq = _close("QQQ")
    if qqq is not None and vix is not None:
        rv = (np.log(qqq).diff(1) ** 2).rolling(20).mean() * 252  # annualised realised var
        iv = (vix / 100) ** 2
        out["vrp_proxy"] = iv - rv  # Bollerslev/Tauchen/Zhou 2009

    return pd.DataFrame(out)


# =============================================================================
# 4.  Yield curve, credit, fixed income
# =============================================================================

def _yields_credit_features(downloaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Estrella & Mishkin 1998 — yield-curve slope predicts recessions.

    Welch & Goyal 2008 — term spread + default spread are equity premium
    predictors. Adrian, Crump & Moench 2013 — term-premium decomposition.
    """
    out: Dict[str, pd.Series] = {}
    def c(t: str): return downloaded[t]["Close"] if t in downloaded else None

    tnx = c("^TNX")  # 10y
    fvx = c("^FVX")  # 5y
    tyx = c("^TYX")  # 30y
    irx = c("^IRX")  # 13w (3m proxy)

    if tnx is not None:
        out["yld_10y"] = tnx
        out["yld_10y_5d"] = tnx.diff(5)
        out["yld_10y_20d"] = tnx.diff(20)
    if fvx is not None: out["yld_5y"] = fvx
    if tyx is not None: out["yld_30y"] = tyx
    if irx is not None: out["yld_3m"] = irx

    if tnx is not None and irx is not None:
        out["term_10y_3m"] = tnx - irx  # recession indicator
    if tnx is not None and fvx is not None:
        out["term_10y_5y"] = tnx - fvx
    if tyx is not None and tnx is not None:
        out["term_30y_10y"] = tyx - tnx

    # Bond-ETF returns (smoothed yield-change proxies)
    for tic, name in [("TLT", "tlt"), ("IEF", "ief"), ("SHY", "shy"), ("TIP", "tip")]:
        if tic in downloaded:
            close = downloaded[tic]["Close"]
            out[f"{name}_logret_5d"] = np.log(close).diff(5)
            out[f"{name}_logret_20d"] = np.log(close).diff(20)

    # Credit risk appetite: HYG / LQD ratio (HY vs IG)
    if "HYG" in downloaded and "LQD" in downloaded:
        ratio = downloaded["HYG"]["Close"] / downloaded["LQD"]["Close"]
        out["hyg_over_lqd"] = ratio
        out["hyg_over_lqd_5d"] = np.log(ratio).diff(5)

    # 10y real-yield proxy: TIP price — when TIP rises, real yields fall
    if "TIP" in downloaded:
        out["tip_logret_20d"] = np.log(downloaded["TIP"]["Close"]).diff(20)

    return pd.DataFrame(out)


# =============================================================================
# 5.  Macro / commodities / FX
# =============================================================================

def _macro_fx_features(downloaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Driesprong, Jacobsen & Maat 2008 — oil predicts stocks.
    Akram 2009 — commodity prices and US monetary policy.
    """
    out: Dict[str, pd.Series] = {}
    pairs = [
        ("DX-Y.NYB", "dxy"),
        ("GC=F", "gold"),
        ("SI=F", "silver"),
        ("CL=F", "wti"),
        ("BZ=F", "brent"),
        ("HG=F", "copper"),
        ("EURUSD=X", "eurusd"),
        ("USDJPY=X", "usdjpy"),
    ]
    for tic, name in pairs:
        if tic not in downloaded: continue
        close = downloaded[tic]["Close"]
        log_c = np.log(close)
        out[f"{name}_logret_1d"] = log_c.diff(1)
        out[f"{name}_logret_5d"] = log_c.diff(5)
        out[f"{name}_logret_20d"] = log_c.diff(20)

    # Cross ratios useful for risk-on / risk-off
    if "GC=F" in downloaded and "QQQ" in downloaded:
        out["gold_over_qqq"] = downloaded["GC=F"]["Close"] / downloaded["QQQ"]["Close"]
    if "HG=F" in downloaded and "GC=F" in downloaded:
        out["copper_gold_ratio"] = downloaded["HG=F"]["Close"] / downloaded["GC=F"]["Close"]

    return pd.DataFrame(out)


# =============================================================================
# 6.  Cross-sectional / breadth — sectors, benchmarks, dispersion
# =============================================================================

def _cross_sectional_features(downloaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Brown & Cliff 2004 — sector rotation predicts index returns.

    Adds: per-sector 5d returns, sector return dispersion (standard deviation
    across sector ETFs), breadth (# sectors with positive 20d return),
    QQQ-vs-SPY relative strength (small/large + tech-tilt).
    """
    out: Dict[str, pd.Series] = {}

    # Benchmark relative strength
    benchmark_pairs = [("SPY", "spy"), ("DIA", "dia"), ("IWM", "iwm"), ("EFA", "efa"), ("EEM", "eem")]
    qqq = downloaded.get("QQQ")
    for tic, name in benchmark_pairs:
        if tic not in downloaded: continue
        c = downloaded[tic]["Close"]
        out[f"{name}_logret_5d"] = np.log(c).diff(5)
        if qqq is not None:
            out[f"qqq_over_{name}"] = qqq["Close"] / c
            out[f"qqq_over_{name}_5d"] = (np.log(qqq["Close"]) - np.log(c)).diff(5)

    # Sectors — collect 5d returns into a matrix for dispersion / breadth
    sector_tickers = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC"]
    sector_5d: Dict[str, pd.Series] = {}
    sector_20d: Dict[str, pd.Series] = {}
    for tic in sector_tickers:
        if tic not in downloaded: continue
        c = downloaded[tic]["Close"]
        ret5 = np.log(c).diff(5)
        ret20 = np.log(c).diff(20)
        out[f"sec_{tic.lower()}_logret_5d"] = ret5
        sector_5d[tic] = ret5
        sector_20d[tic] = ret20

    if sector_5d:
        sec_df_5d = pd.DataFrame(sector_5d)
        sec_df_20d = pd.DataFrame(sector_20d)
        out["sector_dispersion_5d"] = sec_df_5d.std(axis=1)
        out["sector_dispersion_20d"] = sec_df_20d.std(axis=1)
        out["sector_breadth_20d"] = (sec_df_20d > 0).sum(axis=1) / sec_df_20d.shape[1]
        # Cyclicals minus defensives spread
        if "XLK" in sector_5d and "XLP" in sector_5d:
            out["xlk_minus_xlp_5d"] = sector_5d["XLK"] - sector_5d["XLP"]
        if "XLY" in sector_5d and "XLP" in sector_5d:
            out["xly_minus_xlp_5d"] = sector_5d["XLY"] - sector_5d["XLP"]

    # International risk pulse
    for tic, name in [("^N225", "n225"), ("^FTSE", "ftse"), ("^GDAXI", "dax"), ("^HSI", "hsi")]:
        if tic in downloaded:
            c = downloaded[tic]["Close"]
            out[f"{name}_logret_5d"] = np.log(c).diff(5)

    return pd.DataFrame(out)


# =============================================================================
# 7.  Calendar / seasonality (cited)
# =============================================================================

# Approximate FOMC meeting weeks 2003-2025 (one per ~6 weeks). Calendar source:
# https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm. Coarse but
# adequate — features mark the *week containing* a meeting.
FOMC_DATES_2003_2025: list[str] = [
    # 2003
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12",
    "2003-09-16", "2003-10-28", "2003-12-09",
    # 2004
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10",
    "2004-09-21", "2004-11-10", "2004-12-14",
    # 2005
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09",
    "2005-09-20", "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08",
    "2006-09-20", "2006-10-25", "2006-12-12",
    # 2007
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07",
    "2007-09-18", "2007-10-31", "2007-12-11",
    # 2008
    "2008-01-22", "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25",
    "2008-08-05", "2008-09-16", "2008-10-08", "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12",
    "2009-09-23", "2009-11-04", "2009-12-16",
    # 2010
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10",
    "2010-09-21", "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09",
    "2011-09-21", "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01",
    "2012-09-13", "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31",
    "2013-09-18", "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30",
    "2014-09-17", "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29",
    "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
]


def _calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar / seasonality features.

    Day-of-week effect: French 1980, Cross 1973. Turn-of-the-month effect:
    Ariel 1987. January effect: Rozeff & Kinney 1976. FOMC drift:
    Lucca & Moench 2015 JF — equity returns concentrated around FOMC.
    Options expiration: Stivers & Sun 2002. Santa rally / December effect:
    Haug & Hirschey 2006.
    """
    df = pd.DataFrame(index=idx)
    df["dow_mon"] = (idx.dayofweek == 0).astype(int)
    df["dow_tue"] = (idx.dayofweek == 1).astype(int)
    df["dow_wed"] = (idx.dayofweek == 2).astype(int)
    df["dow_thu"] = (idx.dayofweek == 3).astype(int)
    df["dow_fri"] = (idx.dayofweek == 4).astype(int)

    df["month"] = idx.month
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    df["jan_effect"] = (idx.month == 1).astype(int)
    df["dec_effect"] = (idx.month == 12).astype(int)

    df["turn_of_month"] = ((idx.day <= 3) | (idx.day >= 28)).astype(int)
    df["santa_rally"] = ((idx.month == 12) & (idx.day >= 24)).astype(int)

    # FOMC week dummy (week containing a FOMC meeting decision day)
    fomc = pd.to_datetime(FOMC_DATES_2003_2025)
    fomc_weeks = pd.PeriodIndex(fomc, freq="W")
    idx_weeks = pd.PeriodIndex(idx, freq="W")
    df["fomc_week"] = idx_weeks.isin(fomc_weeks).astype(int)
    # FOMC day +/- 1 dummy
    nearest = pd.DatetimeIndex(fomc)
    fomc_set = set(nearest.normalize())
    fomc_day_arr = np.array([d.normalize() in fomc_set for d in idx])
    df["fomc_day"] = fomc_day_arr.astype(int)

    # Options expiration week — third Friday of the month
    third_fri = (idx.dayofweek == 4) & (idx.day >= 15) & (idx.day <= 21)
    df["opex_friday"] = third_fri.astype(int)
    df["opex_week"] = (
        pd.Series(third_fri, index=idx)
        .rolling(5, min_periods=1).max()
        .astype(int)
        .values
    )

    # Earnings season: weeks 2-5 of Jan/Apr/Jul/Oct (rough)
    df["earnings_season"] = (
        idx.month.isin([1, 4, 7, 10]) & (idx.day >= 8) & (idx.day <= 35)
    ).astype(int)

    return df


# =============================================================================
# 8.  Variance ratio + lagged-target autoregressive features
# =============================================================================

def _autoregressive_features(qqq_close: pd.Series) -> pd.DataFrame:
    """Lo & MacKinlay 1988 — variance ratio test of random walk.

    Lagged returns (Conrad & Kaul 1988) — autoregressive predictability.
    """
    out: Dict[str, pd.Series] = {}
    log_c = np.log(qqq_close)
    r1 = log_c.diff(1)
    r5 = log_c.diff(5)
    r20 = log_c.diff(20)

    out["lag_ret_1d"] = r1.shift(1)  # yesterday's return
    out["lag_ret_2d"] = r1.shift(2)
    out["lag_ret_5d"] = r5.shift(1)

    # Variance ratios — VR(q) = Var(r_q)/(q*Var(r_1)) ; tests random walk.
    var1 = r1.rolling(252, min_periods=126).var()
    var5 = r5.rolling(252, min_periods=126).var()
    var20 = r20.rolling(252, min_periods=126).var()
    out["var_ratio_5_1"] = var5 / (5 * var1.replace(0, np.nan))
    out["var_ratio_20_1"] = var20 / (20 * var1.replace(0, np.nan))

    # Drawdown depth
    rolling_max = qqq_close.rolling(252, min_periods=20).max()
    out["dd_from_252max"] = qqq_close / rolling_max - 1.0

    return pd.DataFrame(out)


# =============================================================================
# 9.  Public API
# =============================================================================

def compute_qqq_features(downloaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the full ~120-feature matrix for QQQ.

    Parameters
    ----------
    downloaded : output of ``download.download_all()`` — dict of OHLCV frames
        keyed by ticker. Missing tickers are tolerated; their features are
        simply absent from the output.

    Returns
    -------
    DataFrame indexed by date (NYSE business days), warmup rows dropped,
    NaN rows dropped, all features finite.
    """
    if "QQQ" not in downloaded:
        raise ValueError("QQQ ticker missing — cannot build features.")

    qqq = downloaded["QQQ"]
    primary_feats = _ohlcv_features(qqq, "qqq", full=True)
    primary_feats = primary_feats.replace([np.inf, -np.inf], np.nan)

    # Sector ETFs — compact feature set per sector
    sector_blocks: list[pd.DataFrame] = []
    for tic in ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC"]:
        if tic in downloaded:
            sector_blocks.append(_ohlcv_features(downloaded[tic], tic.lower(), full=False))

    vol_block = _vol_regime_features(downloaded)
    yc_block = _yields_credit_features(downloaded)
    macro_block = _macro_fx_features(downloaded)
    cross_block = _cross_sectional_features(downloaded)
    auto_block = _autoregressive_features(qqq["Close"])

    blocks = [primary_feats] + sector_blocks + [
        vol_block, yc_block, macro_block, cross_block, auto_block,
    ]
    blocks = [b for b in blocks if b is not None and not b.empty]

    combined = pd.concat(blocks, axis=1, join="outer")
    combined = combined.replace([np.inf, -np.inf], np.nan)

    # Calendar features depend only on the index — compute after concat
    cal = _calendar_features(combined.index)
    combined = pd.concat([combined, cal], axis=1)

    # Forward-fill then back-fill macro signals across non-trading-day gaps
    # (e.g. ^TNX has slightly different holiday calendar). Limit to 5d so we
    # never create stale data.
    combined = combined.ffill(limit=5).bfill(limit=2)

    # Drop COLUMNS whose underlying signal starts after 2006-12-31 — these
    # would otherwise force the early rows out via dropna(). The 2007 cutoff
    # preserves the GFC-onset regime (fold 1) for training. Casualties of
    # this filter: XLRE (Oct 2015), XLC (Jun 2018), ^VIX9D (~2011), ^VVIX
    # (~2007 borderline), ^OVX / ^GVZ (2007/2008) — all dropped. We still
    # keep ^VIX, ^SKEW, sector ETFs (1998), bond ETFs (2002).
    cutoff = pd.Timestamp("2007-01-01")
    drop_late: list[str] = []
    for c in combined.columns:
        first_valid = combined[c].first_valid_index()
        if first_valid is None or first_valid > cutoff:
            drop_late.append(c)
    if drop_late:
        logger.info("[features] dropping %d post-2009 columns: %s",
                    len(drop_late),
                    ", ".join(drop_late[:10]) + (" ..." if len(drop_late) > 10 else ""))
        combined = combined.drop(columns=drop_late)

    # Drop warmup + remaining NaN rows
    combined = combined.iloc[WARMUP:]
    combined = combined.dropna()

    logger.info("[features] %d rows × %d features (after late-column drop)",
                *combined.shape)
    return combined


def compute_qqq_targets(downloaded: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute the four target variants A / B / C / D from CLAUDE.md.

    A — fwd_ret_1d  : 1-day forward log-return (PRIMARY optimisation target)
    B — fwd_ret_5d  : 5-day forward log-return (auxiliary head)
    C — fwd_dual    : sign-aligned dual-head label (1 if both 1d & 5d > 0)
    D — fwd_voladj  : (1d return) / (rolling realised vol over 20d) — vol-adjusted
    """
    if "QQQ" not in downloaded:
        raise ValueError("QQQ missing")
    close = downloaded["QQQ"]["Close"]
    log_c = np.log(close)
    r1 = log_c.diff(1).shift(-1)
    r5 = log_c.diff(5).shift(-5)
    rv20 = (log_c.diff(1) ** 2).rolling(20, min_periods=20).mean().pow(0.5)
    rv20 = rv20.replace(0, np.nan)
    targets = pd.DataFrame({
        "fwd_ret_1d":     r1,
        "fwd_ret_5d":     r5,
        # C: sign agreement — used only as a side-channel metric.
        "fwd_sign_concordance": ((np.sign(r1) == np.sign(r5)).astype(float)),
        # D: vol-adjusted 1d (orthogonalises the trend bias from skill).
        "fwd_voladj_ret_1d": r1 / rv20,
    }, index=close.index)
    return targets.dropna()
