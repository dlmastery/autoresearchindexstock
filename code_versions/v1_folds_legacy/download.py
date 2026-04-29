"""Equity-index data download for QQQ and supporting signals.

Downloads OHLCV via yfinance, caches as parquet. Hard-caps at 2025-12-31
because CLAUDE.md forbids any 2026 data — verified at the end of each fetch.

Tickers grouped by purpose so a future contributor can see the design rather
than guess. Each group has a one-line citation pointing to the literature
that motivated its inclusion.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------

# The asset we are forecasting.
PRIMARY: Dict[str, str] = {
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
}

# Cross-asset benchmarks for relative-strength + breadth features.
#   Lo & MacKinlay 1990 — relative-strength reversal between indices.
BENCHMARKS: Dict[str, str] = {
    "SPY":  "SPDR S&P 500",
    "DIA":  "SPDR Dow Jones",
    "IWM":  "iShares Russell 2000",
    "EFA":  "iShares MSCI EAFE (international DM)",
    "EEM":  "iShares MSCI Emerging Markets",
}

# Sector ETFs — used for breadth + dispersion features.
#   Brown & Cliff 2004 — sector rotation predicts index returns.
SECTORS: Dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLI":  "Industrials",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}

# Volatility-regime signals.
#   Bollerslev, Tauchen & Zhou 2009 RFS — variance risk premium predicts
#   equity returns. Pan & Poteshman 2006 — option-implied signals predict
#   future returns. Whaley 2009 — VIX as an "investor fear gauge".
VOL_REGIME: Dict[str, str] = {
    "^VIX":  "CBOE VIX (30-day implied)",
    "^VIX9D": "CBOE VIX 9-day",
    "^VIX3M": "CBOE 3-month VIX",
    "^VIX6M": "CBOE 6-month VIX",
    "^VVIX": "CBOE Vol-of-Vol",
    "^SKEW": "CBOE SKEW (tail risk)",
    "^OVX":  "CBOE Crude Oil ETF VIX",
    "^GVZ":  "CBOE Gold ETF VIX",
}

# Fixed income / credit / yield curve.
#   Estrella & Mishkin 1998 RES — 10y-3m spread predicts recessions.
#   Welch & Goyal 2008 RFS — term spread + default spread are equity premium
#   predictors. Adrian, Crump & Moench 2013 — term premium decomposition.
YIELDS_CREDIT: Dict[str, str] = {
    "^TNX":  "US 10Y Treasury Yield",
    "^FVX":  "US 5Y Treasury Yield",
    "^TYX":  "US 30Y Treasury Yield",
    "^IRX":  "US 13W Treasury Yield (proxy for 3m)",
    "TLT":   "iShares 20+Y Treasury Bond ETF",
    "IEF":   "iShares 7-10Y Treasury Bond ETF",
    "SHY":   "iShares 1-3Y Treasury Bond ETF",
    "TIP":   "iShares TIPS Bond ETF (10y real proxy)",
    "HYG":   "iShares iBoxx HY Corporate (credit risk appetite)",
    "LQD":   "iShares iBoxx IG Corporate",
}

# Macro / commodities / currency.
#   Driesprong, Jacobsen & Maat 2008 — oil predicts equity returns.
#   Akram 2009 — commodity prices and US monetary policy joint dynamics.
MACRO_FX: Dict[str, str] = {
    "GC=F":     "Gold Futures",
    "SI=F":     "Silver Futures",
    "CL=F":     "WTI Crude Oil Futures",
    "BZ=F":     "Brent Crude Futures",
    "HG=F":     "Copper Futures (Dr. Copper)",
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
}

# International risk pulse.
INTL_RISK: Dict[str, str] = {
    "^N225":  "Nikkei 225 (Asia)",
    "^FTSE":  "FTSE 100 (UK)",
    "^GDAXI": "DAX (Germany)",
    "^HSI":   "Hang Seng (HK)",
}

# Combine. Order matters only for log readability.
ALL_SIGNALS: Dict[str, Dict[str, str]] = {
    "primary":      PRIMARY,
    "benchmarks":   BENCHMARKS,
    "sectors":      SECTORS,
    "vol_regime":   VOL_REGIME,
    "yields_credit": YIELDS_CREDIT,
    "macro_fx":     MACRO_FX,
    "intl_risk":    INTL_RISK,
}


# ---------------------------------------------------------------------------
# Date constants — NO 2026 DATA.
# ---------------------------------------------------------------------------

DEFAULT_START = "2004-01-01"
# Hard cap. Anything past this is dropped post-fetch — see _enforce_no_2026.
DEFAULT_END = "2025-12-31"
HARD_CUTOFF = pd.Timestamp("2026-01-01")


DEFAULT_CACHE_DIR = str(
    Path(__file__).resolve().parent.parent / ".data_cache_qqq"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, start: str, end: str, cache_dir: str) -> Path:
    safe = (
        ticker.replace("=", "_").replace("^", "_").replace(".", "_").replace("-", "_")
    )
    return Path(cache_dir) / f"{safe}_{start}_{end}.parquet"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def _enforce_no_2026(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop any rows >= 2026-01-01 with a logged warning. CLAUDE.md mandate."""
    if df is None or df.empty:
        return df
    n_before = len(df)
    df = df.loc[df.index < HARD_CUTOFF]
    n_after = len(df)
    if n_after < n_before:
        logger.info(
            "[no-2026] %s: dropped %d rows >= 2026-01-01",
            ticker, n_before - n_after,
        )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_ticker(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download a single ticker, cache as parquet, drop any 2026 rows."""
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cp = _cache_path(ticker, start, end, cache_dir)
        if cp.exists():
            df = pd.read_parquet(cp)
            return _enforce_no_2026(df, ticker)

    raw = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False,
        threads=False,
    )
    raw = _flatten_columns(raw)
    raw = _enforce_no_2026(raw, ticker)

    if cache_dir is not None and not raw.empty:
        cp = _cache_path(ticker, start, end, cache_dir)
        raw.to_parquet(cp)
    return raw


def download_all(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> Dict[str, pd.DataFrame]:
    """Download every ticker in ALL_SIGNALS, return dict ticker -> DataFrame.

    Tickers that fail (delisted, network error, no data) get a warning and
    are skipped — features.py is defensive about missing tickers.
    """
    out: Dict[str, pd.DataFrame] = {}
    for group_name, group in ALL_SIGNALS.items():
        for ticker in group:
            try:
                df = download_ticker(ticker, start=start, end=end, cache_dir=cache_dir)
            except Exception as e:  # pragma: no cover — yf network flake
                logger.warning("Skip %s (%s): %s", ticker, group_name, e)
                continue
            if df is None or df.empty:
                logger.warning("Empty %s (%s)", ticker, group_name)
                continue
            out[ticker] = df
    logger.info("[download] fetched %d / %d tickers",
                len(out),
                sum(len(g) for g in ALL_SIGNALS.values()))
    return out
