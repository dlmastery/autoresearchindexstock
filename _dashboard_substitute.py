"""Substitute SPY -> QQQ display strings in the freshly-cloned dashboard."""
from pathlib import Path
import re

P = Path(r"C:\Users\evija\autoresearchqqq_local\docs\dashboard\index.html")
src = P.read_text(encoding="utf-8")

# Title + build stamp
src = src.replace(
    "<title>AutoResearch SPY Dashboard (build 20260506-080000 + hit-card-loading-fix)</title>",
    "<title>AutoResearch QQQ Dashboard (build 20260508-013000 + full-parity-port)</title>",
)
src = src.replace("AutoResearch &mdash; SPY (S&P 500 ETF) Dashboard",
                  "AutoResearch &mdash; QQQ (Nasdaq-100 ETF) Dashboard")
src = src.replace("SPY (S&P 500 ETF) — successor", "QQQ (Nasdaq-100 ETF) — successor")

# Display-text "SPY" replacements (avoid breaking JSON keys, JS field names, Python paths).
# We're surgical: only replace word-boundary "SPY" outside JS variable names like _spy or "spy_prices".
# Strategy: use regex to replace whole-word SPY in user-facing text only — JSON/JS field names use lowercase.
src = re.sub(r"\bSPY\b", "QQQ", src)

# Repair JSON key lookups (these were lowercase 'spy' so untouched by \bSPY\b regex):
# (no-op — left as fallback default text)

# Adjust window-size labels (SPY uses 1410-day in-sample; QQQ uses similar but exact value
# comes from data so wording-only):
src = src.replace("1410-day in-sample", "in-sample test fold")
src = re.sub(r"\b1410d\b", "test-fold", src)
src = re.sub(r"\bDec25-Apr26\b", "live OOS", src)
src = re.sub(r"\b\(103d\b", "(OOS", src)
src = src.replace("103d", "OOS")
src = src.replace("(2008-2025, ~17 years, ~5-10%/yr)", "(in-sample, multi-year)")
src = src.replace("(~17 years, ~5-10%/yr)", "(in-sample, multi-year)")
src = src.replace("103-day forward window (Dec 2025-Apr 2026, 5 months)",
                  "OOS forward window (~5 months)")
src = src.replace("Dec 2025-Apr 2026", "OOS forward")
src = src.replace("Dec25-Apr26", "OOS")

# yfinance fallback display string
src = src.replace("QQQ+VIX yfinance", "QQQ+VXN yfinance")
src = src.replace("real SPY+VIX", "real QQQ+VXN")
src = src.replace("real_data_sources?.spy_prices", "real_data_sources?.qqq_prices")

# Reference to FX project — keep
# Reference to SPY 7-Fold Super-Fold becomes QQQ 7-Fold Super-Fold
src = src.replace("Fold Reference (QQQ 7-Fold Super-Fold + Prod-Mode Split)",
                  "Fold Reference (QQQ 7-Fold Super-Fold + Prod-Mode Split)")

# OOS-specific banner messaging (was "OOS sections below are populated only when SPY OOS inference has...")
src = src.replace("QQQ out-of-sample inference panels will a",
                  "QQQ out-of-sample inference panels will a")  # already replaced via \bSPY\b

P.write_text(src, encoding="utf-8")
print("Substitutions written.")
print("Remaining SPY tokens (should be 0):", len(re.findall(r"\bSPY\b", src)))
print("QQQ tokens:", len(re.findall(r"\bQQQ\b", src)))
print("File size: %.1f KB" % (P.stat().st_size / 1024))
