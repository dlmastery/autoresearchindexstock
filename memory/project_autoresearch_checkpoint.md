---
name: AutoResearch QQQ Checkpoint
description: Session 2026-05-09 final. 259 experiments, 8 archived winners. Full SPY-parity dashboard ported (deep ensemble, smart strategies, OOS top-30, metrics glossary, hit-rate champion). 3 of 30 OOS top-30 completed (exp52/48/17 — only archived). 392 ensembles + 686 smart strategies built with 16-col CSVs. Hit rate is now SOLE winner metric (Directive 65). Build stamp 20260509-090000.
---

# QQQ AutoResearch — Session Checkpoint (2026-05-09 final)

> **READ THIS FIRST on session resume.** Self-contained for QQQ resumption.

---

## Repo / live-dashboard pointers

| Item | Value |
|---|---|
| Local working dir | `C:\Users\evija\autoresearchqqq_local\` |
| GitHub repo | https://github.com/dlmastery/autoresearchindexstock |
| Branch | `master` |
| Latest commit (2026-05-09) | `cec2785` (mojibake repair) |
| Live Pages dashboard | https://dlmastery.github.io/autoresearchindexstock/dashboard/ |
| Local dashboard server | `python -m http.server 8889 --directory docs/dashboard` |
| Snapshot zip (this session) | `C:\Users\evija\autoresearch_qqq_snapshot_20260509_011440.zip` (63.2 MB) |
| Companion repo (SPY) | https://github.com/dlmastery/autoresearchspy @ `2a8a188` |

---

## State summary (this session 2026-05-08 → 2026-05-09)

| Counter | Value |
|---|---|
| Total experiments in log | 259 |
| Auto-archived checkpoints in `winners/` | 8 |
| OOS Top-30 completed rows | **3 of 30** (exp52, exp48, exp17 — only 3 archived match top-30) |
| Smart strategies built | **686** (with full 16-col CSVs) |
| Ensemble strategies built | **392** (with full 16-col CSVs) |

### Champion (DEPLOYABLE)

| Metric | Value |
|---|---|
| **Hit-rate champion (Directive 65 — sole winner metric)** | DLinear / mamba family — see `experiment_log.jsonl[hit].max()` |
| In-sample composite-leader (legacy secondary) | mamba dmamba exp 52 composite +1.32 (single-seed lucky basin) |
| OOS hit-rate champion (98 trade days) | exp52 / exp48 mamba at **55.1%** hit each |
| Best smart strategy ($) | `*__protective_put` cluster at **$1423** Sharpe +8.14 hit 55.1% |
| Top ensemble | `all3_*__sma200filter` $1107 Sharpe +7.84 hit 67.9% |

---

## Dashboard ports completed this session

The QQQ dashboard was full-parity ported from SPY (was 57 KB → 173 KB). It now has ALL the SPY panels:

- 🎯 **Hit Rate Champion ★** prominent gold cards (in-sample + OOS)
- 🛰️ OOS Live-Data Inference panel
- 📋 OOS Top-30 — Per-Winner Inference Results (3 completed)
- 🎯 OOS Deep Ensemble — Lakshminarayanan 2017 (392 strategies)
- 💡 Smart Trading Strategies — Overlays & Options-Stock Hedging (686)
- 📖 Metrics Glossary (10x expanded)
- Trading Strategies docs (8 overlays + 6 hedging w/ Black-Scholes)
- 📁 Fold Reference (7-fold super-fold + prod-mode split)

Every clickable row produces an inline detail card with **3-line equity chart**:
- Strategy (purple/green) + Buy & Hold (grey) + Ensemble champion (gold)

---

## CLAUDE.md Directives 64-67 (QQQ-mirrored from SPY)

| # | Rule |
|---|---|
| **64** | Per-strategy CSVs MUST carry full 16-col schema |
| **65** | HIT RATE is the SOLE winner-identification metric |
| **66** | Per-experiment OOS CSVs (`oos_exp<N>.csv`) MUST carry the 16-col schema; row dicts MUST have `equity_curve` with both `_dollars` and `_pct` arrays |
| **67** | Every clickable row → inline detail card with 3-line equity chart |

### Mandatory 16-column CSV schema

```
date, position, pred_direction, traded, actual_ret_1d, bh_log_ret,
strategy_pnl, correct, equity_dollars, buy_hold_dollars, excess_dollars,
cumret_pct, bh_cumret_pct, excess_cumret_pct, drawdown_pct, underwater
```

---

## QQQ-specific scripts (this session)

- `run_oos_top30_qqq.py` — OOS Top-30 inference. Adapted from SPY. Uses 2-yr download (2024-01-01 → 2026-04-30) for SMA-200 warmup. Auto-detects mamba `d_state`/`expand`/`variant` from state_dict shapes (handles dmamba `trend_mlp.*` layers).
- `_build_qqq_ensemble_summary.py` — adapted from SPY ensemble builder
- `_build_qqq_smart_strategies.py` — adapted from SPY smart builder. Uses real **QQQ + ^VXN** (Nasdaq vol index, not VIX) as IV proxy for Black-Scholes options pricing.

Bridge to SPY package: `sys.path.insert(0, r'C:\Users\evija\autoresearchindexspy\autoresearchspy')` makes `autoresearchspy.*` (model.backbone, _pin_to_safe_cores) importable for QQQ scripts.

---

## Pending work (next session)

1. **Backfill retrain 27 missing OOS Top-30 checkpoints** — most QQQ winners weren't archived under the always-archive Directive 62 (which was added in SPY this session). Same patch needs porting to QQQ's `run_autoresearch.py`.
2. **Multi-seed verification** of any QQQ hit-rate champion (binomial std error at n~100 is ~5pp).
3. **DSR correction** for the 686 smart strategies.

---

## How to resume

```bash
cd C:/Users/evija/autoresearchqqq_local
git pull origin master
# Read this checkpoint + CLAUDE.md
python run_oos_top30_qqq.py    # if want to re-run OOS
python -m http.server 8889 --directory docs/dashboard
```

---

## Mojibake gotcha (also affected QQQ on 2026-05-09)

**NEVER edit `index.html` using PowerShell `Get-Content -Raw` + `Set-Content -Encoding UTF8`** — it silently corrupts multi-byte UTF-8 chars. Use Python or `[System.IO.File]::WriteAllText` with `UTF8Encoding(false)` instead. If corruption happens, run `C:\Users\evija\_fix_mojibake_v2.py` then `C:\Users\evija\_fix_remaining3.py` to repair.
