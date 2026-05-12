---
name: AutoResearch QQQ Checkpoint
description: Session 2026-05-11 final. 259 experiments, 8 archived winners. Dashboard at http://localhost:8889 with 363 ensemble + 686 smart strategies, ALL CAUSAL/DEPLOYABLE (0 leaky per Directives 70+74). 31 train-fit strategies (15 simple D72 + 16 advanced D74 incl. 5d-on-1d, stoploss+trailing, regime k-means, bandit Thompson, HRP, risk reversal, iron condor, calendar spread). Directive 73: transparency block ALWAYS renders on row click.
---

# QQQ AutoResearch — Session Checkpoint (2026-05-11 final)

> Self-contained QQQ resumption — fresh session reads only this + CLAUDE.md.

---

## Repo / live-dashboard pointers

| Item | Value |
|---|---|
| Local working dir | `C:\Users\evija\autoresearchqqq_local\` |
| GitHub repo | https://github.com/dlmastery/autoresearchindexstock (branch `master`) |
| Latest commit | `6dafdf8` (Directive 74 advanced train-fit) — about to push docs commit |
| Live Pages dashboard | https://dlmastery.github.io/autoresearchindexstock/dashboard/ |
| Local server | `python -m http.server 8889 --directory docs/dashboard` |
| Companion repo (SPY) | https://github.com/dlmastery/autoresearchspy @ `e18d879` |

---

## Champion roster (all CAUSAL — Directives 70+74)

| Metric | Strategy | Value |
|---|---|---|
| **OOS hit-rate champion** (D65) | exp52 / exp48 mamba | **55.1%** hit / 98 traded days |
| Best train-fit by hit (D72 simple) | `train_optim_kelly_frac_exp52__0.10` | 55.1% hit / Sh +0.84 |
| Best smart strategy ($) | `*__protective_put` cluster | $1423 / Sh +8.14 |
| Best ensemble | `all3_*__sma200filter` | $1107 / Sh +7.84 / hit 67.9% |
| New advanced (D74) | 8 categories × 3 members + 5 ensemble = 16 strategies | top performers in dashboard |

---

## Counts (after Directive 74)

| Panel | Total | Train-fit (D72 simple) | Train-fit (D74 advanced) | Leaky |
|---|---|---|---|---|
| Ensemble strategies | **363** | 15 | 16 | **0** |
| Smart strategies | 686 | 0 | 0 | **0** |
| OOS Top-30 rows | 3 of 30 completed | — | — | — |
| Auto-archived checkpoints | 8 | — | — | — |
| Total experiments in JSONL | 259 | — | — | — |

---

## CLAUDE.md Directives 64-74 (binding rules — full table)

| # | Date | Rule (one-line) |
|---|---|---|
| **64** | 2026-05-06 | Per-strategy CSVs MUST carry **16-col schema** |
| **65** | 2026-05-06 | **HIT RATE is the SOLE winner-identification metric** |
| **66** | 2026-05-08 | Per-experiment OOS CSVs same 16-col schema; row dicts MUST have `equity_curve` with `_dollars` AND `_pct` |
| **67** | 2026-05-09 | Every clickable row → inline detail card with 3-line equity chart |
| **68** | 2026-05-09 | Look-ahead audit: leaky strategies tagged with red ⚠️ badge |
| **69** | 2026-05-09 | Causal-only ensemble selection: `by_train_*` family added |
| **70** | 2026-05-10 | **REMOVE all leaky strategies** — builders amputated |
| **71** | 2026-05-10 | Full transparency on every row click (`buildNarrativeHTML` + `buildTransparencyHTML`) |
| **72** | 2026-05-10 | TRAIN-FIT strategies (simple): per-member sweeps + meta-learners + isotonic calibration |
| **73** | 2026-05-10 | Transparency must ALWAYS render — JS `_pct` fallback + transparency outside chart-conditional |
| **74** | 2026-05-10 | **8 advanced train-fit strategies**: 5d-on-1d, stoploss+trailing, regime k-means, bandit Thompson, HRP, risk reversal, iron condor, calendar spread. Forensic causality audit verified. |

---

## Files of record

| File | Purpose |
|---|---|
| `CLAUDE.md` | Full directive history (1-74) |
| `_build_qqq_ensemble_summary.py` | Ensemble builder. Now emits `_pct` (D73). |
| `_build_qqq_smart_strategies.py` | Smart-strategy builder |
| `_build_train_fit_strategies_qqq.py` | Train-fit builder (D72) |
| `_build_advanced_train_fit_strategies_qqq.py` | **NEW (D74)**: 8 advanced train-fit (with .shift(1) on spot+IV) |
| `run_oos_top30_qqq.py` | OOS top-30 inference (16-col + equity_curve, D66) |
| `docs/dashboard/index.html` | Build stamp `20260510-090000`. Same JS contract as SPY |

---

## Pending work (next session)

1. **Sector rotation** — needs sector ETF closes (yfinance fetch)
2. **Long-short pairs** — needs 2nd asset spec
3. **Walk-forward refit** (Cat E from D72)
4. **DSR multi-testing correction** for all 363 ensemble Sharpes
5. **Backfill retrain 27 missing OOS Top-30 checkpoints** — needs Directive-62 always-archive port to QQQ runner
6. **Multi-seed verification** of QQQ hit-rate champion

---

## Resume command

```bash
cd C:/Users/evija/autoresearchqqq_local
git pull origin master
python -m http.server 8889 --directory docs/dashboard
```

---

## Mojibake gotcha

NEVER use PowerShell `Get-Content -Raw` + `Set-Content -Encoding UTF8` on `index.html`. Use Python's `Path.write_text(encoding='utf-8')` or the Edit tool. Repair: `_fix_mojibake_v2.py` + `_fix_remaining3.py`.
