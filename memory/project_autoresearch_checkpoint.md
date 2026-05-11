---
name: AutoResearch QQQ Checkpoint
description: Session 2026-05-10 final. 259 experiments, 8 archived winners. Dashboard at http://localhost:8889 with 347 ensemble + 686 smart strategies, ALL CAUSAL/DEPLOYABLE (0 leaky per Directive 70). NEW Directive 72: 15 train-fit strategies (parameters/weights LEARNED on in-sample test fold, locked, evaluated OOS). Directive 71: full narrative panel + transparency block on every clickable row. Best train-fit: train_optim_kelly_frac_exp52__0.10 hit 55.1% / Sh +0.84 (98 trade days).
---

# QQQ AutoResearch — Session Checkpoint (2026-05-10 final)

> Self-contained QQQ resumption — fresh session reads only this + CLAUDE.md.

---

## Repo / live-dashboard pointers

| Item | Value |
|---|---|
| Local working dir | `C:\Users\evija\autoresearchqqq_local\` |
| GitHub repo | https://github.com/dlmastery/autoresearchindexstock (branch `master`) |
| Latest commit (2026-05-10) | `bd5e883` (Directive 72 train-fit strategies) |
| Live Pages dashboard | https://dlmastery.github.io/autoresearchindexstock/dashboard/ |
| Local server | `python -m http.server 8889 --directory docs/dashboard` |
| Snapshot zip (latest) | `C:\Users\evija\autoresearch_qqq_snapshot_20260510_112616.zip` (62.8 MB / 3203 files) |
| Companion repo (SPY) | https://github.com/dlmastery/autoresearchspy @ `58ce0d1` |

---

## Champion roster (all CAUSAL — Directive 70+72)

| Metric | Strategy | Value |
|---|---|---|
| **OOS hit-rate champion** (Directive 65) | exp52 / exp48 mamba | **55.1%** hit (98 trade days each) |
| Best train-fit by hit | `train_optim_kelly_frac_exp52__0.10` | 55.1% hit / Sh +0.84 / $1000 |
| Best train-fit by Sharpe | `train_optim_kelly_frac_exp52__0.10` (tie cluster) | Sh +0.84 |
| Best smart strategy ($) | `*__protective_put` cluster | $1423 / Sh +8.14 |
| Best ensemble | `all3_*__sma200filter` | $1107 / Sh +7.84 / hit 67.9% |

---

## Counts (after Directive 72, 2026-05-10)

| Panel | Total | Train-fit | Leaky |
|---|---|---|---|
| Ensemble strategies | **347** | 15 (`train_optim_*` 9, `meta_*` 3, `calib_*` 3) | **0** |
| Smart strategies | 686 | 0 (next session) | **0** |
| OOS Top-30 rows | 3 of 30 completed | — | — |
| Auto-archived checkpoints | 8 | — | — |
| Total experiments in JSONL | 259 | — | — |

---

## CLAUDE.md Directives 64-72 (binding rules — mirrored from SPY)

| # | Date | Rule (one-line) |
|---|---|---|
| **64** | 2026-05-06 | Per-strategy CSVs MUST carry **16-col schema** |
| **65** | 2026-05-06 | **HIT RATE is the SOLE winner-identification metric** |
| **66** | 2026-05-08 | Per-experiment OOS CSVs same 16-col schema; row dicts MUST have `equity_curve` with `_dollars` AND `_pct` arrays |
| **67** | 2026-05-09 | Every clickable row → inline detail card with 3-line equity chart |
| **68** | 2026-05-09 | Look-ahead audit: leaky strategies tagged with red ⚠️ badge |
| **69** | 2026-05-09 | Causal-only ensemble selection: `by_train_*` family added |
| **70** | 2026-05-10 | **REMOVE all leaky strategies** — builders amputated |
| **71** | 2026-05-10 | Full transparency on every row click (`buildNarrativeHTML` + `buildTransparencyHTML`) |
| **72** | 2026-05-10 | TRAIN-FIT strategies — `train_optim_*` + `meta_*` + `calib_*` (parameters fit on in-sample, locked, evaluated OOS) |

---

## Files of record

| File | Purpose |
|---|---|
| `CLAUDE.md` | Full directive history (Directives 1-72) |
| `_build_qqq_ensemble_summary.py` | Ensemble builder — only `by_train_*` selection (D70) |
| `_build_qqq_smart_strategies.py` | Smart-strategy builder — only causal criteria + train-time enrichment |
| `_build_train_fit_strategies_qqq.py` | **NEW (D72)**: train-fit builder (sweeps + meta + calib) |
| `run_oos_top30_qqq.py` | OOS top-30 inference — emits 16-col CSV + equity_curve (D66); auto-detects mamba `d_state`/`expand`/`variant` from state_dict |
| `docs/dashboard/index.html` | Build stamp `20260510-070000`. Same JS contract as SPY |

---

## Pending work (next session)

1. Categories D + E from Directive 72 (regime detection k-means + walk-forward refit)
2. **Backfill retrain 27 missing OOS Top-30 checkpoints** — same Directive-62 always-archive patch needs porting to QQQ runner (currently runner only auto-archives composite-champions)
3. DSR multi-testing correction
4. Multi-seed verification of any QQQ hit-rate champion

---

## Resume command

```bash
cd C:/Users/evija/autoresearchqqq_local
git pull origin master
# Read this checkpoint + CLAUDE.md
python -m http.server 8889 --directory docs/dashboard
# Pick from Pending work above
```

---

## Mojibake gotcha

NEVER use PowerShell `Get-Content -Raw` + `Set-Content -Encoding UTF8` on `index.html`. Use Python's `Path.write_text(encoding='utf-8')` or the Edit tool. Repair scripts: `_fix_mojibake_v2.py` + `_fix_remaining3.py`.
