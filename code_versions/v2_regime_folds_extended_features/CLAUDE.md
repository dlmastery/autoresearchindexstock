# CLAUDE.md — AutoResearch Index Stock (QQQ)

> **This file inherits all rules from the parent project's `C:/Users/evija/autoresearch/CLAUDE.md`.**
> Below documents only what is *different* for the equity-index variant.
> Read the parent first, then read this.

## Project identity

- **Asset**: QQQ (Invesco QQQ Trust, Nasdaq-100 ETF)
- **Data window**: 2004-01-01 → **2025-12-31** (no 2026 data anywhere — hard cap, enforced in `download.py` and `splits.py`)
- **Goal**: meet or beat the FX project's mega-ensemble headline (Sharpe **+9.7071** on the FX dashboard) on a fair-comparison basis. Because QQQ trends, the *fair* comparison is **excess-Sharpe over a long-only buy-and-hold baseline**, tracked alongside raw Sharpe.
- **Optimization target**: **1-day forward log-return** (`ret_1d`). KEEP / DISCARD decisions are driven by the 1-day-return composite. We *additionally* compute, track, and plot four target variants on every experiment so we never lose visibility:
  - **A — `ret_1d`**: 1-day forward log-return (primary)
  - **B — `ret_5d`**: 5-day forward log-return (auxiliary head, trained jointly)
  - **C — dual-head joint**: combined 1d + 5d direction prediction; secondary metric
  - **D — vol-adjusted 1d**: `ret_1d / rolling_vol_20` (orthogonalises trend from skill, common in equity quant)

The runner writes Sharpe / hit-rate / equity / per-fold metrics for **all four** to the JSONL. The dashboard renders one chart per target.

## Why index ≠ FX (what changed vs the parent)

| Aspect | FX (autoresearch) | Index (autoresearchindexstock) |
|---|---|---|
| Drift | ≈ 0 | strongly positive (~+11%/yr) — must beat buy-and-hold, not zero |
| Vol regime | smile around 8-12% | clusters: 12-15% normal, 30-50% in crisis |
| Calendar | 24/5 | NYSE 252 trading days, holidays, FOMC, OpEx, earnings season |
| Driver structure | rates differential, carry, USD index | macro (CPI/Fed/yields), credit, breadth, options-implied vol |
| Feature universe | 104 FX-specific | ~80-120 equity-index-native (cited literature, see `features.py`) |
| Composite metric | `min(test_sharpe, val_sharpe) - 0.1·n_neg_folds` | same form, BUT computed on `ret_1d` AND tracked simultaneously on B/C/D |

## Mandatory equity-index guardrails

1. **No 2026 data.** All downloaders hard-cap `end="2025-12-31"`. If yfinance returns 2026 rows, drop them. Verified at startup.
2. **Buy-and-hold parity baseline.** Every experiment logs `bh_sharpe`, `bh_return_pct`, `excess_sharpe = strategy_sharpe - bh_sharpe`. The dashboard surfaces both.
3. **Multi-target evaluation.** `evaluate_per_window()` returns metrics for A, B, C, D. The runner writes all four into the JSONL entry per fold.
4. **Calendar features cite literature.** Day-of-week, FOMC week, OpEx week, January effect, Santa rally, earnings season — each feature has a comment with the seminal paper and arXiv / journal id.
5. **Hardware constraints from parent CLAUDE.md still apply.** P-cores only, `_pin_to_safe_cores()` first-thing, 4 threads default.

## Splits (different fold dates)

7 walk-forward folds over 2004-01 → 2025-12. Last test window ends **2025-12-31**. Folds (`splits.py`):

| Fold | Regime | Train end | Val | Test |
|---|---|---|---|---|
| 1 | Pre-GFC bull / GFC onset | 2006-12 | 2007-04 → 2007-09 | 2008-01 → 2008-06 |
| 2 | Post-GFC recovery | 2009-12 | 2010-04 → 2010-09 | 2011-01 → 2011-06 |
| 3 | EU debt + taper tantrum | 2012-12 | 2013-04 → 2013-09 | 2014-01 → 2014-06 |
| 4 | Strong dollar / oil crash | 2015-12 | 2016-04 → 2016-09 | 2017-01 → 2017-09 |
| 5 | Low-vol → COVID shock | 2019-09 | 2019-10 → 2020-03 | 2020-04 → 2020-12 |
| 6 | Inflation + Fed hikes | 2021-12 | 2022-04 → 2022-09 | 2023-01 → 2023-09 |
| 7 | AI rally + recent | 2024-12 | 2025-01 → 2025-04 | **2025-05 → 2025-12** |

Same purge=90d, embargo=21d, label-buffer=10d. Zero overlap verified programmatically each run.

## Backbones (SOTA selection mirroring FX)

Same 15-backbone roster as FX. Order is the FX final-ranking — strongest first:

1. **XGBoost** (FX single-model winner)        — Chen & Guestrin 2016 KDD (arXiv:1603.02754)
2. **LightGBM**                                — Ke et al. 2017 NeurIPS
3. **CatBoost**                                — Prokhorenkova et al. 2018 NeurIPS (arXiv:1706.09516)
4. **LSTM** (bidirectional, residual head)     — Fischer & Krauss 2018 EJOR
5. **MLP** residual (sanity baseline)          — Gu, Kelly & Xiu 2020 RFS
6. **Mamba** / dMamba                          — Gu & Dao 2024 COLM (arXiv:2312.00752); Liu 2025 (arXiv:2602.09081)
7. **xLSTM**                                   — Beck et al. 2024 NeurIPS (arXiv:2405.04517)
8. **iTransformer**                            — Liu et al. 2024 ICLR (arXiv:2310.06625)
9. **PatchTST** (seq_len ≥ 60)                 — Nie et al. 2023 ICLR (arXiv:2211.14730)
10. **TSMixer / PatchTSMixer**                 — Ekambaram et al. 2023 KDD (arXiv:2306.09364)
11. **TimesNet**                               — Wu et al. 2023 ICLR (arXiv:2210.02186)
12. **DLinear**                                — Zeng et al. 2023 AAAI (arXiv:2205.13504)
13. **N-BEATS**                                — Oreshkin et al. 2020 ICLR (arXiv:1905.10437)
14. **N-HiTS**                                 — Challu et al. 2023 AAAI (arXiv:2201.12886)
15. **TFT**                                    — Lim et al. 2021 IJF (arXiv:1912.09363)

Foundation TS models (TimesFM, Chronos, Moirai, MOMENT, Time-MoE, Sundial, TiRex)
are **deferred** — they underperformed on FX at our n. Add only if a ceiling appears.

## Experiment-budget mandate (per backbone)

**25 experiments per backbone** (down from FX's 50 — tighter sprint scope per
user instruction 2026-04-26):

- 1 SOTA-recipe baseline at the per-backbone defaults from the parent
  CLAUDE.md table (epochs / patience / lr / warmup / scheduler / batch /
  weight-decay / loss as cited from each backbone's own paper).
- 24 **hill-climb** experiments around the running champion of that backbone.
  Each hill-climb experiment changes ONE hyperparameter, follows the
  7-step process (diagnose → cite → hypothesise → predict → run → analyse →
  checkpoint), and either becomes the new backbone champion (KEEP) or
  reverts (DISCARD).

15 backbones × 25 exps = **~375 total experiments** before phase-b ensemble.

Hill-climb axes (pick from this menu, one per experiment):
| Axis              | When to try                                        |
|-------------------|----------------------------------------------------|
| seq_len           | for any sequence-aware backbone (PatchTST etc.)    |
| learning rate     | when val loss not converging or oscillating        |
| batch size        | flat-minima probe (Keskar 2017)                    |
| weight decay      | overfit symptoms                                   |
| dropout / head_dropout | overfit symptoms                              |
| hidden size       | underfit symptoms                                  |
| n_layers          | depth ablation                                     |
| warmup epochs     | transformer instability                            |
| scheduler         | cosine vs linear vs ReduceLROnPlateau              |
| seed              | variance characterisation (≥3 seeds before champion declared) |
| huber delta       | rarely useful at our return scale                  |
| GBM-specific: max_depth, n_estimators, min_child_weight, subsample, colsample_bytree, gamma, reg_lambda | structural HPs |

## SOTA modeling setup (locked BEFORE first experiment per backbone)

Per parent CLAUDE.md "Per-Backbone SOTA Training Recipes" — every new
backbone re-derives its baseline recipe from its own published paper, NOT
from another backbone's defaults. Recipe table below is the starting point;
each first-experiment reasoning annotation must cite the exact paper and
explain any deviation.

| Backbone   | Epochs | Patience | LR    | Warmup | Sched   | Batch | WD    | Optim  | Loss   |
|------------|-------:|---------:|------:|-------:|---------|------:|------:|--------|--------|
| mlp        | 50     | 10       | 3e-4  | 0      | cosine  | 32    | 1e-5  | AdamW  | Huber  |
| lstm       | 100    | 15       | 1e-3  | 0      | cosine  | 16    | 7e-4  | AdamW  | Huber  |
| xgboost    | 1500*  | 50*      | 0.03  | —      | —       | —     | —     | —      | sq-err |
| lightgbm   | 2000*  | 50*      | 0.03  | —      | —       | —     | —     | —      | sq-err |
| catboost   | 2000*  | 100*     | 0.03  | —      | —       | —     | —     | —      | RMSE   |
| mamba      | 100    | 20       | 5e-4  | 10     | cosine  | 32    | 0.1   | AdamW  | MSE    |
| xlstm      | 80     | 15       | 5e-4  | 5      | cosine  | 16    | 1e-3  | AdamW  | Huber  |
| itransformer | 150  | 20       | 5e-5  | 10     | cosine  | 32    | 0     | AdamW  | MSE    |
| patchtst   | 100    | 20       | 1e-4  | 10     | cosine  | 32    | 1e-4  | AdamW  | MSE    |
| patchtsmixer | 100  | 15       | 1e-3  | 5      | cosine  | 32    | 1e-5  | AdamW  | MSE    |
| timesnet   | 100    | 20       | 1e-4  | 5      | cosine  | 32    | 1e-4  | AdamW  | MSE    |
| dlinear    | 100    | 15       | 1e-3  | 0      | cosine  | 32    | 0     | AdamW  | MSE    |
| nbeats     | 100    | 15       | 1e-3  | 0      | cosine  | 32    | 0     | AdamW  | MSE    |
| nhits      | 100    | 15       | 1e-3  | 0      | cosine  | 32    | 0     | AdamW  | MSE    |
| tft        | 100    | 15       | 1e-3  | 5      | cosine  | 32    | 1e-4  | AdamW  | Quantile|

(*) GBMs measure rounds (n_estimators) + early-stopping rounds, not epochs.

**Iteration discipline (this CLAUDE.md is a living document):** every new
empirical finding — confirmed-good config, dead-end axis, surprise about
QQQ vs FX behaviour, regime where a backbone dies — gets appended to
the **Session Learnings** section below as soon as it is observed. Future
sessions read CLAUDE.md before any other file.

## File layout (parallel to FX)

```
autoresearchindexstock/
  CLAUDE.md                         # this file
  __init__.py
  data/
    download.py                     # QQQ + ~30 macro/breadth signals (no 2026)
    features.py                     # ~80-120 equity-native features
    splits.py                       # 7-fold walk-forward, last test 2025-12
  model/
    __init__.py                     # re-exports parent's backbone + train
  evaluation/
    metrics.py                      # composite + excess-over-buy-and-hold
  run_autoresearch.py               # runner, own JSONL + best_config + winners
  _sync_dashboard_to_docs.py        # mirrors to docs/index_stock_dashboard/
  _ensemble_trade_logs.py           # to be added in phase b
  autoresearch_results/
    experiment_log.jsonl
    best_config.json
    best_model.pt
    dashboard.html                  # adapted from FX, plots A/B/C/D
    reasoning_annotations.json
    research_journal.md
    experiment_summary.md
    trade_logs/                     # per-day CSVs (same schema as FX)
    winners/
  code_versions/
  memory/
    project_autoresearch_checkpoint.md
    project_hardware_crash_log.md   # link/copy of FX one
```

## Live dashboard route

`docs/index_stock_dashboard/index.html` → served at
**`https://dlmastery.github.io/autoresearch/index_stock_dashboard/`**

Mirrors the FX dashboard's UX: backbone tabs, sortable experiment table,
per-fold breakdown, equity curve, reasoning panel. Adds:

- **Target selector** (A / B / C / D buttons) above the equity chart;
  selecting one re-renders the chart + per-fold table for that target.
- **Buy-and-hold baseline line** drawn alongside strategy equity per fold.
- **Trades column** linking each experiment row to
  `trade_logs/expN_trades.csv`.
- **Excess-Sharpe** column alongside Test-Sharpe so the user sees
  whether we beat passive QQQ.

## Run command

```bash
cd C:/Users/evija/autoresearch
"C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock.run_autoresearch \
  --backbone <name> [--lr ... --bs ... etc.] \
  --description "..."
```

Same flag surface as FX runner. Different package path.

## Crash-recovery checkpoint

`autoresearchindexstock/memory/project_autoresearch_checkpoint.md` —
parallel to the FX one but tracks QQQ-only state. Read at every session
start.

## Session Learnings (append-only)

> Update this section whenever an experiment confirms a hypothesis, kills an
> axis, or surfaces a QQQ-vs-FX behavioural difference. The dashboard +
> research journal are the canonical detail; this is the executive summary
> that future sessions read first.

_(empty — no experiments run yet. First entry: bootstrap session 2026-04-26.)_

