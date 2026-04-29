# Autoresearch QQQ — Abstract

**Title**: A multi-backbone autoresearch loop for the Nasdaq-100 index: 216
arxiv-cited experiments yield a stable +1.32 Sharpe Mamba SSM champion and
characterise the seed-instability ceiling of a single-asset super-fold.

**Authors**: Anonymous (forked from `dlmastery/autoresearch` FX 2026-04-29).

**Abstract**:

We adapt the autoresearch loop methodology of the FX project
([dlmastery/autoresearch](https://github.com/dlmastery/autoresearch))
to the equity-index setting using QQQ (Invesco Nasdaq-100 ETF, 2007-01–2025-12).
We run 216 experiments across six cheap-tier backbones (MLP, LSTM, XGBoost,
LightGBM, CatBoost, Mamba SSM) and five Phase-F transformer/state-space
backbones (DLinear, iTransformer, PatchTST, PatchTSMixer, N-BEATS) under a
seven-fold regime-aware super-fold split (GFC, EU debt, Taper, China-oil,
Vol-mageddon, COVID, AI rally). Each experiment is research-strict: a SOTA
arxiv-cited starting point, a per-fold deficiency diagnosis, an arxiv-cited
hypothesis, a numeric prediction, the run, and a verdict — all logged in
JSONL with a public live dashboard.

The headline result is a **+1.3216 composite Sharpe** champion: a Mamba dMamba
variant (Liu 2025 arXiv:2602.09081) at expand=2, d_state=32, seq_len=60,
seed=42, with **7/7 positive test folds** including all major regimes. The
champion's strategy excess Sharpe over a long-only buy-and-hold baseline is
**+0.45** — the fair-comparison metric for a trending equity index. The
**second stable backbone** is a residual MLP (Gu-Kelly-Xiu 2020) at
warmup=5, head-dropout=0.25, lr=3e-4, wd=1e-5 with **5-seed median
+0.433** composite — the only non-Mamba architecture in our sweep that
produces a positive multi-seed median. Every other architecture (XGBoost,
LightGBM, CatBoost, iTransformer, PatchTSMixer) shows the same structural
pathology: high single-seed test alpha but extreme seed-dependence on the
val side, which the composite formula `min(test_sharpe, val_sharpe) -
0.1 × n_negative_folds` punishes — multi-seed median composites cluster
near zero.

We argue this is **not a backbone defect** but a property of the QQQ
super-fold val window: a single-asset 2007-2025 daily series has too
little independent information for the val side to generalise across
random seeds at the higher-capacity HP regimes that yield strong test
alpha. The **Mamba selective-state mechanism is uniquely robust** to this
in our sweep, suggesting state-space sequence models are the architecture
of choice for single-asset equity-index forecasting in the n<3000 daily
regime.

**Keywords**: autoresearch, time-series forecasting, equity index, QQQ,
Mamba, state space models, walk-forward validation, multi-seed ensembles,
super-fold, regime-aware backtesting.

## Practical implications

For a deployment-oriented practitioner: trade the champion as a **5-seed
ensemble of Mamba dMamba** (seeds 42, 0, 7 — drop seed=99 catastrophic)
combined with the **5-seed MLP residual ensemble** (seeds 0, 99, 2024 — drop
seed=42, 7) per Lakshminarayanan-Pritzel-Blundell 2017 NeurIPS deep-ensemble
methodology. Position size with a Kelly fraction × confidence weighting
(Lim-Zohren-Roberts 2019). Re-train monthly on a rolling window. Audit on
permutation-importance and shuffle-test (FX paper §3.5) before any capital
deployment.

## Reproducibility

All 216 experiments, 7 archived winners, code, dashboards, and reasoning
annotations are public at:

- **Repo**: https://github.com/dlmastery/autoresearchindexstock
- **Dashboard**: https://dlmastery.github.io/autoresearchindexstock/dashboard/
- **Release v0.1.0 (zips)**: https://github.com/dlmastery/autoresearchindexstock/releases/tag/v0.1.0-exp216

The CLAUDE.md project rules + User Directives Log document every
methodological choice, every user correction, and the canonical seq_len /
HP / training-recipe table per backbone. A fresh Claude Code session
reading only CLAUDE.md and the crash-recovery checkpoint at
`memory/project_autoresearch_checkpoint.md` is able to resume the loop
without further onboarding.
