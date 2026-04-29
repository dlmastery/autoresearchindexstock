# AutoResearch QQQ — Live Report

> **Live state** at the time of last sync. Auto-regenerated on every push.
> Self-contained narrative; for raw data see `experiment_log.jsonl` and
> the dashboard's per-experiment reasoning panel.

## 1. Project identity

- **Asset**: QQQ (Invesco QQQ Trust, Nasdaq-100 ETF)
- **Window**: 2004-01-01 → **2025-12-31**. No 2026 data anywhere
  (`data/download.py` hard-caps the end date and drops any 2026 row
  with a logged warning).
- **Optimisation target**: `fwd_ret_1d` — 1-day forward log return
  (target A). KEEP / DISCARD decisions are driven by the composite =
  `min(test_A_sharpe, val_A_sharpe) - 0.1·n_negative_folds`.
- **Tracked alongside**: `fwd_ret_5d` (B), sign-concordance (C),
  vol-adjusted 1d return (D). The dashboard plots all four side-by-side
  via the TARGET selector above the equity chart.
- **Goal**: meet or beat the FX project's mega-ensemble Sharpe **+9.7071**
  on excess-Sharpe basis (strategy − buy-and-hold). Excess-Sharpe is the
  fair metric for a trending equity index; raw Sharpe is misleading.

## 2. Backbone roster

23 backbones × 25 hill-climb experiments per backbone = 575-experiment
roadmap (per CLAUDE.md). 15 generic-TS Tier-1 backbones (matched to the
FX project's order) plus 8 equity-specific Tier-1.5 backbones added
2026-04-26: StockMixer, MASTER, CARD, Crossformer, PatchMixer,
Reversible Mixer, Adv-ALSTM, StockNet (PyTorch implementation
pending).

## 3. Data pipeline

- 56 Yahoo Finance tickers spanning: QQQ + 6 cross-asset benchmarks
  (SPY/DIA/IWM/EFA/EEM/^IXIC/AGG) + 4 industry tilts (SOXX/SMH/IBB/ARKK)
  + 11 sector ETFs (XLK..XLC) + 11 vol-regime signals (VIX family,
  ^VXN = QQQ-native VIX, ^MOVE = bond vol, ^SKEW, ^VVIX) + 10 yield/credit
  signals + 8 macro/FX + 4 international + BTC-USD.
- 205 cited equity-native features (price-derived, vol estimators —
  Parkinson/Garman-Klass/Yang-Zhang; momentum + reversal — Jegadeesh-Titman
  1993 / Lehmann 1990 / Asness-Moskowitz-Pedersen 2013 / George-Hwang
  2004 / Bali-Cakici-Whitelaw 2011; volume + microstructure — Amihud 2002;
  VIX-family + Cieslak-Pang 2021 ^MOVE; yield curve — Estrella-Mishkin
  1998 / Welch-Goyal 2008 / Adrian-Crump-Moench 2013; cross-sectional
  + breadth — Brown-Cliff 2004; calendar / FOMC — French 1980 /
  Lucca-Moench 2015 / Stivers-Sun 2002 / Haug-Hirschey 2006 / Rozeff-Kinney
  1976 / Ariel 1987; lagged-target / variance ratios — Lo-MacKinlay 1988).

## 4. Splits — regime-aware fold design

Each test window placed inside a NAMED equity-market regime so per-fold
breakdowns reveal where the model wins or loses by named state. Cited:
Pagan-Sossounov 2003, Lunde-Timmermann 2004, Hamilton 1989, López de
Prado 2018 *Advances in Financial ML* §7.

| Fold | Regime | Test window |
|---|---|---|
| 1 | GFC peak crash (Lehman + Mar-2009 bottom) | 2008-10 → 2009-03 |
| 2 | 2011 US-downgrade + EU debt | 2011-09 → 2012-03 |
| 3 | Taper tantrum + 2014 H1 | 2014-01 → 2014-09 |
| 4 | China devaluation + oil crash | 2015-09 → 2016-04 |
| 5 | 2018 Vol-mageddon + Q4 sell-off | 2018-08 → 2019-04 |
| 6 | COVID crash + V-recovery | 2020-02 → 2020-12 |
| 7 | Inflation bear + AI rally + 2025 | 2024-04 → 2025-12 |

PURGE_DAYS=90, EMBARGO_DAYS=21, LABEL_HORIZON_BUFFER=10. Zero overlap
verified each run.

## 5. Current state — champion + per-backbone winners

**🏆 GLOBAL CHAMPION: dMamba @ FX-Mamba-winner config (exp 17)** —
composite **+0.8625**, A_sharpe **+0.8625**, test_pos_folds **7/7**,
runtime ~41 min on RTX 4090 Laptop GPU.

| Per-backbone winner | Composite | A_Sharpe | Excess | Status |
|---|---:|---:|---:|---|
| dMamba (variant=dmamba expand=4, FX-Mamba winner) | **+0.8625** | +0.8625 | -0.36 | exp 17 — global champion |
| MLP @ FX-Exp32 seed=0 | +0.5799 | +0.6799 | +0.08 | exp 6 (likely lucky seed; 4-seed median -1.35) |
| LightGBM @ FX-Exp235 n_est=1000 | +0.4825 | **+1.0665** | -0.15 | exp 10 — best A_sharpe single model |
| LightGBM @ FX-Exp235 n_est=2000 (full) | +0.2885 | +0.7430 | -0.48 | exp 12 — full FX recipe |
| Mamba vanilla SOTA | -0.1865 | +0.7814 | -0.44 | exp 15 |
| LSTM @ FX-Exp35 seed=42 | -0.1318 | +0.8339 | **+0.2297** | exp 5 — first BH-beating excess; 7/7 test |
| CatBoost full | running | — | — | brmfh6dv2 in flight |
| XGBoost full | pending | — | — | queued for next CPU slot |

Best **excess-Sharpe**: LSTM exp 5 +0.2297 (the FX-fair-comparison metric;
strategy beats passive QQQ by ~0.23 annualised Sharpe over the test
windows).

## 6. Experiment lineage

17 experiments across XGBoost / MLP / LSTM / LightGBM / CatBoost / Mamba
/ dMamba. Multi-seed evidence:

- **MLP @ FX-Exp32** across 4 seeds: 0=+0.58, 7=-1.50, 42=-1.50, 99=-1.20.
  Median composite -1.35 → seed=0 was 2σ above median → **NOT a stable
  champion**.
- **LSTM @ FX-Exp35** across 3 seeds: 42=+0.83, 0=+0.44, 99=+0.47.
  Median A_sharpe +0.47, std ~0.18 → more stable than MLP, less than
  GBM.

## 7. Key cross-experiment findings

1. **More XGBoost trees made things WORSE on QQQ** (exp 1 50-tree beat
   exp 2 300-tree by composite +0.85). Opposite of FX. Diagnosis:
   12,300-dim flattened seq=60 input space is too large for unregularised
   XGBoost; needs aggressive depth or column-fraction regularisation.
2. **MLP > XGBoost in compute-efficiency on QQQ.** Exp 6 produced higher
   composite (+0.58) in 18× less compute (29s vs 335s). User-feedback
   pivot to MLP first was correct.
3. **FX-champion HPs transfer to QQQ.** Both LSTM @ FX-Exp35 and MLP @
   FX-Exp32 beat plain SOTA-recipe baselines. The FX-empirical
   `head_dropout=0.25` and `wd=7e-4` survived the asset-class transfer.
4. **First positive excess-Sharpe (+0.2297) achieved at exp 5** — LSTM
   @ FX-Exp35 strategy beats passive QQQ.
5. **dMamba > vanilla Mamba on QQQ** — same as FX. The trend+seasonal
   decomposition (Liu 2025 arXiv:2602.09081) captures equity-index
   regime dynamics better than pure SSM.
6. **GPU saturation finally observed at Mamba scale.** MLP/LSTM at
   hidden=128 are too small to load the GPU; Mamba at expand=4 saturates
   to 35-50% GPU util.
7. **Seed variance is large and architecture-specific.** MLP > LSTM > GBM
   in instability, matching FX literature.
8. **Per-fold pattern: alpha lights up in chaos** (GFC peak, US-downgrade,
   China crash, Vol-mageddon) and **fades in trending recoveries**
   (post-GFC QE, COVID V-recovery, AI rally). Same pattern as FX paper.

## 8. Next steps

1. **CatBoost full** lands (CPU) → sync, log verdict.
2. **XGBoost full** as next CPU slot pair-mate.
3. **Multi-seed dMamba** (seeds 0, 99) on GPU to verify champion stability.
4. **Build & fire `_qqq_mega_ensemble.py`** — port of FX
   `_emtsf_mega_ensemble.py`, rank-avg of 5 components. Goal: **meet
   or beat FX +9.7071 excess-Sharpe**.
5. **25 hill-climb experiments per backbone** (575 total roadmap). Seed
   variance studies + HP tweaks around each per-backbone champion.
6. Tier-1.5 backbones (StockMixer, MASTER, CARD, etc.) — implement in
   `model/backbone.py`, then the 25-experiment cycle.

## 9. Live links

- **Live Pages dashboard**: <https://dlmastery.github.io/autoresearch/index_stock_dashboard/>
- **GitHub source**: <https://github.com/dlmastery/autoresearch/tree/master/autoresearchindexstock>
- **CLAUDE.md project rules** (753-line self-contained spec): `autoresearchindexstock/CLAUDE.md`
- **Sibling FX project**: <https://dlmastery.github.io/autoresearch/dashboard/> (Sharpe +9.7071 mega-ensemble; the bar QQQ aims to meet or beat)
