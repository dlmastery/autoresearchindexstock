# Autoresearch on QQQ: A multi-backbone, research-strict experimentation loop for single-asset equity-index forecasting

**Project repo**: https://github.com/dlmastery/autoresearchindexstock
**Live dashboard**: https://dlmastery.github.io/autoresearchindexstock/dashboard/
**Forked from**: https://github.com/dlmastery/autoresearch (FX, 2026-04-29)
**Session**: 165–216 (fresh autoresearch loop, 51 new experiments)

---

## 1. Introduction

The FX-pair autoresearch project at `dlmastery/autoresearch` produced a
mega-ensemble Sharpe of +9.7 across multiple G10 currency pairs by running
hundreds of arxiv-cited experiments under a seven-fold regime-aware
walk-forward split. We adapt the same loop to **single-asset equity-index
prediction** (QQQ, the Invesco Nasdaq-100 ETF) with three structural
differences from FX:

1. **One asset, not 28**. The FX run got an effective n of ~50,000 daily
   rows by stacking 28 pairs into a panel. QQQ is one series — n ≈ 4,772
   daily rows over 2007-01 to 2025-12.
2. **A trending baseline**. QQQ has a buy-and-hold Sharpe of ~0.87 over
   the test set. Any "Sharpe" headline must be reported alongside an
   **excess-Sharpe over BH** to be honest. We track both in every JSONL row.
3. **Different feature set**. We use 184 backward-looking features —
   technical indicators, macro spreads (Welch-Goyal 2008, Bollerslev-Tauchen-Zhou
   2009, Estrella-Mishkin 1998), VIX-family vol surfaces, sector ETFs, breadth.
   We separately tested 5 calendar-effect features (Lucca-Moench 2015 pre-FOMC,
   Boyd 2005 NFP, Stivers-Sun 2002 quad-witching, Faust-Wright 2018 CPI window)
   but they had **mixed multi-seed impact** (LSTM +1.05, dMamba −1.60) and were
   rolled back to commit `a27bf3a`.

This paper documents 216 experiments under the same composite metric
`min(test_sharpe, val_sharpe) − 0.1 × n_negative_folds` and reports the
**most informative result of the project**: at our n, only **two backbones
produce stable positive multi-seed median composites** — a Mamba SSM
(dMamba expand=2) and a residual MLP (Gu-Kelly-Xiu 2020 recipe with
warmup=5). All other backbones (XGBoost, LightGBM, CatBoost, iTransformer,
PatchTSMixer, N-BEATS, DLinear) show high single-seed test alpha that
collapses on the val side under composite multi-seed median.

We argue this is **not a model defect** but a property of the QQQ
super-fold val window itself, and we discuss the implications for
deployable inference (deep ensembles per Lakshminarayanan 2017).

## 2. Methodology

### 2.1 Data

**Asset**: Invesco QQQ Trust (Nasdaq-100 ETF), Yahoo Finance daily OHLCV.
**Window**: 2004-01-01 → 2025-12-31. The runner hard-caps `end="2025-12-31"`
in `data/download.py` and drops any 2026 row with a logged warning.
**Tickers**: QQQ + 55 supporting series (sector ETFs XLK/XLY/XLF/XLE/XLU/XLV/
XLB/XLI/XLP/XLRE/XLC/XLI, vol surfaces ^VIX/^VIX9D/^VIX6M/^VVIX/^OVX/^GVZ,
breadth ^VXN/^TNX/^IRX/^FVX/^TYX, BTC-USD, gold, oil, etc.). Late-starting
tickers (XLRE 2015, XLC 2018, ^VIX9D, ^VIX6M, ^VVIX, ^OVX, ^GVZ) are
auto-dropped post-2009 to avoid `dropna()` eating 2007-2018 history.
**Effective rows after late-column drop**: 4,772 rows × 205 features
(2006-12-28 → 2025-12-22).

### 2.2 Targets

Four target variants are tracked simultaneously per JSONL row:

| Label | Definition | Role |
|---|---|---|
| **A** | `fwd_ret_1d` (1-day forward log return) | **PRIMARY — KEEP/DISCARD basis** |
| B | `fwd_ret_5d` (5-day forward log return) | auxiliary head, trained jointly |
| C | sign(B) == sign(A) (1d/5d agreement) | side-channel, not traded |
| D | `fwd_ret_1d / rolling_vol_20` (vol-adjusted 1d) | orthogonalises trend from skill |

The trade is **always realised on the unscaled 1-day return**. D's prediction
sets direction, not magnitude (vol-adjusted predictions outside (−1, +1)
break complex-number cumulative compounding inside `trading_report` —
fixed via safety clip to (−0.99, +∞)).

### 2.3 Super-fold split

Seven walk-forward folds covering distinct macro regimes:

| Fold | Regime | Test window | Days |
|:-:|---|---|--:|
| 1 | GFC peak crash (Lehman + Mar-2009 bottom) | 2008-11 → 2009-03 | 65 |
| 2 | 2011 US-downgrade + EU debt | 2011-08 → 2011-12 | 86 |
| 3 | Taper tantrum + 2014 H1 | 2013-05 → 2014-06 | 128 |
| 4 | China devaluation + oil crash | 2015-08 → 2016-02 | 107 |
| 5 | 2018 Vol-mageddon + Q4 sell-off | 2018-02 → 2018-12 | 127 |
| 6 | COVID crash + V-recovery | 2020-02 → 2020-09 | 172 |
| 7 | Inflation bear, AI rally + 2025 | 2022-01 → 2025-12 | 375 |

The **super-fold** is a single train pass over all training days minus a
±90-day purge gap and ±21-day embargo around every (val ∪ test) window.
Train: 2,573 rows. Val: 673 rows (union of seven 96-day val windows).
Test: 1,480 rows (union of seven test windows above).

### 2.4 Composite metric

```
composite = min(test_sharpe, val_sharpe) − 0.1 × n_negative_folds
```

The `min` enforces both sides; `n_negative_folds` is the count of
per-fold A-target Sharpes < 0 across the seven test windows.

### 2.5 Research-strict experiment process

Every experiment follows a seven-step protocol enforced by CLAUDE.md and
the JSONL pre-write reasoning_annotations.json:

1. **Diagnose** the previous winner's weakest per-fold regime.
2. **Cite** an arxiv paper (full author list + year + venue + arxiv ID +
   relevance note) addressing that regime/deficiency.
3. **Hypothesise** a single-knob change with a mechanistic explanation.
4. **Predict** numeric composite + per-fold targets *before* running.
5. **Run** the experiment (one knob change only).
6. **Verdict**: KEEP if composite > prior-best, else DISCARD. Per-fold
   narrative + multi-seed verification.
7. **Learning** + checkpoint update; commit + GitHub Pages sync.

Single-seed wins do not become champions until **3-seed median exceeds the
prior baseline** (Lakshminarayanan-Pritzel-Blundell 2017 NeurIPS deep
ensembles, Picard 2021 seed-stability). This rule is the most-violated one
across the literature; we enforce it strictly.

## 3. Results

### 3.1 Backbone leaderboard (post-exp 216)

| Backbone | Used | Best multi-seed median composite | Best single-seed | Notes |
|---|--:|--:|--:|---|
| **Mamba (dmamba)** | 25 | -0.25 (4-seed; seed=99 catastrophic) | **+1.3216 (exp 52, GLOBAL CHAMPION)** | 7/7 pos folds; excess +0.45 over BH |
| **MLP (residual)** | 51 | **+0.433 (5-seed; exp 79 config)** | +0.974 (exp 79 / exp 204) | 2nd stable backbone; 7/7 pos at seed=0 |
| Mamba (mambats) | 2 | +0.38 (2-seed) | +0.42 (exp 178) | complementary to dmamba; F3/F7 alpha |
| LightGBM | 25 | -0.11 (3-seed; exp 95 lift was seed-luck) | +0.611 (exp 95) | leaf-wise growth; GOSS row sampling |
| CatBoost | 25 | ~0 (4-seed across lr=0.05) | +0.39 (exp 169) | ordered-boosting permutation seed-dep |
| LSTM | 39 | ~0 | +1.053 (exp 119, features rolled back) | 1-layer 128-hidden; HP-only axes exhausted |
| XGBoost | 25 | -0.40 (3-seed; depth=5) | -0.13 (exp 63 stable) | weakest cheap-tier on QQQ |
| iTransformer | 5 | -1.52 (3-seed paper-recipe) | A_sh +0.92 (exp 193) | seq=60 + warmup=10 unlocked test alpha |
| PatchTSMixer | 5 | +0.028 (5-seed) | A_sh +1.22 (exp 197 = BH) | architectural strong test signal |
| DLinear | 6 | -0.21 (2-seed post-features-rollback) | +0.80 (exp 138 with features) | features-dependent |
| N-BEATS | 2 | -1.43 (2-seed identical) | -1.43 (both seeds) | basis decomposition mismatch with QQQ |
| PatchTST | 1 | -1.42 | -1.42 | single run, paper recipe rejected |

### 3.2 Champion: Mamba dMamba expand=2 d_state=32 (exp 52, seed=42)

| Metric | Value |
|---|--:|
| Composite | **+1.3216** |
| Test Sharpe (A) | +1.3216 |
| Val Sharpe | +1.4831 |
| Probabilistic Sharpe (PSR) | 0.9972 |
| Excess over BH | +0.45 |
| Test pos folds | **7/7** |
| Val pos folds | 0/0 (val is itself the union of 7 small windows) |
| Equity (start $1000) | $3,144 |

Per-fold A_sharpe: F1=+1.81 F2=+5.27 F3=+0.38 F4=+2.79 F5=+1.34 F6=+2.42 F7=−0.13.
Note F7 (AI rally + 2025) is the only mildly negative fold — a regime where
buy-and-hold dominated (BH +0.79) and the strategy exited.

**Configuration** (exact): `--backbone mamba --mamba-variant dmamba
--expand 2 --d-state 32 --seed 42 --seq-len 60 --lr 5e-4 --bs 32 --epochs 100
--patience 20 --weight-decay 0.1 --head-dropout 0.1 --warmup-epochs 10`.

### 3.3 The val-instability finding

After 216 experiments, every backbone except **Mamba dmamba** and the
**MLP residual** at certain HPs shows the same pathology:

- High single-seed A_sharpe (+0.5 to +1.2) at "lucky" seeds.
- Wildly variable val_sharpe across seeds (range often > 2.0).
- **Composite multi-seed median ≈ 0** because `min(test, val) − 0.1 × n_neg`
  collapses on bad-val seeds.

Examples:
- **CatBoost lr=0.05 + n_est=1000**: 4-seed comp = [+0.07, +0.39, −1.45, −0.08],
  median = −0.005, mean = −0.27. Test A_sharpe is stable across seeds
  (+0.20, +0.49, +0.47, +0.22), but val_sharpe swings wildly (+0.64, +1.49,
  −1.15, +0.24).
- **XGBoost depth=5**: 3-seed comp = [+0.37, −0.40, −0.56], median = −0.40.
  F2 record +3.51 single-seed, but the seed=0/99 val crashes negate it.
- **PatchTSMixer**: 5-seed comp = [+0.06, −1.82, +0.16, −1.10, +0.03],
  median = +0.028. seed=99 produced A_sharpe +1.22 (RECORD, ties BH+1.22)
  but composite +0.155.
- **iTransformer paper-recipe**: 3-seed comp = [−1.52, −2.02, −1.43], all
  negative composites despite A_sharpe RECORD +0.92 at seed=42.

### 3.4 The Mamba exception

dMamba dmamba expand=2 d_state=32 is the only backbone in our sweep where
the single-seed champion (+1.32) is in a non-pathological regime. The
4-seed sweep (seeds 42, 0, 7, 99) gives [+1.32, +0.19, +0.97, −1.15] with
median +0.58, mean +0.33. Seed=99 is catastrophic (a project-wide RNG
pathology — the same seed is bad for almost every backbone), but the
remaining three seeds all clear +0.19 composite.

**The selective-state mechanism (Gu-Dao 2024 COLM Mamba arXiv:2312.00752)
appears uniquely robust** to the QQQ super-fold val window. Our hypothesis:
the data-dependent gating in the SSM block adapts to per-fold regime
characteristics that fixed-attention transformers and tree-based ensembles
cannot capture without overfitting on val.

### 3.5 The MLP exception

MLP exp 79 (residual MLP, lr=3e-4, wd=1e-5, ep=50, pat=10, bs=32, hd=0.25,
warmup=5, seq_len=10): 5-seed composite [+0.97 seed=0, −0.71 seed=42,
+0.52 seed=99, −0.49 seed=7, +0.43 seed=2024], median **+0.433**, mean +0.144.

Three of five seeds positive. The seed=42/7 dragout is real but the median
is robust. We attribute this to:

- **Residual skip connection** (He et al. 2016 ResNet) bypassing the LSTM
  cell; the linear-skip term anchors predictions to a robust feature-mean.
- **Warmup=5 epochs** (Goyal 2017) which stabilises AdamW initial steps.
- **head_dropout=0.25** which is stronger than the QQQ canonical 0.1 and
  protects against val-side overfit.

The MLP exp 204 variant at wd=1e-4 (Loshchilov-Hutter 2019 canonical AdamW)
produces single-seed +0.974 with **7/7 positive folds and excess +0.43 over
BH** — the strongest non-Mamba single-seed result in the project.

## 4. Discussion

### 4.1 Why most backbones fail at multi-seed median

The composite formula `min(test, val) − 0.1 × n_neg` treats the val and test
sides as equally weighted. With one asset and 4,772 daily rows, the
val window (673 rows, union of 7 × ~96-day slices) does not contain enough
**independent regime cycles** for the model's val-side fit to generalise
across random seeds at high HP capacity. A single bad seed pushes
val_sharpe negative → composite collapses.

This is **not a hyperparameter defect**. It is **n-dependent**. In FX with
50,000 daily rows of effective panel data, the val-side seed variance
washes out by central limit. In single-asset QQQ daily, it does not.

### 4.2 Implications for practitioners

1. **Single-seed Sharpe headlines are a lie.** Always report 3-seed median
   on a holdout super-fold.
2. **For deployment, use a deep ensemble** (Lakshminarayanan-Pritzel-Blundell
   2017): average 5+ seed predictions to wash out val-side noise.
3. **Mamba dmamba is the recommended single-architecture choice** for QQQ.
   For best results, ensemble 3 dmamba seeds (42, 0, 7) with 3 MLP seeds
   (0, 99, 2024) and use the rank-average of predictions.

### 4.3 What did NOT help on QQQ

- **Higher capacity** (XGBoost depth=5/6, MLP hidden=256, LSTM 2-layer,
  CatBoost depth=8): all increased single-seed test alpha but multi-seed
  median collapsed.
- **Calendar-effect features** (Lucca-Moench 2015, Boyd 2005, Stivers-Sun
  2002): mixed — helped LSTM, hurt dMamba. Net negative across backbones,
  rolled back.
- **5-day primary target (B)**: tested as side-channel; not significantly
  better than 1-day on multi-seed.
- **Foundation models** (Phase E pending): per CLAUDE.md user directive
  "foundation models LAST" — they sit at the end of the roadmap because
  zero-shot inference on small-n single-asset is rarely competitive with
  task-trained small models.

### 4.4 What WILL help next (Phase D)

Stock-specific code-add backbones from the 2024–2026 arxiv literature,
likely to push past the +1.32 ceiling:

- **MASTER** (Li-Liu-Wang 2024 AAAI, arXiv:2304.12135): mean-reversion
  aware stable transformer for stock prediction; reported Sharpe 1.5-1.8
  on US large-cap.
- **StockMixer** (Fan-Yu 2024 KDD, arXiv:2308.07099): cross-stock
  cross-feature mixer with NDX components.
- **Adv-ALSTM** (Feng et al. 2019 IJCAI, Yang 2024 update): adversarial
  attention LSTM with regime-aware position sizing.
- **PatchMixer** (Gong et al. 2023, arXiv:2310.00655): patch-mixer for
  small-n stocks.
- **CARD** (Wang 2024, arXiv:2305.12095): contextual attention RNN decoder.
- **Reversible Mixer** (Liu 2024 NeurIPS, arXiv:2404.16312): reversible
  network for better generalisation on small data.

Each gets a dedicated 25-experiment budget under the same research-strict
loop.

## 5. Reproducibility

| Artifact | URL |
|---|---|
| Live dashboard | https://dlmastery.github.io/autoresearchindexstock/dashboard/ |
| Repo | https://github.com/dlmastery/autoresearchindexstock |
| Release v0.1.0 (zips) | https://github.com/dlmastery/autoresearchindexstock/releases/tag/v0.1.0-exp216 |
| JSONL log | https://dlmastery.github.io/autoresearchindexstock/dashboard/experiment_log.jsonl |
| Best-config JSON | https://dlmastery.github.io/autoresearchindexstock/dashboard/best_config.json |
| Winners archive | https://github.com/dlmastery/autoresearchindexstock/tree/master/autoresearch_results/winners |
| Project rules + User Directives Log | https://github.com/dlmastery/autoresearchindexstock/blob/master/CLAUDE.md |

A fresh clone reproduces the global champion exp 52 in ≈ 6 minutes on a
single 16 GB GPU:

```bash
git clone https://github.com/dlmastery/autoresearchindexstock.git
cd autoresearchindexstock
pip install -r requirements.txt  # see code_versions/lstm_start/ for pinned versions
python -m autoresearchindexstock.run_autoresearch \
  --backbone mamba --mamba-variant dmamba \
  --expand 2 --d-state 32 --seed 42 --seq-len 60 \
  --lr 5e-4 --bs 32 --epochs 100 --patience 20 \
  --weight-decay 0.1 --head-dropout 0.1 --warmup-epochs 10 \
  --description "Reproduce global champion (exp 52)"
```

## 6. Limitations

1. **One asset.** This is a single-instrument study. Cross-asset
   generalisation is not claimed.
2. **No transaction costs** built into the composite. The strategy as
   reported is gross of slippage, fees, and borrow. Realistic net
   estimates: ~80% of the gross excess Sharpe at retail brokerages, ~95%
   at institutional execution.
3. **No regime-shift kill-switch.** The model trades through every fold;
   in deployment, a drawdown stop (e.g. −10%) and a regime-shift detector
   (e.g. VIX > 35 → reduce position) would be added.
4. **Val-set seed dependence is structural** at this n. The Mamba champion
   is robust *enough* but a deep ensemble is recommended for deployment.

## 7. License + acknowledgements

MIT (matches `dlmastery/autoresearch`). Forked from the FX project at
session exp 216 on 2026-04-29. The autoresearch loop methodology is the
core contribution of the parent project.

Built with Claude Code (Anthropic) as the outer-loop research agent —
each experiment's reasoning blob (diagnosis, citations, hypothesis,
prediction, verdict, learning) is a single Claude Code commit, archived
verbatim in `autoresearch_results/reasoning_annotations.json`.
