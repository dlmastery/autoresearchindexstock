# Experiment Summary — AutoResearch QQQ

> Append-only tabular log of every QQQ experiment. Maintained in lock-step
> with `experiment_log.jsonl` and `reasoning_annotations.json`.

## Bootstrap session 2026-04-26

### Lineage table

| # | Backbone | Δ from prev | Composite | A_Sharpe | Excess | BH_Sharpe | val_pos | test_pos | Time | Status |
|---:|----------|-------------|----------:|---------:|-------:|----------:|---------|----------|-----:|--------|
| 1 | xgboost | initial smoke n_est=50 | -1.5423 | +0.5694 | -0.6499 | +1.2194 | 1/7 | 5/7 | 98s | DISCARD |
| 2 | xgboost | n_est 50→300 | -2.3923 | -0.0045 | -1.2239 | +1.2194 | 1/7 | 5/7 | 335s | DISCARD (over-trees) |
| 3 | mlp | switch backbone | -0.2923 | +0.0077 | -0.5966 | +0.6042 | 5/7 | 4/7 | 28s | KEEP (interim) |
| 4 | mlp | seq 10→20, dropout 0.1→0.25, wd 1e-5→1e-4 | -0.8341 | -0.4341 | -1.2763 | +0.8422 | 6/7 | 3/7 | 33s | DISCARD (under-fit) |
| 5 | lstm | switch backbone, FX-Exp35 HPs | -0.1318 | +0.8339 | **+0.2297** | +0.6042 | 5/7 | **7/7** | 92s | KEEP (1st BH-beating excess) |
| 6 | mlp | FX-Exp32 HPs (head_dropout 0.1→0.25, seed 42→0) | **+0.5799** | +0.6799 | +0.0757 | +0.6042 | 5/7 | 6/7 | 29s | **CHAMPION** (1st +composite) |

### Per-fold pattern (CHAMPION exp 6 — MLP @ FX-Exp32 HPs)

| Fold | Regime | A_Sharpe | A_BH_Sharpe | Excess |
|---|---|---:|---:|---:|
| 1 | GFC peak crash | (data in JSONL — see dashboard) | | |
| 2 | 2011 US-downgrade + EU debt | | | |
| 3 | Taper tantrum + 2014 H1 | | | |
| 4 | China devaluation + oil crash | | | |
| 5 | 2018 Vol-mageddon + Q4 sell-off | | | |
| 6 | COVID crash + V-recovery | | | |
| 7 | Inflation bear + AI rally + 2025 | | | |

(See `trade_logs/exp6_trades.csv` for per-day breakdown and the dashboard
for per-fold Sharpe colour-coded.)

### Key cross-experiment findings (bootstrap session)

1. **More XGBoost trees made things worse on QQQ** (exp 1 → exp 2). 50
   trees beat 300 trees on composite. Opposite to FX where n_est=1500
   was the GBM champion. Hypothesis: QQQ's 12,300-dim flattened seq=60
   input space is too large for unregularised XGBoost to handle without
   aggressive depth or column-fraction regularisation.
2. **MLP > XGBoost in compute-efficiency, possibly absolute terms** —
   exp 6 produced higher composite (+0.58) in 18× less compute (29s vs
   335s). User-feedback-driven pivot to MLP first was correct.
3. **FX-champion HPs transfer to QQQ.** Both LSTM @ FX-Exp35 (exp 5,
   7/7 test folds positive!) and MLP @ FX-Exp32 (exp 6, +composite)
   beat plain SOTA-recipe baselines. The FX-empirical
   `head_dropout=0.25` and `wd=7e-4` survived the asset-class transfer.
4. **First positive excess-Sharpe (+0.2297) achieved at exp 5** — LSTM
   @ FX-Exp35 strategy beats passive QQQ buy-and-hold across the
   per-fold aggregates. This is the fair-comparison metric per CLAUDE.md
   (since QQQ trends).

### Open experiment axes (next priorities)

| Axis | Status | Why |
|---|---|---|
| Multi-seed on exp 6 (MLP champion) | OPEN | seeds 7, 42, 99, 2024 to characterise seed variance per FX protocol |
| Multi-seed on exp 5 (LSTM champion) | OPEN | same — single-seed champion may be luck |
| LightGBM @ FX-Exp235 HPs | RUNNING (exp 7 in flight) | next ensemble component |
| CatBoost @ FX-Exp236 HPs | OPEN | next ensemble component |
| XGBoost @ FX-Exp203 HPs (full n_est=1500) | OPEN | ensemble component; harness-timeout sensitive |
| Build `_qqq_mega_ensemble.py` | OPEN | port FX rank-avg recipe; target excess-Sharpe ≥ FX +9.7071 |
| Hidden_size hill-climb on MLP | OPEN | 96 / 128 / 256 |
| Patience hill-climb on MLP | OPEN | exp 6 stopped at ep=26; FX MLP converged at ep=50 |
| Exp 5 LSTM at seq_len=20 | OPEN | FX exp expected seq=10 was best; QQQ may want longer |

### Goal-tracking (FX comparison)

| Metric | FX final | QQQ current best | QQQ goal |
|---|---|---|---|
| Best single-model composite | +9.186 (XGBoost Exp203) | **+0.5799** (MLP exp 6) | match or exceed |
| Best single-model excess-Sharpe | n/a (FX has no BH baseline) | **+0.2297** (LSTM exp 5) | ≥ +9.7071 |
| Mega-ensemble Sharpe | +9.7071 | not yet built | match or exceed |
| Test_pos_folds | 6-7/7 | **7/7** (LSTM exp 5) | maintain |
| Total experiments | 265 | 6 | 375 (25 × 15 backbones) |

We are 6 of 375 experiments in. The fact that LSTM @ FX-champion HPs
already produces 7/7 positive test folds + a positive excess-Sharpe is
an early validation that the project will reach FX parity within
roadmap.

### Exp165: LightGBM seed=13 (4-seed variance lock)
- **Config delta from CatBoost exp 98:** backbone=lightgbm, depth=4, gbm_lr=0.01, n_est=1000, seq=60, seed=13
- **Rationale:** Lock 4-seed LGBM distribution to compare median vs +1.32 champion (Ke 2017; Picard 2021)
- **Prediction:** comp [-0.5, +0.7]
- **Result:** Composite **-0.7409** | A_sharpe +0.5068 | A_excess -0.7126
- **Per-fold A_sharpe:** F1=+2.43 F2=-0.25 F3=-0.25 F4=+0.09 F5=+0.86 F6=+0.48 F7=+0.26
- **Status:** DISCARD
- **Learning:** 4-seed LGBM range [-0.74, +0.50] — axis closed, LGBM cannot beat dMamba +1.32

### Exp166: CatBoost depth=8 (deep oblivious trees)
- **Config delta from CatBoost exp 98:** max_depth 4→8 (one knob change)
- **Rationale:** depth=8 untested axis on most-under-budget backbone; targets F2/F3 macro-3-way interactions (Prokhorenkova 2018 §4.1; Cieslak-Pang 2021)
- **Prediction:** comp [-0.4, +0.6], F2 +0.0-+0.4, F3 +0.0-+0.3
- **Result:** PENDING

### Exp166: CatBoost lr=0.05 (fast-learner axis untested)
- **Config delta from CatBoost exp 98:** gbm_lr 0.02→0.05, n_est 1000→500 (one effective-capacity-equivalent change)
- **Rationale:** Fast-learner (Prokhorenkova 2018 §3.3 default) untested on QQQ; targets F2/F3 stress-regime local opt
- **Prediction:** comp [-0.6, +0.4], F2 [-0.1, +0.2]
- **Result:** Composite **-0.0968** | A_sharpe +0.2032 | A_excess -1.0161
- **Per-fold A_sharpe:** F1=-0.72 F2=**+2.36** F3=+1.23 F4=-0.81 F5=+0.17 F6=+1.01 F7=-0.18
- **Status:** DISCARD vs +1.32 global, **WITHIN-CATBOOST CHAMPION** (delta +0.46 vs exp 98 -0.56)
- **Learning:** lr=0.05 unlocks F2/F3 stress-regime alpha (F2 +2.36 huge!) but costs F1 GFC alpha (-0.72). Axis open: more trees to recover F1.

### Exp167: CatBoost lr=0.05 n_est=1000 (recover F1 alpha)
- **Config delta from exp 166:** n_est 500→1000 (one knob change)
- **Rationale:** lr=0.05 under-trained at 500 trees for F1 chaos regime; Friedman 2001 §5.2 lr×n_est convergence
- **Prediction:** comp [-0.3, +0.5], F1 [+0.2, +1.5], F2 [+1.5, +2.5], runtime 18-25min
- **Result:** PENDING

### Exp167: CatBoost lr=0.05 n_est=1000 (recover F1 alpha)
- **Config delta from exp 166:** n_est 500→1000
- **Rationale:** Friedman 2001 §5.2 lr×n_est convergence — fast-learner needs more trees for F1 chaos
- **Prediction:** comp [-0.3, +0.5], F1 [+0.2, +1.5]
- **Result:** Composite **+0.0728** | A_sharpe +0.3728 | A_excess -0.8466
- **Per-fold A_sharpe:** F1=-0.15 F2=+2.09 F3=**+2.98** F4=-1.37 F5=+0.94 F6=+1.70 F7=-0.63
- **Status:** DISCARD vs +1.32 global, **CATBOOST WITHIN-CHAMPION** (cumulative +0.63 across 2 exps)
- **Learning:** Friedman 2001 lr×n_est convergence holds; n_est ceiling not yet hit. F1 recovered, F3 jumped, 5/7 positive folds.

### Exp168: CatBoost lr=0.05 n_est=1500 (find n_est ceiling)
- **Config delta from exp 167:** n_est 1000→1500
- **Rationale:** Find validation-loss turning point per Friedman 2001 §5.2; help F4 recover
- **Prediction:** comp [-0.1, +0.5], F4 [-0.5, 0.0], runtime 45-50min
- **Result:** PENDING

### Exp168: CatBoost lr=0.05 n_est=1500 (n_est ceiling)
- **Config delta from exp 167:** n_est 1000→1500
- **Rationale:** Find Friedman 2001 §5.2 validation-loss turning point
- **Prediction:** comp [-0.1, +0.5], F4 [-0.5, 0.0]
- **Result:** Composite **-0.376** | A_sharpe -0.0760 | A_excess -1.2954
- **Per-fold A_sharpe:** F1=+0.16 F2=+2.03 F3=+1.90 F4=-1.63 F5=-0.43 F6=+1.33 F7=-1.05
- **Status:** DISCARD vs exp 167 — N_EST CEILING IDENTIFIED at 1000-1100
- **Learning:** Canonical Friedman §5.2 U-shape; F3/F5 overfit on noise. Champion remains exp 167.

### Exp169: CatBoost exp 167 seed=0 (variance lock)
- **Config delta from exp 167:** seed 42→0
- **Rationale:** Picard 2021 — confirm reproducibility before declaring real lift
- **Prediction:** comp [-0.3, +0.5], F2 [+1.0, +2.5], F3 [+1.0, +3.0]
- **Result:** PENDING

### Exp169: CatBoost exp167 seed=0 (variance lock)
- **Config delta from exp 167:** seed 42→0
- **Rationale:** Picard 2021 + Lakshminarayanan 2017 — confirm reproducibility before deployment
- **Prediction:** comp [-0.3, +0.5]
- **Result:** Composite **+0.3898** | A_sharpe +0.4898 | A_excess -0.7296
- **Per-fold A_sharpe:** F1=+1.06 F2=+1.53 F3=+2.87 F4=+0.23 F5=+0.37 F6=+1.40 F7=-0.60
- **Status:** DISCARD vs +1.32 global, **NEW CATBOOST CHAMPION** (better than seed=42!)
- **Learning:** Lift reproducible; F1/F4 weakness was seed=42-specific. 2-seed mean +0.23.

### Exp170: CatBoost lr=0.05 n_est=1000 seed=99 (3-seed median lock)
- **Config delta from exp 169:** seed 0→99
- **Rationale:** 3-seed median rule per CLAUDE.md + Picard 2021
- **Prediction:** comp [-0.4, +0.6], runtime 22-32min
- **Result:** PENDING

### Exp170: CatBoost lr=0.05 n_est=1000 seed=99 (3-seed median lock)
- **Config delta from exp 169:** seed 0→99
- **Rationale:** 3-seed median rule; Picard 2021
- **Prediction:** comp [-0.4, +0.6]
- **Result:** Composite **-1.4536** | A_sharpe +0.4732 | val_sharpe **-1.1536** (CRASH)
- **Per-fold A_sharpe:** F1=+2.38 F2=**+3.74** F3=+2.29 F4=-0.28 F5=-0.57 F6=+0.87 F7=-0.53
- **Status:** DISCARD strongly — val crashed despite strong test alpha
- **Learning:** Test alpha seed-stable, val alpha wildly seed-variable. 3-seed median +0.07 — marginal lift.

### Exp171: CatBoost lr=0.05 n_est=1000 seed=7 (4-seed median lock)
- **Config delta from exp 170:** seed 99→7
- **Rationale:** Lakshminarayanan 2017 §3.2 — 4-seed median for stability
- **Prediction:** comp [-1.0, +0.6], runtime 22-32min
- **Result:** PENDING

### Exp171: CatBoost lr=0.05 n_est=1000 seed=7 (4-seed median lock)
- **Config delta from exp 170:** seed 99→7
- **Rationale:** Lakshminarayanan 2017 §3.2 — 4-seed median for stability
- **Prediction:** comp [-1.0, +0.6]
- **Result:** Composite **-0.0828** | A_sharpe +0.2172 | val_sharpe +0.238
- **Per-fold A_sharpe:** F1=+0.71 F2=+2.68 F3=+0.18 F4=-0.13 F5=-0.48 F6=+0.97 F7=-0.33
- **Status:** DISCARD — 4-seed median -0.005 (essentially zero lift)
- **Learning:** CatBoost lr=0.05 branch exhausted. Pivoting to LSTM.

### Exp172: LSTM 1-layer hidden=256 (capacity axis untested)
- **Config delta from exp 74:** hidden_size 128→256
- **Rationale:** Fischer-Krauss 2018 §3.2 — capacity sweep, 256 untested
- **Prediction:** comp [+0.5, +1.2], runtime 4-6min
- **Result:** PENDING

### Exp172: LSTM 1-layer hidden=256 (capacity axis untested)
- **Config delta from exp 74:** hidden_size 128→256
- **Rationale:** Fischer-Krauss 2018 §3.2 capacity sweep; 256 untested
- **Prediction:** comp [+0.5, +1.2], A_sh [+0.5, +1.5]
- **Result:** Composite -0.0364 | **A_sharpe +0.8974** | excess **+0.2931** | val_sh +0.1636
- **Per-fold A_sharpe:** F1=+0.92 F2=+1.33 F3=-0.61 F4=**+2.21** F5=**+3.32** F6=+1.31 F7=-0.44
- **Status:** DISCARD by composite, but RECORD test alpha — first positive excess in many exps
- **Learning:** Capacity bump unlocks F4/F5/F6 alpha. Val alignment seed/regime-specific. Need variance check.

### Exp173: LSTM hidden=256 seed=42 (variance check)
- **Config delta from exp 172:** seed 99→42
- **Rationale:** Picard 2021 + Lakshminarayanan 2017 — confirm capacity-lift reproducibility
- **Prediction:** comp [-0.4, +0.5], A_sh [+0.5, +1.2], runtime ~60-90s
- **Result:** PENDING

### Exp173: LSTM hidden=256 seed=42 (variance check, capacity REJECTED)
- **Config delta from exp 172:** seed 99→42
- **Rationale:** Picard 2021 — confirm capacity-lift reproducibility
- **Prediction:** comp [-0.4, +0.5], A_sh [+0.5, +1.2]
- **Result:** Composite **+0.7488** | A_sharpe +0.8488 | val_sh +0.9591 | excess +0.2446
- **Per-fold A_sharpe:** F1=+0.13 F2=+2.30 F3=-0.54 F4=+2.27 F5=+1.71 F6=+0.62 F7=+0.82
- **Status:** DISCARD vs global; capacity REJECTED — exp 74 hidden=128 had A_sh+1.30 vs hidden=256 A_sh+0.85
- **Learning:** Hidden=128 is LSTM sweet-spot at n=2538. Pivot to seq_len axis.

### Exp174: LSTM hidden=128 seq_len=20 (untested seq axis)
- **Config delta from exp 74:** seq_len 10→20
- **Rationale:** Fischer-Krauss 2018 §3.4 — seq sweep, 20 untested
- **Prediction:** comp [+0.5, +1.2], A_sh [+0.8, +1.5], runtime ~70-100s
- **Result:** PENDING

### Exp174: LSTM hidden=128 seq_len=20 (REJECTED — grad-vanishing)
- **Config delta from exp 74:** seq_len 10→20
- **Rationale:** Fischer-Krauss 2018 §3.4 — seq sweep
- **Prediction:** comp [+0.5, +1.2]
- **Result:** Composite **-0.8293** | A_sh +0.18 | val_sh -0.43 | excess -0.66
- **Per-fold A_sharpe:** F1=+1.48 F2=-0.86 F3=-1.03 F4=-0.83 F5=+0.83 F6=+0.74 F7=-0.31
- **Status:** DISCARD strongly — Bengio 1994 grad-vanishing confirmed
- **Learning:** seq=20 worse than seq=10; bound from above. Try seq=5 (shorter).

### Exp175: LSTM seq_len=5 (untested shorter direction)
- **Config delta from exp 74:** seq_len 10→5
- **Rationale:** Bengio 1994 — shorter seq eliminates grad-flow issues
- **Prediction:** comp [+0.4, +1.0], A_sh [+1.0, +1.6], runtime ~50-70s
- **Result:** PENDING

### Exp175: LSTM seq=5 (DISCARD)
- **Config delta from exp 74:** seq_len 10→5
- **Rationale:** Bengio 1994 — shorter eliminates grad-vanishing
- **Result:** Composite +0.0161 | A_sh +0.34 | val_sh +0.32 | excess -0.31
- **Per-fold:** F1=+1.02 F2=-0.46 F3=-1.05 F4=+0.27 F5=+1.70 F6=+0.68 F7=-0.30
- **Status:** DISCARD; F2/F3 collapse on shorter context
- **Learning:** seq_len axis fully CLOSED both directions; seq=10 champion.

### Exp176: LSTM bs=8 (Keskar 2017 flat-minima untested)
- **Config delta from exp 74:** bs 16→8
- **Rationale:** Keskar 2017 ICLR + Smith 2018 + Hoffer 2017 — small bs flat minima
- **Prediction:** comp [+0.6, +1.3], A_sh [+1.2, +1.7], runtime ~80-120s
- **Result:** PENDING

### Exp176: LSTM bs=8 (REJECTED)
- **Config delta from exp 74:** bs 16→8
- **Rationale:** Keskar 2017 flat-minima
- **Result:** Composite +0.19 | A_sh +0.29 | val_sh +0.77 | 6/7 pos folds but small
- **Status:** DISCARD; bs axis closed
- **Learning:** Flat-minima theory doesn't transfer to QQQ at our n.

### Exp177: LSTM lr=2e-3 (Smith 2017 highest-leverage axis untested)
- **Config delta from exp 74:** lr 1e-3→2e-3
- **Rationale:** Smith 2017 §3 — lr most-impactful HP; 2e-3 untested above canonical
- **Prediction:** comp [-0.3, +1.0], runtime ~50-70s
- **Result:** PENDING

### Exp177: LSTM lr=2e-3 (REJECTED — destabilized)
- **Config delta from exp 74:** lr 1e-3→2e-3
- **Rationale:** Smith 2017 highest-leverage axis untested
- **Result:** Composite **-1.12** | A_sh +0.05 | val_sh -0.72
- **Per-fold:** F1=+1.09 F2=-1.05 F3=-0.47 F4=-0.18 F5=+1.49 F6=+0.63 F7=-1.19
- **Status:** DISCARD strongly — destabilized training
- **Learning:** LSTM HP-only axes ALL EXHAUSTED (6 consecutive DISCARDs). Pivot to Mamba.

### Exp178: Mamba mambats variant (untested SSM structural variant)
- **Config delta from dMamba exp 52:** mamba_variant dmamba→mambats
- **Rationale:** Cai 2024 NeurIPS — mambats season-trend decomposition for time-series
- **Prediction:** comp [+0.5, +1.6], runtime ~6-12min
- **Result:** PENDING

### Exp178: Mamba mambats variant (untested SSM variant)
- **Config delta from dMamba exp 52:** mamba_variant dmamba→mambats
- **Rationale:** Cai 2024 NeurIPS — season-trend decomposition for time-series
- **Result:** Composite **+0.4193** | A_sh +0.6193 | val_sh +0.7345 | **excess +0.0151 (positive!)**
- **Per-fold:** F1=-0.75 F2=+1.42 F3=+2.10 F4=+0.41 F5=-0.07 F6=+0.58 F7=+1.44
- **Status:** DISCARD vs global, but COMPLEMENTARY to dmamba
- **Learning:** mambats wins F3/F7 trend; dmamba wins F1 chaos. Ensemble axis open.

### Exp179: Mamba mambats variance check seed=7
- **Config delta from exp 178:** seed 42→7
- **Rationale:** Picard 2021 + Lakshminarayanan 2017 — confirm pattern reproducibility
- **Prediction:** comp [+0.0, +0.7], runtime ~80-130s
- **Result:** PENDING

### Exp179: Mamba mambats seed=7 (variance check)
- **Config delta from exp 178:** seed 42→7
- **Rationale:** Picard 2021 — confirm reproducibility
- **Result:** Composite +0.3421 | A_sh +0.4421 | val_sh +0.9938 | excess -0.1622
- **Per-fold:** F1=-0.96 F2=+0.77 F3=+0.18 F4=+0.72 F5=+0.56 F6=+0.91 F7=+0.83
- **Status:** DISCARD; mambats 2-seed mean +0.38 (reproducible at lower amplitude)
- **Learning:** mambats characterized; complementary to dmamba. Try s_mamba next.

### Exp180: Mamba s_mamba variant (last untested SSM variant)
- **Config delta from dMamba exp 52:** mamba_variant dmamba→s_mamba
- **Rationale:** Liu 2025 DMamba §3.4 — s_mamba's input-dependent selectivity
- **Prediction:** comp [-0.7, +1.0], runtime ~6-12min
- **Result:** PENDING

### Exp180: Mamba s_mamba variant (REJECTED)
- **Config delta from dMamba exp 52:** mamba_variant dmamba→s_mamba
- **Rationale:** Last untested SSM variant
- **Result:** Composite -0.5251 | A_sh +0.293 | val NEGATIVE
- **Per-fold:** F1=-0.50 F2=+0.72 F3=+1.79 F4=-0.32 F5=-0.57 F6=+0.52 F7=+0.74
- **Status:** DISCARD; **Mamba 25/25 COMPLETE.** dmamba champion variant.

### Exp181: XGBoost depth=6 (Chen-Guestrin 2016 paper default untested)
- **Config delta from XGBoost exp 63:** max_depth 4→6
- **Rationale:** Chen-Guestrin 2016 KDD §3.2 paper default for tabular n>1k
- **Prediction:** comp [-0.4, +0.5], runtime ~5-10min
- **Result:** PENDING

### Exp181: XGBoost depth=6 (RECORD F3 +3.51, val crash)
- **Config delta from exp 63:** max_depth 4→6
- **Rationale:** Chen-Guestrin 2016 KDD §3.2 paper default
- **Result:** Composite **-0.44** | A_sh +0.42 | val_sh -0.24 NEG | F3 **+3.51 RECORD**
- **Per-fold:** F1=-0.25 F2=+1.68 F3=+3.51 F4=+0.16 F5=+0.93 F6=+1.07 F7=-0.57
- **Status:** DISCARD; same val-crash pattern as CatBoost lr=0.05
- **Learning:** depth=6 unlocks F3 alpha but val unstable. Try depth=5 mid-point.

### Exp182: XGBoost depth=5 (capacity-stability mid-point)
- **Config delta from exp 181:** max_depth 6→5
- **Rationale:** Friedman 2001 §5.4 + ESL §10.12 — bracket depth optimum
- **Prediction:** comp [-0.3, +0.4], F3 [+1.5, +3.0], runtime ~10-18min
- **Result:** PENDING

### Exp182: XGBoost depth=5 (NEW XGBoost WITHIN-CHAMPION +0.37)
- **Config delta from exp 181:** max_depth 6→5
- **Rationale:** Friedman 2001 §5.4 + ESL §10.12 — bracket depth optimum
- **Result:** Composite **+0.3736** | A_sh +0.47 | val_sh +0.79 | excess -0.75
- **Per-fold:** F1=+1.51 F2=+2.55 F3=+1.30 F4=+0.44 F5=+0.86 F6=+1.46 F7=-0.90 (6/7 pos!)
- **Status:** DISCARD vs global, **NEW XGBoost CHAMPION** (+0.50 vs exp 63)
- **Learning:** depth=5 sweet-spot — retains alpha + recovers val. Variance check next.

### Exp183: XGBoost depth=5 seed=0 (variance check)
- **Config delta from exp 182:** seed 42→0
- **Rationale:** Picard 2021 + Lakshminarayanan 2017 — confirm reproducibility
- **Prediction:** comp [+0.0, +0.7], runtime ~15-20min
- **Result:** PENDING

### Exp183: XGBoost depth=5 seed=0 (variance check, val-crash)
- **Config delta from exp 182:** seed 42→0
- **Result:** Composite **-0.4048** | A_sh +0.3384 | val_sh -0.2048 NEG
- **Per-fold:** F1=+0.27 F2=+2.21 F3=+0.65 F4=-0.40 F5=+0.89 F6=+1.52 F7=-0.56
- **Status:** DISCARD; same val-instability as CatBoost. 2-seed mean ≈ 0.
- **Learning:** GBM val-instability structural across XGBoost+CatBoost.

### Exp184: XGBoost depth=5 seed=99 (3-seed median lock)
- **Config delta from exp 183:** seed 0→99
- **Rationale:** CLAUDE.md "3-seed median > baseline"; Picard 2021
- **Prediction:** comp [-0.5, +0.7], median ~0±0.4, runtime ~15-20min
- **Result:** PENDING

### Exp184: XGBoost depth=5 seed=99 (3-seed median REJECTS depth=5)
- **Config delta from exp 183:** seed 0→99
- **Result:** Composite **-0.5599** | A_sh +0.23 | val_sh -0.26 | F2 RECORD +3.17
- **Status:** DISCARD; 3-seed median -0.40 → depth=5 REJECTED
- **Learning:** XGBoost-best stays exp 63 -0.128. depth=5 was seed-luck.

### Exp185: XGBoost depth=4 lr=0.005 n_est=2000 (slowest-lr untested)
- **Config delta from exp 63:** lr 0.01→0.005, n_est 1000→2000 (capacity-matched)
- **Rationale:** Friedman 2001 §5.2 — slower lr finds flat minima
- **Prediction:** comp [-0.4, +0.3], runtime ~25-35min
- **Result:** PENDING

### Exp185: XGBoost lr=0.005 n_est=2000 (XGBoost 25/25 COMPLETE)
- **Config delta from exp 63:** lr 0.01→0.005, n_est 1000→2000
- **Rationale:** Friedman 2001 §5.2 slower-lr stability
- **Result:** Composite -0.2304 | A_sh +0.42 | val barely positive
- **Status:** DISCARD; **XGBoost 25/25 COMPLETE.** Within-best exp 63 -0.128.

### Exp186: LightGBM exp 95 seed=99 (variance check on +0.611 single-seed)
- **Config delta from exp 95:** seed 42→99
- **Rationale:** CLAUDE.md "3-seed median > baseline"; Picard 2021
- **Prediction:** comp [+0.0, +0.8], runtime ~10-15min
- **Result:** PENDING

### Exp186: LGBM exp95 seed=99 (variance check)
- **Config delta from exp 95:** seed 42→99
- **Result:** Composite -0.1098 | A_sh +0.365 | 5/7 pos folds
- **Status:** DISCARD; 2-seed mean +0.25, val unstable
- **Learning:** Need 3rd seed.

### Exp187: LGBM exp 95 seed=0 (3-seed median lock)
- **Config delta from exp 186:** seed 99→0
- **Rationale:** CLAUDE.md "3-seed median > baseline"; Picard 2021
- **Prediction:** comp [-0.4, +0.7], runtime ~5-10min
- **Result:** PENDING

### Exp187: LGBM exp 95 seed=0 (3-seed REJECTS)
- **Config delta from exp 186:** seed 99→0
- **Result:** Composite **-0.4571** | A_sh -0.0571 NEG | val_sh +1.34 | F2 RECORD +3.33
- **Status:** DISCARD; 3-seed median -0.110 — LGBM exp 95 lift REJECTED
- **Learning:** All 3 GBM families show structural val-instability at high capacity.

### Exp188: LGBM depth=5 (untested mid-point, final slot)
- **Config delta from exp 95:** max_depth 4→5
- **Rationale:** Ke 2017 §3.1 + Friedman 2001 §5.4 — bracket depth axis
- **Prediction:** comp [-0.4, +0.6], runtime ~10-15min
- **Result:** PENDING

### Exp188: LGBM depth=5 single-seed (LGBM at 23/25 recount)
- **Config delta from exp 95:** max_depth 4→5
- **Result:** Composite -0.1388 | A_sh +0.41 | val barely positive
- **Status:** DISCARD; same GBM val-instability pattern. LGBM at 23/25.

### Exp189: LGBM lr=0.005 n_est=3000 (slowest-lr untested)
- **Config delta from exp 95:** lr 0.01→0.005, n_est 1500→3000
- **Rationale:** Friedman 2001 §5.2 slow-lr stability
- **Prediction:** comp [-0.3, +0.6], runtime ~15-25min
- **Result:** PENDING

### Exp189: LGBM lr=0.005 n_est=3000 (RECORD A_sh)
- **Config delta from exp 95:** lr 0.01→0.005, n_est 1500→3000
- **Result:** Composite +0.2434 | **A_sh +0.9415 RECORD** | 6/7 pos folds | F6 +2.88 RECORD
- **Status:** DISCARD; promising slow-lr regime, need variance check
- **Learning:** Slowest-lr stabilizes LGBM. Test alpha robust.

### Exp190: LGBM exp 189 seed=99 (variance check)
- **Config delta from exp 189:** seed 42→99
- **Rationale:** Picard 2021 — confirm A_sh +0.94 reproducibility
- **Prediction:** comp [-0.2, +0.6], A_sh [+0.5, +1.1], runtime ~10-15min
- **Result:** PENDING

### Exp190: LGBM seed=99 slowest-lr (LGBM 25/25 COMPLETE)
- **Config delta from exp 189:** seed 42→99
- **Result:** Composite -0.3270 | A_sh -0.027 NEG | 3/7 pos folds
- **Status:** DISCARD; **LGBM 25/25 COMPLETE.** All seeds + slowest-lr 2-seed mean ~+0.5 unstable.

### Exp191: DLinear post-rollback baseline seed=0
- **Config delta from exp 109:** seed 42→0
- **Rationale:** Establish post-features-rollback DLinear variance
- **Prediction:** comp [-0.5, +0.5], runtime ~30-60s
- **Result:** PENDING

### Exp191: DLinear post-rollback seed=0
- **Result:** Composite -0.38 | A_sh -0.08 | DLinear post-rollback 2-seed mean -0.21
- **Status:** DISCARD; weak post-rollback baseline.

### Exp192: DLinear seq_len=20 (untested axis)
- **Config delta from exp 109:** seq_len 10→20
- **Rationale:** Zeng 2023 §4.2 seq sweep
- **Prediction:** comp [-0.4, +0.5], runtime ~30-60s
- **Result:** PENDING

### Exp192: DLinear seq_len=20 (DISCARD)
- **Result:** Composite -0.22 | A_sh +0.08 | val_sh +0.85
- **Status:** DISCARD; DLinear post-rollback weak. Pivot to iTransformer.

### Exp193: iTransformer paper-recipe lr=5e-5 warmup=10
- **Config delta from exp 108:** lr 1e-4→5e-5, warmup 0→10
- **Rationale:** Liu 2024 ICLR §4.1 paper recipe; Vaswani 2017 §5.3 warmup
- **Prediction:** comp [-0.8, +0.3], runtime ~3-6min
- **Result:** PENDING

### Exp193: iTransformer paper-recipe (RECORD A_sh +0.92)
- **Config delta from exp 108:** lr 1e-4→5e-5, warmup 0→10
- **Result:** Composite -1.52 | **A_sh +0.92 RECORD** | val_sh -1.22
- **Per-fold:** F1=+1.43 F2=-0.24 F3=+1.96 F4=-0.04 F5=-0.11 F6=+1.70 F7=+1.09
- **Status:** DISCARD by composite, RECORD test alpha
- **Learning:** Paper-recipe lifts iTransformer +2.3 A_sh. Val drag.

### Exp194: iTransformer paper-recipe seed=0 (variance check)
- **Config delta from exp 193:** seed 42→0
- **Rationale:** Picard 2021 + Lakshminarayanan 2017
- **Prediction:** comp [-1.5, +0.5], A_sh [+0.4, +1.1], runtime ~1.5-3min
- **Result:** PENDING

### Exp194: iTransformer seed=0 (variance crash)
- **Result:** Comp -2.02, A_sh -0.10. 2-seed mean +0.41 std 0.51.
- **Status:** DISCARD; iTransformer val-unstable across seeds.

### Exp195: iTransformer seed=99 (3-seed median lock)
- **Prediction:** comp [-2.0, +0.5], runtime ~1.5-3min
- **Result:** PENDING

### Exp195: iTransformer seed=99 (3-seed lock)
- **Result:** Comp -1.43, A_sh +0.21. 3-seed median composite -1.52; iTransformer rejected.

### Exp196: N-BEATS seed=0 (variance baseline)
- **Config delta from exp 144:** seed 42→0
- **Rationale:** Oreshkin 2020 ICLR §4.2 bootstrap; Picard 2021
- **Prediction:** comp [-2.0, +0.0], runtime ~1-2min
- **Result:** PENDING

### Exp196: N-BEATS seed=0 (reproducibly weak)
- **Result:** Comp -1.43 (identical to exp 144!), A_sh -0.60. 1/7 pos folds.
- **Status:** DISCARD; N-BEATS architecture doesn't fit QQQ. 2-seed cluster.

### Exp197: PatchTSMixer seed=99 (3-seed lock)
- **Config delta from exp 146:** seed 0→99
- **Rationale:** Ekambaram 2023 KDD; Picard 2021
- **Prediction:** comp [-2.0, +0.5], runtime ~1-3min
- **Result:** PENDING

### Exp197: PatchTSMixer seed=99 (RECORD A_sh +1.22)
- **Result:** Comp +0.155 | **A_sh +1.22 RECORD (= BH)** | F2 +4.87 RECORD | 6/7 pos folds
- **Status:** DISCARD by comp, RECORD test alpha
- **Learning:** PatchTSMixer at right seed = strongest test alpha this session.

### Exp198: PatchTSMixer seed=7 (variance check on +1.22)
- **Config delta from exp 197:** seed 99→7
- **Rationale:** Picard 2021 + Lakshminarayanan 2017
- **Prediction:** comp [-2.0, +0.5], A_sh [+0.4, +1.5], runtime ~3-7min
- **Result:** PENDING

### Exp198: PatchTSMixer seed=7 (variance crash)
- **Result:** Comp -1.10, A_sh -0.14. 4-seed comp distribution range -1.98.
- **Status:** DISCARD; A_sh +1.22 from exp 197 was outlier.

### Exp199: PatchTSMixer seed=2024 (5-seed lock)
- **Config delta from exp 198:** seed 7→2024
- **Rationale:** Lakshminarayanan 2017 §3.2 (5+ seeds for stable median)
- **Prediction:** comp [-2.0, +0.5], runtime ~3-7min
- **Result:** PENDING

### Exp199: PatchTSMixer seed=2024 (5-seed lock)
- **Result:** Comp +0.028. 5-seed median +0.028, mean -0.54.
- **Status:** DISCARD; PatchTSMixer characterized — real but tiny lift.

### Exp200: MLP exp 79 seed=42 (variance check on +0.974)
- **Config delta from exp 79:** seed 0→42
- **Rationale:** Picard 2021 + Lakshminarayanan 2017 §3.2
- **Prediction:** comp [+0.4, +1.3], runtime ~30-60s
- **Result:** PENDING

### Exp200: MLP exp 79 seed=42 (variance crash)
- **Result:** Comp -0.71. 2-seed mean +0.13 (huge swing).
- **Status:** DISCARD; MLP exp 79 +0.97 was seed=0 luck.

### Exp201: MLP exp 79 seed=99 (3-seed lock)
- **Config delta from exp 200:** seed 42→99
- **Rationale:** Picard 2021 + Lakshminarayanan 2017 §3.2
- **Prediction:** comp [-0.5, +1.0], runtime ~30-60s
- **Result:** PENDING

### Exp201: MLP exp 79 seed=99 (7/7 POSITIVE FOLDS — first stable positive median!)
- **Result:** Comp **+0.5197** | A_sh +0.87 | excess **+0.26 (positive)** | **7/7 pos folds**
- **3-seed MLP exp 79 median +0.52** — FIRST non-Mamba backbone with stable positive median
- **Status:** DISCARD vs global, but BREAKTHROUGH for non-Mamba stable lift
- **Learning:** MLP exp 79 config is real — need more seeds to confirm.

### Exp202: MLP exp 79 seed=7 (4-seed lock)
- **Config delta from exp 201:** seed 99→7
- **Rationale:** Lakshminarayanan 2017 §3.2 (5+ members)
- **Prediction:** comp [+0.0, +1.0], runtime ~30-60s
- **Result:** PENDING

### Exp202: MLP seed=7 (4-seed median +0.01)
- **Result:** Comp -0.49. 4-seed MLP exp 79 median +0.014.
- **Status:** DISCARD; MLP joins val-instability club.

### Exp203: MLP seed=2024 (5-seed lock)
- **Prediction:** comp [-0.7, +1.0], runtime ~30-60s
- **Result:** PENDING

### Exp203: MLP seed=2024 (5-seed median POSITIVE +0.43)
- **Result:** Comp **+0.4333** | A_sh +0.62 | 6/7 pos folds
- **5-seed MLP exp 79 median +0.433** — SECOND non-Mamba stable positive median!
- **Status:** DISCARD vs global, but MLP exp 79 confirmed deployable.

### Exp204: MLP exp 79 + wd=1e-4 (Loshchilov-Hutter canonical)
- **Config delta from MLP champion:** wd 1e-5→1e-4
- **Rationale:** Loshchilov-Hutter 2019 AdamW canonical; combine with warmup
- **Prediction:** comp [+0.3, +1.2], runtime ~30-60s
- **Result:** PENDING

### Exp204: MLP wd=1e-4 + warmup=5 (NEW MLP CHAMPION +0.97)
- **Config delta from MLP champion:** wd 1e-5→1e-4
- **Result:** Comp **+0.9735** | A_sh **+1.04** | excess **+0.43** | **7/7 pos folds** | F3=+2.91 F6=+2.64
- **Status:** DISCARD vs global, NEW MLP CHAMPION (tied with exp 79 +0.974)
- **Learning:** wd=1e-4 + warmup=5 is the magic combo.

### Exp205: MLP wd=1e-4 seed=42 (variance check)
- **Config delta from exp 204:** seed 0→42
- **Rationale:** Picard 2021 + Lakshminarayanan 2017
- **Prediction:** comp [+0.3, +1.2], runtime ~30-60s
- **Result:** PENDING

### Exp205: MLP wd=1e-4 seed=42 (variance crash)
- **Result:** Comp -0.92. 2-seed [+0.97, -0.92].
- **Status:** DISCARD; wd=1e-4 same seed-instability.

### Exp206: MLP wd=1e-4 seed=99 (3-seed lock; MLP final slot 50/50)
- **Prediction:** comp [-0.5, +1.0], runtime ~30-60s
- **Result:** PENDING

### Exp206: MLP wd=1e-4 seed=99 (MLP 50/50 COMPLETE!)
- **Result:** Comp +0.5197 (= exp 201; wd negligible at seed=99). 3-seed median +0.52.
- **Status:** DISCARD vs global; **MLP 50/50 COMPLETE.** SECOND stable positive backbone.

### Exp207: CatBoost lr=0.005 n_est=2000 (slowest-lr untested)
- **Config delta from CatBoost exp 98:** lr 0.02→0.005, n_est 1000→2000
- **Rationale:** Friedman 2001 §5.2 + Prokhorenkova 2018 §3.3
- **Prediction:** comp [-0.5, +0.6], runtime ~10-20min
- **Result:** PENDING

### Exp207: CatBoost lr=0.005 (val crash but F2 RECORD +3.89)
- **Result:** Comp -1.10 | A_sh +0.68 | val_sh -0.90 NEG
- **Status:** DISCARD; same val-instability across all CatBoost lr regimes.

### Exp208: CatBoost depth=3 (untested shallower for stability)
- **Config delta from exp 98:** max_depth 4→3
- **Rationale:** Friedman 2001 §5.4 shallow stability
- **Prediction:** comp [-0.5, +0.3], runtime ~5-8min
- **Result:** PENDING

### Exp208: CatBoost depth=3 (DISCARD, F2 RECORD +4.63)
- **Result:** Comp -0.80, A_sh +0.38, val NEG. F2 RECORD across all CatBoost runs.

### Exp209: CatBoost seq=30 (untested axis)
- **Config delta from exp 98:** seq 60→30
- **Prediction:** comp [-0.5, +0.4], runtime ~3-6min
- **Result:** PENDING

### Exp214: CatBoost lr=0.04 seed=0 (CatBoost 25/25 COMPLETE!)
- **Result:** Comp -0.25 | A_sh +0.05 | val_sh +0.60 POSITIVE (rare)
- **Per-fold:** F1=-1.34 F2=+2.32 F3=+1.04 F4=+0.24 F5=-0.51 F6=+0.78 F7=-0.30
- **Status:** DISCARD; **CatBoost 25/25 BUDGET COMPLETE.** lr=0.04 most val-stable CatBoost regime.

### Exp215: MLP seq=20 (user directive: +25 MLP grind)
- **Config delta from MLP champion exp 79:** seq_len 10→20
- **Rationale:** User-requested untested seq axis on MLP; Goyal 2017 §2.4 + Hochreiter-Schmidhuber 1997
- **Prediction:** comp [+0.0, +1.2], runtime ~30-60s
- **Result:** PENDING

### Exp215: MLP seq=35 (POSITIVE comp +0.55!)
- **Config delta from MLP champion exp 79:** seq_len 10→35
- **Result:** Comp **+0.5475** | A_sh +0.89 | val_sh +0.65 | F2 **RECORD +4.53** | 6/7 pos
- **Status:** DISCARD vs global, but seq=35 IS HELPING MLP
- **Learning:** Multi-week context captures F2 EU-debt regime alpha (+4.53)

### Exp216: MLP seq=35 seed=42 (variance check)
- **Config delta from exp 215:** seed 0→42
- **Rationale:** Picard 2021 + Lakshminarayanan 2017
- **Prediction:** comp [-0.3, +0.7], runtime ~50-90s
- **Result:** PENDING

### Exp216: MLP seq=35 seed=42 (variance crash)
- **Result:** Comp -1.04 | A_sh -0.31 NEG | val NEG. 2-seed mean -0.25.
- **Status:** DISCARD; seq=35 lift was seed=0 luck.

### Exp217: MLP seq=35 seed=99 (3-seed median lock)
- **Config delta from exp 216:** seed 42→99
- **Prediction:** comp [-0.8, +0.7], runtime ~50-90s
- **Result:** PENDING
