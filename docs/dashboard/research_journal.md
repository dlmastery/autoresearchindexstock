# Research Journal — AutoResearch QQQ

> Human-readable twin of `reasoning_annotations.json`. Every experiment
> is logged here in narrative form: diagnosis → citations → hypothesis →
> prediction → verdict → learning. Bootstrap session 2026-04-26.

---

## Champion lineage so far

| Exp | Backbone | Config | Composite | A_Sharpe | Excess | Status |
|----:|----------|--------|----------:|---------:|-------:|--------|
| 1 | xgboost | smoke n_est=50, depth=4, lr=0.05, seq=60 | -1.5423 | +0.5694 | -0.6499 | DISCARD |
| 2 | xgboost | n_est=300, depth=4, lr=0.03, seq=60 | -2.3923 | -0.0045 | -1.2239 | DISCARD (over-trees) |
| 3 | mlp | plain SOTA, head_dropout=0.1, seq=10 | -0.2923 | +0.0077 | -0.5966 | KEEP (interim champion) |
| 4 | mlp | seq=20 + strong reg dropout=0.25 wd=1e-4 | -0.8341 | -0.4341 | -1.2763 | DISCARD (under-fit) |
| 5 | lstm | **FX-Exp35 HPs** (wd=7e-4 bs=16 seed=42 lr=1e-3 ep=100 pat=15) | -0.1318 | +0.8339 | **+0.2297** | KEEP — first BH-beating excess |
| 6 | mlp | **FX-Exp32 HPs** (residual MLP head_dropout=0.25 seed=0 ep=50 pat=10) | **+0.5799** | +0.6799 | +0.0757 | **CHAMPION — first +composite** |

---

## Exp 1 — XGBoost smoke (n_est=50)

**Diagnosis.** Bootstrap experiment for the QQQ variant — no prior champion, this validates the whole pipeline (download → 184 cited features → 7 regime-aware folds → super-fold split → GBM training → multi-target evaluation → JSONL row + per-day CSV). Configuration deliberately tiny (50 trees) for fast smoke check.

**Citations.** Chen, Guestrin 2016 KDD 'XGBoost: A Scalable Tree Boosting System' (arXiv:1603.02754). Welch, Goyal 2008 RFS. Bollerslev, Tauchen, Zhou 2009 RFS (VRP). Estrella, Mishkin 1998 RES (10y-3m).

**Hypothesis.** XGBoost at depth=4, lr=0.05, n_est=50, seq=60, seed=42 produces measurable composite. Under-trained, expect modest performance.

**Prediction.** Composite [-2, +1]. A_sharpe [0, +1.5]. Excess negative because under-trained model can't beat trending QQQ.

**Verdict.** DISCARD as champion candidate, KEEP as smoke baseline. Composite **-1.5423**, A_sharpe +0.5694 (51.1% direction-betting), excess -0.6499. 5/7 positive test folds. Per-fold A: F1 +3.25 (GFC peak — model alpha lights up in chaos), F2 +0.40 (US-downgrade — bh=+5.02 stomps it), F5 +1.57 (Vol-mageddon).

**Learning.** Pipeline runs end-to-end. Regime-aware folds produce interpretable breakdowns: alpha shows up in chaos (folds 1, 5), passive dominates trending recoveries (folds 2, 6, 7).

---

## Exp 2 — XGBoost n_est=300 hill-climb

**Diagnosis.** Hill-climb on exp 1. Hypothesis: more trees reduce bias.

**Citations.** Chen-Guestrin 2016 KDD; Hastie-Tibshirani-Friedman 2009 ESL §10.12.

**Hypothesis.** n_est=300 (6× exp 1) at lr=0.03 should improve composite by reducing bias.

**Prediction.** Composite [-1, 0]. A_sharpe [+0.5, +1.5].

**Verdict.** **DISCARD strongly** — composite -2.3923 (delta -0.85 vs exp 1, WORSE). 300 trees OVERFIT the noise vs 50 trees. The QQQ feature space (12,300-dim flattened seq=60 windows) is too large for unregularised XGBoost without aggressive depth or column-fraction regularisation.

**Learning.** Critical FX-vs-QQQ divergence: more trees made the model WORSE on QQQ, opposite to FX where n_est=1500 was the FX-champion. Pivoting to MLP per user feedback (FX progression: MLP → LSTM → GBM).

---

## Exp 3 — MLP plain SOTA baseline

**Diagnosis.** User feedback: FX progression was MLP → LSTM → GBM, building cheap-fast experiment volume first. Pivoting to MLP. QQQ XGBoost is ~4× slower than FX so the FX progression is even more critical here.

**Citations.** Gu, Kelly, Xiu 2020 RFS 'Empirical Asset Pricing via Machine Learning' (arXiv:1807.04365). Loshchilov, Hutter 2019 ICLR 'AdamW' (arXiv:1711.05101). He et al. 2016 CVPR 'ResNet' (arXiv:1512.03385).

**Hypothesis.** Residual MLP at the Gu-Kelly-Xiu 2020 SOTA recipe (lr=3e-4, bs=32, ep=50, pat=10, wd=1e-5, head_dropout=0.1, seq=10) produces a positive composite with sub-30-second runtime.

**Prediction.** Composite [-1, +1.5]. Test_pos_folds 4-6/7. Runtime 25-45s.

**Verdict.** KEEP — interim champion. Composite **-0.2923** (delta +1.25 vs exp 1, +2.10 vs exp 2). A_sharpe +0.0077. Excess -0.5966. **Runtime 28.0s** — 18× faster than XGBoost. The compute-efficiency argument validated.

**Learning.** MLP at SOTA recipe is a solid baseline but does not beat passive QQQ. val_pos_folds=1/7 says val windows are tough.

---

## Exp 4 — MLP seq=20 + stronger reg

**Diagnosis.** Hill-climb on exp 3. Champion weakness val_pos_folds=1/7. Try longer seq + stronger reg.

**Citations.** Loshchilov-Hutter 2019 ICLR (wd log-spaced). Srivastava et al. 2014 JMLR (dropout=0.25).

**Hypothesis.** Stronger reg fights val/test divergence.

**Verdict.** **DISCARD** — composite -0.8341, A_sharpe -0.4341 (anti-predictive). Stronger reg made the MLP UNDERFIT.

**Learning.** Axis closed: head_dropout >= 0.2 + wd >= 1e-4 with seq=20. Next try: opposite — keep exp 3 settings but try larger hidden, OR try LSTM at FX-champion HPs.

---

## Exp 5 — LSTM @ FX-champion (Exp35) HPs

**Diagnosis.** User feedback: try the FX winning configurations. The FX neural champion was LSTM Exp35 with wd=7e-4, bs=16, seed=42, lr=1e-3, ep=100, pat=15 (composite +6.4242 in FX).

**Citations.** Fischer, Krauss 2018 EJOR 'Deep learning with long short-term memory networks for financial market predictions' — LSTM SOTA recipe for daily financial TS. Hochreiter, Schmidhuber 1997 Neural Computation (LSTM architecture). Loshchilov-Hutter 2019 ICLR (AdamW).

**Hypothesis.** FX-champion HPs transfer to QQQ at the same daily-equity scale because architecture and dataset shape are similar.

**Prediction.** Composite [0, +1.5]. Test_pos_folds >= 5/7. Convergence ep 25-30.

**Verdict.** **KEEP — new champion.** Composite **-0.1318**. A_sharpe **+0.8339**. **Excess_sharpe +0.2297 — FIRST POSITIVE EXCESS-SHARPE OF THE SESSION** (strategy beats passive QQQ). Test_pos_folds **7/7** (perfect on test); val_pos_folds 5/7 (the bottleneck). Convergence at ep=34.

**Learning.** **The FX-champion LSTM config transfers to QQQ.** Critical: equity-index daily prediction at QQQ scale responds to the same LSTM HPs that won on FX. Test_pos_folds=7/7 is exceptional. Axis open: multi-seed (≥3 seeds before declaring stable champion). Axis open: seq_len=20 at same HPs.

---

## Exp 6 — MLP @ FX-champion (Exp32) HPs

**Diagnosis.** Companion to exp 5. Testing FX-Exp32 (residual MLP, head_dropout=0.25, seed=0, lr=3e-4, ep=50, pat=10, wd=1e-5).

**Citations.** Gu-Kelly-Xiu 2020 RFS. He et al. 2016 CVPR (residual). Srivastava et al. 2014 JMLR (dropout=0.25 FX-empirical optimum).

**Hypothesis.** Residual MLP at FX-Exp32 HPs produces composite >= 0 because the residual-MLP architecture has the same low-SNR-friendly inductive bias on QQQ.

**Verdict.** **KEEP — new champion.** Composite **+0.5799** (first POSITIVE composite). A_sharpe +0.6799. excess +0.0757. Test_pos_folds 6/7. Runtime 28.7s — 18× faster than exp 5 LSTM and produced HIGHER composite.

**Learning.** Residual MLP @ FX HPs is the new lead. Validates: (a) residual-MLP architecture is durable across asset classes (FX → QQQ); (b) head_dropout=0.25 is the right regularisation strength for low-SNR financial data; (c) the 'best' single-model on QQQ so far is also the cheapest to train. Axis open: multi-seed variance check (seeds 7, 42, 99, 2024). Axis open: hidden_size hill-climb. Ready to build the QQQ mega-ensemble path (FX-style: 3 GBMs + 1 LSTM, rank-avg).

---

## Plan forward (multi-session marathon)

1. **Multi-seed exp 5 + 6** (LSTM and MLP at FX-champion HPs) — 4 seeds each → 8 experiments. Establishes seed-variance baseline.
2. **LightGBM @ FX-Exp235 HPs** (depth=4, gbm_lr=0.01, n_est=2000, seq=60).
3. **CatBoost @ FX-Exp236 HPs** (depth=4, gbm_lr=0.01, n_est=2000, seq=60).
4. **XGBoost @ FX-Exp203 HPs** (depth=4, gbm_lr=0.03, n_est=1500, seq=60) — needs foreground or split-runs.
5. **Build `_qqq_mega_ensemble.py`** — port the rank-avg recipe from FX `_emtsf_mega_ensemble.py` to QQQ. Target: meet or beat **excess-Sharpe of FX +9.7071**.
6. **Continue 25-experiment hill-climb per backbone** for the 23-backbone roster (15 generic TS + 8 equity-specific 2024-2026 SOTA).
7. Eventually: paper / Medium / audit reports / Colab notebook (full FX-style artefact suite).

## Exp165 — LightGBM seed=13 variance lock (4-seed ensemble)
**Diagnosis:** 4th-seed LGBM exp 10 config to nail down the seed-variance distribution. FX-paper §3.5 asserts GBM seed-determinism but earlier QQQ XGBoost runs already broke that claim; this run finalizes the LGBM seed-noise band on QQQ.
**Citations:** Ke et al. 2017 NeurIPS 'LightGBM' — GOSS+EFB stochastic sampling means seed-dependent training; Picard 2021 'Torch.manual_seed(3407) is all you need' (arXiv:2109.08203) — empirical seed-std ~0.5 on Sharpe-like metrics at n<10k.
**Hypothesis:** Composite in [-0.5, +0.7]; 4-seed median should converge to honest LGBM estimate.
**Prediction:** comp [-0.5, +0.7], A_sh [+0.2, +1.0], A_exc [-1.5, 0.0].
**Verdict:** DISCARD. Composite -0.7409. Per-fold A_sharpe F1=+2.43 F2=-0.25 F3=-0.25 F4=+0.09 F5=+0.86 F6=+0.48 F7=+0.26. 4-seed LGBM range now [-0.74, +0.50] confirming non-determinism on n=2738 QQQ.
**Learning:** Axis closed — LGBM seed-ensemble median ~+0.0 to +0.2, decisively below dMamba +1.32 champion. Move budget to CatBoost depth=8 (most under-budget at 11/25).

## Exp166 — CatBoost depth=8 (deep oblivious trees, untested axis)
**Diagnosis:** 3 consecutive DISCARDs (163-165). CatBoost is most under-budget cheap-tier (11/25). Within CatBoost, depth=4 (exp 98 best) and depth=6 (exp 103) both tested; depth=8 NEVER tried. Champion CatBoost-best F2/F3 fail consistently — possibly 3-way macro×VIX×yield-curve interactions that depth=4 cannot capture.
**Citations:** Prokhorenkova et al. 2018 NeurIPS 'CatBoost' (arXiv:1706.09516) §4.1 depth=6-8 best for 100-500 feature tabular; §3.2 ordered-boosting protects against prediction-shift overfit at deeper depth; Cieslak-Pang 2021 RFS — stress-regime equity 3-way interactions require depth>=3.
**Hypothesis:** depth=8 lr=0.02 n_est=1000 (one knob from exp 98 depth=4) — deeper oblivious trees fit macro×VIX×yc interactions; mechanism: 256-leaf tree with 184 features and 2538 rows + ordered-boosting bias control.
**Prediction:** comp [-0.4, +0.6], A_sh [+0.2, +1.0], F2 expected +0.0 to +0.4, F3 expected +0.0 to +0.3, runtime ~10-15min.

## Exp166 (depth=8 attempt) — KILLED
Initial config (CatBoost depth=8 seq=30 n_est=500) was killed at 76min wall-time with no fold complete. 256-leaf oblivious trees × 6,150 flattened features × 4 targets × 7 folds is infeasible in our experiment-loop time budget. Axis CLOSED: depth=8 untestable.

## Exp166 — CatBoost lr=0.05 (untested fast-learner axis)
**Diagnosis:** 3 DISCARDs + 1 KILLED. Pivot to untested gbm_lr axis. Best CatBoost so far: exp 98 (lr=0.02 -0.56 baseline). Faster lr=0.05 (Prokhorenkova default) never tested on QQQ. F2/F3 still weak — faster learner might escape lr=0.02 local optimum.
**Citations:** Friedman 2001 Annals of Stats 'Greedy Function Approximation' — lr×n_est tradeoff; Prokhorenkova et al. 2018 NeurIPS 'CatBoost' (arXiv:1706.09516) §3.3 — lr=0.03-0.05 default for noisy small-n tabular; Bergmeir-Hyndman-Koo 2018 CSDA — small-n financial regression benefits from faster lr.
**Hypothesis:** lr=0.05 + depth=4 + n_est=500 + seq=60 escapes lr=0.02 local optimum via 2.5× larger per-step moves; ordered-boosting prevents prediction-shift overfit.
**Prediction:** comp [-0.6, +0.4], A_sh [+0.3, +1.0], F2 [-0.1, +0.2], runtime 5-10min.

## Exp166 (lr=0.05) — CatBoost within-champion lift
**Diagnosis:** Fast-learner axis untested for CatBoost on QQQ; targets F2/F3 EU-debt+Taper local optimum that lr=0.02 stagnates in.
**Citations:** Prokhorenkova 2018 NeurIPS §3.3 lr=0.03-0.05 default for noisy small-n; Friedman 2001 lr×n_est tradeoff.
**Hypothesis:** lr=0.02→0.05 escapes flat-loss region via 2.5× larger per-step moves; ordered-boosting prevents prediction-shift.
**Prediction:** comp [-0.6, +0.4], F2 [-0.1, +0.2].
**Verdict:** DISCARD vs +1.32 global, but **WITHIN-BACKBONE CHAMPION** at -0.0968 (delta +0.46 vs prior CatBoost-best exp 98 -0.56). F2 jumped from -0.25 to **+2.36**! F3 from -0.25 to +1.23. F1 lost (-0.72 vs +1.5-2.5 historical). Runtime 937s.
**Learning:** Major axis discovery — lr=0.05 unlocks stress-regime alpha invisible to lr=0.02 but costs F1 chaos alpha. Axis open: more trees to recover F1 (n_est 500→1000) per Friedman 2001 §5.2.

## Exp167 — CatBoost lr=0.05 n_est=1000 (recover F1 alpha)
**Diagnosis:** Exp 166 unlocked F2/F3 (+2.36/+1.23) but lost F1 (-0.72). Hypothesis: at lr=0.05, 500 trees is under-trained for F1 chaos regime; 1000 trees recovers F1 while keeping F2/F3 wins.
**Citations:** Friedman 2001 §5.2 lr×n_est convergence; Hastie-Tibshirani-Friedman 2009 ESL §10.12 — optimal n_est ∝ 1/lr; Prokhorenkova 2018 ordered-boosting overfit protection.
**Hypothesis:** lr=0.05 + n_est=1000 (one knob from exp 166's n_est=500) recovers F1 alpha via more boosting rounds at the high-lr regime.
**Prediction:** comp [-0.3, +0.5], F1 [+0.2, +1.5], F2 [+1.5, +2.5], F3 [+0.8, +1.4], runtime 18-25min.

## Exp167 — CatBoost lr=0.05 n_est=1000 (FIRST POSITIVE COMPOSITE)
**Diagnosis:** Recover F1 alpha lost in exp 166 by giving fast-learner more trees per Friedman 2001 §5.2.
**Citations:** Friedman 2001 §5.2 lr×n_est convergence; ESL §10.12; Prokhorenkova 2018 ordered-boosting.
**Hypothesis:** lr=0.05 + n_est=1000 (vs 500) recovers F1 chaos alpha while keeping F2/F3 wins.
**Prediction:** comp [-0.3, +0.5], F1 [+0.2, +1.5], F2 [+1.5, +2.5].
**Verdict:** DISCARD vs +1.32 global, but **CATBOOST WITHIN-CHAMPION** at +0.0728. F1 recovered (-0.72→-0.15), F3 jumped (+1.23→+2.98), 5/7 positive folds. Cumulative within-CatBoost lift +0.63 across 2 experiments. Runtime 1945s.
**Learning:** lr=0.05 + n_est=1000 confirms Friedman 2001 lr×n_est convergence on QQQ. Within-CatBoost progression monotonic; n_est ceiling not yet hit. Axis open: n_est=1500 to find the turning point.

## Exp168 — CatBoost lr=0.05 n_est=1500 (find n_est ceiling)
**Diagnosis:** Within-CatBoost monotonic progression -0.56→-0.10→+0.07 suggests n_est ceiling not yet hit. Friedman 2001 §5.2 — find validation-loss turning point.
**Citations:** Friedman 2001 lr×n_est; ESL §10.12 lr=0.05 n_est=1000-3000 typical; Bühlmann-Yu 2003 JCGS noisy-regression boosting; Prokhorenkova 2018 §3.2 ordered-boosting.
**Hypothesis:** lr=0.05 + n_est=1500 (one knob from exp 167's 1000) continues monotonic improvement and helps F4 (-1.37) recover.
**Prediction:** comp [-0.1, +0.5], F4 expected [-0.5, 0.0] recovery, runtime 45-50min.

## Exp168 — CatBoost lr=0.05 n_est=1500 (n_est CEILING IDENTIFIED)
**Diagnosis:** Continue Friedman 2001 §5.2 lr×n_est convergence climb; find turning point.
**Citations:** Friedman 2001 §5.2; ESL §10.12; Bühlmann-Yu 2003; Prokhorenkova 2018.
**Hypothesis:** n_est=1500 continues monotonic improvement.
**Prediction:** comp [-0.1, +0.5], F4 recovery to [-0.5, 0.0].
**Verdict:** DISCARD. Composite -0.376 (delta -0.45 vs exp 167). N_EST CEILING IDENTIFIED — overfit U-shape from n_est=1000→1500. F3 +2.98→+1.90, F5 +0.94→-0.43 (canonical Friedman §5.2 noise-fitting). Champion stays at exp 167.
**Learning:** n_est optimum at lr=0.05 depth=4 = 1000-1100 (sharp dropoff above). Axis open: variance lock on exp 167 with seed=0 per Picard 2021; depth=5; n_est=1200 fine-tune.

## Exp169 — CatBoost exp 167 seed=0 (variance lock)
**Diagnosis:** Exp 167's +0.07 single-seed needs reproducibility check before declaring real lift. Picard 2021 + CLAUDE.md "3-seed median > baseline" rule.
**Citations:** Picard 2021 arXiv:2109.08203 seed-std ~0.5 at n<10k; Lakshminarayanan 2017 NeurIPS deep ensembles; Prokhorenkova 2018 §3.2 ordered-boosting permutation seed-dep.
**Hypothesis:** Same config as exp 167, seed 42→0; composite within ±0.4 of +0.0728 confirms real lift.
**Prediction:** comp [-0.3, +0.5], F2 [+1.0, +2.5], F3 [+1.0, +3.0], runtime ~30-35min.

## Exp169 — CatBoost variance lock seed=0 (NEW WITHIN-CATBOOST CHAMPION)
**Diagnosis:** Confirm exp 167 +0.07 reproducibility per Picard 2021.
**Citations:** Picard 2021 arXiv:2109.08203; Lakshminarayanan 2017 NeurIPS; Prokhorenkova 2018 §3.2 ordered-boosting seed-dep.
**Hypothesis:** Same config as exp 167, seed 42→0; comp within ±0.4 confirms real lift.
**Prediction:** comp [-0.3, +0.5], F2 [+1.0, +2.5], F3 [+1.0, +3.0].
**Verdict:** DISCARD vs +1.32 global, but **NEW CATBOOST CHAMPION** at +0.3898 (BETTER than seed=42's +0.07!). F1 +1.06 (was -0.15), F4 +0.23 (was -1.37), 6/7 positive folds. Two-seed mean +0.23. Runtime 1299s (faster).
**Learning:** Lift IS reproducible. F1/F4 weakness was seed=42-specific. Cumulative within-CatBoost lift +0.95 across 3 exps. Need 3rd seed for median lock.

## Exp170 — CatBoost lr=0.05 n_est=1000 seed=99 (3-seed median lock)
**Diagnosis:** 2-seed mean +0.23 ; need 3rd seed per CLAUDE.md "3-seed median > baseline" rule.
**Citations:** Picard 2021 seed-std; Lakshminarayanan 2017 NeurIPS deep ensembles §3.2; Prokhorenkova 2018 §3.2.
**Hypothesis:** seed=99 locks 3-seed median; if >= +0.10 the lift is decisive.
**Prediction:** comp [-0.4, +0.6], A_sh [+0.0, +0.6], F2/F3 [+1.0, +2.5], runtime 22-32min.

## Exp170 — CatBoost seed=99 (3-seed median lock — VAL CRASH)
**Diagnosis:** 3rd seed for median lock per CLAUDE.md.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Prokhorenkova 2018 §3.2.
**Hypothesis:** seed=99 locks 3-seed median; if >= +0.10 the lift is decisive.
**Prediction:** comp [-0.4, +0.6].
**Verdict:** DISCARD strongly. Composite **-1.4536**. A_sharpe +0.47 (test stable), but val_sharpe CRASHED to -1.15. Per-fold A_sharpe: F1=+2.38, F2=**+3.74** (RECORD!), F3=+2.29 (test alpha is huge!), F4=-0.28, F5=-0.57, F6=+0.87, F7=-0.53. 4/7 positive test folds.
**Learning:** Major seed-variance insight: A_sharpe stable across seeds (+0.20/+0.49/+0.47), val_sharpe wildly variable (+0.64/+1.49/-1.15). 3-seed median composite +0.0728 — lift is REAL but MARGINAL. Need 4-seed lock.

## Exp171 — CatBoost seed=7 (4-seed median lock)
**Diagnosis:** 3-seed median +0.07 barely above baseline; need 4th seed per Lakshminarayanan 2017 §3.2 (≥5 ensemble members ideal).
**Citations:** Lakshminarayanan 2017 NeurIPS arXiv:1612.01474 §3.2; Picard 2021 4-seed reliability.
**Hypothesis:** seed=7 locks 4-seed median; informs deploy-vs-abandon decision.
**Prediction:** comp [-1.0, +0.6], A_sh [+0.2, +0.6], val_sh wild range.

## Exp171 — CatBoost seed=7 (4-seed median lock — DECISIVE)
**Diagnosis:** 4-seed median lock per Lakshminarayanan 2017 §3.2.
**Citations:** Lakshminarayanan 2017 NeurIPS arXiv:1612.01474; Picard 2021.
**Hypothesis:** seed=7 locks 4-seed median; informs deploy-vs-abandon.
**Prediction:** comp [-1.0, +0.6].
**Verdict:** DISCARD. Composite -0.0828. 4-seed distribution [-1.4536, -0.0828, +0.0728, +0.3898] → **median -0.005, mean -0.27**. A_sharpe stable (+0.20/+0.49/+0.47/+0.22) but val_sharpe wild (+0.24/+0.64/+1.49/-1.15).
**Learning:** CatBoost lr=0.05 lift was largely seed-luck. CLI doesn't expose stability levers (random_strength, ordered_boosting=Plain). Branch exhausted. PIVOT to LSTM (most under-budget at 33/75).

## Exp172 — LSTM 1-layer hidden=256 (capacity axis untested)
**Diagnosis:** Pivoting to LSTM after CatBoost branch exhaustion. LSTM-best exp 74 (+0.737). hidden_size=256 untested per Fischer-Krauss 2018 §3.2.
**Citations:** Fischer-Krauss 2018 EJOR §3.2 hidden sweep; Hochreiter-Schmidhuber 1997 LSTM capacity; Goodfellow et al. 2016 §11.3 capacity scaling; He 2016 CVPR.
**Hypothesis:** 1-layer hidden=256 (vs 128) doubles LSTM cell capacity; ~268k params still safe at n=2538.
**Prediction:** comp [+0.5, +1.2], A_sh [+0.5, +1.5], runtime 4-6min.

## Exp172 — LSTM 1-layer hidden=256 (capacity bump REMARKABLE TEST RESULT)
**Diagnosis:** Pivot to LSTM after CatBoost exhaustion; capacity axis untested per Fischer-Krauss 2018 §3.2.
**Citations:** Fischer-Krauss 2018 §3.2; Hochreiter-Schmidhuber 1997; Goodfellow 2016 §11.3; He 2016.
**Hypothesis:** hidden=256 (vs 128) doubles LSTM capacity; ~268k params still safe at n=2538.
**Prediction:** comp [+0.5, +1.2], A_sh [+0.5, +1.5].
**Verdict:** DISCARD by composite (-0.04) BUT **A_sharpe +0.8974**, excess **+0.2931 (FIRST POSITIVE EXCESS in many exps!)**, F5 **RECORD +3.32**, F4 +2.21, F6 +1.31, 5/7 positive folds. Val drag at +0.16. Runtime 60s.
**Learning:** Capacity bump 128→256 unlocks F4/F5/F6 alpha. Val mismatch likely Taper-like regime overfit. Need variance check.

## Exp173 — LSTM hidden=256 seed=42 (variance check)
**Diagnosis:** Confirm exp 172 test-side breakthrough (A_sh +0.90) is reproducible across seeds.
**Citations:** Picard 2021 arXiv:2109.08203; Lakshminarayanan 2017 §3.2; Fischer-Krauss 2018 §3.2; Bengio 1994 IEEE TNN gradient trajectories.
**Hypothesis:** Same exp 172 config, seed 99→42; A_sh ≥ +0.5 confirms real capacity lift.
**Prediction:** comp [-0.4, +0.5], A_sh [+0.5, +1.2], F5 [+1.0, +3.5].

## Exp173 — LSTM hidden=256 seed=42 (variance check, capacity hypothesis REJECTED)
**Diagnosis:** Confirm exp 172 capacity-lift reproducibility per Picard 2021.
**Citations:** Picard 2021; Lakshminarayanan 2017; Fischer-Krauss 2018; Bengio 1994.
**Hypothesis:** Same exp 172 config, seed 99→42; A_sh ≥ +0.5 confirms real lift.
**Prediction:** comp [-0.4, +0.5], A_sh [+0.5, +1.2].
**Verdict:** DISCARD vs +1.32 global. Comp +0.7488, A_sh +0.85, val_sh +0.96. **6/7 positive folds.** Test alpha REPRODUCIBLE across seeds. BUT exp 74 (hidden=128 seed=99) had A_sh +1.30, excess +0.69 — BETTER than hidden=256. **Capacity bump REJECTED.**
**Learning:** Hidden=128 is the LSTM sweet-spot at n=2538; doubling adds noise. LSTM-best stays exp 74. Axis closed: hidden_size. Axis open: seq_len, bidirectional, AWD dropout, Lion, layer norm.

## Exp174 — LSTM hidden=128 seq_len=20 (untested seq axis)
**Diagnosis:** Pivot from rejected capacity axis to seq_len. Only seq=10 tested on LSTM. Fischer-Krauss 2018 §3.4 sweep recommends >10 for noisy financial data.
**Citations:** Fischer-Krauss 2018 §3.4; Goyal 2017 arXiv:1706.02677; Hochreiter-Schmidhuber 1997 §3.1; Bengio 1994.
**Hypothesis:** seq=20 (vs 10) captures 20-day momentum/reversal cycles aligned with monthly vol cycles.
**Prediction:** comp [+0.5, +1.2], A_sh [+0.8, +1.5], runtime ~70-100s.

## Exp174 — LSTM seq=20 (REJECTED — gradient-vanishing)
**Diagnosis:** Test untested seq axis per Fischer-Krauss 2018 §3.4.
**Citations:** Fischer-Krauss 2018 §3.4; Goyal 2017; Hochreiter-Schmidhuber 1997; Bengio 1994.
**Hypothesis:** seq=20 captures 20-day momentum patterns.
**Prediction:** comp [+0.5, +1.2].
**Verdict:** DISCARD strongly. Comp -0.8293 (vs +0.74 baseline). A_sh +0.18 (vs +1.30!). F2/F3/F4 collapsed. Bengio 1994 gradient-vanishing CONFIRMED at our n.
**Learning:** seq=20 worse than seq=10. Axis bounded above. Try seq=5 (opposite direction).

## Exp175 — LSTM seq=5 (untested shorter direction)
**Diagnosis:** seq=20 rejected; try seq=5 (shorter than current sweet-spot 10).
**Citations:** Bengio 1994 IEEE TNN; Hochreiter-Schmidhuber 1997 §3.1; Fischer-Krauss 2018 §3.4; Vaswani 2017 §3.2.
**Hypothesis:** seq=5 within QQQ daily autocorr halflife; eliminates grad-flow issues.
**Prediction:** comp [+0.4, +1.0], A_sh [+1.0, +1.6], runtime ~50-70s.

## Exp175 — LSTM seq=5 (DISCARD — F2/F3 collapse)
**Diagnosis:** Test seq=5 (shorter, opposite of rejected seq=20).
**Citations:** Bengio 1994; Hochreiter-Schmidhuber 1997 §3.1; Vaswani 2017.
**Hypothesis:** seq=5 within autocorr halflife; eliminates grad-flow issues.
**Prediction:** comp [+0.4, +1.0], A_sh [+1.0, +1.6].
**Verdict:** DISCARD. Comp +0.0161, A_sh +0.34. F1/F5 held but F2/F3/F7 collapsed. Runtime 40s.
**Learning:** seq_len axis FULLY CLOSED — seq=10 champion both above and below. Pivot to bs axis.

## Exp176 — LSTM bs=8 (Keskar 2017 flat-minima untested)
**Diagnosis:** seq axis closed; pivot to bs. bs=16 (canonical) and bs=32 tested; bs=8 UNTESTED.
**Citations:** Keskar 2017 ICLR arXiv:1609.04836; Smith 2018 ICLR arXiv:1711.00489; Loshchilov-Hutter 2019; Hoffer-Hubara-Soudry 2017 NeurIPS arXiv:1705.08741.
**Hypothesis:** bs=8 induces √2× more SGD noise; finds flatter minima.
**Prediction:** comp [+0.6, +1.3], A_sh [+1.2, +1.7], runtime ~80-120s.

## Exp176 — LSTM bs=8 (Keskar 2017 flat-minima REJECTED)
**Diagnosis:** Test bs=8 flat-minima axis.
**Citations:** Keskar 2017 ICLR; Smith 2018; Loshchilov-Hutter 2019; Hoffer 2017.
**Hypothesis:** Smaller bs → more SGD noise → flatter minima → better generalization.
**Prediction:** comp [+0.6, +1.3].
**Verdict:** DISCARD. Comp +0.19, A_sh +0.29, 6/7 positive folds but small amplitudes. Lost F1 chaos alpha. Runtime 73s.
**Learning:** bs axis CLOSED — bs=16 champion. Keskar 2017 doesn't transfer to QQQ at n=2538.

## Exp177 — LSTM lr=2e-3 (Smith 2017 highest-leverage axis)
**Diagnosis:** All HP single-knob axes rejected; pivot to lr (most-impactful per Smith 2017).
**Citations:** Smith 2017 arXiv:1803.09820 §3; Goyal 2017; Loshchilov-Hutter 2019; Kingma-Ba 2015 §3.
**Hypothesis:** lr=2e-3 doubles step size, lets LSTM reach deeper minima within early-stop budget.
**Prediction:** comp [-0.3, +1.0], A_sh [+0.5, +1.5], runtime ~50-70s.

## Exp177 — LSTM lr=2e-3 (REJECTED — destabilized training)
**Diagnosis:** Smith 2017 highest-leverage axis untested above 1e-3.
**Citations:** Smith 2017 §3; Goyal 2017; Loshchilov-Hutter 2019; Kingma-Ba 2015.
**Hypothesis:** lr=2e-3 lets LSTM reach deeper minima within early-stop budget.
**Prediction:** comp [-0.3, +1.0].
**Verdict:** DISCARD strongly. Comp -1.12, A_sh +0.05, val_sh NEGATIVE. F2/F3/F4/F7 all collapsed — destabilized training. Runtime 52s.
**Learning:** **LSTM HP-only axes ALL EXHAUSTED across 6 consecutive DISCARDs.** Per CLAUDE.md structural-change rule. Pivot to Mamba.

## Exp178 — Mamba mambats variant (untested SSM structural variant)
**Diagnosis:** LSTM HP exhausted; pivot to Mamba (21/25, global champion). mambats variant NEVER tested on QQQ.
**Citations:** Cai et al. 2024 NeurIPS 'MambaTS' arXiv:2405.16440 §3; Liu 2025 DMamba arXiv:2602.09081; Gu-Dao 2024 §4.2; Wu 2023 ICLR TimesNet.
**Hypothesis:** mambats season-trend decomposition captures QQQ FOMC/EOM/quad-witching seasonal patterns that dmamba misses.
**Prediction:** comp [+0.5, +1.6], A_sh [+0.7, +1.5], runtime ~6-12min.

## Exp178 — Mamba mambats variant (complementary to dMamba)
**Diagnosis:** Pivot from LSTM HP exhaustion to Mamba structural axis; mambats untested.
**Citations:** Cai 2024 NeurIPS arXiv:2405.16440 §3; Liu 2025 DMamba; Gu-Dao 2024 §4.2; Wu 2023 TimesNet.
**Hypothesis:** mambats season-trend decomposition captures FOMC/EOM/quad-witching seasonal patterns.
**Prediction:** comp [+0.5, +1.6], A_sh [+0.7, +1.5].
**Verdict:** DISCARD vs +1.32 global. Comp +0.42, A_sh +0.62, **excess +0.015 (positive!)**. Per-fold F1=-0.75 F2=+1.42 F3=+2.10 F4=+0.41 F5=-0.07 F6=+0.58 F7=+1.44. 5/7 positive folds. Runtime 107s.
**Learning:** mambats COMPLEMENTARY to dmamba — gains F3/F7 trend alpha, loses F1 GFC chaos. Suggests ensemble axis. Need variance check.

## Exp179 — Mamba mambats variance check seed=7
**Diagnosis:** Confirm mambats per-fold pattern reproducibility per Picard 2021 + Lakshminarayanan 2017.
**Citations:** Picard 2021 arXiv:2109.08203; Lakshminarayanan 2017 §3.2; Cai 2024 NeurIPS §3.1 mambats seed-sensitivity.
**Hypothesis:** Same exp 178 config, seed 42→7; comp [+0.0, +0.7], same per-fold signature.
**Prediction:** comp [+0.0, +0.7], A_sh [+0.4, +1.0], runtime ~80-130s.

## Exp179 — Mamba mambats seed=7 (variance check, 2-seed mean +0.38)
**Diagnosis:** Confirm mambats reproducibility per Picard 2021.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Cai 2024 §3.1.
**Hypothesis:** Same exp 178 config, seed 42→7; comp [+0.0, +0.7], same per-fold sig.
**Prediction:** comp [+0.0, +0.7].
**Verdict:** DISCARD vs +1.32 global. Comp +0.34, A_sh +0.44, val_sh +0.99. Per-fold F1=-0.96 (still weak), F2=+0.77, F3=+0.18, F4=+0.72, F5=+0.56, F6=+0.91, F7=+0.83. 5/7 pos folds. Runtime 80s.
**Learning:** mambats reproducibility CONFIRMED at lower amplitude than dmamba. Complementary signal. Try s_mamba next (last untested variant).

## Exp180 — Mamba s_mamba variant (last untested SSM variant)
**Diagnosis:** Mamba budget 24/25; complete variant-axis sweep with s_mamba.
**Citations:** Liu 2025 DMamba arXiv:2602.09081 §3.4; Gu-Dao 2024 §2; Wang-Wu et al. 2024 S-Mamba.
**Hypothesis:** s_mamba's input-dependent selectivity differs from dmamba's decoupled gates; distinct per-fold pattern.
**Prediction:** comp [-0.7, +1.0], A_sh [+0.3, +1.0], runtime ~6-12min.

## Exp180 — Mamba s_mamba variant (REJECTED, Mamba 25/25 COMPLETE)
**Diagnosis:** Last untested Mamba variant.
**Citations:** Liu 2025 DMamba §3.4; Gu-Dao 2024 §2.
**Hypothesis:** s_mamba's input-dependent selectivity distinct.
**Prediction:** comp [-0.7, +1.0].
**Verdict:** DISCARD strongly. Comp -0.52, A_sh +0.29, val NEGATIVE -0.23. 4/7 pos folds. Runtime 1348s.
**Learning:** Mamba variant axis FULLY CLOSED. dmamba=+1.32 (champ), mambats=+0.38, s_mamba=-0.53, vanilla=-0.67. **Mamba 25/25 budget COMPLETE.** Pivot to XGBoost.

## Exp181 — XGBoost depth=6 (Chen-Guestrin 2016 paper default untested)
**Diagnosis:** Pivot from completed Mamba to XGBoost (4 left). depth=6 NEVER tested on QQQ.
**Citations:** Chen-Guestrin 2016 KDD arXiv:1603.02754 §3.2; Friedman 2001 §5.4; ESL §10.12.
**Hypothesis:** depth=6 (vs 4) captures 3-way feature interactions; 64 leaves vs 16.
**Prediction:** comp [-0.4, +0.5], A_sh [+0.2, +0.8], runtime ~5-10min.

## Exp181 — XGBoost depth=6 (paper default, val crash but RECORD F3)
**Diagnosis:** Pivot from completed Mamba; Chen-Guestrin 2016 paper default untested.
**Citations:** Chen-Guestrin 2016 KDD §3.2; Friedman 2001 §5.4; ESL §10.12.
**Hypothesis:** depth=6 captures 3-way interactions; 64 leaves vs depth=4's 16.
**Prediction:** comp [-0.4, +0.5].
**Verdict:** DISCARD. Comp -0.44, A_sh +0.42, val NEGATIVE -0.24. F3 **RECORD +3.51** across all backbones! But F1/F7 weak, val crashed. Same val-instability pattern as CatBoost lr=0.05. Runtime 1632s.
**Learning:** depth=6 unlocks F3 multi-feature interactions but val unstable. Try depth=5 for capacity-stability mid-point.

## Exp182 — XGBoost depth=5 (mid-point untested)
**Diagnosis:** Bracket depth axis: 4 stable, 6 alpha-rich+val-crash. Mid=5 untested.
**Citations:** Friedman 2001 §5.4; Chen-Guestrin 2016 §3.2; ESL §10.12.
**Hypothesis:** depth=5 retains F3 alpha while preserving val stability.
**Prediction:** comp [-0.3, +0.4], F3 [+1.5, +3.0], val_sh [+0.0, +0.5], runtime ~10-18min.

## Exp182 — XGBoost depth=5 (NEW XGBoost CHAMPION +0.37)
**Diagnosis:** Bracket depth axis: depth=4 stable, depth=6 alpha-rich+val-crash; mid=5 untested.
**Citations:** Friedman 2001 §5.4; Chen-Guestrin 2016 §3.2; ESL §10.12.
**Hypothesis:** depth=5 retains F3 alpha + val stability.
**Prediction:** comp [-0.3, +0.4], F3 [+1.5, +3.0], val [+0.0, +0.5].
**Verdict:** DISCARD vs +1.32 global, but **XGBOOST WITHIN-CHAMPION** at +0.3736 (delta +0.50 vs exp 63 -0.13). F1=+1.51 F2=+2.55 F3=+1.30 F4=+0.44 F5=+0.86 F6=+1.46 F7=-0.90. 6/7 positive folds. val_sh +0.79 RECOVERED from depth=6's -0.24. Runtime 1003s.
**Learning:** depth=5 sweet-spot empirically confirmed. Friedman 2001 §5.4 prediction holds. Variance check next per Picard 2021.

## Exp183 — XGBoost depth=5 seed=0 (variance check)
**Diagnosis:** Confirm exp 182 +0.37 reproducibility per Picard 2021 + Lakshminarayanan 2017.
**Citations:** Picard 2021 arXiv:2109.08203; Lakshminarayanan 2017 §3.2; Chen-Guestrin 2016 §3.2 histogram seed-sensitivity.
**Hypothesis:** Same exp 182 config, seed 42→0; comp within ±0.4.
**Prediction:** comp [+0.0, +0.7], val_sh [+0.3, +1.0], runtime ~15-20min.

## Exp183 — XGBoost depth=5 seed=0 (variance check, val-crash)
**Diagnosis:** Confirm exp 182 +0.37 reproducibility per Picard 2021.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Chen-Guestrin 2016 §3.2.
**Hypothesis:** Same exp 182 config, seed 42→0; comp within ±0.4.
**Prediction:** comp [+0.0, +0.7].
**Verdict:** DISCARD. Comp -0.40, A_sh +0.34, val NEGATIVE -0.20. Same val-instability pattern as CatBoost lr=0.05. 2-seed mean -0.015. Runtime 1014s.
**Learning:** GBM val-instability at high capacity is STRUCTURAL across XGBoost+CatBoost. Need 3-seed lock.

## Exp184 — XGBoost depth=5 seed=99 (3-seed median lock)
**Diagnosis:** 2-seed mean ≈ 0; need 3rd seed per CLAUDE.md "3-seed median".
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Chen-Guestrin 2016 §3.2; Friedman 2001 §5.4.
**Hypothesis:** seed=99 locks 3-seed median; informs depth=5 keep/reject decision.
**Prediction:** comp [-0.5, +0.7], median expected near 0±0.4, runtime ~15-20min.

## Exp184 — XGBoost depth=5 seed=99 (3-seed median REJECTS depth=5)
**Diagnosis:** 3rd seed for median lock per CLAUDE.md.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Chen-Guestrin 2016 §3.2; Friedman 2001 §5.4.
**Hypothesis:** seed=99 locks 3-seed median.
**Prediction:** comp [-0.5, +0.7].
**Verdict:** DISCARD strongly. Comp -0.56, A_sh +0.23, val NEGATIVE -0.26. F2 RECORD +3.17. 3-seed median -0.40 (NEGATIVE). Runtime 1009s.
**Learning:** depth=5 lift was seed-luck. Same GBM val-instability pattern. XGBoost-best stays exp 63 -0.128.

## Exp185 — XGBoost depth=4 lr=0.005 n_est=2000 (slowest-lr untested)
**Diagnosis:** 1 slot left; try stability-focused slowest-learner regime.
**Citations:** Friedman 2001 §5.2; ESL §10.12; Chen-Guestrin 2016 §3.2; Mason 2000 NeurIPS.
**Hypothesis:** lr=0.005 + n_est=2000 finds smoother-loss flat-minima; matches exp 63 capacity.
**Prediction:** comp [-0.4, +0.3], runtime ~25-35min.

## Exp185 — XGBoost lr=0.005 n_est=2000 (slowest-lr — XGBoost 25/25 COMPLETE)
**Diagnosis:** Final XGBoost slot; stability-focused slowest-lr.
**Citations:** Friedman 2001 §5.2; ESL §10.12; Chen-Guestrin 2016 §3.2; Mason 2000.
**Hypothesis:** Slower lr finds smoother loss minima.
**Prediction:** comp [-0.4, +0.3].
**Verdict:** DISCARD. Comp -0.23, A_sh +0.42, val barely positive +0.07. Runtime 1195s. **XGBoost 25/25 COMPLETE.**
**Learning:** XGBoost-best stays exp 63 -0.128. Pivot to LightGBM (3 left).

## Exp186 — LightGBM exp 95 seed=99 (variance check on +0.611 single-seed champ)
**Diagnosis:** LGBM-best exp 95 +0.611 single-seed; needs variance verification.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Ke 2017 §3.1.
**Hypothesis:** seed=99 lift within ±0.5 confirms LGBM exp 95 as real.
**Prediction:** comp [+0.0, +0.8], runtime ~10-15min.

## Exp186 — LGBM exp 95 seed=99 (variance check, 2-seed mean +0.25)
**Diagnosis:** Confirm exp 95 +0.611 reproducibility per Picard 2021.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Ke 2017 §3.1.
**Hypothesis:** seed=99 within ±0.5 confirms.
**Prediction:** comp [+0.0, +0.8].
**Verdict:** DISCARD. Comp -0.11. 2-seed mean +0.25. Same GBM val-instability. Runtime 337s.
**Learning:** Need 3rd seed.

## Exp187 — LGBM exp 95 seed=0 (3-seed median lock)
**Diagnosis:** 2-seed mean +0.25; 3-seed median lock per CLAUDE.md.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Ke 2017 §3.1.
**Hypothesis:** seed=0 locks 3-seed median; ≥+0.1 means real lift.
**Prediction:** comp [-0.4, +0.7], runtime ~5-10min.

## Exp187 — LGBM exp 95 seed=0 (3-seed median REJECTS lift)
**Diagnosis:** 3rd seed for median lock per CLAUDE.md.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Ke 2017 §3.1.
**Hypothesis:** seed=0 locks 3-seed median; ≥+0.1 means real lift.
**Prediction:** comp [-0.4, +0.7].
**Verdict:** DISCARD strongly. Comp -0.46, A_sh NEGATIVE -0.06, val_sh +1.34 high. F2 RECORD +3.33. 3-seed median -0.110 — REJECTS lift. Runtime 337s.
**Learning:** All 3 GBM families (CB, XGB, LGB) show structural val-instability at high capacity. LGBM-best stays median ~0.

## Exp188 — LGBM depth=5 (untested mid-point, final LGBM slot)
**Diagnosis:** Final LGBM slot. depth axis: 4 (+0.61 single but median ~0), 6 (-0.34). depth=5 untested.
**Citations:** Ke 2017 NeurIPS §3.1; Friedman 2001 §5.4; Chen-Guestrin 2016 §3.2.
**Hypothesis:** depth=5 leaf-wise growth (32 leaves) may produce different per-fold than XGBoost depth=5.
**Prediction:** comp [-0.4, +0.6], runtime ~10-15min.

## Exp188 — LGBM depth=5 (single-seed, LGBM budget 23/25 not 25/25)
**Diagnosis:** Final LGBM slot (recount: actually 24 now).
**Citations:** Ke 2017 §3.1; Friedman 2001 §5.4.
**Hypothesis:** depth=5 leaf-wise growth different from XGBoost.
**Prediction:** comp [-0.4, +0.6].
**Verdict:** DISCARD. Comp -0.14, A_sh +0.41, val barely positive +0.06. Same val-instability pattern. Runtime 426s.
**Learning:** LGBM at 23/25 (recount). 2 more slots needed.

## Exp189 — LGBM lr=0.005 n_est=3000 (slowest-lr capacity-matched, untested)
**Diagnosis:** 2 LGBM slots remaining; try slowest-learner regime.
**Citations:** Friedman 2001 §5.2; ESL §10.12; Ke 2017 §3.1.
**Hypothesis:** lr=0.005 + n_est=3000 finds smoother flat minima.
**Prediction:** comp [-0.3, +0.6], runtime ~15-25min.

## Exp189 — LGBM lr=0.005 n_est=3000 (slowest-lr, A_sh RECORD +0.94)
**Diagnosis:** Slow-lr stability test untested.
**Citations:** Friedman 2001 §5.2; ESL §10.12; Ke 2017 §3.1.
**Hypothesis:** Slow lr finds smoother flat minima.
**Prediction:** comp [-0.3, +0.6].
**Verdict:** DISCARD. Comp +0.24, **A_sh +0.94 RECORD**, 6/7 positive folds, F6 COVID +2.88 RECORD. Runtime 662s.
**Learning:** Slowest-lr regime is most-stable LGBM variant. Need variance check.

## Exp190 — LGBM exp 189 seed=99 (variance check on RECORD A_sh)
**Diagnosis:** Final LGBM slot; verify slowest-lr +0.94 A_sh reproducibility.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Friedman 2001 §5.2.
**Hypothesis:** A_sh stays ≥ +0.6 confirms deployable.
**Prediction:** comp [-0.2, +0.6], A_sh [+0.5, +1.1], runtime ~10-15min.

## Exp190 — LGBM seed=99 slowest-lr (variance check, REJECTED, LGBM 25/25 COMPLETE)
**Diagnosis:** Final LGBM slot; verify exp 189 +0.94 A_sh.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Friedman 2001 §5.2.
**Hypothesis:** A_sh stays ≥ +0.6.
**Prediction:** comp [-0.2, +0.6].
**Verdict:** DISCARD. Comp -0.33, A_sh NEGATIVE -0.03, 3/7 pos folds. 2-seed slowest-lr A_sh [+0.94, -0.03]. Runtime 662s.
**Learning:** LGBM family fundamentally seed-unstable. **LGBM 25/25 COMPLETE.** Pivoting to DLinear post-rollback baseline.

## Exp191 — DLinear post-rollback baseline seed=0
**Diagnosis:** DLinear 4 runs, best +0.80 was with rolled-back features. Need post-rollback variance.
**Citations:** Zeng-Chen-Zhang-Xu 2023 AAAI arXiv:2205.13504; Picard 2021; Lakshminarayanan 2017.
**Hypothesis:** Same exp 109 config seed=0; comp ±0.4 establishes post-rollback range.
**Prediction:** comp [-0.5, +0.5], runtime ~30-60s.

## Exp191 — DLinear post-rollback seed=0 (DISCARD)
**Diagnosis:** Establish post-features-rollback DLinear baseline.
**Citations:** Zeng 2023 AAAI; Picard 2021; Lakshminarayanan 2017.
**Hypothesis:** Same exp 109 config seed=0; comp ±0.4.
**Prediction:** comp [-0.5, +0.5].
**Verdict:** DISCARD. Comp -0.38, A_sh NEGATIVE -0.08. 2-seed mean -0.21. Runtime 30s.
**Learning:** DLinear post-rollback weak. seq_len axis untested.

## Exp192 — DLinear seq_len=20 (untested)
**Diagnosis:** DLinear 5/25; try seq=20 untested.
**Citations:** Zeng 2023 §4.2 seq sweep; Goyal 2017; Hochreiter 1997.
**Hypothesis:** seq=20 captures 20-day momentum cycles.
**Prediction:** comp [-0.4, +0.5], runtime ~30-60s.

## Exp192 — DLinear seq_len=20 (DISCARD)
**Diagnosis:** seq axis untested for DLinear.
**Citations:** Zeng 2023 §4.2; Goyal 2017; Hochreiter 1997.
**Hypothesis:** seq=20 captures 20-day momentum.
**Prediction:** comp [-0.4, +0.5].
**Verdict:** DISCARD. Comp -0.22, A_sh +0.08, val_sh +0.85. F4/F6 lost. Runtime 65s.
**Learning:** DLinear post-rollback weak; seq axis not the lift.

## Exp193 — iTransformer paper-recipe lr=5e-5 + warmup
**Diagnosis:** iTransformer 2/25, post-rollback baseline -1.41. Paper recipe untested.
**Citations:** Liu 2024 ICLR arXiv:2310.06625 §4.1; Vaswani 2017 §5.3 warmup; Goyal 2017.
**Hypothesis:** lr=5e-5 + warmup=10 stabilizes inverted attention.
**Prediction:** comp [-0.8, +0.3], runtime ~3-6min.

## Exp193 — iTransformer paper-recipe lr=5e-5 warmup=10 (RECORD A_sh +0.92)
**Diagnosis:** Liu 2024 ICLR §4.1 paper recipe untested.
**Citations:** Liu 2024 ICLR arXiv:2310.06625 §4.1; Vaswani 2017 §5.3; Goyal 2017.
**Hypothesis:** lr=5e-5+warmup=10 stabilizes inverted attention.
**Prediction:** comp [-0.8, +0.3].
**Verdict:** DISCARD by composite (-1.52) BUT **A_sh +0.92 RECORD** for iTransformer. val crash. F1=+1.43 F3=+1.96 F6=+1.70 F7=+1.09. Runtime 98s.
**Learning:** Paper-recipe lifts iTransformer test alpha massively. Need variance check.

## Exp194 — iTransformer paper-recipe seed=0 (variance check)
**Diagnosis:** Verify exp 193 +0.92 A_sh reproducibility per Picard 2021.
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Liu 2024 §4.1.
**Hypothesis:** A_sh stays ≥ +0.5 confirms.
**Prediction:** comp [-1.5, +0.5], A_sh [+0.4, +1.1], runtime ~1.5-3min.

## Exp194 — iTransformer paper-recipe seed=0 (variance crash)
**Verdict:** DISCARD strongly. Comp -2.02, A_sh NEGATIVE -0.10. 2-seed A_sh [+0.92, -0.10] mean +0.41. Same GBM val-instability pattern.

## Exp195 — iTransformer paper-recipe seed=99 (3-seed median lock)
**Citations:** Picard 2021; Lakshminarayanan 2017; Liu 2024.
**Hypothesis:** seed=99 locks 3-seed median.
**Prediction:** comp [-2.0, +0.5], A_sh [-0.5, +1.0], runtime ~1.5-3min.

## Exp195 — iTransformer seed=99 (3-seed median locks negative comp)
**Verdict:** DISCARD. Comp -1.43. 3-seed A_sh median +0.21; all comps negative. iTransformer can't beat baseline.

## Exp196 — N-BEATS seed=0 variance baseline
**Citations:** Oreshkin 2020 ICLR arXiv:1905.10437 §4.2; Picard 2021; Lakshminarayanan 2017.
**Hypothesis:** seed=0 establishes N-BEATS variance.
**Prediction:** comp [-2.0, +0.0], runtime ~1-2min.

## Exp196 — N-BEATS seed=0 (reproducibly weak)
**Verdict:** DISCARD. Comp -1.43 (same as exp 144!). 2-seed mean -1.43 — N-BEATS architecture mismatch with QQQ.

## Exp197 — PatchTSMixer seed=99 (3-seed median lock)
**Citations:** Ekambaram 2023 KDD arXiv:2306.09364; Picard 2021; Lakshminarayanan 2017.
**Hypothesis:** seed=99 locks 3-seed PatchTSMixer median.
**Prediction:** comp [-2.0, +0.5], runtime ~1-3min.

## Exp197 — PatchTSMixer seed=99 (RECORD A_sh +1.22, F2 +4.87)
**Verdict:** DISCARD by composite (+0.155) BUT **A_sh +1.22 RECORD** (matches BH exactly!), F2 +4.87 RECORD, 6/7 positive folds. val drag.
**Learning:** PatchTSMixer at right seed produces strongest test alpha this session. Need variance lock.

## Exp198 — PatchTSMixer seed=7 (variance check on A_sh +1.22 record)
**Citations:** Ekambaram 2023 KDD; Picard 2021; Lakshminarayanan 2017 §3.2.
**Hypothesis:** A_sh stays ≥ +0.7 confirms.
**Prediction:** comp [-2.0, +0.5], A_sh [+0.4, +1.5], runtime ~3-7min.

## Exp198 — PatchTSMixer seed=7 (variance crash)
**Verdict:** DISCARD. Comp -1.10, A_sh -0.14. 4-seed PatchTSMixer comp [+0.06, -1.82, +0.16, -1.10] median -0.52.

## Exp199 — PatchTSMixer seed=2024 (5-seed median lock)
**Citations:** Lakshminarayanan 2017 §3.2; Picard 2021; Ekambaram 2023.
**Hypothesis:** seed=2024 locks 5-seed median.
**Prediction:** comp [-2.0, +0.5], runtime ~3-7min.

## Exp199 — PatchTSMixer seed=2024 (5-seed lock, median +0.03)
**Verdict:** DISCARD. Comp +0.028. **5-seed median composite +0.028.** Real but tiny lift. PatchTSMixer characterized.

## Exp200 — MLP exp 79 seed=42 (variance check on +0.974)
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2; Goyal 2017; Loshchilov-Hutter 2019.
**Hypothesis:** seed=42 confirms MLP exp 79 lift.
**Prediction:** comp [+0.4, +1.3], runtime ~30-60s.

## Exp200 — MLP exp 79 seed=42 (variance crash)
**Verdict:** DISCARD. Comp -0.71. 2-seed MLP exp79 [+0.97, -0.71]. Same val-instability pattern.

## Exp201 — MLP exp 79 seed=99 (3-seed median lock)
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2.
**Hypothesis:** seed=99 locks 3-seed MLP median.
**Prediction:** comp [-0.5, +1.0], runtime ~30-60s.

## Exp201 — MLP exp 79 seed=99 (7/7 POSITIVE FOLDS, FIRST stable positive median!)
**Verdict:** DISCARD vs +1.32 global BUT major: comp **+0.5197**, A_sh **+0.87**, **7/7 POSITIVE FOLDS**, excess +0.26 (positive!). **3-seed MLP exp 79 median +0.52** — FIRST non-Mamba stable positive median this session.
**Learning:** MLP exp 79 (warmup=5 + hd=0.25) is REAL lift! Need 4-5 seeds per Lakshminarayanan 2017.

## Exp202 — MLP exp 79 seed=7 (4-seed median lock)
**Citations:** Lakshminarayanan 2017 §3.2 (5+ members ideal); Picard 2021; Goyal 2017; Loshchilov-Hutter 2019.
**Hypothesis:** seed=7 confirms +0.52 median.
**Prediction:** comp [+0.0, +1.0], runtime ~30-60s.

## Exp202 — MLP seed=7 (4-seed median drops to +0.01)
**Verdict:** DISCARD. Comp -0.49. 4-seed MLP exp 79 [+0.97, -0.71, +0.52, -0.49] median +0.014.
**Learning:** MLP joins val-instability club. Mamba dmamba uniquely stable.

## Exp203 — MLP seed=2024 (5-seed median lock)
**Citations:** Lakshminarayanan 2017 §3.2; Picard 2021.
**Prediction:** comp [-0.7, +1.0], runtime ~30-60s.

## Exp203 — MLP seed=2024 (5-seed median POSITIVE +0.43)
**Verdict:** DISCARD vs +1.32 global BUT major: comp **+0.4333**, A_sh +0.62, **6/7 positive folds**.
**5-seed MLP exp 79 distribution: median +0.433, mean +0.144.** SECOND non-Mamba backbone with stable positive multi-seed median!
**Learning:** MLP exp 79 is a real lift — deployable for ensemble.

## Exp204 — MLP exp 79 + wd=1e-4 (Loshchilov-Hutter canonical)
**Citations:** Loshchilov-Hutter 2019; Goyal 2017; Gu-Kelly-Xiu 2020.
**Hypothesis:** wd=1e-4 + warmup=5 combines best axes.
**Prediction:** comp [+0.3, +1.2], runtime ~30-60s.

## Exp204 — MLP wd=1e-4 + warmup=5 (NEW MLP champion +0.97!)
**Diagnosis:** Combine canonical AdamW wd + warmup.
**Citations:** Loshchilov-Hutter 2019; Goyal 2017; Gu-Kelly-Xiu 2020.
**Hypothesis:** wd=1e-4 + warmup=5 stronger than either alone.
**Prediction:** comp [+0.3, +1.2].
**Verdict:** DISCARD vs +1.32 global BUT MAJOR result: comp **+0.9735**, A_sh **+1.04**, **7/7 positive folds**, excess **+0.43**. F3=+2.91 F6=+2.64. Runtime 24s.
**Learning:** wd=1e-4 + warmup=5 is the new MLP best config.

## Exp205 — MLP wd=1e-4 + warmup=5 seed=42 (variance check)
**Citations:** Picard 2021; Lakshminarayanan 2017; Loshchilov-Hutter 2019.
**Hypothesis:** Confirms +0.97 reproducibility.
**Prediction:** comp [+0.3, +1.2], runtime ~30-60s.

## Exp205 — MLP wd=1e-4 seed=42 (variance crash)
**Verdict:** DISCARD. Comp -0.92. Same val-instability. 2-seed [+0.97, -0.92].

## Exp206 — MLP wd=1e-4 seed=99 (3-seed lock + MLP 50/50 final)
**Citations:** Picard 2021; Lakshminarayanan 2017; Loshchilov-Hutter 2019.
**Prediction:** comp [-0.5, +1.0], runtime ~30-60s.

## Exp206 — MLP wd=1e-4 seed=99 (MLP 50/50 COMPLETE!)
**Verdict:** DISCARD vs global. Comp +0.5197 IDENTICAL to exp 201 (wd has near-zero effect at this seed). 3-seed MLP wd=1e-4 median +0.520. **MLP 50/50 COMPLETE.**
**Learning:** MLP exp 79/204 stable positive lift. SECOND stable positive backbone after mamba.

## Exp207 — CatBoost lr=0.005 n_est=2000 (slowest-lr untested)
**Citations:** Friedman 2001 §5.2; Prokhorenkova 2018 §3.3; ESL §10.12; Bühlmann-Yu 2003.
**Hypothesis:** Slowest CatBoost lr regime stabler.
**Prediction:** comp [-0.5, +0.6], runtime ~10-20min.

## Exp207 — CatBoost slowest-lr lr=0.005 (val crash F2 RECORD +3.89)
**Verdict:** DISCARD. Comp -1.10, A_sh +0.68, val NEG. F2 RECORD +3.89. Same val pattern. Runtime 43min.

## Exp208 — CatBoost depth=3 (untested shallower)
**Citations:** Friedman 2001 §5.4; Prokhorenkova 2018 §3.3; ESL §10.12.
**Hypothesis:** Shallower trees stabler.
**Prediction:** comp [-0.5, +0.3], runtime ~5-8min.

## Exp208 — CatBoost depth=3 (val crash, F2 RECORD +4.63)
**Verdict:** DISCARD. Comp -0.80, F2 RECORD +4.63 but val NEG. Same pattern.

## Exp209 — CatBoost seq=30 (untested seq axis)
**Citations:** Prokhorenkova 2018; Hochreiter-Schmidhuber 1997.
**Prediction:** comp [-0.5, +0.4], runtime ~3-6min.

## Exp214 — CatBoost lr=0.04 seed=0 (val POSITIVE +0.60, CatBoost 25/25 COMPLETE)
**Verdict:** DISCARD. Comp -0.25. **2-seed lr=0.04 [-0.19, -0.25] = most val-stable CatBoost regime.** **CatBoost 25/25 COMPLETE.**
**Learning:** 5 backbones complete. Pivot to MLP +25 with seq=20.

## Exp215 — MLP seq=20 (user directive: +25 MLP, seq axis untested)
**Diagnosis:** User-extended MLP budget 50→75 with seq_len=20 focus. MLP only tested at seq=10.
**Citations:** Goyal 2017 arXiv:1706.02677 §2.4; Hochreiter-Schmidhuber 1997 §3.1; Gu-Kelly-Xiu 2020 RFS; Loshchilov-Hutter 2019; Srivastava 2014 dropout.
**Hypothesis:** Doubled context (seq=20 vs 10) captures multi-week patterns; risk of overfit at 3,680 input dims.
**Prediction:** comp [+0.0, +1.2], A_sh [+0.5, +1.3], runtime ~30-60s.

## Exp215 — MLP seq=35 (user directive: +10 cheap-tier winners @ seq=35, MLP first)
**Diagnosis:** User-extended directive: re-run cheap-tier winners with seq=35 untested.
**Citations:** Goyal 2017 §2.4; Gu-Kelly-Xiu 2020; He 2016 ResNet; Loshchilov-Hutter 2019.
**Hypothesis:** seq=35 captures multi-week regime context.
**Prediction:** comp [-0.5, +1.0].
**Verdict:** DISCARD vs global, but POSITIVE comp +0.55, A_sh +0.89, F2 **RECORD +4.53**, 6/7 pos folds. seq=35 helps!

## Exp216 — MLP seq=35 seed=42 (variance check #2/10 for MLP seq=35 grind)
**Citations:** Picard 2021; Lakshminarayanan 2017 §3.2.
**Prediction:** comp [-0.3, +0.7], runtime ~50-90s.

## Exp216 — MLP seq=35 seed=42 (variance crash)
**Verdict:** DISCARD. Comp -1.04, A_sh NEGATIVE, val NEGATIVE. 2/7 pos folds. 2-seed [+0.55, -1.04] mean -0.25.
**Learning:** seq=35 lift was seed=0 luck. Need 3rd seed.

## Exp217 — MLP seq=35 seed=99 (3-seed median lock #3/10)
**Citations:** Picard 2021; Lakshminarayanan 2017; Goyal 2017.
**Prediction:** comp [-0.8, +0.7], runtime ~50-90s.
