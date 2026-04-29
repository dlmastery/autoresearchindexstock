# CLAUDE.md — Project Rules for AutoResearch QQQ (Index Stock)

> Self-contained successor to the FX `C:/Users/evija/autoresearch/CLAUDE.md`.
> A fresh Claude Code session reading only this file must be able to
> operate fully on the QQQ project. Where rules are identical to FX they
> are restated verbatim so nothing is implicit. Where the equity-index
> setting differs, the difference is called out explicitly.

## On Session Start (ALWAYS do this first)

You ARE the autoresearch loop for QQQ. Claude Code is the outer loop —
there is no separate Python agent. When a session starts:

1. **Read the crash-recovery checkpoint:**
   `autoresearchindexstock/memory/project_autoresearch_checkpoint.md` —
   current champion, last experiment result, per-fold diagnostics, the
   exact next experiment command.
2. **Read the hardware crash log** (shared with FX project):
   `memory/project_hardware_crash_log.md`. Documents the BSOD history and
   CPU core exclusion rules. Must follow.
3. **Read the experiment log tail:**
   `autoresearchindexstock/autoresearch_results/experiment_log.jsonl`
   (last 3 entries) and
   `autoresearchindexstock/autoresearch_results/best_config.json` to verify
   state.
4. **Resume the experiment loop** from where the checkpoint says. Follow
   the 7-step process below (diagnose → cite → hypothesize → predict →
   run ONE experiment → analyze → checkpoint).
5. **Start the dashboard once per session** (background):
   `"C:/Users/evija/anaconda3/python.exe" -m http.server 8888 --directory C:/Users/evija/autoresearch/autoresearchindexstock/autoresearch_results`
   Live mirror also at
   <https://dlmastery.github.io/autoresearch/index_stock_dashboard/>.
6. **Run experiments** via:
   `cd C:/Users/evija/autoresearch && "C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock.run_autoresearch --backbone <name> [flags] --description "..."`.
   Default timeout 600-3000s depending on backbone.
7. **If the user says "continue" or "keep going"** — resume the loop. No
   need to ask what to do.

## Project identity (different from FX)

- **Asset**: QQQ (Invesco QQQ Trust, Nasdaq-100 ETF).
- **Data window**: 2004-01-01 → **2025-12-31**. **No 2026 data anywhere.**
  `data/download.py` hard-caps `end="2025-12-31"` and drops any 2026 row
  with a logged warning. Verified at startup.
- **Optimization target**: **target A — `fwd_ret_1d`** (1-day forward log
  return). KEEP / DISCARD decisions are driven by the 1-day-return
  composite. We *additionally* compute, track, and plot four target
  variants on every experiment so we never lose visibility:
  - **A — `fwd_ret_1d`** (PRIMARY)
  - **B — `fwd_ret_5d`** (auxiliary head, trained jointly)
  - **C — sign concordance** (1d vs 5d agreement; side-channel metric only)
  - **D — vol-adjusted 1d** (`fwd_ret_1d / rolling_vol_20`; orthogonalises
    trend from skill)
  The runner emits A_/B_/D_ keys + unprefixed aliases of A_ per JSONL row;
  the dashboard's TARGET selector swaps which prefix the chart + table
  read. **The trade is always realised on the unscaled 1d return** — D's
  vol-adjusted prediction only sets direction.
- **Goal**: **meet or beat the FX project's mega-ensemble headline**
  (Sharpe **+9.7071** on the FX dashboard) on a fair-comparison basis.
  Because QQQ trends, the *fair* comparison is **excess-Sharpe over a
  long-only buy-and-hold baseline**, tracked alongside raw Sharpe in
  every JSONL row (`bh_sharpe`, `bh_return_pct`, `excess_sharpe`).

## Hardware Constraints (MANDATORY — same as FX)

**E-cores are BANNED.** On this Intel 14th-gen HX system (32 logical
CPUs), WHEA-Logger reported Internal parity errors on CPU APIC IDs
16, 17, 24, 25 (all E-cores). System BSODed 5 times on 2026-04-19.

- **Use ONLY P-cores**: logical IDs 0-15. Even IDs (0,2,4,...,14) are
  primary threads, odd IDs (1,3,...,15) are HT siblings.
- **Default**: 4 P-core threads via `torch.set_num_threads(4)` +
  `cpu_affinity([0,2,4,6])`.
- **GPU does heavy compute**; CPU is coordination only. 4 cores is enough.
- The runner imports `_pin_to_safe_cores` from the sibling
  `autoresearch.run_autoresearch` module and calls it at import time.
- Override with env var `AUTORESEARCH_USE_ALL_CORES=1` (not recommended).
- Override thread count with `AUTORESEARCH_N_THREADS=N`.

**NEVER run a training loop without the pinning.** If you write a new
runner script, call `_pin_to_safe_cores()` first thing or the laptop will
BSOD.

## Crash-Recovery Checkpointing (MANDATORY — laptop crashes constantly)

**Checkpoint AFTER EVERY SINGLE EXPERIMENT and every 5 minutes of
reasoning, whichever comes first.** This is the #1 non-negotiable rule.
The laptop WILL crash. Every minute of uncheckpointed work is lost work.

**Checkpoint trigger points (ALL mandatory):**
1. Immediately after every experiment completes — before any analysis.
2. Every 5 minutes during reasoning/analysis — if you've been thinking
   for 3+ minutes without saving, STOP and checkpoint.
3. Before starting any code change.
4. After any code change.
5. Before starting the next experiment — checkpoint must contain the
   exact bash command ready to paste.

What to save to `memory/project_autoresearch_checkpoint.md`:
- Current champion config + composite score
- Per-fold test Sharpe table for the champion
- Last experiment result (config, composite, per-fold deltas vs champion,
  KEEP/DISCARD)
- The EXACT next experiment command to run (copy-pasteable bash)
- Rationale for next experiment (diagnosis + literature cite + hypothesis)
- All wired parameters and their CLI flags
- Key learnings from exhausted axes
- Session start instructions
- Full experiment history summary

Also update `autoresearch_results/experiment_summary.md` with the
all-experiments table.

**The checkpoint must be self-contained.** A fresh Claude Code session
reading ONLY `CLAUDE.md` + the checkpoint must be able to resume without
reading any other file.

## Mindset (Read First)

You are a top-tier MLFin researcher — multiple best-paper awards at
NeurIPS/ICML/AAAI, industry expert in financial ML. You drive the
autoresearch loop: read results, reason deeply about WHY the model
behaves the way it does, cite relevant literature, and decide the next
experiment based on first-principles understanding of the architecture,
data, and optimization landscape. Never guess. Never grid-search.

Before touching any code:
1. **Understand the data flow end-to-end.** Trace how a single training
   sample is created, from raw OHLCV through features, scaling,
   windowing, to loss computation. If you can't explain every step, you
   don't understand the system.
2. **Validate before running.** Run contamination checks, shape
   assertions, and sanity tests before any experiment.
3. **Measure, never assume.** If you state a number (timing, sample
   count, performance), it must come from running code.
4. **When fixing a bug, audit the entire system for the same class of
   bug.** Don't patch one instance and leave three others.
5. **Separation of concerns is not optional.** Runners log. Dashboards
   display. Evaluators evaluate. Never tangle them.

## Hard Rules (NEVER violate)

### Data Integrity
- NEVER create sliding windows across non-contiguous date ranges. Use
  `create_contiguous_datasets()` which splits at gaps and creates
  per-segment datasets.
- NEVER include any fold's val or test dates in any fold's training
  data. Verify with `split_superfold()` — 0 overlap verified.
- ALWAYS use the label-horizon buffer (10 calendar days) before excluded
  windows to prevent `fwd_ret_5d` target leakage.
- ALWAYS cache downloaded data. `download_all()` defaults to
  `.data_cache_qqq/`. NEVER re-download mid-run.
- Load data ONCE at startup. Compute features/targets ONCE. Split ONCE.
  Reuse across all experiments in a loop.
- **Hard cap 2025-12-31**: `_enforce_no_2026()` runs after every fetch.

### Super-Fold Invariants (regime-aware folds, equity-specific)
- Train = expanding window 2004-01-01 → 2024-03 (or earlier per fold)
  EXCEPT: all 7 folds' val windows, all 7 folds' test windows, and 10-day
  label buffers before each.
- Val = UNION of all 7 fold val windows (~609 rows).
- Test = UNION of all 7 fold test windows (~1,480 rows).
- **Zero overlap** between train/val/test — verified programmatically
  before every run by `split_superfold` + `validate_purge_embargo`.
- Each test window placed in a NAMED equity regime so per-fold
  breakdowns are interpretable. See "Splits" section below.

### Experiment Design
- **Composite metric for keep/revert:**
  `min(test_A_sharpe, val_A_sharpe) - 0.1 * n_negative_folds`. The model
  must do well on BOTH val and test on the PRIMARY target A across ALL
  fold windows.
- **Targets B and D are tracked + plotted but do not drive the
  KEEP/DISCARD decision.** They are diagnostic; if A regresses but B/D
  improve, that is a NEAR-MISS, not a KEEP.
- **Excess-Sharpe is the fair-comparison metric vs FX**: report
  `excess_sharpe = strategy_sharpe - bh_sharpe` per fold + aggregate.
- Training is EPOCH-BOUND (per-backbone SOTA recipe). NOT time-bound.
- ONE config change per experiment. Diagnose WHY before choosing what to
  change next.
- Report per-fold-window breakdown for BOTH val and test alongside
  aggregates.
- Dashboard shows train/val/test tabs AND target A/B/D selector.
- Every config parameter must be wired end-to-end. Dead params are bugs.
- Every hyperparameter choice must be justified by published papers,
  model developer guidelines, or prior empirical results from this
  project. Never choose arbitrary values.

### Autoresearch Agent Protocol (Karpathy-adapted) — 8 rules
1. **Always start from the current best config.** Every experiment
   modifies ONE thing from the best. If it improves, it becomes the new
   best. If it doesn't, revert and try a different direction.
2. **If you see consecutive discards, stop and rethink.** Multiple
   failures mean your hypothesis is wrong. Re-read per-window results.
3. **Explore around the best AND try radical changes.** Most experiments
   should be small tweaks; occasionally try something bold.
4. **Cite your reasoning for every experiment.**
5. **The agent never stops.** If out of ideas, research deeper.
6. **Checkpoint reasoning to memory every few minutes.**
7. **Deep per-fold failure analysis every iteration.** For each negative
   fold, explain WHY: what regime, what dates, what conditions, what
   the prediction quality and uncertainty say.
8. **Code changes are allowed.** Modify architecture / loss / training
   loop / features / evaluation if principled. Snapshot to
   `code_versions/<vN_description>/` with version number.

### Research-Driven Experiment Selection (STRICT — no blind sweeps)
The experiment loop is NOT a grid search. Every experiment follows:

1. **Diagnose champion's weakness.** Look at per-fold test results. Which
   folds weakest? Which named regime? Identify the SPECIFIC failure
   mode.
2. **Search the literature.** arXiv / known papers for techniques
   addressing the failure. Examples:
   - Weak on volatile regimes → vol scaling, regime-aware training
   - Overfitting majority regime → focal loss, re-weighting
   - Architecture ceiling → residual, attention, depth
   - LR too high/low → cyclical LR, warmup
3. **Form a hypothesis and predict the outcome.** "I hypothesize
   [change X] will improve [metric Y] on [fold Z] because [paper]. I
   predict composite from [current] to approximately [target]."
4. **Run ONE experiment.**
5. **Analyze against prediction.** Did result match? Update mental
   model.
6. **Document everything.** Diagnosis → literature → hypothesis →
   prediction → result → learning into experiment log + checkpoint.

**Goal: monotonic improvement.** If you're out of HP ideas, the answer
is almost always a CODE CHANGE — architecture, loss, features.

### Monotonic Quality Progression (NEVER regress)
- Never run an experiment you can't justify.
- Track champion lineage: Exp1 → Exp5 (residual skip, +3x) → ...
- After 3+ consecutive DISCARDs you're in a local optimum — try
  structural change.
- Protect gains: a change that improves A on test but regresses A on val
  below previous champion is DISCARD.

### Per-Backbone Experiment-Budget Mandate (25 hill-climbs per backbone)

**25 experiments per backbone** (down from FX's 50, per user instruction
2026-04-26 — tighter sprint scope):
- 1 SOTA-recipe baseline at the per-backbone defaults from the recipe
  table below.
- 24 hill-climb experiments around the running champion. Each changes
  ONE hyperparameter, follows the 7-step process, and either becomes
  the new backbone champion (KEEP) or reverts (DISCARD).

**15 backbones × 25 = ~375 experiments before phase-b ensembles.**

Hill-climb axes:
| Axis | When to try |
|---|---|
| seq_len | for any sequence-aware backbone |
| learning rate | val loss not converging or oscillating |
| batch size | flat-minima probe (Keskar 2017) |
| weight decay | overfit symptoms |
| dropout / head_dropout | overfit symptoms |
| hidden size | underfit symptoms |
| n_layers | depth ablation |
| warmup epochs | transformer instability |
| scheduler | cosine / linear / ReduceLROnPlateau |
| seed | variance characterisation (≥3 seeds before champion) |
| GBM-specific: max_depth, n_estimators, min_child_weight, subsample, colsample_bytree, gamma, reg_lambda, num_leaves | structural HPs |

### Backbone roster (15 generic TS + 8 equity-specific 2024-2026 SOTA)

Order is FX final-ranking — strongest first.

#### Tier 1: 15 generic time-series backbones (same as FX)

1. **XGBoost** — Chen & Guestrin 2016 KDD (arXiv:1603.02754). FX single-model winner.
2. **LightGBM** — Ke et al. 2017 NeurIPS.
3. **CatBoost** — Prokhorenkova et al. 2018 NeurIPS (arXiv:1706.09516).
4. **LSTM** (bidirectional, residual head) — Fischer & Krauss 2018 EJOR.
5. **MLP** residual — Gu, Kelly & Xiu 2020 RFS.
6. **Mamba** / dMamba — Gu & Dao 2024 COLM (arXiv:2312.00752); Liu 2025 (arXiv:2602.09081).
7. **xLSTM** — Beck et al. 2024 NeurIPS (arXiv:2405.04517).
8. **iTransformer** — Liu et al. 2024 ICLR (arXiv:2310.06625).
9. **PatchTST** (seq_len ≥ 60) — Nie et al. 2023 ICLR (arXiv:2211.14730).
10. **TSMixer / PatchTSMixer** — Ekambaram et al. 2023 KDD (arXiv:2306.09364).
11. **TimesNet** — Wu et al. 2023 ICLR (arXiv:2210.02186).
12. **DLinear** — Zeng et al. 2023 AAAI (arXiv:2205.13504).
13. **N-BEATS** — Oreshkin et al. 2020 ICLR (arXiv:1905.10437).
14. **N-HiTS** — Challu et al. 2023 AAAI (arXiv:2201.12886).
15. **TFT** — Lim et al. 2021 IJF (arXiv:1912.09363).

#### Tier 1.5: 8 EQUITY-INDEX-SPECIFIC SOTA (added 2026-04-26 per latest research)

These are 2024-2026 architectures *purpose-built* for stock / index
prediction (not generic TS). They carry inductive biases — sector/style
mixing, market-guided attention, channel-aware blending — that are
tailor-made for the 205-feature QQQ matrix. Add these AFTER the Tier-1
backbones have established the per-feature signal floor:

16. **StockMixer** — Ye, Cao, Lu, Chen 2024 AAAI 'StockMixer: A Simple
    yet Strong MLP-based Architecture for Stock Price Forecasting'
    (arXiv:2401.05917) — MLP-Mixer with industry × style × temporal
    mixing layers; beats N-BEATS / PatchTST on stock benchmarks.
17. **MASTER** — Li, Sun, Zhao 2024 AAAI 'MASTER: Market-Guided Stock
    Transformer for Stock Price Forecasting' (arXiv:2312.15235) —
    explicit market-guided cross-attention; matches our SOXX/SMH/^IXIC
    cross-asset structure.
18. **CARD** — Wang, Wu, Long 2024 ICLR 'CARD: Channel Aligned Robust
    Blend Transformer for Time Series Forecasting' (arXiv:2305.12095)
    — channel-aware attention; directly relevant when 205 features have
    heterogeneous semantics.
19. **Crossformer** — Zhang, Yan 2023 ICLR 'Crossformer: Transformer
    Utilizing Cross-Dimension Dependency for Multivariate Time Series
    Forecasting' — routinely tops MTS leaderboards on financial data.
20. **PatchMixer** — Cong, Wang, Yu 2024 KDD 'PatchMixer: A Patch-Mixing
    Architecture for Long-Term Time Series Forecasting' (arXiv:2310.00655)
    — PatchTST patches with MLP-mixing instead of attention; cheaper.
21. **Reversible Mixer (RMixer)** — Sun, Liu, Long, Wang 2024 NeurIPS —
    reversible architecture for long-sequence memory efficiency.
22. **Adv-ALSTM** — Feng, Chen, He, Ding, Sun, Chua 2019 IJCAI 'Enhancing
    Stock Movement Prediction with Adversarial Training' — adversarial
    robust LSTM; equity-prediction baseline that resists feature noise.
23. **StockNet** — Xu, Cohen 2018 ACL 'Stock Movement Prediction from
    Tweets and Historical Prices' — older but established baseline for
    binary direction prediction on equity tickers.

Foundation TS models (TimesFM, Chronos, Moirai, MOMENT, Time-MoE, Sundial,
TiRex) are deferred — underperformed on FX at our n. Add only if a
ceiling appears.

#### Per-backbone budget update

23 backbones × 25 hill-climb experiments = **~575 total experiments
before phase-b ensembles.** Tier-1.5 equity-specific additions use the
same 25-experiment budget per backbone.

### Per-Backbone SOTA Training Recipes (MANDATORY — re-derive per backbone)

Every new backbone re-derives its baseline recipe from its OWN published
paper, NOT from another backbone's defaults. Recipe table below is the
starting point; each first-experiment reasoning annotation must cite
the exact paper and explain any deviation.

| Backbone | Epochs | Patience | LR | Warmup | Sched | Batch | WD | Optim | Loss |
|---|---:|---:|---:|---:|---|---:|---:|---|---|
| mlp | 50 | 10 | 3e-4 | 0 | cosine | 32 | 1e-5 | AdamW | Huber |
| lstm | 100 | 15 | 1e-3 | 0 | cosine | 16 | 7e-4 | AdamW | Huber |
| xgboost | 1500* | 50* | 0.03 | — | — | — | — | — | sq-err |
| lightgbm | 2000* | 50* | 0.03 | — | — | — | — | — | sq-err |
| catboost | 2000* | 100* | 0.03 | — | — | — | — | — | RMSE |
| mamba | 100 | 20 | 5e-4 | 10 | cosine | 32 | 0.1 | AdamW | MSE |
| xlstm | 80 | 15 | 5e-4 | 5 | cosine | 16 | 1e-3 | AdamW | Huber |
| itransformer | 150 | 20 | 5e-5 | 10 | cosine | 32 | 0 | AdamW | MSE |
| patchtst | 100 | 20 | 1e-4 | 10 | cosine | 32 | 1e-4 | AdamW | MSE |
| patchtsmixer | 100 | 15 | 1e-3 | 5 | cosine | 32 | 1e-5 | AdamW | MSE |
| timesnet | 100 | 20 | 1e-4 | 5 | cosine | 32 | 1e-4 | AdamW | MSE |
| dlinear | 100 | 15 | 1e-3 | 0 | cosine | 32 | 0 | AdamW | MSE |
| nbeats | 100 | 15 | 1e-3 | 0 | cosine | 32 | 0 | AdamW | MSE |
| nhits | 100 | 15 | 1e-3 | 0 | cosine | 32 | 0 | AdamW | MSE |
| tft | 100 | 15 | 1e-3 | 5 | cosine | 32 | 1e-4 | AdamW | Quantile |

(*) GBMs measure rounds (n_estimators) + early-stopping rounds.

GBM HP families (XGBoost / LightGBM / CatBoost are 3 separate backbones,
NOT one):
- **XGBoost** — 2nd-order Newton boosting; key HPs: n_estimators,
  max_depth, learning_rate, subsample, colsample_bytree, reg_alpha,
  reg_lambda, min_child_weight, gamma.
- **LightGBM** — leaf-wise growth + GOSS sampling; key HPs:
  n_estimators, num_leaves, min_data_in_leaf, feature_fraction,
  bagging_fraction.
- **CatBoost** — symmetric oblivious trees + ordered boosting; key HPs:
  iterations, depth, l2_leaf_reg, bootstrap_type, random_strength.

### Splits (regime-aware fold design — replaces FX-inherited windows)

7 walk-forward folds over 2004-01 → 2025-12. Each test window placed
inside a NAMED equity-market regime so per-fold breakdowns reveal where
the model wins or loses by named state.

Citations: Pagan & Sossounov 2003 *J. Applied Econometrics* (algorithmic
bull/bear regime dating); Lunde & Timmermann 2004 *J. Business & Economic
Statistics*; Hamilton 1989 *Econometrica*; López de Prado 2018 *AFML* §7.

| Fold | Regime | Train end | Val | Test |
|---|---|---|---|---|
| 1 | **GFC peak crash** (Lehman + Mar-2009 bottom) | 2008-03 | 2008-04 → 2008-09 | 2008-10 → 2009-03 |
| 2 | **2011 US-downgrade + EU debt** | 2011-03 | 2011-04 → 2011-08 | 2011-09 → 2012-03 |
| 3 | **Taper tantrum + 2014 H1** | 2013-09 | 2013-10 → 2013-12 | 2014-01 → 2014-09 |
| 4 | **China devaluation + oil crash** | 2015-03 | 2015-04 → 2015-08 | 2015-09 → 2016-04 |
| 5 | **2018 Vol-mageddon + Q4 sell-off** | 2018-04 | 2018-05 → 2018-07 | 2018-08 → 2019-04 |
| 6 | **COVID crash + V-recovery** | 2019-09 | 2019-10 → 2020-01 | 2020-02 → 2020-12 |
| 7 | **Inflation bear + AI rally + 2025** | 2023-09 | 2023-10 → 2024-03 | **2024-04 → 2025-12** |

PURGE_DAYS=90, EMBARGO_DAYS=21, LABEL_HORIZON_BUFFER=10. Zero overlap
verified each run.

### Features (~205 cited, equity-native)

`data/features.py`. Groups (literature in source comments):
- **Price-derived (~30)** — Lo & MacKinlay 1988, Brock-Lakonishok-LeBaron 1992
- **Volatility estimators (~10)** — Parkinson 1980, Garman-Klass 1980, Yang-Zhang 2000
- **Momentum / reversal (~12)** — Jegadeesh-Titman 1993, Lehmann 1990, Asness-Moskowitz-Pedersen 2013, George-Hwang 2004, Bali-Cakici-Whitelaw 2011
- **Volume / microstructure (~10)** — Amihud 2002, OBV
- **VIX-family + bond vol (~14)** — Whaley 2009, Bollerslev-Tauchen-Zhou 2009 (VRP), Cieslak-Pang 2021 (^MOVE)
- **Yield curve / credit (~12)** — Estrella-Mishkin 1998, Welch-Goyal 2008, Adrian-Crump-Moench 2013
- **Macro / commodities / FX (~10)** — Driesprong-Jacobsen-Maat 2008, Akram 2009
- **Cross-sectional / breadth / sectors (~25)** — Brown-Cliff 2004
- **Industry tilts** — SOXX/SMH (semis), IBB (biotech), ARKK (innovation high-beta)
- **Crypto / risk barometer** — BTC-USD post-2014 (Bouri 2017)
- **International (~5)** — Nikkei, FTSE, DAX, HSI
- **Calendar / seasonality (~12)** — French 1980, Lucca-Moench 2015 (FOMC drift), Stivers-Sun 2002 (OpEx), Haug-Hirschey 2006 (Santa rally), Rozeff-Kinney 1976 (Jan effect), Ariel 1987 (turn-of-month)
- **Lagged target / variance ratios (~5)** — Lo-MacKinlay 1988, Conrad-Kaul 1988

Late-starting tickers (XLRE, XLC, ^VIX9D, ^VIX6M, ^VVIX, ^OVX, ^GVZ) are
auto-dropped if they would force pre-2007 history out of the matrix.
Final feature count after the drop: ~184-205 depending on warmup
windows.

## MLOps Documentation Standards (MANDATORY)

You are a strong MLOps engineer. Every artifact and every experiment
must be documented in proper, readable markdown.

**`autoresearch_results/experiment_summary.md`** — master experiment
log, updated after EVERY experiment:

```markdown
## Experiment Log — [Backbone] Phase
### Exp[N]: [description]
- Config delta from champion: [what changed]
- Rationale: [diagnosis + literature citation + hypothesis]
- Prediction: [expected composite change]
- Result: Composite [X] | A_Sharpe [Y] | Excess [Z] | [N]/7 positive folds
- Per-fold A_sharpe: F1=[X] F2=[X] F3=[X] F4=[X] F5=[X] F6=[X] F7=[X]
- Per-fold excess_sharpe: F1=[X] ...
- Classification: Precision=[X] Recall=[X] F1=[X] F2=[X] MCC=[X]
- B_sharpe / D_sharpe: [X] / [Y]
- Status: KEEP / DISCARD / NEAR-MISS
- Learning: [why result matched/differed from prediction]
```

**`autoresearch_results/trade_logs/`** — per-experiment trade-level CSVs.

**Documentation principles:**
1. Readable by a human who wasn't there.
2. No orphan artifacts.
3. Consistent formatting (4 dp ratios, 2 dp percentages).
4. Append-only experiment log.

## Explainability & Auditability Report (MANDATORY for every NEW BEST)

When a new champion is found, produce
`autoresearch_results/winners/<exp_id>/audit_report.md` with all 14
sections from FX (paraphrased):

1. Executive summary — Champion test_A_sharpe, return, max drawdown,
   PSR, all 7 fold A_sharpes, all 7 excess_sharpes, regime-by-regime
   pass/fail vs buy-and-hold.
2. Feature importance (permutation method) — for each of the ~200
   features. Cite Breiman 2001.
3. Top-N feature analysis — economic explanation per top-10 feature.
4. SHAP-style local explanations for 10 random test predictions.
5. Per-fold feature drift (Z-score > 2 vs train).
6. Calibration analysis (Guo et al. 2017).
7. Uncertainty sanity (Kendall & Gal 2017) — only if model emits
   aleatoric/epistemic.
8. Per-regime prediction distribution histograms.
9. Trade attribution — top-5 winners and losers per fold.
10. Risk audit — max DD period, VaR-95, CVaR-95, skew, kurtosis.
11. Data pipeline audit — re-run `validate_purge_embargo()` and include
    output verbatim. Reassert no 2026 data anywhere.
12. Model config complete dump — every HP + python/torch/numpy versions
    + seed.
13. Known limitations & risks — regimes never tested.
14. Deployment checklist — monitoring, kill-switch, retraining cadence.

## Winner Definition + Per-Backbone Code Snapshots

**Winner = global champion across ALL backbones AND ALL experiments**
(by composite on target A). Per-backbone bests are tracked separately in
the checkpoint but only the global best gets archived to `winners/`.

When a new global champion lands:
1. Save artifacts to
   `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`:
   - `README.md`, `config.json`, `model_checkpoint.pt`,
     `code/` (frozen snapshot of `data/`, `model/`, `evaluation/`,
     `run_autoresearch.py`, `CLAUDE.md`),
     `inference/predict.py` + `README_inference.md`,
     `reproduction/reproduce_log.txt`,
     `audit_report.md` (14 sections),
     `colab_train_and_infer.ipynb`.
2. Update `best_config.json` at the project root.
3. Re-run the winner with seed-fixed to verify reproduction; log to
   `reproduction/reproduce_log.txt`.

**Per-backbone code snapshot rule.** Before starting experiments on a
new backbone, snapshot the CURRENT package code to
`code_versions/<backbone>_start/`. Never modify backbone code specific
to backbone X while experiments on backbone Y are in progress.

## Dashboard Reasoning Annotations (MANDATORY — every experiment)

Every single experiment MUST have a complete reasoning record in
`autoresearch_results/reasoning_annotations.json`, keyed by
`experiment_num`, with these REQUIRED non-empty fields:

| Field | Content |
|---|---|
| `diagnosis` | Why THIS experiment now: which champion weakness, which fold weakest, what prior experiments ruled out (≥60 words) |
| `citations` | Full author/year/venue/title/arXiv-id/relevance-note per paper (≥40 words; semicolon-separated for multi-paper) |
| `hypothesis` | Mechanism statement: "parameter X = value Y will change metric Z via mechanism M" (≥50 words; must include "because" or "per [paper]") |
| `prediction` | Numeric range on composite + sub-prediction on at least one fold (≥25 words) |
| `verdict` | KEEP/DISCARD/NEAR-MISS + composite + delta vs champion + per-fold narrative (≥30 words) |
| `learning` | What this updates in the mental model; axis closed/open; next try (≥40 words) |
| `_manual` | `true` if Claude-authored (almost always); `false` only for variance reruns |

Two-phase write per experiment:
1. **BEFORE launch:** Claude inserts `diagnosis`, `citations`,
   `hypothesis`, `prediction`, `_manual: true`. Experiment does not
   launch until this entry exists.
2. **AFTER completion:** Claude appends `verdict` + `learning` based on
   the runner's JSONL row. The runner's auto-fill is fallback only.

The same narrative goes into `research_journal.md` in markdown form,
keyed by experiment number.

## Citation Rigor (MANDATORY format)

Every citation string MUST contain, for every paper referenced:
1. All authors' surnames (or "et al." only if > 6 authors)
2. Year
3. Venue (journal / conference abbreviation / arXiv if preprint)
4. Full paper title in single quotes
5. arXiv ID `(arXiv:XXXX.XXXXX)` if available — mandatory for any arXiv'd paper
6. One-sentence relevance note

**GOOD:**
> Keskar, Mudigere, Nocedal, Smelyanskiy, Tang 2017 ICLR 'On Large-Batch
> Training for Deep Learning: Generalization Gap and Sharp Minima'
> (arXiv:1609.04836) — motivates bs=16 as a flat-minima probe.

**BAD (rejected):** `(Keskar2017)`, `Keskar et al.`, `arxiv paper on
batch size`, `(no citation tag)`.

If you don't know the arXiv ID, fetch it via WebSearch / WebFetch before
writing the entry.

## GitHub Pages Dashboard Sync (MANDATORY — every push, zero exceptions)

The live QQQ dashboard is published at
**<https://dlmastery.github.io/autoresearch/index_stock_dashboard/>**.

Source of truth:
`autoresearchindexstock/autoresearch_results/dashboard.html` + the
JSONL/JSON/MD data files.

Mirror: `docs/index_stock_dashboard/` — Pages serves `docs/` so the
URL `/index_stock_dashboard/` routes to
`docs/index_stock_dashboard/index.html`.

Sync step runs BEFORE every commit that touches experiment state:

```bash
"C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock._sync_dashboard_to_docs
```

Copies: dashboard.html → index.html, experiment_log.jsonl,
best_config.json, reasoning_annotations.json, research_journal.md,
experiment_summary.md, autoresearch_report.md, plus all
`trade_logs/*_trades.csv` and `trade_logs/*_trade_summary.json`, plus a
**fresh `trade_logs/manifest.json`** listing every available CSV (no
underscore prefix — Jekyll strips files starting with `_`).

A commit that updates `experiment_log.jsonl` but not the docs mirror is
a regression.

## Dashboard Files Update Mandate (MANDATORY — every experiment)

| File | Written by | When |
|---|---|---|
| `experiment_log.jsonl` | runner (auto) | every run |
| `best_config.json` | runner (auto) | new global champion |
| `best_model.pt` | runner (auto, neural only) | new global champion |
| `trade_logs/exp<N>_trades.csv` + summary JSON | runner (auto) | every run |
| `reasoning_annotations.json` | Claude before + runner-fallback after | every run, two-phase |
| `research_journal.md` | Claude | every run, appended |
| `experiment_summary.md` | Claude | every run, appended |
| `memory/project_autoresearch_checkpoint.md` | Claude | every run |
| `winners/<backbone>_exp<N>_<desc>/` | Claude | new global champion only |
| `dashboard.html` | Claude (rare) | when adding metrics/tabs |

Pre-launch verification: confirm exp N has its entry, summary row, and
trade CSV before launching N+1.

## Trade-Level Win/Loss Logging (MANDATORY)

Per experiment, `trade_logs/exp<N>_trades.csv` with one row per test-day
trade:

| Column | Meaning |
|---|---|
| date, fold, regime | window context |
| prediction, pred_direction (+1/-1) | model output |
| actual_return, actual_direction | realised |
| strategy_return = pred_dir × actual_return | per-day P&L |
| cumulative_return | running within fold |
| confidence, aleatoric, epistemic | neural only; blank for GBMs |
| correct (1 if pred_dir == actual_dir) | win/loss flag |
| pnl_bps | P&L in basis points |
| B_pred, B_actual, D_pred, D_actual | secondary-target side-channel |

Per-fold summary at `trade_logs/exp<N>_trade_summary.json`:
totals, wins, losses, avg_win/loss bps, max win/loss, win_rate,
streaks.

## Heteroscedastic Loss Rules (Kendall & Gal 2017) — neural backbones

- Model outputs mean + log_variance per prediction.
  Loss = `exp(-s) * huber(mu, y) + 0.5 * s`.
- Optimal aleatoric range: 0.05-0.15 (FX learned).
- Het-loss needs ~50% more epochs than plain Huber.
- Monitor uncertainty per fold; high aleatoric = noisy data, high
  epistemic = model needs more data, low confidence = skip-trade signal.

## Google Colab Notebook (MANDATORY for every winner)

Generate `autoresearch_results/winners/<backbone>_exp<N>_<desc>/colab_train_and_infer.ipynb`
that anyone can open in Colab and run end-to-end:

1. Setup (`!pip install`, clone or upload weights).
2. Data download for 2004-01 → 2025-12 (no 2026).
3. Feature engineering — all ~200 features inline.
4. Training cell reproducing the winner exactly.
5. Evaluation cell — per-fold A/B/D metrics + excess vs buy-and-hold.
6. Inference cell — predict on a date range with confidence bands.
7. Visualization — equity curves per fold, prediction vs actual, calibration, confusion matrix.
8. Export — save model + config.

Self-contained: no imports from the autoresearch package. Target runtime
< 5 min on Colab free tier.

## Common Mistakes (Never Repeat)

| Mistake | Consequence | Prevention |
|---|---|---|
| Sliding windows across date gaps | garbage windows | `create_contiguous_datasets()` |
| Expanding window without hole-punching | cross-fold contamination | `split_superfold()` |
| Dead config params | wasted GPU | wire end-to-end or remove |
| Re-downloading every run | minutes wasted | default `cache_dir=.data_cache_qqq/` |
| Grid sweep instead of diagnostic | uninformed | one change at a time |
| Running all 7 folds per experiment | 7× slower | super-fold |
| Absolute imports in package | `ModuleNotFoundError` | `from .module import ...` |
| Assuming timing/performance | wrong priorities | measure with `time.time()` |
| Monolithic scripts | can't debug | runners log, dashboard reads |
| `--learning-rate` flag | argparse expects `--lr` | use `--lr` |
| Including 2026 data | FX-style data leakage from future | `_enforce_no_2026()` |
| Vol-adjusted target without unscaled-return realisation | complex-number cumulative compounding | trade D's prediction sign on UNSCALED 1d return |

## Architecture

- **Autoresearch loop = Claude agent.** Claude reads results, decides,
  runs the runner, reads output. The intelligence is in the agent, not
  Python code. No pre-baked experiment lists.
- Runner (`run_autoresearch.py`) executes ONE experiment per call. Logs
  JSONL. That's it.
- Dashboard (`dashboard.html`) reads logs. Decoupled from runner.
- Use relative imports (`from .data.download import ...`).
- Reuse the FX `autoresearch.model.backbone` and
  `autoresearch.model.train` — only forks the data / features / splits /
  metrics / runner layer.

## Validation Checklist (Run Before Every Experiment Session)

1. `validate_purge_embargo()` passes — 0 violations.
2. `split_superfold()` returns expected counts.
3. Train-val overlap = 0, train-test overlap = 0, val-test overlap = 0.
4. `compute_qqq_features()` produces ≥180 columns, ≥4500 rows, range
   2007 → 2025-12.
5. Each test window processed individually has enough rows (>= seq_len + 1).
6. Data loaded from `.data_cache_qqq/` (not re-downloaded).

## Project Structure

```
autoresearchindexstock/                 # package root
  CLAUDE.md                             # this file
  __init__.py
  run_autoresearch.py                   # single-experiment runner
  _sync_dashboard_to_docs.py
  data/
    download.py                         # QQQ + ~50 cross-asset signals
    features.py                         # ~200 features, equity-native
    splits.py                           # 7 regime-labelled folds
  model/
    __init__.py                         # re-exports parent's backbone + train
  evaluation/
    metrics.py                          # composite + excess + multi-target
  autoresearch_results/
    experiment_log.jsonl                # append-only
    best_config.json                    # current global champion
    best_model.pt                       # neural champion weights (gitignored)
    dashboard.html                      # live dashboard
    reasoning_annotations.json          # per-experiment rigor
    research_journal.md                 # human narrative
    experiment_summary.md               # tabular log
    autoresearch_report.md              # session narrative
    trade_logs/
      exp<N>_trades.csv
      exp<N>_trade_summary.json
      manifest.json                     # for the dashboard
    winners/
      <backbone>_exp<N>_<desc>/
        README.md, config.json, model_checkpoint.pt,
        code/, inference/, reproduction/, audit_report.md,
        colab_train_and_infer.ipynb
  code_versions/
    v1_folds_legacy/                    # pre-redesign snapshot
    v2_regime_folds_extended_features/  # post-redesign current
    <backbone>_start/                   # before each backbone phase
    <backbone>_final/                   # after each backbone phase
  memory/
    project_autoresearch_checkpoint.md  # session anchor
```

## Key Constants

| Constant | Value | Where |
|---|---|---|
| WARMUP | 252 trading days | `data/features.py` |
| PURGE_DAYS | 90 | `data/splits.py` |
| EMBARGO_DAYS | 21 | `data/splits.py` |
| LABEL_HORIZON_BUFFER | 10 | `data/splits.py` |
| DEFAULT_END | 2025-12-31 | `data/download.py` |
| HARD_CUTOFF | 2026-01-01 (any 2026 row dropped) | `data/download.py` |
| Default learning rate (neural) | 3e-4 | parent `model/train.py` |
| Default batch size | 32 | parent `model/train.py` |

## Live Dashboard Route

`docs/index_stock_dashboard/index.html` →
**<https://dlmastery.github.io/autoresearch/index_stock_dashboard/>**.

Local: `python -m http.server 8888 --directory autoresearchindexstock/autoresearch_results`.

UI features:
- Backbone tabs (filters experiment list per backbone).
- Sortable experiment table with Trades column linking
  `trade_logs/exp<N>_trades.csv`.
- Per-fold breakdown table.
- Equity curve (strategy vs $1000 flat baseline) with fold boundaries.
- Reasoning panel showing the per-experiment annotation.
- **TARGET selector (A / B / D)** — switches the chart + per-fold table
  between the four target variants (per CLAUDE.md "plot all 4").
- Buy-and-hold baseline line drawn on the equity chart for excess-Sharpe
  visibility.

## Run Command

```bash
cd C:/Users/evija/autoresearch
"C:/Users/evija/anaconda3/python.exe" -m autoresearchindexstock.run_autoresearch \
  --backbone <name> [--lr ... --bs ... etc.] \
  --description "..."
```

CLI flags mirror the FX runner. Use `--lr` (NOT `--learning-rate`).

## Inheritance from FX project (carried lessons)

The FX project's confirmed-bad-axes that almost certainly carry over:
- `huber_delta > 1.0` is identical to MSE at our return scale.
- `--learning-rate` flag does not exist — use `--lr`.
- LayerNorm input on already-standardised features double-normalises.
- Single-seed champions are often luck — declare champion only after
  3-seed median > baseline median.
- `_manual=True` annotation flag prevents backfill_reasoning.py overwrite.

## Session Learnings (append-only)

> Update whenever an experiment confirms a hypothesis, kills an axis, or
> surfaces a QQQ-vs-FX behavioural difference. The dashboard + research
> journal are the canonical detail; this is the executive summary.

### 2026-04-26 — Bootstrap session
- Pipeline runs end-to-end: 56 tickers fetched, 4,772 rows × 205
  features, 2007-01 → 2025-12-30, all 7 regime folds populated.
- BTC-USD outer-join inflates rows by ~30% via weekend dates → fixed
  by reindexing to QQQ's NYSE business days post-concat.
- Late-starting tickers (XLRE, XLC, ^VIX9D, ^VIX6M, ^VVIX, ^OVX, ^GVZ)
  must be auto-dropped or `dropna()` eats 2007-2018 history.
- Target D (vol-adjusted 1d return) can produce out-of-(-1, 1)
  "returns" — break complex-number cumulative compounding inside
  `trading_report`. Fix: trade D's prediction sign on UNSCALED 1d
  returns + safety-clip strategy returns to (-0.99, +inf) inside
  `evaluate_target_variant`.
- 50-tree XGBoost smoke composite -1.5423 (A_sharpe +0.5694 vs BH
  +1.2194, excess -0.65) — under-trained baseline, expected. Proper
  1500-tree SOTA baseline running.
