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

## User Directives Log (Consolidated — read first, every session)

This section consolidates all explicit user directives received during the QQQ project. Each is binding. New session starts here.

### Session 2026-04-27 directives

1. **Cheap-first 25-experiment-per-backbone protocol**: order is cheap (MLP, LSTM) → medium (XGBoost, LightGBM, CatBoost) → heavy (CatBoost full, Mamba). Within tier, hill-climb from per-backbone champion (composite metric, NOT A_sharpe).
2. **Do not use seq>60**: QQQ super-fold val windows are too small (F3=63d, F5=63d). Above seq=60, val folds get skipped — composite metric becomes incomparable to historical experiments. (See "Per-backbone seq_len" section.)
3. **Per-backbone industry SOTA seq_len norm**: each backbone family has its own canonical seq from the founding paper. MLP=10 (Gu-Kelly-Xiu 2020), LSTM=10-240 (Fischer-Krauss 2018), Mamba=60+ (Gu-Dao 2024). Pick from the paper, capped at QQQ's seq=60 ceiling.
4. **Saturate GPU not CPU**: Intel HX silicon degradation (5 BSODs on 2026-04-19, WHEA E-core errors). Neural backbones (MLP, LSTM, Mamba, xLSTM) use CUDA. CPU pinned to P-cores [0,2,4,6] only. Future opportunity to GPU-accelerate GBMs.
5. **Don't stop and ask**: when forking, default to (b) and continue. Apply CLAUDE.md research-driven protocol per experiment.
6. **For each new experiment**: start from previous winner → deep per-fold analysis → SOTA arxiv research targeting the specific deficiency → arxiv-cited justification → THEN run. Dashboard's reasoning_annotations.json must reflect this fully.
7. **Read FX docs and apply rigorously**: paper.md, medium_article.md, research_journal.md, autoresearch_report.md. The FX project's 265-experiment trace is the protocol exemplar; QQQ should match its rigor.
8. **Architectural variants matter**: residual MLP (FX MLP champion), bidirectional LSTM, num_layers/hidden_size — the runner exposes `--num-layers` and `--hidden-size` flags; use them to explore architectural axes, not just optimisation HPs.

### Session 2026-04-28 directives

9. **Foundation models LAST**: order is cheap-tier → medium-tier → Mamba → Phase F (other-already-in-runner: PatchTST, PatchTSMixer, iTransformer, DLinear, N-BEATS) → Phase D (Stock-specific code adds: Adv-ALSTM, StockMixer, MASTER, PatchMixer, CARD, Reversible Mixer) → **Phase E SOTA April 2026 foundation models LAST** (Sundial, TimesFM 2.5, Chronos-2, Moirai 2.0, TiRex, MOMENT, Time-MoE, TimeMixer/++).
10. **Add 25 more MLP experiments (50 total)**: extended from 25 to 50 budget per user. MLP runs ~30s on GPU; cheap.
11. **Add 50 more LSTM experiments (75 total)**: extended from 25 to 75 budget per user. Try various SOTA LSTM types — vanilla LSTM, xLSTM (already in runner), and consider future code adds for AWD-LSTM, Mogrifier, etc.
12. **Always start every backbone with SOTA HP from arxiv latest research**: Bootstrap Rule. The first experiment of any new backbone must use the recipe from the latest arxiv paper for that architecture. Deviations require arxiv-cited justification. (See "Backbone Bootstrap Rule" section.)
13. **Always update dashboard + commit + push BEFORE moving to next experiment**: per-experiment commit rule. No batching. (See "Per-Experiment Sync + Commit Rule" section.)
14. **Phase priority within Phase D / Phase E**:
    - Phase D order: Adv-ALSTM (simplest port) → StockMixer → MASTER → PatchMixer → CARD → Reversible Mixer.
    - Phase E order: MOMENT-small → TimeMixer → TiRex → Chronos-2-bolt-small → Time-MoE-base → Moirai-small → TimesFM small → Sundial.
15. **Create proper task lists with meticulous checkpoints**: 14 tasks created (#32-45) covering all backbone budgets + cross-cutting work (deep-ensemble, confidence-weighted sizing, shuffle-test audit, calibration). Plus #46 for arxiv-cited feature engineering.
16. **For each backbone winner — detailed weak-points analysis, arxiv research deep, arxiv justification before starting**: re-emphasis of the 7-step research-driven protocol. Each new experiment requires per-fold deficiency identification + SOTA arxiv paper match + numeric prediction + verdict + learning. No grid search disguised as research.
17. **Research arxiv for features that might boost Sharpe**: per FX paper §6.5 'it pays more to engineer features than to replace the model'. QQQ feature gaps documented in task #46: VIX term-structure slopes, HYG-AGG credit spread, FOMC/CPI/NFP calendar dummies, yield curve slopes, cross-sectional NDX momentum, style factor returns, 0DTE options flow, news/LLM embeddings.

### Sticky operational rules (cumulative)

- **Per-experiment commit + push BEFORE next experiment** (rule 13)
- **SOTA-from-arxiv recipe for every new backbone Experiment 1** (rule 12)
- **Research-driven protocol per experiment** (rules 6, 16) — diagnose-cite-hypothesize-predict-run-analyse-checkpoint
- **seq_len ≤ 60 hard ceiling** (rule 2)
- **GPU not CPU** (rule 4)
- **Foundation models LAST** (rule 9)
- **Cheap → medium → heavy → Phase F → Phase D → Phase E** (rules 1, 9, 14)
- **Track everything in tasks** (rule 15)
- **Feature engineering is a parallel workstream** (rule 17, task #46)

### Mistakes I've made and the user has corrected

- ❌ Grid-searching HP without per-fold deficiency analysis (corrected after exps 58-60)
- ❌ Trying seq=120 without checking fold-size constraint (corrected by user)
- ❌ Treating seed instability as deficiency when actual deficiency was per-fold regime failure (corrected)
- ❌ Foundation models ordered first instead of last (corrected)
- ❌ FX→QQQ HP transfer naïveté (didn't realize paper-defaults > FX-tuned for smaller QQQ data)
- ❌ Missing dashboard fields for 56 experiments (classification metrics undetected, fixed retroactively)
- ❌ KEEP/DISCARD bug undetected for 23 experiments (Status column blank, fixed retroactively)
- ❌ Parallel-launch experiments causing annotation key drift (re-keyed retroactively)

## Per-Experiment Sync + Commit Rule (UPDATED 2026-04-28 — MANDATORY)

**Before launching every new experiment, sync the dashboard to docs/ AND commit + push to GitHub.** No batching. No "I'll commit after 5 more". Every experiment's full state lands on GitHub before the next one starts.

### Workflow (strict, no shortcuts)

After every experiment completes:
1. Read result, write verdict + learning into `reasoning_annotations.json`
2. Run `python autoresearchindexstock/_sync_dashboard_to_docs.py`
3. `git add autoresearchindexstock/autoresearch_results docs/index_stock_dashboard autoresearchindexstock/memory autoresearchindexstock/CLAUDE.md` (whatever changed)
4. `git commit -F .commit_msg.txt` with a focused message describing this single experiment + verdict
5. `git push origin master`
6. Verify `git status` clean
7. THEN pre-author next experiment annotation and launch

### Rationale

- Dashboard at https://dlmastery.github.io/autoresearch/index_stock_dashboard/ is the project's institutional memory. Stale dashboard = lying about what's been done.
- Per-experiment commits give precise rollback granularity if a code change breaks a future run.
- Forces a deliberate pause between experiments — prevents the rapid-fire grid-search drift CLAUDE.md prohibits.
- If the laptop crashes mid-experiment-batch (Intel HX silicon known issue, 5 BSODs on 2026-04-19), the worst loss is the SINGLE running experiment. Without per-experiment commits, ~10 experiments of state could be lost.

### Allowed exception

For very-cheap (<60s) MLP/LSTM rapid-fire bursts (3-5 sequential experiments at the same baseline) — single commit covering the burst is acceptable IF AND ONLY IF dashboard is synced + pushed at the END of the burst BEFORE moving to a different backbone.

### Enforcement

If you find yourself launching an experiment with `git status` not clean (uncommitted changes from prior experiment), STOP. Commit + push first. Re-read this section. The protocol exists because the institutional-memory and crash-recovery costs of skipping commits compound.

## Backbone Bootstrap Rule (UPDATED 2026-04-28 — MANDATORY)

**Every new backbone MUST start with the SOTA hyperparameter recipe from the latest arXiv paper for that architecture.** No "let's try the runner defaults". No "let's adapt FX-tuned values". The research-driven protocol begins from the paper's recommended recipe; deviation from the paper requires arXiv-cited justification in the reasoning annotation.

### Workflow per new backbone (Experiment 1 of its 25-budget)

1. **Pull the paper.** Read the experiments section and find the recommended HP recipe (lr, batch, epochs, patience, warmup, weight_decay, seq_len, head_dropout, optimizer, scheduler, loss).
2. **Record the recipe** in `reasoning_annotations.json` as a separate "sota_recipe" block. Include paper citation with arXiv ID.
3. **Note any QQQ-required deviations** explicitly:
   - seq_len ceiling at 60 (QQQ fold-size constraint — see "Per-backbone seq_len" section)
   - VRAM ceiling at 16 GB (QQQ hardware — see "GPU Memory Constraint" section)
   - Multi-pair vs single-asset training (most papers train on multi-instrument panels; QQQ is single asset)
4. **Run Experiment 1 at the paper recipe** (with documented deviations only). This is the bootstrap baseline.
5. **All subsequent 24 experiments** hill-climb from this paper-recipe baseline, ONE knob per experiment, arXiv-cited deficiency-driven per CLAUDE.md research-driven protocol.

### Examples of paper-recipes already used

| Backbone | Paper-recipe used in QQQ Exp 1 | Citation |
|---|---|---|
| MLP | lr=3e-4 bs=32 ep=50 pat=10 wd=1e-5 hd=0.25 hidden=128 (residual) | Gu, Kelly, Xiu 2020 RFS |
| LSTM | lr=1e-3 bs=16 ep=100 pat=15 wd=7e-4 hd=0.1/0.25 num_layers=2 hidden=128 (bidirectional) | Fischer & Krauss 2018 EJOR |
| dMamba | lr=5e-4 bs=32 ep=100 pat=20 wd=0.1 warmup=10 expand=4 d_state=16 num_layers=2 | Gu, Dao 2024 COLM (arXiv:2312.00752) + Liu 2025 dMamba (arXiv:2602.09081) |
| XGBoost | lr=0.03 max_depth=6 n_est=1500 (later refined to 0.01/4/1500) | Chen, Guestrin 2016 KDD (arXiv:1603.02754) |
| LightGBM | lr=0.01 depth=4 n_est=1000 leaf-wise growth | Ke et al. 2017 NeurIPS LightGBM |
| CatBoost | lr=0.01 depth=4 n_est=2000 ordered-boosting | Prokhorenkova et al. 2018 NeurIPS (arXiv:1706.09516) |
| xLSTM | lr=1e-3 bs=32 ep=100 pat=20 wd=7e-4 hd=0.1 (Beck 2024 default) | Beck et al. 2024 NeurIPS (arXiv:2405.04517) |
| PatchTST (when run) | lr=1e-4 bs=32 ep=100 pat=20 seq=60 patch_len=16 | Nie et al. 2023 ICLR (arXiv:2211.14730) |

### Required deviations table per backbone

| Constraint | Affected backbones | Deviation forced |
|---|---|---|
| seq_len ≤ 60 (QQQ fold-size ceiling) | All seq-models | Most papers used seq=240+ or 512+; we cap at 60 |
| 16 GB VRAM ceiling | Foundation models > 200M params | Use smallest checkpoint OR PEFT/LoRA |
| Single-asset training | All | Most papers used multi-instrument panels; we have just QQQ |
| 2,200-row training set | All | Many papers had 5000+ rows; smaller-data favours simpler architectures |

**If a paper's recipe explicitly requires a longer seq_len than 60 OR more parameters than fit in 16 GB**, document the deviation in the Experiment 1 reasoning annotation — and choose the smallest viable variant. Do NOT silently adapt; the arXiv-cited recipe is the starting truth.

## Post-Cheap-Tier Roadmap (UPDATED 2026-04-28 by user directive)

After cheap-tier 25-experiment budgets are met for fast backbones (MLP, LSTM at minimum), the priority order for new backbones is **STRICT**:

**Order — non-negotiable per user directive 2026-04-28:**
1. **Cheap-tier completion** (MLP, LSTM, xLSTM)
2. **Medium-tier completion** (XGBoost, LightGBM, CatBoost) + **Mamba family**
3. **Phase F other-already-in-runner** (PatchTST, PatchTSMixer, iTransformer, DLinear, N-BEATS) — small/cheap, in-runner
4. **Phase D Stock-specific backbones** (Adv-ALSTM, StockMixer, MASTER, PatchMixer, CARD, Reversible Mixer) — most QQQ-relevant, code-adds required
5. **Phase E SOTA April 2026 Foundation models LAST** (Sundial, TimesFM 2.5, Chronos-2, Moirai 2.0, TiRex, MOMENT, Time-MoE, TimeMixer/++) — heaviest VRAM, most code work, most uncertain transfer to QQQ

**Foundation models come LAST. Do not jump ahead.** They are the heaviest, hardest-to-debug, and most likely to need PEFT/LoRA infrastructure builds. Stock-specific architectures are MORE likely to pay off on QQQ since they are designed for equity prediction.

### Phase D — Stock-specific backbones (priority over foundation models)
Each requires CODE ADDS to `model/backbone.py` + `create_model()` routing + SOTA recipe + 25-exp budget.

| # | Backbone | Paper | arXiv | Why prioritize on QQQ |
|---|---|---|---|---|
| 1 | **MASTER** | Li, Sun, Liu, Yang, Sun 2024 AAAI 'Market-Guided Stock Transformer' | 2312.15235 | Market-guided attention specifically designed for stock prediction. Uses macro signals as guides to attend over equity features — exactly the QQQ feature panel structure. |
| 2 | **StockMixer** | Ye, Liu, Liu, Tian, Pang 2024 AAAI 'StockMixer: A Simple yet Strong MLP-based Architecture for Stock Price Forecasting' | 2401.05917 | MLP-mixer applied to industry × style × temporal axes of stock returns. Architecturally matches QQQ's 56-ticker cross-section + 205 features. |
| 3 | **CARD** | Wang, Wu, Long, Long 2024 ICLR 'CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting' | 2305.12095 | Channel-aligned attention robust to noisy financial features; outperforms PatchTST on noisy benchmarks. |
| 4 | **PatchMixer** | Cong, Yu et al. 2024 KDD 'PatchMixer: Patch-MLP-Mixer Time Series Forecasting' | 2310.00655 | Patches + MLP-mixing — combines PatchTST's patch tokenisation with TSMixer's MLP backbone. |
| 5 | **Adv-ALSTM** | Feng, Chen, He, Ding, Sun, Chua 2019 IJCAI 'Adversarial Attention LSTM for Stock Prediction' | 1810.09936 | Older but stock-specific: adversarial training + attentive LSTM specifically for daily stock returns. |
| 6 | **Reversible Mixer** | Sun et al. 2024 NeurIPS | TBD | Reversible long-sequence mixer. |

### Phase E — SOTA April 2026 foundation / general TS backbones (LAST per user directive)
Each requires CODE ADDS + recipe selection + memory-pre-flight (per "GPU Memory Constraint" — most are 200M-1B params).

| # | Backbone | Paper | arXiv | Notes |
|---|---|---|---|---|
| 7 | **Sundial** | Liu, Zhang, Wu, Long 2025 'Sundial' | 2502.00816 | 1T-token TimeBench pretrain + flow-matching loss; 500M-1B params (PEFT only) |
| 8 | **TimesFM 2.5** | Google 2025 (~500M) | (no arXiv yet — model card) | Decoder-only TS foundation, continuous quantile heads; PEFT |
| 9 | **Chronos-2** | Ansari et al. 2025 'Chronos-2' | 2510.15821 | Universal TS, top zero-shot; multi-quantile heads |
| 10 | **Moirai 2.0** | Woo et al. 2025 'Moirai 2.0' | 2511.11698 | Sparse MoE, multi-token prediction |
| 11 | **TiRex** | Auer, Pöppel, Pflüger, Brandstetter, Hochreiter 2025 'TiRex' | (xLSTM-decoder, NXAI 2025) | xLSTM-based decoder, retrieval-augmented; ~300M |
| 12 | **MOMENT** | Goswami, Szafer, Choudhry, Cai, Li, Dubrawski 2024 ICML 'MOMENT' | 2402.03885 | T5 masked-TS pretrain; small (40M) and large (385M) variants |
| 13 | **Time-MoE** | Shi et al. 2024 ICLR'25 'Time-MoE' | 2409.16040 | Sparse MoE decoder; 113M-453M variants |
| 14 | **TimeMixer / TimeMixer++** | Wang, Wu, Shi, Hu, Luo, Ma, Zhang, Zhou 2024 ICLR 'TimeMixer' | 2405.14616 | Multi-scale decomposition; small footprint |

### Implementation protocol per new backbone

For EACH new backbone in Phase 1 / Phase 2:
1. **Read the paper.** Document architecture, SOTA recipe, key inductive bias.
2. **Implement class** in `autoresearchindexstock/model/backbone.py` (or import from a HF/external package if licensed). Add to `BACKBONE_REGISTRY` and `create_model()` routing.
3. **Snapshot code** to `code_versions/<backbone>_start/` per the per-backbone-isolation rule.
4. **Pre-flight VRAM check** for 200M+ models (record in reasoning_annotation experiment 1).
5. **Run SOTA-recipe baseline** (experiment 1 — defaults from the paper).
6. **Hill-climb 25 experiments** per CLAUDE.md research-driven protocol (one knob per experiment, arXiv-cited rationale, per-fold deficiency-driven).
7. **Snapshot code** to `code_versions/<backbone>_final/` after the 25-exp budget closes.

Total effort: 6 stock-specific × 25 exps + 8 SOTA-foundation × 25 exps = **350 experiments** of additional research budget.

Within current Claude Code session, this is many sessions of work. The roadmap is committed; sequencing is bottom-up by complexity (Adv-ALSTM is older + simpler so easiest port; foundation models are heaviest).

### Order of attack within each phase

**Phase 1 within order:** Adv-ALSTM (simplest, 5-day port) → StockMixer (clean MLP-mixer port) → MASTER (transformer with market-guide) → PatchMixer → CARD → Reversible Mixer.

**Phase 2 within order:** MOMENT-small (40M, fits easily) → TimeMixer (no foundation pretrain, simpler) → TiRex (xLSTM-decoder, builds on existing xlstm code) → Chronos-2-bolt-small (9M-48M trivially fits) → Time-MoE-base (113M) → Moirai-small (14M-91M) → TimesFM small → Sundial.

## FX Project Learnings — Apply Rigorously to QQQ (UPDATED 2026-04-27)

The FX project (in sibling folder `autoresearch/`) reached composite **+9.186** (XGBoost Exp 203) with a 3-way GBM rank-average ensemble at test Sharpe **+9.4708**. QQQ peaks at **+1.32** (dMamba Exp 52). The 8-Sharpe-unit gap is partially structural (asset class) but also reflects FX-protocol learnings I have not yet rigorously applied to QQQ. This section is the consolidated transfer list. Deviation from these is itself a regression — if you find yourself bypassing one of these rules, STOP and re-read this section.

### Methodology — non-negotiable

1. **One change per iteration.** Strict. Composition of changes is forbidden. The FX protocol followed this for 265 experiments. I violated it once on QQQ (exp 28 changed both epochs and patience) — never again.

2. **Three consecutive DISCARDs = STOP, rethink mechanism.** Not "try the next HP axis". The FX paper explicitly warns: "Multiple failures mean your hypothesis about what to change is wrong. The answer is NOT more hyperparameter tweaks — it's a structural change: different architecture, different loss, different features, different training procedure." I violated this on the MLP grind (exps 27-30 + 58-60 = 7 consecutive DISCARDs) by continuing HP tweaks.

3. **Diagnose-before-cite-before-hypothesize.** Per-fold failure surface FIRST, then literature search, then hypothesis. Generic optimization papers (Smith 2017, Loshchilov 2019, Keskar 2017) without a fold-specific deficiency match are insufficient — that's grid search wearing research-driven costume.

4. **Citation rigor.** Every non-trivial change carries an explicit paper citation with full author/year/venue/title/arXiv ID. Generic "Bouthillier 2019 says LR is the variance lever" is too thin if the proposed change isn't actually about LR. Map paper to specific failure mode.

5. **Reasoning annotation is the primary scientific artifact.** The FX paper explicitly: "the reasoning log is the science, not the model". Every experiment writes verdict + learning post-hoc, not just diagnosis pre-hoc. If a session ends without verdict written, that's institutional-memory loss.

### Specific FX findings I have NOT yet applied to QQQ

These are direct gaps. Each is a candidate next experiment.

a) **Tree-friendly seq_len monotonic uplift (FX paper §4.3 Table 4).** FX trees showed composite +7.34 at seq=5 → +9.19 at seq=60 — monotonic improvement because flattened windows give trees a 6240-column feature matrix and trees split axis-aligned on any column. **QQQ trees have only been tested at seq=60.** GBM seq sweep on QQQ is a clear gap — try seq=5, 10, 20, 30, 45 against the current LightGBM exp 10 baseline.

b) **3-way GBM rank-average ensemble at seq=60 (FX paper §4.5 Table 5).** FX champion XGBoost +9.186 → ensemble +9.471 by averaging XGBoost + LightGBM + CatBoost at the SAME seq=60 with rank aggregation. **QQQ tried MEGA-5 (3 GBMs + LSTM + MLP at +0.876)** but NEVER the pure 3-way GBM rank-average. For QQQ this might underperform MEGA-5 (since GBMs don't dominate as in FX) — but it's the FX-paper deployment artifact and worth measuring.

c) **Shuffle-test audit for tree champions (FX paper §3.5).** FX runs an explicit leakage check: train tree model on permuted training labels, evaluate on real test set. FX result: aggregate test Sharpe +0.006 across permuted runs (no leakage). **QQQ has never run this audit.** Per FX-paper protocol: "if a tree model's reported test Sharpe exceeds the strongest deep-learning baseline by > +1.0, run shuffle test." On QQQ no GBM exceeds Mamba +1.32, but the shuffle-test discipline applies to ALL champions; should be added as a CI-style check.

d) **Off-by-one alignment audit (FX paper §3.5).** FX's first XGBoost run gave composite -1.61 due to `y = seg_tgt.values[seq_len:]` (lookahead by 1) vs evaluator's `[seq_len-1:]`. Fix gave +8.78 jump. **QQQ runner uses `seq_len - 1` already** (verified at run_autoresearch.py:457) — alignment correct. But mandatory: a `validate_data_contract.py` module that asserts (x, y) pairs from training match evaluator's FXDataset for a random mini-batch. Not yet ported to QQQ.

e) **Multi-pair-style training (FX paper §3.1, Limitations §6.6).** FX trains on 6 currency pairs concatenated → effective n_train ~3-5x larger than QQQ's single-asset 2200 rows. **QQQ's biggest data deficit.** Possible fix: train QQQ champion config on QQQ + SPY + IWM + VTI as panel (4× n_train), evaluate only on QQQ. Architecturally same model, different data composition. Requires runner extension.

f) **Per-backbone code snapshot to `code_versions/`.** FX rule: snapshot `model/backbone.py` and `run_autoresearch.py` before starting experiments on a new backbone. Prevents adjacent-backbone code changes from contaminating provenance. **QQQ has not been doing this.** Going forward: snapshot when switching backbone families.

g) **Median-of-k seed reporting for neural champions.** FX paper recommends k≥3 seed median + range. **QQQ champion exp 52 has 4-seed sweep (median +0.86, range 2.34) — already done.** Document this with the right framing in the audit report (seed=42 single-run +1.32 is the headline; +0.86 is the deployment expectation).

h) **Calibration analysis** — FX §5.4 calibration error 0.013 on champion. **QQQ has not done calibration.** Add to winner audit reports.

i) **Permutation feature importance** — FX §6.5 implicit; tree models reveal feature importance natively. **QQQ has not done this.** With LightGBM exp 10 at +0.48, importance ranking on the 205 features would tell us which features carry signal vs noise — possible feature ablation lever.

### What FX did better (process discipline)

- **265 experiments at the time of paper.** QQQ has 62. FX-level coverage requires ~5× more experiments.
- **DISCARD ratio ~60%** (FX paper §6.1). High DISCARD ratio is a sign of non-trivial hypotheses. QQQ DISCARD ratio is similar (~70%) — OK on this axis.
- **Self-reversal on evidence.** FX agent reversed prior architecture preference (deep-learning bias) to GBM after exp 175 alignment-fix +8.78 jump. QQQ similar reversal: dMamba expand=4 (FX-tuned) → expand=2 (paper default) at exp 48.
- **Append-only log, no rewrites.** Every experiment, including failures, is preserved. QQQ has done this.

### Key structural observation: QQQ's val folds are pathologically small

FX val windows are 180-200 days each. QQQ val windows: F1=96d, F2=105d, F3=63d, F4=105d, F5=63d, F6=84d, F7=123d. The composite metric `min(test_sh, val_sh) - 0.1·n_neg` is high-variance because the `min` is dominated by val_sh which is noisy at tiny n. The seq_len ceiling at 60 (F3/F5 limit) is a direct consequence. **This is the deepest single reason QQQ peaks lower than FX**: even a model with FX-equivalent test_sharpe is composite-clipped by a 4-prediction-day val Sharpe. Future structural fix candidate (out of session scope): enlarge F3/F5/F6 val windows in `data/splits.py` to ≥120 days each.

### Don't-repeat-on-QQQ list (mistakes I made already)

- ❌ Treating "seed instability" as the deficiency for 30+ exps when the actual deficiency was per-fold regime failure (F1 GFC anti-skill).
- ❌ Grid-search on cheap MLP/LSTM HP after 4+ DISCARDs without pivoting to structural change.
- ❌ Launching seq=120 experiment without checking fold-size compatibility (would have skipped 6/7 val folds).
- ❌ Running 56 experiments before noticing missing classification metrics in the JSONL (dashboard schema regression).
- ❌ Not running shuffle-test on Mamba champion +1.32 (still TODO).
- ❌ Building MEGA-5 ensemble with single-seed components instead of multi-seed-median components.
- ❌ Initially analyzing huber_delta=0.3 effect without realizing 0.3 > max QQQ residual 0.10 → loss is effectively MSE → my "F1 fix" hypothesis was based on wrong loss-shape analysis.

## Hard Rules (NEVER violate)

### Compute & GPU saturation (UPDATED 2026-04-27)

**Saturate the GPU, NOT the CPU.** This laptop has known Intel 14th-gen
HX silicon degradation (5 BSODs on 2026-04-19, WHEA parity errors on
E-cores). Treat the CPU as fragile.

- **Neural backbones (MLP, LSTM, Mamba/dMamba)**: must use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` (already wired in runner). Mamba's selective state-space scan is GPU-vectorised; LSTM's recurrence batches well on GPU; even MLP's matmul is GPU-faster.
- **CPU pinning (mandatory)**: 4 P-cores only — logical IDs `[0,2,4,6]` via `_pin_to_safe_cores()`. NEVER lift this without explicit user approval. E-cores 16-31 are BANNED (WHEA parity errors caused 5 BSODs on 2026-04-19).
- **Override knobs**: `AUTORESEARCH_USE_ALL_CORES=1` (NOT recommended), `AUTORESEARCH_N_THREADS=N`.
- **GBMs (XGBoost / LightGBM / CatBoost)**: currently CPU-only via the wrapper — acceptable for short runs. Future opportunity: enable `tree_method='gpu_hist'` (XGBoost), `device='gpu'` (LightGBM), `task_type='GPU'` (CatBoost) to offload further work to the RTX 4090 Laptop GPU and reduce sustained CPU load. Pre-flight any such change with HWiNFO64 monitoring.
- **Pre-flight VRAM check**: 16 GB cap. Per-experiment annotation should record expected peak VRAM at chosen `seq_len` × `batch_size` × precision when adopting any new foundation-scale backbone (>200 M params). See "GPU Memory Constraint" section.

### Metric output schema (parity with FX runner — UPDATED 2026-04-27)

The QQQ runner's JSONL output must include EVERY field the dashboard
columns read. This was a recurring deficiency: the QQQ
`evaluate_target_variant` originally dropped classification metrics that
the FX runner emitted, and `_evaluate_per_window` never aggregated
uncertainty / never ran a `train_eval` pass — leaving 6+ dashboard
columns blank for ALL 56 historical experiments. Fixed in commit
following exp 57.

**Required JSONL top-level fields per experiment** (any new field added
to FX runner's emit MUST also be added to QQQ runner — they share
`evaluation/metrics.classification_metrics`):
- Strategy: `composite`, `sharpe`, `val_sharpe`, `train_sharpe` (TODO if missing), `ic`, `hit`, `psr`, `equity`, `return_pct`
- Buy-and-hold (QQQ-specific): `bh_sharpe`, `bh_return_pct`, `excess_sharpe`
- Multi-target (QQQ-specific): `B_sharpe`, `B_excess_sharpe`, `B_return_pct`, `D_sharpe`, `D_excess_sharpe`, `D_return_pct`
- **Classification (mandatory)**: `precision`, `recall`, `f1`, `f2`, `mcc`, `accuracy` + val-prefixed twins (`val_precision`, etc.)
- Uncertainty (neural backbones via MC Dropout): `confidence_mean`, `aleatoric_mean`, `epistemic_mean` + val-prefixed twins (TODO — currently empty for QQQ)
- Counts: `test_pos_folds`, `val_pos_folds`
- Per-window: `per_window` (test) and `per_window_val` (val) — each window dict contains the same A_/B_/D_ prefix block + unprefixed aliases for target A + per-window classification keys (`precision`, `recall`, `f1`, `f2`, `mcc`, `accuracy`, `tp`, `fp`, `tn`, `fn`)
- Status: `status` ∈ {`KEEP`, `DISCARD`} computed against `best_config.json` BEFORE the JSONL append (commit `0debb50`)

**Audit gate (run before every batch of experiments)**:
1. `wc -l experiment_log.jsonl` matches the next `experiment_num`
2. The most recent JSONL row has every required field non-null
3. Run `_backfill_classification.py` if any historical row is missing classification keys (idempotent)
4. Per-window dicts have `precision`/`recall`/`f1`/`f2`/`mcc`/`accuracy`/`tp`/`fp`/`tn`/`fn`

If ANY required field is missing on a new run, the runner has regressed and must be patched BEFORE launching the next experiment. The dashboard reads these fields directly — empty columns are a runner bug, not a dashboard bug.

### Cheap-first 25-experiment-per-backbone protocol (UPDATED 2026-04-27)

Per user direction (2026-04-27): exhaust cheap backbones BEFORE moving
to heavy ones. Order:

| Tier | Backbones | Per-run cost | Mandate |
|---|---|---|---|
| Cheap | MLP, LSTM | 28-150s | 25 each |
| Medium | XGBoost, LightGBM | 100-2900s | 25 each |
| Heavy | CatBoost, Mamba/dMamba | 700-19000s | 25 each |

Within each tier, ALWAYS hill-climb from the per-backbone champion
(highest composite, NOT highest A_sharpe — composite is the formal
champion criterion per `composite_score` in metrics.py). One config
change per experiment. ONE arXiv-cited rationale per experiment.

When 3+ consecutive DISCARDs from a single per-backbone champion: STOP
hill-climbing that axis — pivot to a different axis OR a different
backbone. Per CLAUDE.md "consecutive discards = local optimum" mandate.

### Per-backbone seq_len (UPDATED 2026-04-27 — STRUCTURAL CEILING)

**seq_len is variable PER backbone — the runner accepts `--seq-len` and
each experiment chooses.** BUT — there is a hard ceiling driven by
the QQQ super-fold structure that no backbone can exceed without
corrupting the metric.

**THE seq_len CEILING IS 60 (test) / 60 (val).** The smallest val
folds (F3 = 63d, F5 = 63d) have wt_len just enough for `seq=60`. The
runner's fold-bounded windowing (`feats.loc[ws:we]`, no straddling —
this is correct, no leakage) refuses to predict any fold where
`wt_len < seq_len + 1`. At seq>60:

| Val fold | wt_len | seq=60 | seq=90 | seq=120 |
|---|---:|---:|---:|---:|
| F1 | 96d | 37 preds | 7 preds | **SKIP** |
| F2 | 105d | 46 preds | 16 preds | **SKIP** |
| F3 | 63d | 4 preds | **SKIP** | **SKIP** |
| F4 | 105d | 46 preds | 16 preds | **SKIP** |
| F5 | 63d | 4 preds | **SKIP** | **SKIP** |
| F6 | 84d | 25 preds | **SKIP** | **SKIP** |
| F7 | 123d | 64 preds | 34 preds | 4 preds |

At seq=90, 3 of 7 val folds skip; at seq=120, 6 of 7 val folds skip.
Composite formula `min(test_sh, val_sh) - 0.1 * n_neg` becomes
mechanically incomparable to historical experiments — fewer val folds
counted, val Sharpe averaged over a tiny prediction set. This is NOT
strict data leakage (the runner doesn't straddle held-out periods)
but it IS metric-integrity violation.

**Hard rule**: **DO NOT run any experiment with seq_len > 60 unless
you have ALSO restructured the folds to enlarge F3/F5 val windows.**
Fold restructure is a structural change requiring re-baseline of
ALL prior experiments — out of scope for HP hill-climbing.

| Backbone | QQQ champion seq | SOTA reference seq | Useful range on QQQ given ceiling |
|---|---:|---|---|
| MLP | 10 | n/a (architectural ceiling — flatten dim grows linearly with seq, hits overfit at seq>10) | 5, 8, 10, 15, 20, 30 — but exp 25 showed seq=60 catastrophic, so MLP's effective ceiling is ~30 |
| LSTM | 10 (champ exp 8) | Fischer & Krauss 2018 EJOR S&P daily LSTM used **seq=240** (5000-row dataset, irrelevant here). | 5, 10, 15, 20, 30, 60 — tested 10/20/60; seq=20 (exp 32) was the best non-baseline |
| XGBoost / LightGBM / CatBoost | 60 | Tabular has no canonical seq_len. | 30, 45, 60 — opportunity to test 30, 45 |
| Mamba / dMamba | 60 (champ exp 52) | Gu-Dao 2024 used L=512-2048 (long-context language). SSM scales O(L) but **CAN'T exceed 60 here without losing val folds**. | 30, 45, 60 — opportunity to test shorter (30, 45) since SSM may not need long context if d_state=32 already encodes regime. |

**Future option (out of session scope)**: enlarging val windows in
`data/splits.py` — F3 val window is currently 2014-Q1 (63 days); could
extend to 2014-H1 (126 days). Similar for F5 around Vol-mageddon Q4
2018. This would re-allow seq=120+ but invalidates all prior
experiments and requires the full 25-exp budget rerun per backbone.

When testing seq_len changes (within the seq≤60 ceiling):
1. Cite the specific paper that recommends the candidate seq for the backbone family.
2. Pre-flight VRAM at the new seq × batch (linear or quadratic per architecture).
3. seq_len change is ONE knob — combine with other changes only after seq_len axis is closed.

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

## User Directives Log — Session 2026-04-28 to 04-29 (exps 165-216+)

This section captures **every user correction/directive** issued during this
session so future Claude Code instances inherit the exact rules.

### Dashboard + GitHub artifact rules

1. **DASHBOARD STATUS COLUMN** — must always be `KEEP` or `DISCARD` based on
   `composite > prev_best_composite`. Runner sets `entry["status"]` on every
   write. Backfilled 23 historical entries.
2. **CLASSIFICATION METRICS COLUMNS** — every entry must have precision /
   recall / f1 / f2 / accuracy / mcc / tp / fp / tn / fn. Backfilled 55/57.
3. **JSONL JS-COMPATIBLE OUTPUT (CRITICAL)** — JavaScript `JSON.parse()`
   rejects raw `NaN`/`Infinity` tokens. Runner must sanitise floats to
   `null` if `math.isnan` or `math.isinf` and write with `allow_nan=False`.
   74 historical entries had `val_loss: NaN` and silently dropped from JS
   dashboard, leaving only 139 of 213 visible. Fixed in
   `run_autoresearch.py`; backup `experiment_log.jsonl.bak_pre_nan_fix`.
4. **DASHBOARD CACHE-BUST STAMP** — embed a build timestamp in
   `<title>AutoResearch QQQ Dashboard (build YYYYMMDD-HHMMSS)</title>` so
   browsers/CDNs invalidate. Bump on every dashboard.html edit.
5. **PER-EXPERIMENT SYNC + COMMIT RULE** — always run
   `_sync_dashboard_to_docs.py`, `git add`, `git commit`, `git push origin master`
   *before launching the next experiment*. Live mirror at the GitHub Pages
   URL must never lag the local JSONL.
6. **PLAYWRIGHT VERIFICATION** — when user reports a dashboard issue,
   verify with the Playwright MCP browser. Validate via DOM rendering
   check, console-error inspection, navigation. Don't assume "I synced
   therefore correct".

### Experiment loop rules (research-strict)

7. **ALWAYS START EVERY BACKBONE WITH SOTA ARXIV HP** — first experiment
   uses recommended HP from latest arXiv paper, citation in reasoning blob.
8. **CHEAP TIER FIRST, EXPENSIVE LAST** — Order: MLP → LSTM → GBMs
   (XGBoost / LightGBM / CatBoost) → Mamba → Phase F (PatchTST etc.) →
   Phase D (stock-specific code adds: Adv-ALSTM, StockMixer, MASTER,
   PatchMixer, CARD, Reversible Mixer) → Phase E (foundation models —
   Sundial, TimesFM, Chronos, Moirai). **FOUNDATION MODELS LAST.**
9. **EACH NEW EXPERIMENT** — start from previous winner, deep per-fold
   analysis, arxiv search targeting deficiency, full reasoning blob
   (diagnosis + citations + hypothesis + prediction) BEFORE launch,
   verdict + learning AFTER.
10. **DEFICIENCY = WEAKEST PER-FOLD REGIME**, not "seed instability" —
    target the specific fold where champion's A_sharpe is most negative.
11. **GPU SATURATION, NOT CPU** — Intel 14th-gen HX has E-core parity
    errors (5 BSODs on 2026-04-19). All compute saturation on GPU; CPU
    pinned to 4 P-cores [0,2,4,6] via `_pin_to_safe_cores()`.

### seq_len discipline

12. **PER-BACKBONE seq_len IS NOT UNIFORM** —
    MLP/LSTM = 10 (FX-port); GBMs/Mamba/iTransformer/PatchTST/PatchTSMixer
    = 60 (paper-default); DLinear/N-BEATS = 10. Variable per backbone is
    intentional.
13. **seq_len ≤ 60 HARD CEILING** — at seq_len > 63 fold val/test windows
    (smallest = 63 days) cannot fit. Killed exp 58 (seq=120) before launch.
14. **2026-04-29 user directive: MLP seq AXIS UNDER-EXPLORED** — extended
    MLP budget +25 (50→75) to test seq=20 then seq=35.
15. **2026-04-29 user directive: 10 ADDITIONAL EXPERIMENTS PER CHEAP-TIER
    WINNER AT seq=35** — re-run all cheap-tier within-backbone winners
    (MLP, LSTM, XGBoost, LightGBM, CatBoost, Mamba) for 10 more
    experiments at seq_len=35 *with arxiv-driven code rewrites where
    relevant* (e.g. He 2016 ResNet bottleneck on MLP). Total +60.
    First result exp 215 (MLP seq=35 seed=0): comp **+0.5475**, A_sh
    +0.89, F2 RECORD +4.53 — seq=35 IS helping MLP.

### Features rules

16. **ARXIV-CITED FEATURE RESEARCH** — implemented 5 calendar-effect
    features (Lucca-Moench 2015 pre-FOMC, Boyd-Hu-Jagannathan 2005 NFP,
    Faust-Wright 2018 CPI, Stivers-Sun 2002 quad-witching).
17. **FEATURES ROLLED BACK if not useful** — mixed impact: LSTM helped
    +1.05, dMamba dropped -1.60. Net negative. Rolled back to commit
    a27bf3a. Re-introduction requires multi-seed median improvement.

### Winner archiving

18. **EVERY GLOBAL CHAMPION GETS A SELF-CONTAINED ARCHIVE** in
    `autoresearch_results/winners/<backbone>_exp<N>_<desc>/`:
    - README.md (headline + config table + per-fold + multi-seed + cites)
    - config.json, experiment_log_entry.json, per_fold_results.json
    - reproduction/multi_seed_summary.json + trade logs CSV/JSON
    - code/ frozen snapshot of run_autoresearch.py + model + data + evaluation + CLAUDE.md
    - inference/predict.py, audit_report.md (14-section), colab notebook
19. **2026-04-29 user requested archiving be re-run** — helper script
    `_archive_session_winners.py` creates archives on demand for new
    champions. Run after every new champion. Currently archived:
    `mamba_exp52_dmamba_e2_d32_seed42` (global +1.32),
    `mlp_exp79_residual_seq10_wd1e5_warmup5` (5-seed median +0.43),
    `mlp_exp204_residual_seq10_wd1e4_warmup5` (single-seed +0.97),
    `mamba_exp178_mambats_e2_d32_seed42` (complementary +0.42).

### Repository structure

20. **2026-04-29 user directive: PACKAGE autoresearchindexstock AS
    TOP-LEVEL PROJECT IN dlmastery, NOT pollute autoresearch (forex)** —
    QQQ project lives in `dlmastery/autoresearchindexstock` (new repo).
    FX project remains at `dlmastery/autoresearch`. New repo includes
    `docs/` for GitHub Pages dashboard. Migration via `gh repo create`
    + initial push from autoresearchindexstock-as-root staging.
21. **CLAUDE.md must be updated with every user directive** — this
    section is append-only on each session correction.

### Memory + checkpoint discipline

22. **2026-04-29 user directive: "make sure you follow the details in
    claude.md please. you should have sharp memory to follow
    instructions"** — re-read CLAUDE.md `User Directives Log` at every
    session start. The 7-step research-strict process is non-negotiable.
    Single-seed "champion" requires **3-seed median > prior baseline**
    before being archived. Failure cases (e.g. CatBoost lr=0.05 +0.39
    single-seed → 4-seed median 0.0) documented in research_journal.md.

23. **2026-04-29 06:30 user directive: 5-day target analysis** —
    investigate whether 5-day forecasting (target B) outperforms 1-day
    (target A). Empirical finding (LSTM seq=35, 7 seeds): B_sh mean
    +0.97 vs A_sh +0.40 — **5-day predictions are 2.4× stronger than
    1-day** in single-asset n=4,772 daily QQQ. Trading 1-day on
    5-day-prediction sign requires either a `--primary-target {1d,5d}`
    CLI flag (TODO) or post-hoc analysis on the trade_log CSVs (which
    contain both A and B prediction columns).

24. **2026-04-29 06:55 user directive: "redo another 10 experiments each
    for xlstm, dmamba, itransformer with prediction of 5 days, seq=60.
    still the target is next day"** — predict B (5d), trade on A (1d).
    Currently the runner trains both A and B heads jointly, so this
    works out of the box; we report B_sh from existing JSONL plus a
    post-hoc `B_signal_A_trade_sharpe` from the trade_logs CSVs. The
    "10 experiments each" must follow the research-strict protocol per
    directive 25 below.

25. **2026-04-29 07:00 user re-emphasis: "make sure each experiment you
    look into latest arxiv literature to setup - to hill climb from
    winner of the backbone. remember this claude.md"** — every
    experiment in the 30-exp batch (10 dMamba + 10 xLSTM + 10
    iTransformer at seq=60) MUST:
    a. Start from the current within-backbone winner (NOT a random seed
       sweep).
    b. Look up latest 2024-2026 arxiv literature for the specific
       backbone family.
    c. Cite the paper (full author + year + venue + arxiv ID + relevance
       note) in the reasoning blob BEFORE launching.
    d. Hill-climb ONE knob (HP, architectural variant, or feature axis).
    e. Multi-seed verification only after a single-seed proves the
       hypothesis via 3-seed median rule.

    For dMamba (champion +1.32): hill-climb candidates from arxiv —
    Liu 2025 DMamba expansion factor ablation; Cai 2024 MambaTS
    season-trend layer; Sundial flow-matching loss (Liu 2025
    arXiv:2502.00816); Time-MoE sparse-MoE upgrade (Shi 2024
    arXiv:2409.16040).

    For xLSTM: Beck 2024 NeurIPS (arXiv:2405.04517) base; TiRex 2025
    NXAI xLSTM-decoder time-series specialisation; sLSTM-mLSTM ratio
    ablation per §4.2 of Beck 2024.

    For iTransformer (best A_sh +0.92): Liu 2024 ICLR (arXiv:2310.06625)
    base; Itransformer + RevIN normalisation; deeper d_model (per Liu
    Table 4).

## GitHub Pages URLs (after 2026-04-29 migration)

Once `dlmastery/autoresearchindexstock` is live with Pages enabled:

- **Dashboard**: <https://dlmastery.github.io/autoresearchindexstock/dashboard/>
- **Repo root**: <https://github.com/dlmastery/autoresearchindexstock>
- **JSONL**: <https://dlmastery.github.io/autoresearchindexstock/dashboard/experiment_log.jsonl>

Until migration complete, current URL stays at
<https://dlmastery.github.io/autoresearch/index_stock_dashboard/>.

## User Directives Log -- Session 2026-05-06 (mirrored from SPY project)

### Directive 64 (2026-05-06) -- Strategy CSVs MUST carry the full 16-column rich schema

User: *make sure all the csvs should have all the columns - update claude.md to make this mandatory.*

**MANDATORY 16-column schema** for every per-strategy daily CSV written by any builder (`run_oos_inference.py`, `run_oos_top30.py`, `run_oos_ensemble.py`, future smart-strategy / hedging builders, etc.):

| # | column | meaning |
|---|---|---|
| 1 | `date` | YYYY-MM-DD trade date |
| 2 | `position` | signed continuous position after sizing/overlay |
| 3 | `pred_direction` | sign(position): -1 / 0 / +1 |
| 4 | `traded` | 1 if position != 0 else 0 (binary participation flag) |
| 5 | `actual_ret_1d` | next-day QQQ log return (the realised market move) |
| 6 | `bh_log_ret` | buy-hold log return per day |
| 7 | `strategy_pnl` | per-day signed log P&L = position * actual_ret_1d |
| 8 | `correct` | 1 if sign(position) == sign(actual) AND traded, else 0 |
| 9 | `equity_dollars` | compounded $ equity from $1000 base |
| 10 | `buy_hold_dollars` | BH compounded $ equity from $1000 base |
| 11 | `excess_dollars` | equity_dollars - buy_hold_dollars (per-day) |
| 12 | `cumret_pct` | (equity_dollars / 1000 - 1) * 100 |
| 13 | `bh_cumret_pct` | (buy_hold_dollars / 1000 - 1) * 100 |
| 14 | `excess_cumret_pct` | cumret_pct - bh_cumret_pct |
| 15 | `drawdown_pct` | peak-to-trough on strategy equity |
| 16 | `underwater` | 1 if eq < peak else 0 |

**Implementation contract:** align all columns to a single `dates` index BEFORE building the dict; reindex `.fillna(0)` if any series is shorter; coerce to numpy arrays of equal length. Use `float_format="%.6f"` on `to_csv()`. Compounded equity uses `$1000 * exp(sum_log_returns)` (NOT linear sum).

**A 6/7-column CSV is a P0 regression -- fix before next sync.** The `correct` column is the per-day input to hit rate (Directive 65) -- losing it loses the winner-metric audit trail.

### Directive 65 (2026-05-06) -- HIT RATE is the SOLE winner-identification metric

User: *update the charter claude.md to focus on hit rate as the sole metric for identifying the winner. update the charts with appropriately. add hit rate as top level prominent box in the dashboard given that is the top metric now*

**MANDATORY CHANGE:** the SOLE metric for identifying a "winner" (per-strategy, per-experiment, per-OOS) is **HIT RATE** (= percent of traded days where `sign(prediction) == sign(actual_ret_1d)`). All prior composite-, Sharpe-, equity-, and PSR-based winner declarations are SUPERSEDED.

**Why hit rate?**
1. **Robust to small-sample Sharpe inflation.** With ~100 OOS days, Sharpe std-error is ~1.56 -- a true Sharpe of +1 can show as +5+ by chance. Hit rate is bounded [0%, 100%] and binomial; std-error at n=100 is ~5%. Inflation-resistant.
2. **No leverage / sizing distortion.** Sharpe and equity reward variance management more than directional skill. Hit rate isolates pure directional accuracy -- the only thing the model is actually trained to predict (sign of `fwd_ret_1d`).
3. **Aligned with deployment intent.** The deployable signal is `sign(prediction)` traded on next-day QQQ return; everything else (sizing, hedging, overlays) is downstream. Pick the model that gets direction right most often, then layer overlays.

**Scope of change:**
- **Per-experiment champion (`best_config.json`)**: rank by `hit_rate_pct` (or `hit` field) on the test set. KEEP/DISCARD becomes: `hit > prev_best_hit` -> KEEP. Composite is now SECONDARY.
- **Headline summary cards**: lead with "HIT RATE CHAMPION" (gold-bordered, gold accent, larger font). Sharpe / $ / Composite become secondary readouts. Composite Champion card explicitly tagged "SECONDARY (Directive 65)".
- **Tables**: default sort by hit-rate descending (`Hit%` column).
- **OOS panels (when added)**: filter OOS Hit-Rate Champion display to strategies with `n_traded_days >= 20` so the rate is statistically meaningful.

**Tie-breaking when hit rates are equal:** prefer (1) higher `n_traded_days` (more chances to be wrong), (2) higher PSR, (3) higher composite. Document the tie chain on every champion archive.

**Days Traded column mandate:** any table that surfaces hit-rate MUST also surface `n_traded_days` immediately to the right. A 60% hit rate over 5 traded days is binomial noise (95% CI: 15%-95%); 60% over 80 traded days is signal (95% CI: 49%-71%). Without `Days Traded` visible, the Hit% column is misleading.

**Build stamp on dashboard updates** must change every time these rules are added, removed, or amended -- enforces browser-cache invalidation.

### Directive 66 (2026-05-08) -- Per-experiment OOS CSVs MUST carry 16-column schema + equity_curve in JSON

User: *missing all the columns that regular one has - this is for oos and ensemble + the graphs are missing in individual rows*

**MANDATORY (effective 2026-05-08):**

1. Every per-experiment OOS CSV must carry the canonical 16-column schema (Directive 64): `date, position, pred_direction, traded, actual_ret_1d, bh_log_ret, strategy_pnl, correct, equity_dollars, buy_hold_dollars, excess_dollars, cumret_pct, bh_cumret_pct, excess_cumret_pct, drawdown_pct, underwater`. Plus prediction columns + legacy aliases (`cumulative_pnl, cumulative_buy_hold`) for backwards-compat = 17-22 columns total.
2. Compounded equity is required: `equity_dollars = $1000 * exp(cumsum(strategy_pnl))`. Linear cumsum is NOT acceptable as the equity proxy.
3. Every per-experiment metrics dict in `oos_top30_table.json` MUST include `equity_curve` with keys: `dates`, `strategy_dollars`, `buy_hold_dollars`, `strategy_pct`, `buy_hold_pct`. The `_pct` arrays are required because `sparklineSVG()` reads them; missing them = `renderOOSTop30()` throws and the table never inserts into the DOM.
4. Row dict must include `train_val_sharpe`, `train_test_pos_folds`, and `oos_psr` so all dashboard table columns render. PSR uses Bailey-López de Prado 2012 (skew/kurt-corrected).
5. **Mamba checkpoint loading**: detect `mamba_d_state`, `mamba_expand`, and `mamba_variant` from state_dict shapes (`.A_log`, `.in_proj.weight`, presence of `trend_mlp.*` keys for dmamba). The training-time config dict often omits these, so state_dict introspection is mandatory.

**Implementation reference:** `run_oos_top30_qqq.py` is the QQQ-specific port. Uses 2-year download window (2024-01-01 to 2026-04-30) so SMA-200 features have proper warmup.

### Directive 67 (2026-05-09) -- Every clickable table row produces inline detail card with strategy / BH / ensemble equity chart

User: *when i click on smart trading strategy i need to see an inline card / the chart is also missing for strategy vs buy and hold - kind of two lines in chart / add equity chart with buy and hold/ensemble - for both qqq and spy dashboards when i click a row in the table*

**MANDATORY:** every row in the OOS Top-30, OOS Deep Ensemble, and Smart Trading Strategies tables MUST be clickable. Click-action MUST open or update an inline detail card with all of:

1. **Headline cards** — lead with **🎯 Hit Rate ★** (gold-bordered, large font, Directive 65) followed by `n_traded_days`, then `$Final/$1k`, `$Excess vs BH`, `Sharpe`, `Excess Sharpe`, `Sortino`, `Return %`, `PSR`, `Max DD`, `Exposure`.
2. **3-line equity chart (SVG)** rendered inline:
   - **Strategy** (purple/green, thicker stroke) — the row's own equity curve.
   - **Buy & Hold** (grey, thinner stroke) — same OOS window, no model.
   - **Ensemble champion** (gold, medium stroke) — picked via `getEnsembleChampion()` (highest `hit_rate_pct`, filtered to `n_traded_days >= 10`). Drawn only if dates overlap the row's dates.
3. **CSV download button** linking to the row's per-day CSV (Directive 64 16-column schema).
4. **Cross-link mini-table** — `buildSmartVariantsHTML(signal_source, contextLabel)` shows other overlays/hedging on the same signal source.
5. **Close button** (✕) to dismiss the inline card.

**JS contract** (must exist on every dashboard):
- `loadSmartDetails(name)` — Smart Strategies panel → renders into `#oos-smart-detail` div above the table.
- `loadOOSDetailsForRow(expNum)` — OOS Top-30 panel → renders into `#oos-section` panel above.
- `loadEnsembleDetails(strategyName)` — OOS Deep Ensemble panel → renders into `#oos-section`.
- `getEnsembleChampion()` helper returns the highest-hit-rate ensemble's record or `null`.

**Why 3 lines?** Strategy alone says whether the model worked; BH says whether doing nothing would have been better; ensemble champion says whether the single-model is competitive against the deep-ensemble baseline (Lakshminarayanan 2017). All three are the minimum honest comparison set per Directive 65.

**Build-stamp bump on every dashboard JS edit** is mandatory for browser cache invalidation. Format: `(build YYYYMMDD-HHMMSS + descriptive-tag)`.

### Directive 68 (2026-05-10) -- LOOK-AHEAD audit on ensemble selection criteria (mirrored from SPY)

User: *how did you come up with strategy to decide to trade or not for each row - did you leak into data?*

**Honest audit:**

| Family | Causal? | Deployable? | Notes |
|---|---|---|---|
| `sma200filter`, `sma200_real`, `vol_regime_gate_15`, `dead_zone`, `hold_until_flip`, `stoploss2pct`, `dd_gate_5pct` | ✅ YES | ✅ YES | Use only data observable at decision-time T. |
| All sizing modes (`raw`, `kelly`, `conf`, `voltarget`) | ✅ YES | ✅ YES | From past predictions / past vol. |
| All hedging (`protective_put`, `covered_call`, `collar`, etc.) | ✅ YES | ✅ YES | Black-Scholes priced at close[T-1] using observable VXN. |
| `all<N>_mean`, `all<N>_vote`, `vote_geq_K_of_N` | ✅ YES | ✅ YES | No member selection. |
| `top<K>_by_train_composite` | ✅ YES | ✅ YES | Train-time metric only. |
| **`top<K>_by_oos_*` (sharpe/return/excess/hit/psr/min_dd/compound/sortino/recency_30d)** | ❌ **NO** | ❌ **NO** | **POST-HOC SELECTION — uses OOS metrics. Hindsight bias. Upper-bound only.** |
| `all<N>_weighted_by_<oos_*>` | ❌ NO | ❌ NO | Weights from OOS perf. |

**Mandatory dashboard display:** `decodeStrategyName(name, s)` detects leaky tags and renders a red ⚠️ LOOK-AHEAD WARNING badge for hindsight-biased strategies; clean strategies get a green ✅ badge. `getEnsembleChampion()` prefers clean strategies for the chart benchmark line.

**Deployable** = `top<K>_by_train_composite__*`, `all<N>_mean__*`, `all<N>_vote__*`, `vote_geq_K_of_N__*` (any sizing/overlay combination is fine — those are causal). The `top<K>_by_oos_*` family stays in the JSON for oracle-bound analysis, NOT for deployment.

### Directive 69 (2026-05-10) -- Causal-only ensemble selection (decide on TRAIN+VAL, evaluate on OOS)

User: *can you fix the look ahead thing - may be you can redo trading strategy by figuring out the criteria on test and validation data only and not on oos data + all oos stuff need to be using test and validation to get the trading strategy and then use it on oos*

**MANDATORY (effective 2026-05-10):**

1. `_build_qqq_ensemble_summary.py:load_completed_members()` enriches every member with TRAIN-TIME metrics from `experiment_log.jsonl`: `train_test_sharpe`, `train_val_sharpe`, `train_train_sharpe`, `train_hit`, `train_psr`, `train_equity`, `train_return_pct`, `train_excess_sharpe`, `train_composite`, `train_ic`. Causal — observable BEFORE OOS.

2. `selection_criteria` and `weight_metrics_full` split into HINDSIGHT vs CAUSAL blocks. New deployable selections: `by_train_test_sharpe`, `by_train_val_sharpe`, `by_train_hit`, `by_train_psr`, `by_train_equity`, `by_train_excess_sharpe`, `by_train_return_pct` (in addition to existing `by_train_composite`).

3. Every strategy record in `oos_ensemble_summary.json` MUST carry `is_leaky` (bool) + `selection_basis` (string). Final-pass tagging walks all strategies after build (catches `stoploss2pct`, `ddgate5pct`, `sma200filter` overlays that bypass `add()`).

4. Summary JSON adds `n_leaky_strategies`, `n_clean_strategies`, `leakage_audit_directive` at top level.

5. Dashboard `decodeStrategyName()` prefers server-side `is_leaky` flag over client regex; renders red ⚠️ vs green ✅ DEPLOYABLE badge. `getEnsembleChampion()` prefers clean strategies for chart benchmark.

6. **Workflow contract:** decide trading-strategy rules using only train+val+in-sample-test. Lock them. Apply mechanically to OOS. Tuning any parameter on OOS = leakage.

**For paper-trading: pick ONLY from green-badged strategies** (`is_leaky: false`).

### Directive 70 (2026-05-10) -- REMOVE all leaky strategies

User: *why leaky strategy ? i dont want. i want just non leaky*

Builders amputated to NEVER compute leaky strategies. `_build_qqq_ensemble_summary.py` and `_build_qqq_smart_strategies.py` no longer include `by_oos_sharpe`, `by_oos_return`, `by_excess`, `by_hit`, `by_psr`, `by_min_dd`, `by_compound`, `by_sortino`, `by_recency_30d` in their selection/weight criteria. Only `by_train_*`, `all_*`, `vote_geq_*` plus all sizings/overlays/hedges remain (all causal).

Smart-strategies builder also enriches members with TRAIN-TIME metrics from `experiment_log.jsonl` so the new `by_train_*` criteria can compute.

Stale leaky CSVs purged from disk on 2026-05-10 cleanup (824 in QQQ).

**Counts after Directive 70 cleanup:** Ensemble 332 (all clean) | Smart 686 (all clean) | Total deployable **1018**.

### Directive 71 (2026-05-10) -- Full transparency on every clickable row

User: *one way you can help me is being very transparent in dashboard. when one clicks on row the detail should explain everthing in there + a detailed description of what and how part is very important + do the same for every row click of every table*

Every clickable row in every table (OOS Top-30, OOS Deep Ensemble, Smart Trading Strategies, main experiment-log table) renders 2 expandable blocks beneath the metric cards + chart:

1. **`buildNarrativeHTML(s, name, contextLabel)`** — 5-section "what / how / when / why / limits" prose: 📖 WHAT (plain-English step-by-step with each strategy-name token decoded), 🔬 HOW (exact formulas at every step), ⏱️ WHEN (observability table proving causality), 📐 WHY (per-metric formula+intuition+caveat), ⚠️ HONEST LIMITS (explicit can/cannot conclude list).

2. **`buildTransparencyHTML(s, ctx)`** — 6 sections: 👥 member roster, 📡 data sources & timing, ⚙️ backtest assumptions (8 ❌-list items NOT modeled), 📊 statistical caveats (binomial CI, Sharpe std-error band, multi-testing inflation), 📋 last 5 trade-days from CSV, 📐 naive baselines.

JS contract: both helpers attached to `window.*`. Wired into `loadEnsembleDetails`, `loadSmartDetails`, `loadOOSDetailsForRow`, `showDetail` (main table). Build stamp must bump on every helper edit.

### Directive 72 (2026-05-10) -- TRAIN-FIT strategies (design on in-sample, lock, evaluate OOS)

User: *now a very important question. you identified some trading strategies using cheating - now i told you to fix that and you fixed but used the same trading strategy. dont you think you need to come up with new trading strategies as well using test/validation data set and apply on oos and ensemble on oos etc.,,.*

**Background:** Directive 70 removed leaky `top<K>_by_oos_*` selection but the surviving strategies still relied on **literature-default parameter values** (200d SMA, 15% vol-target, 0.25 Kelly, 5% OTM) — those defaults were never themselves *fit* to data.

**Pipeline:** in-sample test-fold trades (`trade_logs/exp<N>_trades.csv`) → sweep/fit/calibrate using ONLY in-sample data → LOCK chosen parameter → apply mechanically to OOS → report OOS as the test of the locked strategy. Tag with `is_leaky=False` + `selection_basis` + `fit_method` + `fit_param_pool` + `fit_param_chosen` + `in_sample_sharpe_at_choice`.

**Implemented categories** (`_build_train_fit_strategies_qqq.py`): per-member parameter sweeps (`train_optim_conf_thresh`, `train_optim_kelly_frac`, `train_optim_voltarget`), ensemble meta-learners (`meta_ridgeCV`, `meta_logreg`, `meta_xgb`), per-member isotonic calibration (`calib_isotonic`).

**Pending (Categories D+E next session):** regime detection (k-means on (vol, momentum) → per-regime weights) + walk-forward refit (gold standard).

**Counts after Directive 72:** QQQ ensemble panel total = 347 | train-fit = 15 (9 sweeps + 3 meta + 3 calib) | leaky = 0.

### Directive 73 (2026-05-10) -- Transparency must ALWAYS render on row click

User: *when i see Ensemble Strategy: ... - i click on but i dont see any other details on click - update to have more transparency*

**Bug:** transparency append was inside `if (chartEl && eq && eq.dates...)`, AND chart code crashed silently because ensemble JSON had `_dollars` but not `_pct`. Fix: (1) JS computes `_pct` fallback from `_dollars`; (2) transparency moved OUTSIDE chart-conditional so it always renders; (3) builders now emit BOTH `_dollars` AND `_pct` arrays in `equity_curve`.

### Directive 74 (2026-05-10) -- 8 ADVANCED train-fit strategies (sector-rotation/pairs/RL skipped)

User: *Strategies NOT YET implemented (roadmap) [...] except rl execution can you add remaining strategies in oos ? remember to use validation/test data sets for parameters and then final run on oos + make sure forensic auditor no leakage at any cost + some of these strategies may need additional data which may not be there in data source then skip*

**Implemented (`_build_advanced_train_fit_strategies_qqq.py`):** 5d-on-1d, stop-loss+trailing, regime k-means (k=2,3), bandit Thompson, HRP, risk reversal, iron condor, calendar spread.

**Skipped (per "skip if needs additional data"):** sector rotation (needs XLK/XLF/etc fetch), long-short pairs (no 2nd asset spec), RL execution timing (per user direction).

**Forensic causality audit — every advanced strategy verified leakage-free:**
- 5d-on-1d: σ-threshold sweep on in-sample B-pred only
- Stop-loss+trailing: walks day-by-day, only past P&L seen, sweep on in-sample → lock
- Regime k-means: KMeans+StandardScaler fit on in-sample (vol_20, mom_60d); LOCKED model classifies OOS regime
- Bandit Thompson: Beta posterior pre-loaded on in-sample; on OOS day i, weights sampled BEFORE seeing actual[i]; posterior update AFTER position-decision
- HRP: correlation matrix from in-sample member-PREDICTIONS only (not actuals/future); weights LOCKED
- Risk reversal / Iron condor / Calendar spread: spot+IV via `.shift(1)` (close[T-1] / VXN[T-1] observable at T); strike/tenor sweep on in-sample → lock

**The `.shift(1)` discipline on price+IV** is the single most important guarantee. Spot at "open of day T" is operationally `close[T-1]` (previous-day's close, observable overnight). IV at the same moment is `VXN[T-1]`. 1-day option P&L is realized as price evolves from `close[T-1]` to `close[T]`, where realized return = `actual_ret_1d[T]`.

**Counts after Directive 74:** QQQ ensemble panel total = **363** | train-fit total = **31** (15 simple D72 + 16 advanced D74) | leaky = **0**.

**Audit checklist for any new strategy:**
1. Identify every data source (predictions, actuals, prices, IV, regime features, etc.)
2. For each, verify observable at decision time T (use `.shift(1)` if needed)
3. Confirm parameter/weight choice was made using in-sample data ONLY
4. LOCK the choice; don't re-tune on OOS
5. Tag with `is_leaky=False` + `selection_basis` + `fit_method` + `fit_param_pool` + `fit_param_chosen` + `in_sample_sharpe_at_choice`
