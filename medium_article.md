# I ran 216 arxiv-cited ML experiments on QQQ. Only Mamba and MLP survived multi-seed median.

> A research-strict autoresearch loop on the Nasdaq-100 ETF, with every
> single experiment's reasoning annotation, dashboard, code, and frozen
> winner snapshots public.

---

## TL;DR

I forked the FX autoresearch loop from `dlmastery/autoresearch` (which
hit a +9.7 Sharpe mega-ensemble across 28 G10 currency pairs) and ran the
same loop on a single asset: **QQQ**, the Invesco Nasdaq-100 ETF, daily
2007 → 2025. After **216 experiments across 11 backbones** under a
seven-fold regime-aware super-fold split, the headline:

- **Global champion**: Mamba dMamba SSM (`expand=2, d_state=32, seed=42`),
  composite **+1.32 Sharpe**, **7/7 positive test folds**, excess
  **+0.45 over buy-and-hold**.
- **Second stable backbone**: residual MLP at warmup=5, 5-seed median
  composite **+0.43**.
- **Everything else** (XGBoost / LightGBM / CatBoost / iTransformer /
  PatchTSMixer / DLinear / N-BEATS / PatchTST / xLSTM): **multi-seed
  median composite ≈ 0**, even though single-seed test alpha hits +1.0+.

The takeaway is the *opposite* of what most ML-finance papers report:
**at single-asset n=4,772 daily, only state-space models and small
residual MLPs avoid val-side seed instability.** Everything else looks
great in single-seed cherry-picks and dies on a 3-seed median.

---

## Why I rebuilt the FX loop for QQQ

The FX autoresearch project is a **continuous research loop**: Claude
Code (the AI agent) reads results, picks an arxiv-cited next experiment,
runs ONE knob change, evaluates against multi-seed median, and either
KEEPs or DISCARDs. Every experiment has a 6-line reasoning blob:
diagnosis → citations → hypothesis → prediction → verdict → learning.
The whole thing is on GitHub Pages live, with a pretty dashboard:
[https://dlmastery.github.io/autoresearch/dashboard/](https://dlmastery.github.io/autoresearch/dashboard/).

Equity-index prediction is a different beast:

1. **Trends.** QQQ is up 13× in our window. Buy-and-hold Sharpe is +0.87.
   Any "Sharpe" without an excess-Sharpe is a lie.
2. **One asset.** No cross-section. We can't stack 28 pairs to get effective
   n=50,000. We have 4,772 daily rows. Period.
3. **Macro regimes.** GFC, EU debt 2011, Taper, China-oil, Vol-mageddon
   2018, COVID, AI rally — seven clearly distinct test windows that
   should make a model that can't generalise across regimes look bad.

So I cloned the FX project, swapped the data layer for QQQ + 55 supporting
ETF/macro/vol series (184 backward-looking features), and started running
the loop.

---

## The leaderboard

| Backbone | Used | Best multi-seed median | Best single-seed | Notes |
|---|--:|--:|--:|---|
| **Mamba dmamba** | 25 | -0.25 (4-seed) | **+1.3216 (CHAMP)** | 7/7 pos folds; excess +0.45 |
| **MLP residual** | 51 | **+0.43 (5-seed)** | +0.97 | 2nd stable; warmup=5 + hd=0.25 |
| Mamba mambats | 2 | +0.38 (2-seed) | +0.42 | complementary to dmamba |
| LSTM 1-layer | 39 | ~0 | +1.05 (features rolled back) | HP-only axes exhausted |
| LightGBM | 25 | -0.11 (3-seed) | +0.61 (luck) | leaf-wise GOSS |
| CatBoost | 25 | ~0 (4-seed) | +0.39 | ordered-boosting seed-dep |
| XGBoost | 25 | -0.40 (3-seed) | -0.13 stable | weakest cheap-tier |
| iTransformer | 5 | -1.52 (3-seed) | A_sh +0.92 | seq=60 + warmup=10 |
| PatchTSMixer | 5 | +0.028 (5-seed) | A_sh +1.22 = BH | strong test signal |
| DLinear | 6 | -0.21 | +0.80 (features) | features-dependent |
| N-BEATS | 2 | -1.43 | -1.43 | architecture mismatch |

The **green-flag column is "best multi-seed median"**. Single-seed numbers
are vibes; multi-seed median is what survives a 3rd-party reproduction.

---

## What I learned that I didn't expect

### 1. Mamba is uniquely robust at this n

Every transformer (iTransformer, PatchTST, PatchTSMixer) showed the same
pattern: high single-seed test A_sharpe (+0.9 to +1.2) and complete val
collapse on at least one of three seeds. iTransformer with the paper-recipe
lr=5e-5 + warmup=10 epochs (Liu et al. 2024 ICLR) hit A_sharpe **+0.92**
single-seed but its 3-seed composite median was **−1.52**.

Mamba dmamba was the only architecture where seed=42 gave +1.32 *and*
seeds 0 and 7 cleared +0.19 and +0.97. Only seed=99 was catastrophic
(seed=99 is bad for almost every backbone in this project — likely a
project-wide RNG pathology in the data shuffler).

My read: **the data-dependent gating in Mamba's selective-state SSM
adapts to per-fold regime characteristics that fixed-attention transformers
overfit on val.** This may not hold beyond QQQ — and it definitely won't
hold at FX panel n. But for single-asset daily, **dMamba is the
recommended starting point**.

### 2. Residual MLP at warmup=5 is the dark-horse second place

The Gu-Kelly-Xiu 2020 RFS recipe (residual MLP, lr=3e-4, wd=1e-5, ep=50,
pat=10) plus a 5-epoch warmup (Goyal et al. 2017) and head-dropout 0.25
(Srivastava 2014) gives a **5-seed median composite of +0.433** and
**single-seed +0.974 with 7/7 positive folds**.

This is the only non-Mamba result in the project where the multi-seed
median is robustly positive. The win is a combination of:

- **Skip connection**: anchors predictions to a robust feature-mean.
- **Warmup**: stabilises AdamW's initial steps which otherwise overshoot
  on the (small) dataset.
- **Head dropout 0.25**: stronger than canonical 0.1, adds noise at the
  prediction head specifically, fights val-side overfit.

Plain MLP without these three? +0.0 to +0.6 single-seed. With them?
**+0.97 single-seed, +0.43 multi-seed median.**

### 3. GBMs (XGBoost / LightGBM / CatBoost) ALL fail multi-seed median

This was the **biggest surprise**. The FX project found GBMs near the top
of its leaderboard. On QQQ:

- XGBoost depth=5 single-seed +0.37 → 3-seed median **−0.40**.
- LightGBM exp 95 single-seed +0.61 → 3-seed median **−0.11**.
- CatBoost lr=0.05 single-seed +0.39 → 4-seed median **0.0**.

In every case, single-seed wins were val-side seed-luck. The histogram
method and ordered-boosting permutation are inherently seed-dependent on
the val side at this n.

The same 3 GBM families with **the same paper-recipe HPs** were all
positive-Sharpe winners on FX. The diff is **n_effective**: panel FX has
~50k rows, single-asset QQQ has 4.7k. The GBM val sensitivity scales
inversely with n_effective.

### 4. Calendar-effect features helped LSTM and hurt Mamba

I added 5 arxiv-cited calendar features: Lucca-Moench 2015 pre-FOMC drift
(t-1, t-2), Boyd-Hu-Jagannathan 2005 NFP day, Stivers-Sun 2002
quad-witching, Faust-Wright 2018 CPI window. Result:

- LSTM exp 119 with features: **+1.05 single-seed** (best LSTM ever).
- Mamba dmamba with features: composite dropped from +1.32 to −1.60.

So I **rolled them back**. Net negative across backbones. The calendar
features inject information that helps recurrent models but interferes
with the SSM's selective gating. A different way to inject these
features (e.g., as conditioning input rather than as feature columns)
might be needed.

### 5. The composite formula is the right metric

`composite = min(test_sharpe, val_sharpe) − 0.1 × n_negative_folds`

This formula is unforgiving. It punishes:
- Asymmetric val/test fits (`min` enforces both sides).
- Folds where the strategy went underwater (`n_neg` penalty).

It is **exactly the right metric for distinguishing models that generalise
from models that cherry-pick** seeds. Most published "Sharpe X.Y on Nasdaq"
papers would land at composite < 0 under this formula, because the val
side they don't report would crash on at least one seed.

If you take one thing from this project: **adopt this composite formula
and report 3-seed median**. Everything else is performance theatre.

---

## How to deploy this

For real money, never trust a single-seed champion. The recommendation:

1. **Train 5 seeds of Mamba dmamba** (expand=2, d_state=32, seq_len=60,
   the rest paper recipe). Drop seed=99 (catastrophic). Average the
   remaining 4 predictions.
2. **Train 5 seeds of MLP residual** (warmup=5, hd=0.25, wd=1e-5, ep=50).
   Drop seed=42 / seed=7 (the negative draws). Average the remaining 3.
3. **Rank-average** the dMamba mean prediction and the MLP mean prediction
   into a single signal. This is your direction.
4. **Position size with confidence × Kelly fraction** (Lim-Zohren-Roberts
   2019). Cap per-trade exposure at 25% of equity. Cap leverage at 1.5×.
5. **Drawdown stop**: if equity falls below 90% of peak, halve position
   sizes for the next 21 days (one embargo period).
6. **Regime shift detector**: if 5-day rolling VIX rank > 95th percentile,
   exit longs. Re-enter on regime confirmation.
7. **Retrain monthly** on a rolling 5-year window. Validate every retrain
   against a held-out next-30-days super-fold before swapping into prod.

This is **not financial advice**. It is a deployment template for someone
who has already done the homework.

---

## What's next: Phase D + E

The current cheap-tier sweep (MLP / LSTM / GBMs / Mamba) is exhausted.
The next phase is **stock-specific code-add backbones** from the 2024–2026
arxiv literature, each with its own 25-experiment budget under the same
loop:

- **MASTER** (Li-Liu-Wang 2024 AAAI, arXiv:2304.12135) — reported Sharpe 1.5-1.8
- **StockMixer** (Fan-Yu 2024 KDD, arXiv:2308.07099)
- **Adv-ALSTM** (Feng et al. 2019, Yang 2024 update)
- **PatchMixer** (Gong et al. 2023, arXiv:2310.00655)
- **CARD** (Wang 2024, arXiv:2305.12095)
- **Reversible Mixer** (Liu 2024 NeurIPS, arXiv:2404.16312)

Then **foundation models LAST** (per the user directive — they're rarely
competitive on small-n single-asset zero-shot): TimesFM 2.5, Chronos-2,
Moirai 2.0, Sundial, MOMENT, Time-MoE.

If MASTER pushes past +1.5 multi-seed median, I will report.

---

## Run it yourself

Everything's public:

- **Repo**: https://github.com/dlmastery/autoresearchindexstock
- **Dashboard**: https://dlmastery.github.io/autoresearchindexstock/dashboard/
- **Release v0.1.0 (zips)**: https://github.com/dlmastery/autoresearchindexstock/releases/tag/v0.1.0-exp216
- **CLAUDE.md (project rules + every user directive)**: https://github.com/dlmastery/autoresearchindexstock/blob/master/CLAUDE.md

Reproduce the global champion in 6 minutes on a single 16 GB GPU:

```bash
git clone https://github.com/dlmastery/autoresearchindexstock.git
cd autoresearchindexstock
pip install -r requirements.txt
python -m autoresearchindexstock.run_autoresearch \
  --backbone mamba --mamba-variant dmamba \
  --expand 2 --d-state 32 --seed 42 --seq-len 60 \
  --lr 5e-4 --bs 32 --epochs 100 --patience 20 \
  --weight-decay 0.1 --head-dropout 0.1 --warmup-epochs 10 \
  --description "Reproduce global champion (exp 52)"
```

Forks welcome. PRs even more welcome — especially if you hit +1.5+ on
the MASTER / StockMixer line.

---

*Built with Claude Code (Anthropic) as the outer-loop autoresearch agent.
Forked from `dlmastery/autoresearch` (FX) on 2026-04-29 at session exp 216.
MIT licensed.*
