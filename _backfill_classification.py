"""Backfill classification metrics (precision/recall/f1/f2/mcc/accuracy/tp/fp/tn/fn)
into JSONL entries that pre-date the metrics fix.

Reads each experiment's trade_logs/expN_trades.csv (which has the raw
prediction + actual_return columns) and uses the FX-shared
classification_metrics() helper to compute the dashboard fields, then
merges them into the JSONL entry's top level + each per_window row.

Idempotent: re-running on already-backfilled rows just rewrites the same values.
"""
from __future__ import annotations
import csv, json, sys, shutil
from pathlib import Path
import numpy as np

base = Path('C:/Users/evija/autoresearch/autoresearchindexstock')
log_p = base / 'autoresearch_results' / 'experiment_log.jsonl'
trades_dir = base / 'autoresearch_results' / 'trade_logs'

sys.path.insert(0, 'C:/Users/evija/autoresearch')
from autoresearch.evaluation.metrics import classification_metrics


def load_trades(exp_num: int):
    p = trades_dir / f'exp{exp_num}_trades.csv'
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    return rows


def cm_from_rows(rows, fold_filter=None):
    if not rows:
        return None
    if fold_filter:
        rows = [r for r in rows if r.get('fold') == fold_filter]
    if not rows:
        return None
    pred = np.array([float(r['prediction']) for r in rows])
    act = np.array([float(r['actual_return']) for r in rows])
    return classification_metrics(pred, act)


def merge_cls(target: dict, cm: dict, val_prefix: bool = False):
    if cm is None:
        return target
    pre = 'val_' if val_prefix else ''
    target[pre + 'precision'] = cm['precision']
    target[pre + 'recall']    = cm['recall']
    target[pre + 'f1']        = cm['f1']
    target[pre + 'f2']        = cm['f2']
    target[pre + 'accuracy']  = cm['accuracy']
    target[pre + 'mcc']       = cm['mcc']
    target[pre + 'tp']        = cm['tp']
    target[pre + 'fp']        = cm['fp']
    target[pre + 'tn']        = cm['tn']
    target[pre + 'fn']        = cm['fn']
    return target


def main():
    shutil.copy(log_p, str(log_p) + '.bak_pre_cls_backfill')
    rows = [json.loads(l) for l in open(log_p, encoding='utf-8')]
    n_filled = 0
    n_skipped = 0
    for r in rows:
        n = r.get('experiment_num')
        if n is None:
            continue
        trades = load_trades(n)
        if trades is None:
            n_skipped += 1
            continue
        # Aggregate (test) classification — covers all rows
        agg_cm = cm_from_rows(trades)
        merge_cls(r, agg_cm, val_prefix=False)
        # Per-window
        for w in r.get('per_window', []) or []:
            wcm = cm_from_rows(trades, fold_filter=w.get('fold'))
            merge_cls(w, wcm, val_prefix=False)
        # Also stash the same aliased classification keys on per_window rows
        # using A_-prefix so the multi-target dashboard prefix-swap finds them.
        for w in r.get('per_window', []) or []:
            for k in ['precision', 'recall', 'f1', 'f2', 'accuracy', 'mcc', 'tp', 'fp', 'tn', 'fn']:
                if k in w and 'A_' + k not in w:
                    w['A_' + k] = w[k]
        n_filled += 1
    with open(log_p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'Backfilled {n_filled} entries, skipped {n_skipped} (no trade CSV)')
    # Show example
    if rows:
        e = rows[-1]
        print(f'\nExample (exp {e.get("experiment_num")}):')
        for k in ['precision', 'recall', 'f1', 'f2', 'mcc', 'accuracy', 'tp', 'fp', 'tn', 'fn']:
            print(f'  {k}: {e.get(k)}')


if __name__ == '__main__':
    main()
