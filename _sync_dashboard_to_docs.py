"""Sync the QQQ dashboard + data files into docs/index_stock_dashboard/ so
GitHub Pages serves the latest state at:

  https://dlmastery.github.io/autoresearch/index_stock_dashboard/

Mirror of the FX sync script — same structure, different SRC / DST.
"""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repo root
SRC = ROOT / "autoresearchindexstock" / "autoresearch_results"
DST = ROOT / "docs" / "index_stock_dashboard"
DST.mkdir(parents=True, exist_ok=True)

REQUIRED = ["dashboard.html", "experiment_log.jsonl", "best_config.json"]
OPTIONAL = ["reasoning_annotations.json", "experiment_summary.md",
            "research_journal.md", "autoresearch_report.md"]

for name in REQUIRED:
    src = SRC / name
    if not src.exists():
        # Allow first-bootstrap state where some files are missing.
        # Continue rather than raise so the very first sync (before any
        # experiment) still produces a valid landing page.
        print(f"  [warn] missing {src.name}, skipping")
        continue
    tgt_name = "index.html" if name == "dashboard.html" else name
    shutil.copy2(src, DST / tgt_name)

for name in OPTIONAL:
    src = SRC / name
    if src.exists():
        shutil.copy2(src, DST / name)

# Trade logs + manifest
trade_src = SRC / "trade_logs"
trade_dst = DST / "trade_logs"
if trade_src.exists():
    trade_dst.mkdir(exist_ok=True)
    n_csv = n_ens = n_sum = 0
    for csv in trade_src.glob("exp*_trades.csv"):
        shutil.copy2(csv, trade_dst / csv.name); n_csv += 1
    for csv in trade_src.glob("*_trades.csv"):
        if csv.name.startswith("exp"): continue
        shutil.copy2(csv, trade_dst / csv.name); n_ens += 1
    for js in trade_src.glob("*_trade_summary.json"):
        shutil.copy2(js, trade_dst / js.name); n_sum += 1

    EXP_RE = re.compile(r"^exp(\d+)_trades\.csv$")
    experiments: list[int] = []
    ensembles: list[dict] = []
    for csv in sorted(trade_dst.glob("*_trades.csv")):
        m = EXP_RE.match(csv.name)
        if m:
            experiments.append(int(m.group(1)))
            continue
        stem = csv.stem.replace("_trades", "")
        sum_path = trade_dst / f"{stem}_trade_summary.json"
        sharpe = rows = wr = ret_pct = None
        if sum_path.exists():
            try:
                s = json.loads(sum_path.read_text(encoding="utf-8"))
                sharpe = s.get("test_sharpe") or s.get("test_sharpe_A")
                rows = s.get("total_trades")
                wr = s.get("overall_win_rate")
                ret_pct = s.get("total_return_pct")
            except Exception:
                pass
        ensembles.append({
            "name": stem,
            "label": stem.replace("_", " ").title(),
            "file": csv.name,
            "summary_file": sum_path.name if sum_path.exists() else None,
            "sharpe": sharpe, "rows": rows,
            "win_rate": wr, "return_pct": ret_pct,
        })

    manifest = {
        "experiments": sorted(set(experiments)),
        "ensembles": ensembles,
        "n_experiment_csvs": len(set(experiments)),
        "n_ensemble_csvs": len(ensembles),
    }
    for target_dir in (trade_src, trade_dst):
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  [trade_logs] copied {n_csv} per-exp + {n_ens} ensemble + "
          f"{n_sum} summaries; manifest: {len(set(experiments))} exps + "
          f"{len(ensembles)} ensembles")

# Regenerate the all-experiments Excel download (with embedded charts)
try:
    import subprocess
    subprocess.run([
        "python",
        str(Path(__file__).parent / "_export_equity_excel.py")
    ], check=True, capture_output=True)
    print("  [excel] autoresearch_equity.xlsx refreshed")
except Exception as e:
    print(f"  [excel] WARN: failed to regenerate xlsx: {e}")

total = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
n_files = sum(1 for f in DST.rglob("*") if f.is_file())
print(f"Synced {n_files} files to docs/index_stock_dashboard/ "
      f"({total / 1024 / 1024:.2f} MB).")
