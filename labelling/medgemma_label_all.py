"""
medgemma_label_all.py  (labeller 2)
Run the MedGemma labeller of medgemma_validate.py on every report and save the
12 scores per study to outputs/medgemma_weak_labels_full.csv (saved every 50
reports, so a crash loses little work).
"""
import os, sys, json, time
from pathlib import Path
import pandas as pd, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medgemma_validate import load_model, query, FINDINGS

ROOT = Path(os.environ.get("KNEE_ROOT", "."))  # folder with data/ and outputs/
DATA = ROOT / "data"

def main():
    train_csv = pd.read_csv(DATA / "train.csv")
    all_ids_with_report = train_csv[train_csv['Report'].notna()].reset_index(drop=True)
    print(f"total studies with report text: {len(all_ids_with_report)}", flush=True)

    tok, model = load_model()

    out_path = ROOT / "outputs" / "medgemma_weak_labels_full.csv"
    rows = []
    t0 = time.time()
    n_fail = 0
    N = len(all_ids_with_report)
    for i, row in all_ids_with_report.iterrows():
        preds, raw = query(tok, model, str(row['Report']))
        if preds is None:
            n_fail += 1
            preds = {f: 0.0 for f in FINDINGS}
        rec = {'StudyInstanceUID': row['StudyInstanceUID']}
        rec.update(preds)
        rows.append(rec)
        if (i + 1) % 50 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta_min = (N - (i + 1)) / rate / 60 if rate > 0 else float('nan')
            print(f"  {i+1}/{N} done, {elapsed:.0f}s elapsed, {rate:.2f} studies/s, "
                  f"ETA {eta_min:.1f} min, fails={n_fail}", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"DONE. saved -> {out_path}. total time {(time.time()-t0)/60:.1f} min, "
          f"parse failures {n_fail}/{N}", flush=True)

if __name__ == "__main__":
    main()
