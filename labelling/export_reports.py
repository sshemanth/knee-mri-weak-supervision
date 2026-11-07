"""
export_reports.py
Export study IDs, report text, gold flag and the 12 finding columns from
train.csv into one CSV that the language-model labeller can read.

Usage:
  python export_reports.py --train train.csv --folds folds.csv --report-col Report
"""
import argparse, os, re, sys
import pandas as pd

# The 12 findings scored by the challenge, in submission order
FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]

def norm(s):
    """Lower-case a column name and keep only letters and digits, so that
    'Baker's' and 'bakers' are treated as the same column."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# ---- command-line arguments ----
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True, help="path to the challenge train.csv")
ap.add_argument("--folds", default=None,
                help="CSV listing the 58 gold (expert-labelled) studies")
ap.add_argument("--report-col", default="Report")
ap.add_argument("--id-col", default="StudyInstanceUID")
ap.add_argument("--out", default="reports_for_qwen.csv")
a = ap.parse_args()

# ---- load and check the input ----
df = pd.read_csv(a.train)
print(f"Loaded {a.train}: {df.shape[0]} rows, {df.shape[1]} columns")
for c in (a.report_col, a.id_col):
    if c not in df.columns:
        sys.exit(f"Column '{c}' not found. Columns: {list(df.columns)}")

# ---- find the 12 finding columns, whatever their exact spelling ----
fmap = {norm(f): f for f in FINDINGS}
fcols = [c for c in df.columns if norm(c) in fmap]
print(f"Finding columns ({len(fcols)}/12):", fcols)

# ---- build the output table ----
out = df[[a.id_col, a.report_col] + fcols].rename(
    columns={a.id_col: "study_id", a.report_col: "report"})

# Mark the gold studies: prefer the folds file; otherwise use rows with all labels filled
out["is_gold"] = False
if a.folds and os.path.exists(a.folds):
    folds = pd.read_csv(a.folds)
    key = a.id_col if a.id_col in folds.columns else folds.columns[0]
    out["is_gold"] = out["study_id"].astype(str).isin(folds[key].astype(str))
elif fcols:
    out["is_gold"] = df[fcols].notna().all(axis=1).values

# ---- report a short summary so the export can be checked ----
has_text = out["report"].fillna("").astype(str).str.strip().str.len() > 0
n_text, n_gold = int(has_text.sum()), int(out["is_gold"].sum())
print(f"Rows: {len(out)} | with report text: {n_text} | gold: {n_gold}")
out.to_csv(a.out, index=False)
print("Wrote", a.out)
