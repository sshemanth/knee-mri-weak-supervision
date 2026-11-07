"""
fuse_labels.py
Average the three labellers (Qwen, MedGemma, regex) into one soft label per
study and finding, score each option on the gold set, and save the result.
"""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

F = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
O = "outputs/"

# Study list, gold flag and expert labels (expert labels exist for the 58 gold studies only)
base = pd.read_csv(O + "reports_for_qwen.csv", usecols=["study_id", "is_gold"] + F,
                   dtype={"study_id": str})
ids = base.study_id

# ---- labeller 3: Qwen scores, read from the raw answers file ----
q = {}
for line in open(O + "qwen_raw_all.jsonl"):
    r = json.loads(line)
    if r.get("ok"):
        q[str(r["study_id"])] = r["scores"]
Q = pd.DataFrame.from_dict(q, orient="index").reindex(ids)[F].values.astype(float)

# ---- labeller 2: MedGemma scores (one value per report and finding) ----
M = (pd.read_csv(O + "medgemma_weak_labels_full.csv", dtype={"StudyInstanceUID": str})
       .set_index("StudyInstanceUID").reindex(ids)[F].values.astype(float))

# ---- labeller 1: regex labels (1, 0, or empty when the finding is not mentioned) ----
X = (pd.read_csv(O + "weak_labels_v2.csv", dtype={"StudyInstanceUID": str})
       .set_index("StudyInstanceUID").reindex(ids)[["pred_" + f for f in F]]
       .values.astype(float))
print(f"Qwen scored: {np.isfinite(Q).all(1).sum()}/{len(ids)}")
print(f"Regex answered: {np.isfinite(X).mean():.0%} of report-finding pairs")

# ---- candidate label sets: plain means of the available scores (no learned weights) ----
with np.errstate(all="ignore"):
    blends = {
        # fall back to MedGemma if missing
        "qwen":          np.where(np.isfinite(Q), Q, M),
        "qwen+medgemma": np.nanmean(np.stack([Q, M]), 0),
        # empty regex values are skipped
        "qwen+mg+regex": np.nanmean(np.stack([Q, M, X]), 0),
    }

# ---- score each candidate against the 58 gold studies ----
g = base.is_gold.values
Y = (base.loc[g, F].values > 0.5).astype(int)
table = {k: [roc_auc_score(Y[:, j], v[g, j]) for j in range(12)] for k, v in blends.items()}
t = pd.DataFrame(table, index=F)
print(t.round(3).to_string()); print("\nMacro:"); print(t.mean().round(4).to_string())

# ---- save the chosen label set (L4); file names keep the older v-numbering ----
L = np.clip(blends["qwen+mg+regex"], 0, 1)
out = pd.DataFrame(L, columns=F); out.insert(0, "StudyInstanceUID", ids)
out.to_csv(O + "final_labels_soft_v4.csv", index=False)

# Spread between labellers, saved for the agreement-weighting experiment (v15)
agree = pd.DataFrame(np.nanstd(np.stack([Q, M, X]), 0), columns=F)
agree.insert(0, "StudyInstanceUID", ids)
agree.to_csv(O + "label_agreement_v4.csv", index=False)
