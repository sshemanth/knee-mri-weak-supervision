"""
qwen_score.py
Score each radiology report with a local language model (served by Ollama)
into 12 finding probabilities, then measure per-finding ROC AUC against the
58 gold studies.

Usage:
  python qwen_score.py --model qwen27b:latest --limit 3   # quick test on 3 gold reports
  python qwen_score.py --model qwen27b:latest             # all 58 gold reports
  python qwen_score.py --model qwen27b:latest --all       # all 4,407 reports
The script appends every answer to a .jsonl file and resumes from it if stopped.
"""
import argparse, json, os, time, urllib.request, urllib.error
import pandas as pd
from sklearn.metrics import roc_auc_score

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
URL = "http://localhost:11434/api/chat"          # local Ollama server

# JSON schema: forces the model to answer with exactly 12 numbers between 0 and 1
SCHEMA = {"type": "object",
          "properties": {f: {"type": "number", "minimum": 0, "maximum": 1}
                         for f in FINDINGS},
          "required": FINDINGS}

# Instruction given to the model before every report
SYSTEM = """You label knee MRI radiology reports. The report can be in any language.
For each finding, give the probability (0 to 1) that the report says it is PRESENT.
Use 0 if absent or not mentioned, 1 if clearly present, and a middle value for hedged wording
(e.g. "possible", "cannot exclude", "suspicious for").
Findings:
- ACL: anterior cruciate ligament tear or injury
- MCL: medial collateral ligament tear or injury
- Medial Meniscus: medial meniscus tear
- Lateral Meniscus: lateral meniscus tear
- Medial OA: osteoarthritis / cartilage loss, medial tibiofemoral compartment
- Lateral OA: osteoarthritis / cartilage loss, lateral tibiofemoral compartment
- PF OA: patellofemoral osteoarthritis / cartilage loss
- Effusion: joint effusion
- Synovitis: synovitis
- Baker's: Baker's (popliteal) cyst
- Contusion: bone contusion / bone marrow edema from trauma
- Fracture: fracture
Reply with JSON only, with exactly these 12 keys."""

use_think = True   # turn off the model's "thinking" mode when the server supports the option

def call(model, report):
    """Send one report to the model and return its raw JSON text.
    Temperature 0 makes the answer repeatable."""
    global use_think
    body = {"model": model, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 400},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Report:\n" + report}]}
    if use_think:
        body["think"] = False
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())["message"]["content"]
    except urllib.error.HTTPError as e:
        # this model has no "think" option: retry without it
        if e.code == 400 and use_think:
            use_think = False
            return call(model, report)
        raise

def parse(txt):
    """Read the 12 numbers from the reply and clip them to [0, 1]."""
    d = json.loads(txt)
    return {f: min(1.0, max(0.0, float(d[f]))) for f in FINDINGS}

# ---- arguments and input ----
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--csv", default="reports_for_qwen.csv")
ap.add_argument("--all", action="store_true", help="score all reports, not just gold")
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

df = pd.read_csv(a.csv)
tag = "all" if a.all else "gold"
if not a.all:
    df = df[df["is_gold"]].reset_index(drop=True)
if a.limit:
    df = df.head(a.limit)
outdir = os.path.dirname(os.path.abspath(a.csv))
raw_path = f"{outdir}/qwen_raw_{tag}.jsonl"

# ---- resume: skip reports that were already scored ----
done = {}
if os.path.exists(raw_path):
    for line in open(raw_path):
        r = json.loads(line)
        done[r["study_id"]] = r
print(f"{len(df)} reports to score, {len(done)} already done (resuming)")

# ---- main loop: one request per report, two attempts each ----
t0 = time.time()
with open(raw_path, "a") as fh:
    for i, row in df.iterrows():
        sid = str(row["study_id"])
        if sid in done:
            continue
        rec = {"study_id": sid, "ok": False, "raw": ""}
        for attempt in range(2):
            try:
                rec["raw"] = call(a.model, str(row["report"]))
                rec["scores"] = parse(rec["raw"])
                rec["ok"] = True
                break
            except Exception as e:
                rec["error"] = repr(e)
        done[sid] = rec
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()   # save at once
        print(f"[{i+1}/{len(df)}] {'ok ' if rec['ok'] else 'FAIL'} "
              f"{(time.time()-t0)/60:.1f} min elapsed", flush=True)

# ---- save the scores as a table ----
ok = [r for r in done.values() if r["ok"]]
print(f"\nParsed OK: {len(ok)}/{len(done)}")
pred = pd.DataFrame([{"study_id": r["study_id"], **r["scores"]} for r in ok])
pred.to_csv(f"{outdir}/qwen_scores_{tag}.csv", index=False)

# ---- on the gold set: per-finding AUC against the expert labels ----
if not a.all and len(pred):
    gold = df.assign(study_id=df["study_id"].astype(str))
    m = gold.merge(pred, on="study_id", suffixes=("_gold", "_qwen"))
    print(f"\nPer-finding AUC vs gold ({len(m)} studies):")
    aucs = []
    for f in FINDINGS:
        y = pd.to_numeric(m[f + "_gold"], errors="coerce"); keep = y.notna()
        yb = (y[keep] > 0.5).astype(int)
        if yb.nunique() < 2:          # AUC needs both classes
            continue
        auc = roc_auc_score(yb, m.loc[keep, f + "_qwen"]); aucs.append(auc)
        print(f"  {f:17s} AUC {auc:.4f}")
    if aucs:
        print(f"\nMacro AUC: {sum(aucs)/len(aucs):.4f} over {len(aucs)} findings")
