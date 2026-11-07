"""
medgemma_validate.py  (labeller 2)
Ask MedGemma-4B to read each report and return 12 finding probabilities as JSON,
then score the answers against the 58 gold studies. The functions load_model()
and query() are reused by medgemma_label_all.py for all 4,407 reports.
"""
import os, sys, json, re, time
from pathlib import Path
import pandas as pd, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(os.environ.get("KNEE_ROOT", "."))  # folder with data/ and outputs/
DATA = ROOT / "data"

FINDINGS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
            'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

# instruction-tuned 4B model, used with text-only prompts; fits on one L4 GPU
MODEL_ID = "google/medgemma-4b-it"

# Prompt sent with every report ({report} is replaced by the report text)
PROMPT_TMPL = """You are a musculoskeletal radiologist. Read the knee MRI report below (it may be in Spanish, English, German, Dutch, French, Portuguese, or Italian) and independently assess each of the 12 findings listed, based STRICTLY on the evidence in THIS report. Do not default to any typical or example pattern - every score must be justified by specific text in this report.

Findings and what counts as positive evidence:
- ACL: anterior cruciate ligament tear/sprain/rupture
- MCL: medial collateral ligament injury/sprain/tear
- Medial Meniscus: medial meniscus tear/rupture
- Lateral Meniscus: lateral meniscus tear/rupture
- Medial OA: medial compartment osteoarthritis - osteophytes, joint space narrowing, subchondral sclerosis/cysts, cartilage loss/chondromalacia in the MEDIAL femorotibial compartment
- Lateral OA: same OA changes but in the LATERAL femorotibial compartment
- PF OA: same OA changes in the patellofemoral compartment (patella/trochlea)
- Effusion: joint effusion / fluid in the joint
- Synovitis: synovial thickening/inflammation
- Baker's: Baker's cyst / popliteal cyst
- Contusion: bone contusion / bone marrow edema (traumatic, not degenerative)
- Fracture: any fracture

For each finding, think about what the report actually says about that specific anatomical structure, then output a confidence score from 0.0 (explicitly normal / negated / no abnormality mentioned) to 1.0 (clearly present). Watch for negation across languages (e.g. "sin lesion", "no evidence of", "keine", "sans", "ausencia de", "sem", "senza"). A structure never mentioned at all should score 0.0.

Report:
\"\"\"
{report}
\"\"\"

Respond with ONLY a JSON object with exactly these 12 keys, each mapped to a float between 0.0 and 1.0 that reflects YOUR independent reading of THIS report (values will naturally differ report to report - do not reuse a fixed pattern):
{{"ACL": <float>, "MCL": <float>, "Medial Meniscus": <float>, "Lateral Meniscus": <float>, "Medial OA": <float>, "Lateral OA": <float>, "PF OA": <float>, "Effusion": <float>, "Synovitis": <float>, "Baker's": <float>, "Contusion": <float>, "Fracture": <float>}}
"""

def load_model():
    """Load the tokenizer and the model in bfloat16 on the GPU."""
    print("loading tokenizer+model...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    print("model loaded, dtype/device ok", flush=True)
    return tok, model

def query(tok, model, report_text):
    """Greedy decoding of one report (truncated to 4,000 characters); returns
    a dict of 12 floats, or None if no valid JSON object is found."""
    prompt = PROMPT_TMPL.format(report=report_text[:4000])
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                     return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                             temperature=None, top_p=None, top_k=None)
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    # take the first {...} block in the answer and parse it as JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, text
    try:
        d = json.loads(m.group(0))
        return {f: float(d.get(f, 0.0)) for f in FINDINGS}, text
    except Exception:
        return None, text

def main():
    train_csv = pd.read_csv(DATA / "train.csv")
    gold = train_csv[train_csv[FINDINGS].notna().any(axis=1)].reset_index(drop=True)
    print(f"validating on {len(gold)} gold studies", flush=True)

    tok, model = load_model()

    rows = []
    t0 = time.time()
    n_fail = 0
    for i, row in gold.iterrows():
        preds, raw = query(tok, model, str(row['Report']))
        if preds is None:              # unreadable answer: count it and score 0
            n_fail += 1
            preds = {f: 0.0 for f in FINDINGS}
        rec = {'StudyInstanceUID': row['StudyInstanceUID']}
        rec.update(preds)
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(gold)} done, {time.time()-t0:.0f}s elapsed, "
                  f"{n_fail} parse failures", flush=True)

    preds_df = pd.DataFrame(rows)
    preds_df.to_csv(ROOT / "outputs" / "medgemma_gold_val_preds.csv", index=False)
    print(f"parse failures: {n_fail}/{len(gold)}", flush=True)

    # ---- per-finding AUC against the expert labels ----
    from sklearn.metrics import roc_auc_score
    print("\n===== MEDGEMMA vs GOLD (58 studies) =====")
    aucs = []
    for f in FINDINGS:
        yt = gold[f].values
        yp = preds_df.set_index('StudyInstanceUID').loc[gold['StudyInstanceUID'], f].values
        m = ~pd.isna(yt)
        if m.sum() >= 2 and len(set(yt[m])) > 1:
            a = roc_auc_score(yt[m], yp[m])
            aucs.append(a)
            print(f"{f:20s} n={m.sum():3d} auc={a:.4f}")
        else:
            print(f"{f:20s} n={m.sum():3d} auc=n/a")
    macro = float(np.nanmean(aucs))
    print(f"\nMEDGEMMA MACRO AUC (vs 58 gold): {macro:.4f}")

if __name__ == "__main__":
    main()
