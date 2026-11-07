"""
run_cv.py  (Phase 1, run v8; v9 is the same script with a different random seed)
5-fold cross-validation over the 58 gold studies. For each fold: train on the
report-derived labels (L3) of all non-gold studies plus the expert labels of the
other four gold folds (weight 5), early-stop on the held-out fold, and save its
out-of-fold predictions. masked_bce_loss (losses.py) is a binary cross-entropy
that skips missing labels and multiplies each term by its label weight.
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

ROOT = Path(os.environ.get("KNEE_ROOT", "."))  # folder with data/ and outputs/
DATA = ROOT / "data"
CACHE_DIR = ROOT / "cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import KneeMRIDataset, collate_studies, FINDINGS
from model import KneeMRIModel
from losses import masked_bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device, flush=True)

final_lab = pd.read_csv(ROOT / "outputs" / "final_labels_soft_v3.csv")
sources = pd.read_csv(ROOT / "outputs" / "label_sources_v3.csv")
train_csv = pd.read_csv(DATA / "train.csv")
gold_ids = set(train_csv.loc[train_csv[FINDINGS].notna().any(axis=1), 'StudyInstanceUID'])
folds = pd.read_csv(ROOT / "outputs" / "folds.csv")
weak_ids_all = final_lab.loc[~final_lab['StudyInstanceUID'].isin(gold_ids), 'StudyInstanceUID'].tolist()
series_meta_df = pd.read_csv(DATA / "train_series.csv")

GOLD_WEIGHT, WEAK_WEIGHT = 5.0, 1.0
weights = pd.DataFrame({'StudyInstanceUID': sources['StudyInstanceUID']})
for f in FINDINGS:
    weights[f] = sources[f].map({'gold': GOLD_WEIGHT, 'weak': WEAK_WEIGHT, 'none': 0.0})

MAX_EPOCHS, PATIENCE = 8, 2
IMAGE_SIZE, MAX_SLICES, MAX_SERIES, EMBED_DIM, BATCH, NUM_WORKERS = 224, 16, 8, 256, 4, 12
N_FOLDS = 5
BACKBONE_TYPE = 'radimagenet_resnet50'  # 1-channel input

def build_meta_lists(batch, device):
    return [[m.to(device, non_blocking=True) for m in item['series_meta']] for item in batch]

def run_eval(model, loader):
    model.eval()
    probs, labs, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            series_lists = [[s.to(device, non_blocking=True) for s in item['series']] for item in batch]
            meta_lists = build_meta_lists(batch, device)
            labels = torch.stack([item['label'] for item in batch])
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(series_lists, meta_lists)
            probs.append(torch.sigmoid(logits).float().cpu())
            labs.append(labels)
            ids.extend([item['study_id'] for item in batch])
    if not probs:
        return float('nan'), None, None, None
    preds = torch.cat(probs).numpy(); y = torch.cat(labs).numpy()
    from sklearn.metrics import roc_auc_score
    aucs = []
    for i in range(len(FINDINGS)):
        yt, yp = y[:, i], preds[:, i]
        m = ~np.isnan(yt)
        if m.sum() >= 2 and len(set(yt[m])) > 1:
            aucs.append(roc_auc_score(yt[m], yp[m]))
    return (float(np.nanmean(aucs)) if aucs else float('nan')), preds, y, ids

oof_records = []
(ROOT / "models").mkdir(exist_ok=True)

for FOLD in range(N_FOLDS):
    print(f"\n===== FOLD {FOLD} =====", flush=True)
    val_ids = folds.loc[folds['fold'] == FOLD, 'StudyInstanceUID'].tolist()
    train_gold_ids = folds.loc[folds['fold'] != FOLD, 'StudyInstanceUID'].tolist()
    train_ids = train_gold_ids + weak_ids_all
    print(f"train={len(train_ids)} ({len(train_gold_ids)} gold + {len(weak_ids_all)} weak), val={len(val_ids)}", flush=True)

    # use_25d is False: 1-channel input, reusing the cached slices.
    train_ds = KneeMRIDataset(train_ids, final_lab, DATA / "train_series", image_size=IMAGE_SIZE,
                               max_slices=MAX_SLICES, max_series=MAX_SERIES, weights_df=weights,
                               cache_dir=CACHE_DIR, augment=True, series_meta_df=series_meta_df)
    val_ds = KneeMRIDataset(val_ids, final_lab, DATA / "train_series", image_size=IMAGE_SIZE,
                             max_slices=MAX_SLICES, max_series=MAX_SERIES, cache_dir=CACHE_DIR,
                             augment=False, series_meta_df=series_meta_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_studies,
                               num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_studies,
                             num_workers=2, pin_memory=True)

    model = KneeMRIModel(n_findings=len(FINDINGS), embed_dim=EMBED_DIM, pretrained=True,
                          backbone_type=BACKBONE_TYPE, in_channels=1, use_series_meta=True,
                          meta_dim=32).to(device)
    print("backbone_type:", model.slice_encoder.backbone_type,
          "in_channels:", model.slice_encoder.in_channels,
          "use_series_meta:", model.use_series_meta, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_auc, best_state, best_preds, best_y, best_ids, no_improve = -1, None, None, None, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        t0 = time.time()
        running_loss, n_batches, n_skipped = 0.0, 0, 0
        for batch in train_loader:
            series_lists = [[s.to(device, non_blocking=True) for s in item['series']] for item in batch]
            meta_lists = build_meta_lists(batch, device)
            labels = torch.stack([item['label'] for item in batch]).to(device, non_blocking=True)
            w = torch.stack([item['weight'] for item in batch]).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            try:
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(series_lists, meta_lists)
                    loss = masked_bce_loss(logits, labels, sample_weight=w)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                running_loss += loss.item(); n_batches += 1
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                del series_lists, meta_lists, labels, w
                torch.cuda.empty_cache()
                n_skipped += 1
                continue
        sched.step()
        dt = time.time() - t0
        val_auc, preds, y, ids = run_eval(model, val_loader)
        print(f"  epoch {epoch}: loss={running_loss/max(n_batches,1):.4f} time={dt:.1f}s val_auc={val_auc:.4f} skipped={n_skipped}", flush=True)
        if val_auc > best_auc:
            best_auc, best_preds, best_y, best_ids, no_improve = val_auc, preds, y, ids, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop (no improvement for {PATIENCE} epochs)", flush=True)
                break

    torch.save(best_state, ROOT / "models" / f"fold{FOLD}_v8_best.pt")
    print(f"FOLD {FOLD} best val_auc={best_auc:.4f}", flush=True)
    for row_idx, sid in enumerate(best_ids):
        for f_idx, f in enumerate(FINDINGS):
            if not np.isnan(best_y[row_idx, f_idx]):
                oof_records.append({'StudyInstanceUID': sid, 'finding': f,
                                     'true': best_y[row_idx, f_idx], 'pred': best_preds[row_idx, f_idx],
                                     'fold': FOLD})

oof = pd.DataFrame(oof_records)
oof.to_csv(ROOT / "outputs" / "oof_predictions_v8.csv", index=False)

from sklearn.metrics import roc_auc_score
print("\n===== Pooled out-of-fold results (58 gold studies) =====")
per_finding = {}
aucs = []
for f in FINDINGS:
    sub = oof[oof['finding'] == f]
    if sub['true'].nunique() > 1 and len(sub) >= 4:
        a = roc_auc_score(sub['true'], sub['pred'])
        per_finding[f] = a
        aucs.append(a)
        print(f"{f:20s} n={len(sub):3d} auc={a:.4f}")
    else:
        per_finding[f] = None
        print(f"{f:20s} n={len(sub):3d} auc=n/a")
macro = float(np.nanmean(aucs))
print(f"\nPooled macro AUC (5-fold CV, 58 gold studies): {macro:.4f}")
with open(ROOT / "outputs" / "cv_results_v8.json", "w") as fh:
    json.dump({'per_finding_auc': per_finding, 'macro_auc': macro}, fh, indent=2)
print("saved -> outputs/oof_predictions_v8.csv, outputs/cv_results_v8.json")
