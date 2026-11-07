"""
train.py  (Kaggle notebook, GPU)
Train the 2.5D ConvNeXt model with per-finding attention on report-derived
soft labels. The 58 gold studies are never used for training; they are
scored after every epoch, and the best epoch is saved.
"""
import os, re, glob, time, math, numpy as np, pandas as pd
import torch, torch.nn as nn, timm
from sklearn.metrics import roc_auc_score

# ---------------- settings ----------------
CACHE_NAME = "cache384"                     # slice cache to read
LABEL_FILE = "final_labels_soft_v4.csv"     # soft labels from the three labellers
BACKBONE = "convnext_tiny"
# epochs, studies per batch, windows per study, learning rate
EPOCHS, BS, K, LR = 6, 2, 16, 5e-5
# gradient accumulation steps; start from the 256 px model
ACC, INIT = 2, "v11_best.pt"
TAG = "v13_384"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SIZES = [18, 14, 12, 8, 12]; BND = np.cumsum([0] + SIZES)     # slot sizes and boundaries
# Every possible 3-slice window (centre slice, slot) that stays inside one slot
CENT = [(c, s) for s in range(5) for c in range(BND[s] + 1, BND[s + 1] - 1)]
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

def find(name, maxdepth=4):
    """Find a file or folder by name anywhere under /kaggle/input."""
    hits = []
    def walk(d, depth):
        try: ents = os.listdir(d)
        except Exception: return
        if name in ents: hits.append(os.path.join(d, name))
        if depth < maxdepth:
            for e in ents:
                p = os.path.join(d, e)
                if e not in ("train_series", "test_series", name) and os.path.isdir(p):
                    walk(p, depth + 1)
    walk("/kaggle/input", 0); return sorted(hits)

# ---------------- 1. unpack the cache into one memory-mapped file ----------------
# Reading one large uncompressed file is much faster during training than the npz shards.
shards = sorted(s for d in find(CACHE_NAME) for s in glob.glob(d + "/shard_*.npz"))
assert shards, f"{CACHE_NAME} not found"
IMG = np.load(shards[0])["vol"].shape[-1]; CROP = int(IMG * 0.875)
cap = 100 * len(shards); MM = "/tmp/vol.u8"
vol = np.memmap(MM, np.uint8, "w+", shape=(cap, 64, IMG, IMG))
ids, sm, n = [], [], 0
for s in shards:
    z = np.load(s); v = z["vol"]; vol[n:n + len(v)] = v
    # remember which slots are filled (non-zero) for every study
    sm.append(np.stack([v[:, a:b, ::8, ::8].reshape(len(v), -1).any(1)
                        for a, b in zip(BND[:-1], BND[1:])], 1))
    ids += z["ids"].astype(str).tolist(); n += len(v)
vol.flush(); del vol
SM = np.concatenate(sm); POS = {sid: i for i, sid in enumerate(ids)}
print(f"cache: {n} studies at {IMG}px")

# ---------------- 2. labels ----------------
R = os.path.dirname(find("train_series.csv")[0])
tr = pd.read_csv(f"{R}/train.csv", dtype={"StudyInstanceUID": str})
gold = tr[tr[LAB].notna().all(1)]                         # the 58 expert-labelled studies
gold = gold[gold.StudyInstanceUID.isin(POS)].reset_index(drop=True)
YG = (gold[LAB].values.astype(float) > 0.5).astype(int)

lab = pd.read_csv(find(LABEL_FILE)[0])
idc = next((c for c in lab.columns if "study" in c.lower()), lab.columns[0])
cmap = {f: next(c for c in lab.columns if norm(c) == norm(f)) for f in LAB}
lab[idc] = lab[idc].astype(str)
# Training set = all cached studies EXCEPT the gold studies
keep = lab[idc].isin(POS) & ~lab[idc].isin(set(gold.StudyInstanceUID))
lab = lab[keep].reset_index(drop=True)
Y = lab[[cmap[f] for f in LAB]].apply(pd.to_numeric, errors="coerce")
Y = Y.values.astype(np.float32)
Wt = (~np.isnan(Y)).astype(np.float32)                   # mask: 1 where a label exists
Y = np.clip(np.nan_to_num(Y), 0, 1)
print(f"gold studies: {len(gold)} | training studies: {len(lab)}")

# ---------------- 3. dataset ----------------
class DS(torch.utils.data.Dataset):
    """Returns the 3-slice windows of one study, their slot numbers, labels and mask.
    Training: K random windows from the filled slots. Evaluation: every window."""
    def __init__(s, sids, y=None, w=None, train=False):
        s.sids, s.y, s.w, s.train, s.v = list(sids), y, w, train, None
    def __len__(s): return len(s.sids)
    def __getitem__(s, i):
        if s.v is None: s.v = np.memmap(MM, np.uint8, "r", shape=(cap, 64, IMG, IMG))
        p = POS[s.sids[i]]
        val = [(c, sl) for c, sl in CENT if SM[p, sl]] or CENT       # skip empty slots
        if s.train:
            pick = sorted(np.random.choice(len(val), K, replace=len(val) < K))
            val = [val[j] for j in pick]
        x = np.stack([s.v[p, c - 1:c + 2] for c, _ in val])          # (windows, 3, H, W)
        y = s.y[i] if s.y is not None else np.zeros(12, np.float32)
        w = s.w[i] if s.w is not None else np.ones(12, np.float32)
        slots = torch.tensor([sl for _, sl in val])
        return torch.from_numpy(x), slots, torch.from_numpy(y), torch.from_numpy(w)

dl = torch.utils.data.DataLoader(DS(lab[idc], Y, Wt, True), batch_size=BS, shuffle=True,
                                 drop_last=True, num_workers=4,
                                 persistent_workers=True, pin_memory=True)
dv = torch.utils.data.DataLoader(DS(gold.StudyInstanceUID), batch_size=1, num_workers=2)

# ---------------- 4. model ----------------
dev = "cuda"
# ImageNet statistics
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)

def prep(x, train):
    """Scale to [0, 1]; in training add a random 87.5% crop and small random
    gamma and gain changes; then apply the ImageNet normalisation."""
    x = x.to(dev, non_blocking=True).float().div(255)
    if train:
        B, Kk, H = x.shape[0], x.shape[1], x.shape[-1]
        out = x.new_empty(B, Kk, 3, CROP, CROP)
        for b in range(B):
            i, j = np.random.randint(0, H - CROP + 1, 2)
            g, gain = math.exp(np.random.uniform(-0.25, 0.25)), np.random.uniform(0.9, 1.1)
            out[b] = x[b, :, :, i:i + CROP, j:j + CROP].clamp_min(1e-4) ** g * gain
        x = out
    return (x - MEAN) / STD

class Net(nn.Module):
    """ConvNeXt encoder per window + slot embedding + one attention map per finding."""
    def __init__(s):
        super().__init__()
        # ImageNet weights
        s.bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
        # 768 for ConvNeXt-Tiny
        Fd = s.bb.num_features
        s.slot = nn.Embedding(5, Fd); nn.init.normal_(s.slot.weight, std=0.02)
        s.norm = nn.LayerNorm(Fd)
        s.att = nn.Sequential(nn.Linear(Fd, 256), nn.Tanh(), nn.Dropout(0.2),
                              nn.Linear(256, 12))
        s.W = nn.Parameter(torch.randn(12, Fd) * 0.02); s.b = nn.Parameter(torch.zeros(12))
    def forward(s, x, slots):
        B, Kk = x.shape[:2]
        flat = x.flatten(0, 1).contiguous(memory_format=torch.channels_last)
        # encode windows in chunks of 64
        f = torch.cat([s.bb(c) for c in flat.split(64)])
        # add the slot (view) embedding
        h = s.norm(f.view(B, Kk, -1) + s.slot(slots))
        # (B, windows, 12) attention
        a = s.att(h).float().softmax(1)
        # one pooled feature per finding
        pooled = torch.einsum("bkn,bkf->bnf", a, h.float())
        return (pooled * s.W).sum(-1) + s.b                       # 12 logits

model = Net().to(dev).to(memory_format=torch.channels_last)
# warm start (v13 starts from v11)
if INIT:
    model.load_state_dict(torch.load(find(INIT)[0], map_location="cpu")["model"])

# AdamW with a 10x higher learning rate for the new (non-backbone) layers
head = [p for n_, p in model.named_parameters() if not n_.startswith("bb.")]
opt = torch.optim.AdamW([{"params": model.bb.parameters(), "lr": LR},
                         {"params": head, "lr": LR * 10}], weight_decay=0.05)
total = EPOCHS * len(dl) // ACC; warm = min(300, total // 10)
def lr_factor(t):
    """Linear warm-up for `warm` steps, then cosine decay to zero."""
    if t < warm:
        return (t + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
scaler = torch.amp.GradScaler("cuda")                             # mixed precision

@torch.no_grad()
def evaluate():
    """Predict every gold study from all its windows and return per-finding AUC."""
    model.eval(); P = []
    for x, sl, _, _ in dv:
        with torch.autocast("cuda", dtype=torch.float16):
            P.append(model(prep(x, False), sl.to(dev)).float().sigmoid().cpu().numpy()[0])
    P = np.array(P)
    return P, [roc_auc_score(YG[:, j], P[:, j]) for j in range(12)]

# ---------------- 5. training loop ----------------
best = 0
for ep in range(EPOCHS):
    model.train(); tot = 0; opt.zero_grad(set_to_none=True)
    for it, (x, sl, y, w) in enumerate(dl):
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(prep(x, True), sl.to(dev))
        # binary cross-entropy against the soft labels, only where a label exists
        bce = nn.functional.binary_cross_entropy_with_logits(logit.float(), y.to(dev),
                                                             reduction="none")
        loss = (bce * w.to(dev)).sum() / w.sum().clamp_min(1)
        scaler.scale(loss / ACC).backward(); tot += loss.item()
        # update every ACC batches
        if (it + 1) % ACC == 0:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
    P, aucs = evaluate(); macro = float(np.mean(aucs))
    print(f"epoch {ep+1}/{EPOCHS} | loss {tot/len(dl):.4f} | gold macro AUC {macro:.4f}",
          flush=True)
    if macro > best:                                               # keep the best epoch
        best = macro
        torch.save({"model": model.state_dict(), "backbone": BACKBONE, "img": IMG},
                   f"/kaggle/working/{TAG}_best.pt")
        d = pd.DataFrame(P, columns=LAB)
        d.insert(0, "StudyInstanceUID", gold.StudyInstanceUID)
        d.to_csv(f"/kaggle/working/{TAG}_gold_preds.csv", index=False)
print(f"Best gold macro AUC: {best:.4f}")
