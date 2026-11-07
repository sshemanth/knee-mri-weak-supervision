"""
submit.py  (Kaggle notebook, GPU, internet OFF)
Builds the same 5-slot stack for every test study, runs the trained model on all
windows and writes submission.csv in the order of sample_submission.csv.
The functions hdr(), px(), series() and study() are identical to build_cache.py.
"""
WEIGHTS = "v13_384_best.pt"
ck = torch.load(find(WEIGHTS)[0], map_location="cpu")
# image size and backbone come from the checkpoint
IMG, BACKBONE = int(ck["img"]), ck["backbone"]

# Same network as train.py, except that the backbone is created with pretrained=False:
# internet is off, and all weights come from the checkpoint instead.
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.bb = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
        Fd = s.bb.num_features
        s.slot = nn.Embedding(5, Fd); s.norm = nn.LayerNorm(Fd)
        s.att = nn.Sequential(nn.Linear(Fd, 256), nn.Tanh(), nn.Dropout(0.2),
                              nn.Linear(256, 12))
        s.W = nn.Parameter(torch.zeros(12, Fd)); s.b = nn.Parameter(torch.zeros(12))
    def forward(s, x, slots):                                 # identical to train.py
        B, Kk = x.shape[:2]
        f = torch.cat([s.bb(c) for c in x.flatten(0, 1).split(32)])
        h = s.norm(f.view(B, Kk, -1) + s.slot(slots))
        a = s.att(h).float().softmax(1)
        return (torch.einsum("bkn,bkf->bnf", a, h.float()) * s.W).sum(-1) + s.b

model = Net(); model.load_state_dict(ck["model"])
model = model.to(dev).to(memory_format=torch.channels_last).eval()

ids = pd.read_csv(f"{R}/test.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
P = np.full((len(ids), 12), 0.5, np.float32)            # safe default if a study fails
# CPU workers read DICOM while the GPU predicts
pool = Pool(4)
for n, (sid, vol, filled) in enumerate(pool.imap(study, ids), 1):
    try:
        # filled slots
        sm = [vol[a:b, ::8, ::8].any() for a, b in zip(BND[:-1], BND[1:])]
        val = [(c, s) for c, s in CENT if sm[s]] or CENT
        x = np.stack([vol[c - 1:c + 2] for c, _ in val])
        x = torch.from_numpy(x)[None].to(dev).float().div(255)
        sl = torch.tensor([[s for _, s in val]], device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            P[n - 1] = model((x - MEAN) / STD, sl).float().sigmoid()[0].cpu().numpy()
    except Exception as e:
        print("study failed:", sid[-12:], type(e).__name__, flush=True)
pool.close()

# Write the file with the exact column order of the sample submission
cols = list(pd.read_csv(f"{R}/sample_submission.csv", nrows=1).columns)
sub = pd.DataFrame(P, columns=LAB); sub.insert(0, "StudyInstanceUID", ids); sub = sub[cols]
assert np.isfinite(sub[LAB].values).all()
sub.to_csv("/kaggle/working/submission.csv", index=False)
