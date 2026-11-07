"""
build_cache.py  (Kaggle notebook, CPU only)
Turn every study into one fixed stack of slices: 5 slots, 64 slices,
140 mm physical crop, resized to IMG pixels, stored as 8-bit values.
For large caches, run several copies with PART = 0, 1, ... to stay under
Kaggle's 20 GB output limit.
"""
import os, time, numpy as np, pandas as pd, pydicom, cv2
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

PART, NPARTS = 0, 2                       # which part of the studies this notebook builds
IMG, CROP_MM, SHARD, MAXS = 384, 140.0, 100, 64
# (plane, fluid-sensitive: 1 = prefer yes, 0 = prefer no, -1 = any, number of slices)
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
OUT = "/kaggle/working/cache384"
R = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
if not os.path.exists(R): R = "/kaggle/input/rsna-knee-abnormality-detection"
TAGS = ["ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber", "PixelSpacing"]

def hdr(f):
    """Read only the header of one DICOM file. Returns the slice position along
    the slice normal (for sorting), the file name and the pixel spacing in mm."""
    try:
        h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=TAGS)
        iop, ipp = h.get("ImageOrientationPatient"), h.get("ImagePositionPatient")
        if iop is not None and ipp is not None and len(iop) == 6:
            # slice normal
            n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
            pos = float(np.dot(np.array(ipp, float), n))
        else:
            # fallback order
            pos = float(h.get("InstanceNumber") or 0)
        ps = h.get("PixelSpacing")
        return pos, f, (float(ps[0]) if ps else 0.5)
    except Exception:
        return None

def px(f):
    """Read the pixels of one DICOM file as float32, applying the rescale tags
    and inverting MONOCHROME1 images so that bright always means high signal."""
    try:
        d = pydicom.dcmread(f)
        a = d.pixel_array.astype(np.float32)
        if a.ndim == 3: a = a[a.shape[0] // 2]
        s, i = d.get("RescaleSlope"), d.get("RescaleIntercept")
        if s is not None and i is not None: a = a * float(s) + float(i)
        if str(d.get("PhotometricInterpretation", "")) == "MONOCHROME1": a = a.max() - a
        return a
    except Exception:
        return None

def series(sdir, k, ex):
    """Pick k evenly spaced slices from the middle 96% of one series,
    crop each to 140 mm, normalise and resize. Returns (k, IMG, IMG) uint8."""
    out = np.zeros((k, IMG, IMG), np.uint8)
    fs = [os.path.join(sdir, f) for f in os.listdir(sdir)]
    # sort by position
    recs = sorted([r for r in ex.map(hdr, fs) if r], key=lambda r: r[0])
    n = len(recs)
    if n == 0: return out, 0
    lo = int(n * 0.02); hi = max(int(n * 0.98) - 1, lo)
    picks = np.linspace(lo, hi, k).round().astype(int)
    uniq = sorted(set(picks.tolist()))
    # read pixels in parallel
    arrs = dict(zip(uniq, ex.map(lambda j: px(recs[j][1]), uniq)))
    valid = [a for a in arrs.values() if a is not None]
    if not valid: return out, 0
    # One intensity window per series: 2nd to 98th percentile
    loq, hiq = np.percentile(np.concatenate([a[::2, ::2].ravel() for a in valid]), [2, 98])
    ok = 0
    for t, j in enumerate(picks):
        a = arrs[int(j)]
        if a is None: continue
        ps = recs[j][2] if recs[j][2] > 0 else 0.5
        # 140 mm in pixels
        h, w = a.shape; c = min(int(round(CROP_MM / ps)), h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        a = np.clip((a[y0:y0 + c, x0:x0 + c] - loq) / (hiq - loq + 1e-6), 0, 1)
        interp = cv2.INTER_AREA if c > IMG else cv2.INTER_LINEAR
        out[t] = (cv2.resize(a, (IMG, IMG), interpolation=interp) * 255).astype(np.uint8)
        ok += 1
    return out, ok

def study(sid):
    """Fill the 5 slots of one study. Each series is used at most once; a slot
    with no suitable series stays zero. Returns the stack and slices read per slot."""
    cv2.setNumThreads(1)
    rows = SER.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); filled = []; used = set(); idx = 0
    with ThreadPoolExecutor(16) as ex:
        for plane, fluid, k in SLOTS:
            c = [r for r in rows if r[1] == plane and r[0] not in used]
            pref = [r for r in c if r[2] == fluid] if fluid in (0, 1) else c
            # preferred, else any
            r = (pref or c or [None])[0]
            ok = 0
            if r is not None:
                used.add(r[0])
                try: vol[idx:idx + k], ok = series(f"{R}/train_series/{sid}/{r[0]}", k, ex)
                except Exception: ok = 0
            filled.append(ok); idx += k
    return sid, vol, filled

# ---- series table: (series id, plane, fluid-sensitive) per study ----
ser = pd.read_csv(f"{R}/train_series.csv",
                  dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
SER = {k: list(zip(g.SeriesInstanceUID, g.Anatomical_Plane, g.Fluid_Sensitive))
       for k, g in ser.groupby("StudyInstanceUID")}
ids = pd.read_csv(f"{R}/train.csv",
                  dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
ids = ids[PART::NPARTS]                     # this notebook's share of the studies
os.makedirs(OUT, exist_ok=True)

# ---- process studies on 4 CPU cores and save in compressed shards of 100 studies ----
t0 = time.time(); meta = []; buf_v = []; buf_i = []; shard = 0
def flush():
    global shard, buf_v, buf_i
    if buf_i:
        np.savez_compressed(f"{OUT}/shard_{shard:03d}.npz",
                            vol=np.stack(buf_v), ids=np.array(buf_i))
        shard += 1; buf_v = []; buf_i = []

with Pool(4) as pool:
    for n, (sid, vol, filled) in enumerate(pool.imap(study, ids), 1):
        buf_v.append(vol); buf_i.append(sid); meta.append([sid] + filled)
        if len(buf_i) == SHARD: flush()
        if n % 50 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} | {(time.time()-t0)/60:.1f} min", flush=True)
flush()

# ---- summary: how often each slot could be filled ----
m = pd.DataFrame(meta, columns=["StudyInstanceUID", "sag_fluid", "sag_other",
                                "cor_fluid", "cor_other", "axial"])
m.to_csv(f"{OUT}/meta.csv", index=False)
print((m.iloc[:, 1:] > 0).mean().round(3).to_string())
