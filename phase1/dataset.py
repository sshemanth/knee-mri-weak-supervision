import random
import numpy as np, pandas as pd, pydicom, torch
import torchvision.transforms.functional as TF
from pathlib import Path
from torch.utils.data import Dataset

FINDINGS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
            'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

PLANE_MAP = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2}  # unknown/other -> 3


class KneeMRIDataset(Dataset):
    """One item = one study. Returns all its series as a list of [S,H,W] (1-channel) or
    [S,3,H,W] (2.5D, use_25d=True) tensors (S = slice count, varies per series), an
    optional parallel list of per-series metadata tensors (use_series_meta via
    series_meta_df), plus a [12] label vector (may contain NaN)."""

    def __init__(self, study_ids, labels_df, series_root, findings=FINDINGS,
                 image_size=224, max_slices=24, max_series=None, weights_df=None, cache_dir=None,
                 augment=False, use_25d=False, series_meta_df=None):
        self.study_ids = list(study_ids)
        self.labels = labels_df.set_index('StudyInstanceUID')
        self.weights = weights_df.set_index('StudyInstanceUID') if weights_df is not None else None
        self.series_root = Path(series_root)
        self.findings = findings
        self.image_size = image_size
        self.max_slices = max_slices
        self.max_series = max_series
        self.augment = augment
        self.use_25d = use_25d
        if series_meta_df is not None:
            self.series_meta = series_meta_df.set_index(['StudyInstanceUID', 'SeriesInstanceUID'])
        else:
            self.series_meta = None
        if cache_dir is not None:
            # mode_tag is added only for 2.5D, so 1-channel caches stay reusable.
            mode_tag = "_25d" if use_25d else ""
            self.cache_dir = Path(cache_dir) / f"{image_size}_{max_slices}_{max_series}{mode_tag}"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

    def __len__(self):
        return len(self.study_ids)

    def _list_series_dirs(self, study_dir):
        dirs = []
        if study_dir.exists():
            for series_dir in sorted(study_dir.iterdir()):
                if series_dir.is_dir():
                    dirs.append(series_dir)
                    if self.max_series is not None and len(dirs) >= self.max_series:
                        break
        return dirs

    def _load_series(self, series_dir):
        """Read up to max_slices DICOM slices, scale each to [0,1] and resize."""
        files = sorted(series_dir.glob('*.dcm'))[:self.max_slices]
        slices = []
        for fp in files:
            try:
                ds = pydicom.dcmread(fp)
                arr = ds.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                slices.append(arr)
            except Exception:
                continue
        if not slices:
            if self.use_25d:
                return torch.zeros(1, 3, self.image_size, self.image_size)
            return torch.zeros(1, self.image_size, self.image_size)
        stack = np.stack(slices)
        t = torch.from_numpy(stack).unsqueeze(1)  # [S,1,H,W]
        t = torch.nn.functional.interpolate(t, size=(self.image_size, self.image_size),
                                             mode='bilinear', align_corners=False)
        t = t.squeeze(1)  # [S,H,W]
        if not self.use_25d:
            return t
        S = t.shape[0]
        idx_prev = torch.clamp(torch.arange(S) - 1, min=0)
        idx_next = torch.clamp(torch.arange(S) + 1, max=S - 1)
        t25 = torch.stack([t[idx_prev], t, t[idx_next]], dim=1)  # [S,3,H,W] -- each slice
        # stacked with its immediate neighbors (edges repeat) as a 3-channel 2.5D input.
        return t25

    def _augment_series(self, t):
        """t: [S,H,W] or [S,3,H,W] in [0,1]. Same random rotation/translation/scale/
        intensity jitter applied to every slice (and every channel, in 2.5D mode) in the
        series -- keeps the stack anatomically/temporally coherent. Deliberately NO flips:
        Medial/Lateral findings are orientation-specific, and the DICOMs come from 19
        sites across the challenge with no guaranteed consistent left/right convention, so flipping risks
        silently swapping what medial vs lateral means for that study."""
        squeeze_back = t.dim() == 3
        if squeeze_back:
            t = t.unsqueeze(1)  # [S,1,H,W]
        angle = random.uniform(-12, 12)
        max_shift = 0.05 * self.image_size
        translate = [random.uniform(-max_shift, max_shift), random.uniform(-max_shift, max_shift)]
        scale = random.uniform(0.95, 1.05)
        t = TF.affine(t, angle=angle, translate=translate, scale=scale, shear=[0.0, 0.0])
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        mean = t.mean()
        t = (t - mean) * contrast + mean
        t = t * brightness
        t = t.clamp(0.0, 1.0)
        return t.squeeze(1) if squeeze_back else t

    def _series_meta_vec(self, study_id, series_id):
        if self.series_meta is not None and (study_id, series_id) in self.series_meta.index:
            row = self.series_meta.loc[(study_id, series_id)]
            plane = PLANE_MAP.get(row['Anatomical_Plane'], 3)
            fluid = float(row['Fluid_Sensitive']) if pd.notna(row['Fluid_Sensitive']) else 0.5
            fat = float(row['Fat_Suppression']) if pd.notna(row['Fat_Suppression']) else 0.5
        else:
            plane, fluid, fat = 3, 0.5, 0.5
        return torch.tensor([float(plane), fluid, fat], dtype=torch.float32)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        study_dir = self.series_root / study_id
        series_dirs = self._list_series_dirs(study_dir)
        series_ids = [d.name for d in series_dirs]

        cache_file = self.cache_dir / f"{study_id}.pt" if self.cache_dir is not None else None
        series_tensors = None
        if cache_file is not None and cache_file.exists():
            try:
                series_tensors = torch.load(cache_file, weights_only=True)
            except Exception:
                series_tensors = None
        if series_tensors is None:
            series_tensors = [self._load_series(d) for d in series_dirs]
            if cache_file is not None:
                try:
                    tmp = cache_file.with_suffix('.tmp')
                    torch.save(series_tensors, tmp)
                    tmp.replace(cache_file)
                except Exception:
                    pass
        if self.augment:
            series_tensors = [self._augment_series(s) for s in series_tensors]

        series_meta = [self._series_meta_vec(study_id, sid) for sid in series_ids] \
            if self.series_meta is not None else None

        if study_id in self.labels.index:
            row = self.labels.loc[study_id, self.findings]
        else:
            row = pd.Series([float('nan')] * len(self.findings), index=self.findings)
        y = torch.tensor(row.astype(float).values, dtype=torch.float32)
        if self.weights is not None and study_id in self.weights.index:
            w_row = self.weights.loc[study_id, self.findings].astype(float).fillna(1.0)
            w = torch.tensor(w_row.values, dtype=torch.float32)
        else:
            w = torch.ones(len(self.findings), dtype=torch.float32)
        return {'study_id': study_id, 'series': series_tensors, 'series_meta': series_meta,
                'label': y, 'weight': w}


def collate_studies(batch):
    """Studies have a variable number of series with variable slice counts -> can't
    torch.stack them, so keep the batch as a plain list; the caller stacks labels."""
    return batch
