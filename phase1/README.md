# Phase 1 baseline

Slice-level ResNet (ResNet18, then RadImageNet ResNet50) with attention pooling over slices and
then over series, plus a small series-metadata encoder (plane, fluid-sensitive, fat suppression)
fused through a zero-initialised residual layer.

`run_cv.py` runs 5-fold cross-validation over the 58 gold studies. Each fold trains on the
report-derived labels (L3) of all other studies plus the expert labels of the other four gold
folds (weight 5), and uses the held-out fold for early stopping.

Settings: up to 16 slices x 8 series per study, 224 px.

Note: `run_cv.py` imports `masked_bce_loss` from `losses.py` (binary cross-entropy that ignores
missing labels and accepts per-sample weights). That file is not included yet.
