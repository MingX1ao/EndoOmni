# EndoOmni Metric-Depth Plan

## Boundary

EndoOmni owns monocular metric-depth estimation and checkpoint export. It does
not own VO training, descriptor training, trigger policies, or runtime replay.

The downstream contract is:

```text
RGB -> EndoOmni metric depth -> VO / descriptor / visual-servo runtime
```

## Current Default Checkpoint

Use the `DeletedModel` metric-depth checkpoint as the current default:

```text
database/DeletedModel/depth_estimater/run_20260526_193032/endoomni_metric_best.pt
```

Training source:

```text
database/DeletedModel/collections/vo_synth_224x224_mm/trainset
database/DeletedModel/collections/vo_synth_224x224_mm/testset
CODE/EndoOmni/models/weights/EndoOmni_b.pt
```

Validation metrics:

```text
AbsRel: 0.0140
RMSE: 1.045 mm
RMSElog: 0.0282
delta1: 0.9972
```

## Data Contract

Synthetic and future registered-real depth data should use the same frame-level
shape:

```text
images/frame_<id>.png
depths/frame_<id>.npy
poses.csv
metadata.json
masks/      optional
weights/    optional
```

Depth values are metric millimeters. Invalid synthetic depth is `0`; real
registered data should provide masks when available.

## Output Contract

Metric-depth runs are saved under:

```text
database/<mesh_id>/depth_estimater/<depth_run_id>/
|- config.json
|- train_history.json
|- loss_history.json
|- metrics.json
|- endoomni_metric_latest.pt
`- endoomni_metric_best.pt
```

Paths stored in configs should remain workspace-relative.

## Next Work

1. Keep `run_20260526_193032` as the synthetic-trained default for `DeletedModel`.
2. When registered real depth becomes available, train a mixed synthetic-real model using synthetic data as the scale anchor.
3. Re-run VO, descriptor, and runtime replay selection after replacing the depth checkpoint.
