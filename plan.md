# EndoOmni Metric-Depth Fine-Tuning Plan

## Goal

Fine-tune EndoOmni into a metric-depth estimator for the bronchoscopy localization pipeline.

The downloaded checkpoint:

```text
CODE/EndoOmni/models/weights/EndoOmni_b.pt
```

is a ViT-B EndoOmni checkpoint for endoscopic relative depth. The fine-tuned model should output metric depth, preferably in `mm`, so downstream code can use:

```text
RGB -> EndoOmni metric depth -> VOModel -> visual_servo runtime
```

without per-frame median scaling.

## Boundary

EndoOmni owns only monocular depth estimation and depth checkpoint export.

`visual_localization` still owns synthetic trajectory collection and VO training. `visual_servo` still owns branch landmarks, relocalization, and state-machine logic.

## Data

Use one shared supervised-depth dataset interface for both current synthetic data and future real registered-depth data.

Current synthetic data:

```text
Data/visual-localization/data_collection/AirwayHollow/<split>/
|- images/frame_<id>.png
|- depths/frame_<id>.npy
|- poses.csv
`- metadata.json
```

The synthetic depth maps are float32 ray-hit distances in `mm`; `0` means no hit.

Future real data should follow the same basic shape:

```text
Data/visual-localization/registered_depth/<case>/<split>/
|- images/
|- depths/
|- masks/        optional
|- poses.csv
`- metadata.json
```

For real registered depth, use `mask` when available. If registration confidence is available, load it as an optional pixel weight.

## Model

Start from `EndoOmni_b.pt`.

Use the existing DINOv2-DPT EndoOmni model as the backbone. Adapt the output so the model predicts positive metric depth instead of relative/disparity-style depth.

Initial fine-tuning schedule:

1. Train the metric head / decoder first.
2. Unfreeze the last DINOv2 blocks.
3. Unfreeze more encoder layers only if validation improves.

Use a smaller learning rate for the encoder than for the metric head.

## Loss

Keep the loss simple:

```text
L = L_metric + lambda_grad * L_gradient + lambda_ssi * L_ssi
```

Recommended starting point:

1. `L_metric`: masked SILog or log-L1 on metric depth.
2. `L_gradient`: masked depth-gradient loss for local geometry.
3. `L_ssi`: small auxiliary scale/shift-invariant loss to preserve EndoOmni relative-depth structure.

For synthetic data, mask is `depth > 0`.

For real registered data, mask is the registered valid-depth mask. Optional confidence weights can scale `L_metric`.

## Remaining Training Stage

### Mixed Synthetic And Real Fine-Tuning

Once real RGB + registered depth is available, train with both data sources.

Use synthetic data as a scale anchor and real data as the target visual domain. Start with a balanced sampler, then increase the real-data ratio when there is enough real coverage.

Split real data by case/procedure, not by adjacent frames.

## Outputs

Save fine-tuning artifacts under:

```text
Data/visual-localization/depth/EndoOmniMetric/<run_id>/
|- config.json
|- train_history.json
|- metrics.json
|- qualitative/
|- endoomni_metric_latest.pt
`- endoomni_metric_best.pt
```

Paths stored in configs should stay relative to the workspace root, matching the rest of the project.

## Evaluation

Depth evaluation:

1. AbsRel
2. RMSE in mm
3. RMSElog
4. delta thresholds
5. qualitative RGB / GT / prediction / error grids

Downstream evaluation:

1. Train VO using the fine-tuned EndoOmni metric adapter.
2. Evaluate VO on `data_collection/<mesh>/testset`.
3. Reuse the same adapter in visual-servo planned-route replay.

## Remaining Implementation Order

1. Add real RGB + registered-depth cases when they are available.
2. Train the mixed-domain metric model using synthetic data as the scale anchor.
3. Run downstream VO evaluation and planned-route replay with the selected metric checkpoint.
