from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


ENDOOMNI_ROOT = Path(__file__).resolve().parent
CODE_ROOT = ENDOOMNI_ROOT.parent
WORKSPACE_ROOT = CODE_ROOT.parent
for import_root in (ENDOOMNI_ROOT, CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from metric_finetune.data import MetricDepthDataset
from metric_finetune.losses import depth_metrics
from metric_finetune.model import load_model
from path_utils import relative_to_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an EndoOmni metric-depth checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Frame-level RGB/depth dataset directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="endoomni_metric_*.pt checkpoint.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory. Defaults near the checkpoint.")
    parser.add_argument("--height", type=int, default=None, help="Validation image height. Defaults to checkpoint config.")
    parser.add_argument("--width", type=int, default=None, help="Validation image width. Defaults to checkpoint config.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-visuals", type=int, default=16, help="Number of RGB/GT/inferred triplets to save.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_metric_depth(
        data_dir=args.data,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_visuals=args.num_visuals,
        device_name=args.device,
    )
    print(format_metrics(result["metrics"]))
    print(f"Validation outputs saved to {result['output_dir']}")


def validate_metric_depth(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    height: int | None = None,
    width: int | None = None,
    batch_size: int = 4,
    num_workers: int = 0,
    num_visuals: int = 16,
    device_name: str = "cpu",
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    image_size = (
        int(height if height is not None else config.get("height", 224)),
        int(width if width is not None else config.get("width", 224)),
    )
    output = Path(output_dir) if output_dir is not None else default_output_dir(checkpoint_path, data_dir)
    output.mkdir(parents=True, exist_ok=True)
    visual_dir = output / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    dataset = MetricDepthDataset(data_dir, image_size=image_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device(device_name)
    model = load_model(checkpoint_path, device=device)
    model.eval()

    totals = {"abs_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "delta1": 0.0}
    valid_pixels = 0
    sample_count = 0
    saved_visuals = 0
    records: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="validate metric depth", leave=False):
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            mask = batch["mask"].to(device).bool()
            pred = model(image)
            metrics = depth_metrics(pred, depth, mask)
            batch_size_actual = int(image.shape[0])
            sample_count += batch_size_actual
            pixel_count = int(torch.count_nonzero(mask & (depth > 0)).item())
            valid_pixels += pixel_count
            for key, value in metrics.items():
                totals[key] += float(value) * batch_size_actual

            for index in range(batch_size_actual):
                image_path = str(batch["image_path"][index])
                if saved_visuals < int(num_visuals):
                    visual_path = visual_dir / f"sample_{saved_visuals:04d}.png"
                    save_depth_triplet(
                        visual_path,
                        image[index].detach().cpu(),
                        depth[index].detach().cpu(),
                        pred[index].detach().cpu(),
                        mask[index].detach().cpu(),
                    )
                    saved_visuals += 1
                records.append(
                    {
                        "image_path": relative_to_workspace(image_path),
                        "valid_pixels": int(torch.count_nonzero(mask[index] & (depth[index] > 0)).item()),
                    }
                )

    denom = max(sample_count, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["sample_count"] = sample_count
    metrics["valid_pixel_count"] = valid_pixels
    payload = {
        "data": relative_to_workspace(data_dir),
        "checkpoint": relative_to_workspace(checkpoint_path),
        "output_dir": relative_to_workspace(output),
        "image_size": list(image_size),
        "metrics": metrics,
        "visual_count": saved_visuals,
        "records": records,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "validation.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def save_depth_triplet(
    path: Path,
    image: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_depth: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required to save validation visuals.") from exc

    rgb = image_to_uint8(image)
    gt = depth_to_rgb(gt_depth, mask)
    pred = depth_to_rgb(pred_depth, mask)
    title_height = 20
    panel_height, panel_width = rgb.shape[:2]
    canvas = Image.new("RGB", (panel_width * 3, panel_height + title_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate((("RGB", rgb), ("gt_depth", gt), ("infer_depth", pred))):
        x = index * panel_width
        draw.text((x + 4, 3), title, fill=(0, 0, 0))
        canvas.paste(Image.fromarray(panel), (x, title_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def image_to_uint8(image: torch.Tensor) -> np.ndarray:
    tensor = image.detach().to(dtype=torch.float32)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"image must have shape (3, H, W). Got {tuple(tensor.shape)}.")
    tensor = (tensor * 0.5 + 0.5).clamp(0.0, 1.0)
    return (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def depth_to_rgb(depth: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    values = depth.detach().to(dtype=torch.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    valid = mask.bool() & torch.isfinite(values) & (values > 0)
    normalized = torch.zeros_like(values, dtype=torch.float32)
    if torch.any(valid):
        valid_values = values[valid]
        low = torch.quantile(valid_values, 0.02)
        high = torch.quantile(valid_values, 0.98)
        scale = torch.clamp(high - low, min=1e-6)
        normalized[valid] = torch.clamp((values[valid] - low) / scale, 0.0, 1.0)
    gray = (normalized.numpy() * 255.0).round().astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[~valid.numpy()] = np.array([30, 30, 30], dtype=np.uint8)
    return rgb


def default_output_dir(checkpoint_path: str | Path, data_dir: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    run_id = checkpoint.parent.name
    data_name = Path(data_dir).name
    return WORKSPACE_ROOT / "Data" / "depth_estimater" / "EndoOmniMetric" / run_id / "validation" / data_name


def format_metrics(metrics: dict[str, object]) -> str:
    return (
        f"samples={int(metrics['sample_count'])} "
        f"abs_rel={float(metrics['abs_rel']):.6f} "
        f"rmse={float(metrics['rmse']):.6f} "
        f"rmse_log={float(metrics['rmse_log']):.6f} "
        f"delta1={float(metrics['delta1']):.6f}"
    )


if __name__ == "__main__":
    main()
