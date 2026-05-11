from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from tqdm import tqdm


ENDOOMNI_ROOT = Path(__file__).resolve().parent
CODE_ROOT = ENDOOMNI_ROOT.parent
for import_root in (ENDOOMNI_ROOT, CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from metric_finetune.utils.config import metric_run_config, serialize_paths
from metric_finetune.utils.data import create_loader
from metric_finetune.utils.losses import depth_metrics, gradient_loss, metric_loss, ssi_loss
from metric_finetune.utils.model import load_model, parameter_groups, set_encoder_trainable
from training_curves import save_loss_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune EndoOmni for bronchoscopy metric depth.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=ENDOOMNI_ROOT / "models" / "weights" / "EndoOmni_b.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--encoder", choices=("vits", "vitb", "vitl"), default="vitb")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-grad", type=float, default=0.1)
    parser.add_argument("--lambda-ssi", type=float, default=0.05)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    image_size = (args.height, args.width)
    keys, default_output = metric_run_config(args.train_data)
    output = args.output or default_output
    output.mkdir(parents=True, exist_ok=True)

    config = serialize_paths(vars(args), ("train_data", "val_data", "weights", "output"))
    config["data_keys"] = keys
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_loader = create_loader(args.train_data, image_size, args.batch_size, True, args.num_workers)
    val_loader = None if args.val_data is None else create_loader(args.val_data, image_size, args.batch_size, False, args.num_workers)

    model = load_model(args.weights, encoder=args.encoder, device=device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, lr_head=args.lr_head, lr_encoder=args.lr_encoder),
        weight_decay=args.weight_decay,
    )

    best = float("inf")
    history = []
    loss_history: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        set_encoder_trainable(model, epoch > args.freeze_encoder_epochs)
        train_stats = run_epoch(model, train_loader, device, optimizer, args, f"Epoch {epoch:03d}/{args.epochs} train")
        val_stats = run_epoch(model, val_loader, device, None, args, f"Epoch {epoch:03d}/{args.epochs} val") if val_loader else None
        score = train_stats["loss"] if val_stats is None else val_stats["loss"]
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        loss_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_stats["loss"]),
                "eval_loss": None if val_stats is None else float(val_stats["loss"]),
                "train_abs_rel": float(train_stats["abs_rel"]),
                "eval_abs_rel": None if val_stats is None else float(val_stats["abs_rel"]),
            }
        )
        save_checkpoint(output / "endoomni_metric_latest.pt", model, epoch, score, config)
        if score < best:
            best = score
            save_checkpoint(output / "endoomni_metric_best.pt", model, epoch, score, config)
        print(format_epoch(epoch, args.epochs, train_stats, val_stats))

    save_loss_history(output, loss_history, title="EndoOmni Metric Loss")
    (output / "train_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = history[-1]["val"] or history[-1]["train"]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def run_epoch(model, loader, device, optimizer, args, desc: str) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {"loss": 0.0, "metric": 0.0, "grad": 0.0, "ssi": 0.0}
    metric_totals: dict[str, float] = {"abs_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "delta1": 0.0}
    count = 0
    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        mask = batch["mask"].to(device).bool()
        weight = batch["weight"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(image)
            loss_metric = metric_loss(pred, depth, mask, weight)
            loss_grad = gradient_loss(pred, depth, mask)
            loss_ssi = ssi_loss(pred, depth, mask)
            loss = loss_metric + args.lambda_grad * loss_grad + args.lambda_ssi * loss_ssi
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = int(image.shape[0])
        count += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        totals["metric"] += float(loss_metric.item()) * batch_size
        totals["grad"] += float(loss_grad.item()) * batch_size
        totals["ssi"] += float(loss_ssi.item()) * batch_size
        for key, value in depth_metrics(pred, depth, mask).items():
            metric_totals[key] += value * batch_size
        progress.set_postfix(loss=f"{float(loss.item()):.4f}")
    denom = max(count, 1)
    return {key: value / denom for key, value in {**totals, **metric_totals}.items()}


def save_checkpoint(path: Path, model, epoch: int, loss: float, config: dict[str, object]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss),
            "config": config,
        },
        path,
    )


def format_epoch(epoch: int, epochs: int, train: dict[str, float], val: dict[str, float] | None) -> str:
    text = f"Epoch {epoch:03d}/{epochs} train_loss={train['loss']:.6f} train_absrel={train['abs_rel']:.4f}"
    if val is not None:
        text += f" val_loss={val['loss']:.6f} val_absrel={val['abs_rel']:.4f} val_rmse={val['rmse']:.3f}"
    return text


if __name__ == "__main__":
    main()
