from __future__ import annotations

import torch


def metric_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    valid = mask.bool() & (target > 0)
    if not torch.any(valid):
        return pred.sum() * 0.0
    diff = torch.abs(torch.log(pred[valid] + 1e-6) - torch.log(target[valid] + 1e-6))
    if weight is not None:
        diff = diff * weight[valid]
    return diff.mean()


def gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_x = mask[:, :, 1:] & mask[:, :, :-1]
    valid_y = mask[:, 1:, :] & mask[:, :-1, :]
    grad_x = torch.abs((pred[:, :, 1:] - pred[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1]))
    grad_y = torch.abs((pred[:, 1:, :] - pred[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :]))
    terms = []
    if torch.any(valid_x):
        terms.append(grad_x[valid_x].mean())
    if torch.any(valid_y):
        terms.append(grad_y[valid_y].mean())
    return pred.sum() * 0.0 if not terms else sum(terms) / len(terms)


def ssi_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for pred_i, target_i, mask_i in zip(pred, target, mask):
        valid = mask_i.bool() & (target_i > 0)
        if not torch.any(valid):
            continue
        pred_n = robust_normalize(pred_i[valid])
        target_n = robust_normalize(target_i[valid])
        losses.append(torch.abs(pred_n - target_n).mean())
    return pred.sum() * 0.0 if not losses else sum(losses) / len(losses)


def robust_normalize(values: torch.Tensor) -> torch.Tensor:
    median = torch.median(values)
    centered = values - median
    scale = centered.abs().mean().clamp_min(1e-6)
    return centered / scale


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.bool() & (target > 0)
    if not torch.any(valid):
        return {"abs_rel": 0.0, "rmse": 0.0, "rmse_log": 0.0, "delta1": 0.0}
    pred_v = pred[valid].detach().clamp_min(1e-6)
    target_v = target[valid].detach().clamp_min(1e-6)
    ratio = torch.maximum(pred_v / target_v, target_v / pred_v)
    return {
        "abs_rel": float((torch.abs(pred_v - target_v) / target_v).mean().item()),
        "rmse": float(torch.sqrt(torch.mean((pred_v - target_v) ** 2)).item()),
        "rmse_log": float(torch.sqrt(torch.mean((torch.log(pred_v) - torch.log(target_v)) ** 2)).item()),
        "delta1": float((ratio < 1.25).to(torch.float32).mean().item()),
    }

