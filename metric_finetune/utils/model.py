from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ENDOOMNI_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ENDOOMNI_ROOT.parent
WORKSPACE_ROOT = CODE_ROOT.parent
if str(ENDOOMNI_ROOT) not in sys.path:
    sys.path.insert(0, str(ENDOOMNI_ROOT))


class EndoOmniMetricModel(nn.Module):
    """Thin metric-depth wrapper around the existing EndoOmni DINOv2-DPT model."""

    def __init__(self, encoder: str = "vitb") -> None:
        super().__init__()
        self.encoder = encoder
        with endoomni_workdir():
            from models.depth_anything.dpt import DepthAnything

            self.base = DepthAnything(encoder=encoder)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        depth = self.base(image)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        return F.softplus(depth) + 1e-6


def load_model(
    checkpoint: str | Path | None = None,
    encoder: str | None = None,
    device: str | torch.device = "cpu",
) -> EndoOmniMetricModel:
    checkpoint = resolve_checkpoint_path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu") if checkpoint is not None else None
    if encoder is None:
        encoder = checkpoint_encoder(payload) or "vitb"
    model = EndoOmniMetricModel(encoder=encoder)
    if payload is not None:
        state = payload.get("model", payload.get("model_state_dict", payload))
        if any(key.startswith("base.") for key in state):
            model.load_state_dict(state, strict=False)
        else:
            model.base.load_state_dict(state, strict=False)
    return model.to(device)


def resolve_checkpoint_path(checkpoint: str | Path | None) -> Path | None:
    if checkpoint is None:
        return None
    path = Path(checkpoint).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                WORKSPACE_ROOT / path,
                CODE_ROOT / path,
                ENDOOMNI_ROOT / path,
            ]
        )
        parts = path.parts
        if parts and parts[0] == "EndoOmni":
            candidates.append(CODE_ROOT / path)
            candidates.append(WORKSPACE_ROOT / "CODE" / path)
        elif parts and parts[0] == "models":
            candidates.append(ENDOOMNI_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(candidate) for candidate in unique_paths(candidates))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}\nSearched:\n  - {searched}")


def unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def checkpoint_encoder(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if isinstance(config, dict) and config.get("encoder"):
        return str(config["encoder"])
    return None


def parameter_groups(model: EndoOmniMetricModel, lr_head: float, lr_encoder: float) -> list[dict[str, object]]:
    encoder_params = list(model.base.pretrained.parameters())
    encoder_ids = {id(param) for param in encoder_params}
    head_params = [param for param in model.parameters() if id(param) not in encoder_ids]
    return [
        {"params": encoder_params, "lr": lr_encoder},
        {"params": head_params, "lr": lr_head},
    ]


def set_encoder_trainable(model: EndoOmniMetricModel, trainable: bool) -> None:
    for param in model.base.pretrained.parameters():
        param.requires_grad = trainable


@contextmanager
def endoomni_workdir():
    old_cwd = os.getcwd()
    os.chdir(ENDOOMNI_ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)
