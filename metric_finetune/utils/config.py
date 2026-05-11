from __future__ import annotations

from pathlib import Path
from time import strftime

from data_management import data_keys, data_keys_from_dataset_dir, default_depth_run_id, depth_estimater_dir
from path_utils import relative_to_workspace


def metric_train_keys(train_data: str | Path, timestamp: str | None = None) -> dict[str, str]:
    dataset_keys = data_keys_from_dataset_dir(train_data)
    return data_keys(
        case_id=dataset_keys["case_id"],
        mesh_id=dataset_keys.get("mesh_id"),
        source_mesh_id=dataset_keys.get("source_mesh_id"),
        texture_id=dataset_keys.get("texture_id"),
        dataset_id=dataset_keys.get("dataset_id"),
        dataset_split=dataset_keys.get("dataset_split"),
        depth_run_id=default_depth_run_id(timestamp or strftime("%Y%m%d_%H%M%S")),
    )


def default_metric_output_dir(train_data: str | Path, timestamp: str | None = None) -> Path:
    keys = metric_train_keys(train_data, timestamp=timestamp)
    return depth_estimater_dir(keys["mesh_id"], keys["depth_run_id"])


def metric_run_config(train_data: str | Path, timestamp: str | None = None) -> tuple[dict[str, str], Path]:
    run_timestamp = timestamp or strftime("%Y%m%d_%H%M%S")
    keys = metric_train_keys(train_data, timestamp=run_timestamp)
    return keys, depth_estimater_dir(keys["mesh_id"], keys["depth_run_id"])


def default_validation_output_dir(checkpoint_path: str | Path, data_dir: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    run_id = checkpoint.parent.name
    data_name = Path(data_dir).name
    return checkpoint.parent / "validation" / data_name


def serialize_paths(config: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    payload = dict(config)
    for key in keys:
        payload[key] = relative_to_workspace(payload[key])
    return payload
