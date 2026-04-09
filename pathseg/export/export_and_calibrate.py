from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import click
import torch
import torch.nn.functional as F
import yaml

from pathseg.utils.load_experiment import load_experiment


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _get_cfg_section(cfg: dict[str, Any]) -> dict[str, Any]:
    if "fit" in cfg and isinstance(cfg["fit"], dict):
        return cfg["fit"]
    return cfg


def import_from_string(path: str):
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _get_quantile(scores: torch.Tensor, alpha: float) -> float:
    """
    Standard split-conformal quantile:
        k = ceil((n + 1) * (1 - alpha))
    """
    if scores.numel() == 0:
        raise ValueError("Cannot compute conformal quantile on empty scores.")

    scores = torch.sort(scores).values
    n = scores.numel()
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = min(max(k, 1), n)
    return float(scores[k - 1].item())


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML dict in {path}, got {type(data).__name__}")
    return data


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# -----------------------------------------------------------------------------
# Calibration spec
# -----------------------------------------------------------------------------


@dataclass
class HeadCalibrationSpec:
    name: str
    enabled: bool
    dataloader_idx: int
    logits_key: str
    num_classes: int
    ignore_index: int
    exclude_classes: list[int] = field(default_factory=list)
    filter_background: bool = False

    def excluded_set(self) -> set[int]:
        return set(self.exclude_classes)


# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------


def export_model_bundle(
    cfg: dict[str, Any],
    pl_model: torch.nn.Module,
    output_dir: Path,
    yaml_name: str,
    weights_name: str,
) -> tuple[Path, Path]:
    if not hasattr(pl_model, "network"):
        raise AttributeError("Loaded Lightning model has no attribute 'network'.")

    network = pl_model.network.eval()
    cfg_section = _get_cfg_section(cfg)

    try:
        network_cfg = cfg_section["model"]["init_args"]["network"]
    except KeyError as e:
        raise KeyError(
            "Could not find cfg['model']['init_args']['network'] in the config."
        ) from e

    if "class_path" not in network_cfg:
        raise KeyError("network config has no 'class_path'.")
    if "init_args" not in network_cfg:
        raise KeyError("network config has no 'init_args'.")

    class_path = str(network_cfg["class_path"])
    init_args = _to_builtin(network_cfg["init_args"])

    weights_path = output_dir / weights_name
    yaml_path = output_dir / yaml_name

    # save CPU weights for portability
    state_dict_cpu = {
        k: v.detach().cpu() if torch.is_tensor(v) else v
        for k, v in network.state_dict().items()
    }
    torch.save(state_dict_cpu, weights_path)

    export_cfg = {
        "model": {
            "factory": class_path,
            "init_args": init_args,
        }
    }

    _save_yaml(yaml_path, export_cfg)
    return weights_path, yaml_path


# -----------------------------------------------------------------------------
# Calibration helpers
# -----------------------------------------------------------------------------


def _get_datamodule_from_bundle(bundle: dict[str, Any]):
    dm = bundle.get("datamodule", None)
    if dm is None:
        raise KeyError(
            "load_experiment(...) did not return a 'datamodule'. "
            "Please adapt this script to instantiate the datamodule from config, "
            "or update load_experiment to return it."
        )
    return dm


def _get_selected_dataloader(loaders: Any, dataloader_idx: int):
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    if dataloader_idx < 0 or dataloader_idx >= len(loaders):
        raise IndexError(
            f"dataloader_idx={dataloader_idx} out of range. "
            f"Found {len(loaders)} dataloader(s)."
        )
    return loaders[dataloader_idx]


def _extract_targets_pp(
    pl_model,
    targets_raw,
    ignore_index: int,
) -> list[torch.Tensor]:
    targets_pp = pl_model.to_per_pixel_targets_semantic(targets_raw, ignore_index)
    return [t.long() for t in targets_pp]


@torch.no_grad()
def infer_logits_list_for_head(
    pl_model,
    imgs: torch.Tensor,
    logits_key: str,
) -> list[torch.Tensor]:
    """
    Mirrors TwoHeadSemantic.eval_step logic:
      - window_imgs_semantic
      - network forward on crops
      - interpolate crop logits to img_size
      - revert_window_logits_semantic
    Returns:
      list of logits tensors [(C,H,W), ...]
    """
    device = next(pl_model.parameters()).device

    crops, origins, img_sizes = pl_model.window_imgs_semantic(imgs)
    crops = crops.to(device, non_blocking=True)
    crop_out = pl_model(crops)

    if not isinstance(crop_out, dict):
        raise TypeError("Expected model forward output to be a dict for calibration.")

    if logits_key not in crop_out:
        raise KeyError(
            f"Requested logits_key='{logits_key}' not found in model output keys: "
            f"{list(crop_out.keys())}"
        )

    crop_logits = crop_out[logits_key]
    crop_logits = F.interpolate(
        crop_logits,
        pl_model.img_size,
        mode="bilinear",
    )
    logits_list = pl_model.revert_window_logits_semantic(
        crop_logits,
        origins,
        img_sizes,
    )
    return logits_list


def validate_logits_num_classes(
    logits: torch.Tensor,
    spec: HeadCalibrationSpec,
) -> None:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected per-image logits shape (C,H,W), got {tuple(logits.shape)}"
        )
    if logits.shape[0] != spec.num_classes:
        raise ValueError(
            f"Head '{spec.name}' / logits_key='{spec.logits_key}' produced "
            f"C={logits.shape[0]} channels, but calibration config expects "
            f"num_classes={spec.num_classes}."
        )


@torch.no_grad()
def fit_class_conditional_conformal_for_head(
    pl_model,
    dataloader: Iterable[Any],
    spec: HeadCalibrationSpec,
    alpha: float,
) -> dict[str, Any]:
    excluded = spec.excluded_set()
    scores_by_class: list[list[torch.Tensor]] = [[] for _ in range(spec.num_classes)]

    for batch_idx, batch in enumerate(dataloader):
        imgs, targets, _source_ids, _image_ids = batch

        targets_pp = _extract_targets_pp(pl_model, targets, spec.ignore_index)
        logits_list = infer_logits_list_for_head(pl_model, imgs, spec.logits_key)

        if len(logits_list) != len(targets_pp):
            raise ValueError(
                f"Mismatch between logits ({len(logits_list)}) and targets "
                f"({len(targets_pp)}) in batch {batch_idx}."
            )

        for logits, tgt in zip(logits_list, targets_pp):
            validate_logits_num_classes(logits, spec)

            probs = F.softmax(logits.unsqueeze(0), dim=1).squeeze(0)  # [C,H,W]
            pred = torch.argmax(probs, dim=0)  # [H, W]
            tgt = tgt.to(probs.device).long()

            valid = (tgt != spec.ignore_index) & (tgt >= 0) & (tgt < spec.num_classes)
            if excluded:
                for c in excluded:
                    valid &= tgt != c

            if not valid.any():
                continue

            tgt_safe = tgt.clamp(0, spec.num_classes - 1)
            p_true = probs.gather(0, tgt_safe.unsqueeze(0)).squeeze(0)  # [H,W]
            scores = 1.0 - p_true

            for c in range(spec.num_classes):
                if c in excluded:
                    continue
                mask_c = valid & (tgt == c)
                if spec.filter_background:
                    mask_c &= pred != 0

                if mask_c.any():
                    scores_by_class[c].append(scores[mask_c].detach().cpu())

    thresholds: list[float | None] = []
    counts: list[int] = []

    for c in range(spec.num_classes):
        if c in excluded:
            thresholds.append(None)
            counts.append(0)
            continue

        if len(scores_by_class[c]) == 0:
            click.echo(
                f"[WARN] head='{spec.name}' class={c}: no calibration samples found; "
                f"using threshold=1.0"
            )
            thresholds.append(1.0)
            counts.append(0)
            continue

        class_scores = torch.cat(scores_by_class[c], dim=0)
        thresholds.append(_get_quantile(class_scores, alpha))
        counts.append(int(class_scores.numel()))

    return {
        "method": "class_conditional_pixelwise",
        "alpha": float(alpha),
        "dataloader_idx": int(spec.dataloader_idx),
        "logits_key": spec.logits_key,
        "num_classes": int(spec.num_classes),
        "ignore_index": int(spec.ignore_index),
        "exclude_classes": sorted(excluded),
        "thresholds": thresholds,
        "counts_per_class": counts,
    }


def _build_head_specs(calibration_cfg: dict[str, Any]) -> list[HeadCalibrationSpec]:
    calib = calibration_cfg.get("calibration", calibration_cfg)
    heads_cfg = calib.get("heads", None)
    if not isinstance(heads_cfg, dict):
        raise KeyError("Calibration config must contain calibration.heads")

    specs: list[HeadCalibrationSpec] = []
    for head_name, head_cfg in heads_cfg.items():
        if not isinstance(head_cfg, dict):
            raise TypeError(f"head '{head_name}' config must be a dict")

        exclude_classes = head_cfg.get("exclude_classes", [])
        if exclude_classes is None:
            exclude_classes = []
        if not isinstance(exclude_classes, list):
            raise TypeError(
                f"head '{head_name}' exclude_classes must be a list of ints"
            )

        specs.append(
            HeadCalibrationSpec(
                name=str(head_name),
                enabled=bool(head_cfg.get("enabled", False)),
                dataloader_idx=int(head_cfg["dataloader_idx"]),
                logits_key=str(head_cfg["logits_key"]),
                num_classes=int(head_cfg["num_classes"]),
                ignore_index=int(head_cfg.get("ignore_index", 255)),
                exclude_classes=[int(x) for x in exclude_classes],
                filter_background=bool(head_cfg.get("filter_background", False)),
            )
        )
    return specs


def _setup_dm_for_split(dm, split: str) -> Any:
    split = split.lower()
    if split == "val":
        dm.setup("fit")
        return dm.val_dataloader()
    if split == "test":
        dm.setup("test")
        return dm.test_dataloader()
    raise ValueError(f"Unsupported calibration split: {split!r}. Use 'val' or 'test'.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@click.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the experiment config YAML.",
)
@click.option(
    "--ckpt-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the Lightning checkpoint.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where export + calibration artifacts will be written.",
)
@click.option(
    "--calibration-config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the calibration config YAML.",
)
def main(
    config_path: Path,
    ckpt_path: Path,
    output_dir: Path,
    calibration_config_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_cfg = _load_yaml(calibration_config_path)
    export_cfg = calibration_cfg.get("export", {})
    calib_cfg = calibration_cfg.get("calibration", {})

    yaml_name = str(export_cfg.get("yaml_name", "model.yaml"))
    weights_name = str(export_cfg.get("weights_name", "weights.pt"))

    split = str(calib_cfg.get("split", "val"))
    device = str(
        calib_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    alpha = float(calib_cfg.get("alpha", 0.1))
    save_as_separate_file = bool(calib_cfg.get("save_as_separate_file", True))
    conformal_filename = str(calib_cfg.get("filename", "conformal.yaml"))

    head_specs = _build_head_specs(calibration_cfg)
    enabled_specs = [s for s in head_specs if s.enabled]

    if len(enabled_specs) == 0:
        raise ValueError("No enabled heads found in calibration config.")

    click.echo("[INFO] loading experiment...")
    bundle = load_experiment(
        config_path=str(config_path.resolve()),
        ckpt_path=str(ckpt_path.resolve()),
        stage="fit",
        device=device,
        eval_mode=True,
    )

    cfg = bundle["config"]
    pl_model = bundle["model"].to(device).eval()
    dm = _get_datamodule_from_bundle(bundle)

    click.echo(f"[INFO] preparing dataloaders for split='{split}'...")
    loaders = _setup_dm_for_split(dm, split)

    conformal_artifact: dict[str, Any] = {
        "split": split,
        "device": device,
        "heads": {},
    }

    for spec in enabled_specs:
        click.echo(
            f"[INFO] calibrating head='{spec.name}' "
            f"(dataloader_idx={spec.dataloader_idx}, logits_key={spec.logits_key}, "
            f"exclude_classes={sorted(spec.excluded_set())})..."
        )
        dl = _get_selected_dataloader(loaders, spec.dataloader_idx)

        result = fit_class_conditional_conformal_for_head(
            pl_model=pl_model,
            dataloader=dl,
            spec=spec,
            alpha=alpha,
        )
        conformal_artifact["heads"][spec.name] = result
        click.echo(
            f"[OK] fitted conformal for head='{spec.name}' "
            f"with thresholds={result['thresholds']}"
        )

    click.echo("[INFO] exporting model bundle...")
    weights_path, model_yaml_path = export_model_bundle(
        cfg=cfg,
        pl_model=pl_model,
        output_dir=output_dir,
        yaml_name=yaml_name,
        weights_name=weights_name,
    )
    click.echo(f"[OK] saved weights -> {weights_path}")
    click.echo(f"[OK] saved model yaml -> {model_yaml_path}")

    if save_as_separate_file:
        conformal_path = output_dir / conformal_filename
        _save_yaml(conformal_path, conformal_artifact)
        click.echo(f"[OK] saved conformal yaml -> {conformal_path}")
    else:
        model_yaml = _load_yaml(model_yaml_path)
        model_yaml["conformal"] = conformal_artifact
        _save_yaml(model_yaml_path, model_yaml)
        click.echo(f"[OK] wrote conformal section into -> {model_yaml_path}")


if __name__ == "__main__":
    main()
