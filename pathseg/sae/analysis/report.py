from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pathseg.sae.analysis.attribution import TaskAttribution
from pathseg.sae.analysis.types import AnalysisResult, TaskSpec


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "task"


def _write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def save_analysis_report(
    output_dir: str | Path,
    *,
    result: AnalysisResult,
    task_specs: Mapping[str, TaskSpec],
    attributions: Mapping[str, TaskAttribution],
) -> Path:
    """Save reusable tensors plus human-readable CSV/JSON summaries."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = result.state_dict()
    state["attributions"] = {
        name: attribution.state_dict()
        for name, attribution in attributions.items()
    }
    torch.save(state, output_dir / "analysis.pt")

    summary = {
        "num_latents": result.num_latents,
        "token_mass": result.token_mass,
        "actual_l0": result.actual_l0,
        "dead_latents": int(result.dead_latents.numel()),
        "datasets": {
            name: {"token_mass": statistics.token_mass}
            for name, statistics in result.datasets.items()
        },
        "tasks": {
            name: {
                "class_mass": statistics.class_mass.tolist(),
                "class_names": list(task_specs[name].class_names),
            }
            for name, statistics in result.tasks.items()
        },
        "selected_latents": sorted(result.top_activations),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    latent_rows: list[dict[str, Any]] = []
    density = result.density
    mean_positive = result.mean_positive_activation
    importance = result.importance

    for latent_id in range(result.num_latents):
        row: dict[str, Any] = {
            "latent_id": latent_id,
            "density": _float(density[latent_id]),
            "mean_positive_activation": _float(mean_positive[latent_id]),
            "importance": _float(importance[latent_id]),
            "decoder_norm": (
                _float(result.decoder_norms[latent_id])
                if result.decoder_norms is not None
                else ""
            ),
        }

        for dataset_name, statistics in result.datasets.items():
            prefix = f"dataset_{_slug(dataset_name)}"
            row[f"{prefix}_mean_activation"] = _float(
                statistics.activation_sum[latent_id]
                / max(statistics.token_mass, 1e-12)
            )
            row[f"{prefix}_density"] = _float(
                statistics.firing_sum[latent_id]
                / max(statistics.token_mass, 1e-12)
            )

        for task_name, attribution in attributions.items():
            prefix = _slug(task_name)
            task_spec = task_specs[task_name]
            activation_values = attribution.mean_activation[:, latent_id]
            contribution_values = attribution.mean_contribution[:, latent_id]
            top_activation_class = int(activation_values.argmax().item())
            top_abs_contribution = int(contribution_values.abs().argmax().item())
            row[f"{prefix}_top_activation_class"] = task_spec.class_name(
                top_activation_class
            )
            row[f"{prefix}_top_activation"] = _float(
                activation_values[top_activation_class]
            )
            row[f"{prefix}_top_contribution_class"] = task_spec.class_name(
                top_abs_contribution
            )
            row[f"{prefix}_top_contribution"] = _float(
                contribution_values[top_abs_contribution]
            )

        latent_rows.append(row)

    latent_fields = list(latent_rows[0]) if latent_rows else []
    _write_csv(
        output_dir / "latents.csv",
        fieldnames=latent_fields,
        rows=latent_rows,
    )

    rankings: dict[str, Any] = {}
    for task_name, attribution in attributions.items():
        spec = task_specs[task_name]
        rows: list[dict[str, Any]] = []
        task_rankings: dict[str, Any] = {}

        for class_idx in range(spec.num_classes):
            class_name = spec.class_name(class_idx)
            scores = attribution.mean_contribution[class_idx]
            count = min(20, result.num_latents)
            positive = torch.topk(scores, k=count).indices.tolist()
            negative = torch.topk(-scores, k=count).indices.tolist()
            task_rankings[class_name] = {
                "positive": positive,
                "negative": negative,
            }

            class_mass = result.tasks[task_name].class_mass[class_idx]
            for latent_id in range(result.num_latents):
                rows.append(
                    {
                        "class_index": class_idx,
                        "class_name": class_name,
                        "latent_id": latent_id,
                        "class_mass": _float(class_mass),
                        "mean_activation": _float(
                            attribution.mean_activation[class_idx, latent_id]
                        ),
                        "firing_rate": _float(
                            attribution.firing_rate[class_idx, latent_id]
                        ),
                        "head_alignment": _float(
                            attribution.alignment[class_idx, latent_id]
                        ),
                        "head_contrast": _float(
                            attribution.contrast[class_idx, latent_id]
                        ),
                        "mean_contribution": _float(
                            attribution.mean_contribution[class_idx, latent_id]
                        ),
                    }
                )

        rankings[task_name] = task_rankings
        _write_csv(
            output_dir / f"task_{_slug(task_name)}_latents.csv",
            fieldnames=list(rows[0]) if rows else [],
            rows=rows,
        )

    with (output_dir / "rankings.json").open("w", encoding="utf-8") as stream:
        json.dump(rankings, stream, indent=2)
        stream.write("\n")

    with (output_dir / "top_activations.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for latent_id in sorted(result.top_activations):
            for rank, example in enumerate(result.top_activations[latent_id], start=1):
                record = example.as_dict()
                record["rank"] = rank
                stream.write(json.dumps(record))
                stream.write("\n")

    return output_dir
