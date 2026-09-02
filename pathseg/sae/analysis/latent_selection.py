from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class LatentSpec:
    """One latent plus optional semantic metadata used by analysis commands."""

    latent_id: int
    task: str | None = None
    class_index: int | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.latent_id < 0:
            raise ValueError("latent_id must be non-negative.")
        if self.task is not None and not self.task.strip():
            raise ValueError("task cannot be empty.")
        if self.class_index is not None and self.class_index < 0:
            raise ValueError("class_index must be non-negative.")
        if self.label is not None and not self.label.strip():
            raise ValueError("label cannot be empty.")

    @property
    def key(self) -> tuple[int, str | None, int | None]:
        return self.latent_id, self.task, self.class_index

    def as_dict(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "task": self.task,
            "class_index": self.class_index,
            "label": self.label,
        }


def parse_latent_spec(value: str) -> LatentSpec:
    """Parse ``ID``, ``ID:TASK``, or ``ID:TASK:CLASS_INDEX``."""

    parts = value.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(
            "Latent specifications must be ID, ID:TASK, or "
            "ID:TASK:CLASS_INDEX."
        )
    if not parts[0]:
        raise ValueError("Latent specification is missing its ID.")

    latent_id = int(parts[0])
    task = parts[1] if len(parts) >= 2 and parts[1] else None
    class_index = (
        int(parts[2]) if len(parts) == 3 and parts[2] else None
    )
    return LatentSpec(
        latent_id=latent_id,
        task=task,
        class_index=class_index,
    )


def _latent_spec_from_value(value: Any, *, index: int) -> LatentSpec:
    if isinstance(value, int):
        return LatentSpec(latent_id=value)
    if isinstance(value, str):
        return parse_latent_spec(value)
    if not isinstance(value, Mapping):
        raise TypeError(
            f"latents[{index}] must be an integer, string, or mapping."
        )

    values = dict(value)
    if "latent_id" not in values and "id" in values:
        values["latent_id"] = values.pop("id")
    if "class" in values and "class_index" not in values:
        values["class_index"] = values.pop("class")
    try:
        return LatentSpec(
            latent_id=int(values["latent_id"]),
            task=(str(values["task"]) if values.get("task") is not None else None),
            class_index=(
                int(values["class_index"])
                if values.get("class_index") is not None
                else None
            ),
            label=(
                str(values["label"])
                if values.get("label") is not None
                else None
            ),
        )
    except KeyError as error:
        raise ValueError(
            f"latents[{index}] is missing latent_id."
        ) from error


def load_latent_manifest(path: str | Path) -> list[LatentSpec]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Latent manifest not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)

    if isinstance(value, Mapping):
        value = value.get("latents")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Latent manifest must contain a 'latents' sequence.")
    return [
        _latent_spec_from_value(item, index=index)
        for index, item in enumerate(value)
    ]


def resolve_latent_specs(
    *,
    manifest_path: str | Path | None = None,
    values: Sequence[str] = (),
) -> list[LatentSpec]:
    specs: list[LatentSpec] = []
    if manifest_path is not None:
        specs.extend(load_latent_manifest(manifest_path))
    specs.extend(parse_latent_spec(value) for value in values)
    if not specs:
        raise ValueError("Provide --manifest and/or at least one --latent.")

    result: list[LatentSpec] = []
    seen: set[tuple[int, str | None, int | None]] = set()
    for spec in specs:
        if spec.key in seen:
            continue
        seen.add(spec.key)
        result.append(spec)
    return result


def load_top_activation_records(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Top-activation file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"{path}:{line_number} must contain a JSON object."
                )
            records.append(dict(value))
    return records


def select_top_activation_records(
    records: Sequence[Mapping[str, Any]],
    spec: LatentSpec,
    *,
    max_examples: int,
    include_class_mismatches: bool = False,
) -> list[dict[str, Any]]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive.")

    selected: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        if int(record["latent_id"]) != spec.latent_id:
            continue
        if spec.task is not None and str(record["task_name"]) != spec.task:
            continue
        if (
            spec.class_index is not None
            and not include_class_mismatches
            and record.get("target_class") != spec.class_index
        ):
            continue
        selected.append(record)

    selected.sort(
        key=lambda record: (
            int(record.get("rank", 10**9)),
            -float(record["activation"]),
        )
    )
    return selected[:max_examples]


def unique_latent_ids(specs: Sequence[LatentSpec]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(spec.latent_id for spec in specs))
