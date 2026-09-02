from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from pathseg.sae.analysis.targets import to_dense_semantic_targets


@dataclass(frozen=True, slots=True)
class AnalysisBatch:
    images: torch.Tensor
    targets: torch.Tensor
    task_names: tuple[str, ...]
    image_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]


def _stack_tensors(value: Any, *, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and all(
        torch.is_tensor(item) for item in value
    ):
        try:
            return torch.stack(tuple(value))
        except RuntimeError as error:
            raise ValueError(
                f"Cannot stack {name}; all validation samples must have the "
                "same spatial shape within a batch."
            ) from error
    raise TypeError(f"Expected {name} to be a tensor or sequence of tensors.")


def unpack_multitask_batch(
    batch: Any,
    *,
    ignore_idx: int,
    target_converter: Callable[[Any], Any] | None = None,
) -> AnalysisBatch:
    """Normalize both training and ``eval_collate`` multitask batches."""

    if not isinstance(batch, (tuple, list)) or len(batch) < 4:
        raise TypeError(
            "Expected a multitask batch containing "
            "(images, targets, task_names, image_ids)."
    )

    images = _stack_tensors(batch[0], name="images")
    batch_size = images.shape[0]
    raw_targets = batch[1]
    if target_converter is not None:
        raw_targets = target_converter(raw_targets)
    targets = to_dense_semantic_targets(
        raw_targets,
        ignore_idx=ignore_idx,
        batch_size=batch_size,
    )

    raw_tasks = batch[2]
    if isinstance(raw_tasks, str):
        task_names = (raw_tasks,) * batch_size
    else:
        task_names = tuple(str(task) for task in raw_tasks)

    raw_ids = batch[3]
    if isinstance(raw_ids, (str, int)):
        image_ids = (str(raw_ids),) * batch_size
    else:
        image_ids = tuple(str(image_id) for image_id in raw_ids)

    if len(task_names) != batch_size:
        raise ValueError(
            f"Received {len(task_names)} task names for a batch of {batch_size}."
        )
    if len(image_ids) != batch_size:
        raise ValueError(
            f"Received {len(image_ids)} image IDs for a batch of {batch_size}."
        )
    return AnalysisBatch(
        images=images,
        targets=targets,
        task_names=task_names,
        image_ids=image_ids,
        sample_ids=image_ids,
    )
