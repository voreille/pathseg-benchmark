from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pathseg.sae.analysis.types import AnalysisResult, TaskSpec


@dataclass(frozen=True, slots=True)
class TaskAttribution:
    task_name: str
    alignment: torch.Tensor
    contrast: torch.Tensor
    mean_activation: torch.Tensor
    firing_rate: torch.Tensor
    mean_contribution: torch.Tensor

    def state_dict(self) -> dict[str, torch.Tensor | str]:
        return {
            "task_name": self.task_name,
            "alignment": self.alignment,
            "contrast": self.contrast,
            "mean_activation": self.mean_activation,
            "firing_rate": self.firing_rate,
            "mean_contribution": self.mean_contribution,
        }


def _linear_head_weight(head: nn.Module) -> torch.Tensor:
    weight = getattr(head, "weight", None)
    if not torch.is_tensor(weight):
        raise TypeError(
            f"Expected a linear layer or 1x1 convolution, got {type(head).__name__}."
        )
    if weight.ndim == 2:
        return weight
    if weight.ndim == 4 and weight.shape[-2:] == (1, 1):
        return weight.flatten(1)
    raise ValueError(
        "Task-head attribution requires nn.Linear or Conv2d(kernel_size=1); "
        f"got weight shape {tuple(weight.shape)}."
    )


def _matching_linear_modules(
    module: nn.Module,
    *,
    input_dim: int,
    output_dim: int,
) -> list[tuple[str, nn.Module]]:
    matches: list[tuple[str, nn.Module]] = []
    for name, child in module.named_modules():
        if not name:
            continue
        try:
            weight = _linear_head_weight(child)
        except (TypeError, ValueError):
            continue
        if weight.shape == (output_dim, input_dim):
            matches.append((name, child))
    return matches


def resolve_task_heads(
    decoder: nn.Module,
    *,
    task_specs: Mapping[str, TaskSpec],
    input_dim: int,
) -> dict[str, nn.Module]:
    """Resolve task-specific final linear layers from common decoder layouts.

    Explicit task containers are preferred.  A named-module search is used as
    a fallback, but ambiguous matches fail loudly instead of silently assigning
    the wrong head.
    """

    containers: list[object] = [decoder]
    for attribute in ("heads", "task_heads", "decoders", "heads_by_task"):
        value = getattr(decoder, attribute, None)
        if value is not None:
            containers.insert(0, value)

    resolved: dict[str, nn.Module] = {}
    for task_name, spec in task_specs.items():
        task_module: nn.Module | None = None
        for container in containers:
            try:
                candidate = container[task_name]  # type: ignore[index]
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(candidate, nn.Module):
                task_module = candidate
                break

        if task_module is not None:
            try:
                weight = _linear_head_weight(task_module)
            except (TypeError, ValueError):
                matches = _matching_linear_modules(
                    task_module,
                    input_dim=input_dim,
                    output_dim=spec.num_classes,
                )
                if len(matches) != 1:
                    names = [name for name, _ in matches]
                    raise ValueError(
                        f"Task {task_name!r}: expected one final linear layer "
                        f"inside {type(task_module).__name__}, found {names}."
                    )
                resolved[task_name] = matches[0][1]
            else:
                if weight.shape != (spec.num_classes, input_dim):
                    raise ValueError(
                        f"Task {task_name!r} head has shape {tuple(weight.shape)}, "
                        f"expected ({spec.num_classes}, {input_dim})."
                    )
                resolved[task_name] = task_module
            continue

        matches = _matching_linear_modules(
            decoder,
            input_dim=input_dim,
            output_dim=spec.num_classes,
        )
        named_matches = [
            (name, module)
            for name, module in matches
            if task_name.lower() in name.lower().split(".")
            or task_name.lower() in name.lower()
        ]
        candidates = named_matches or matches
        if len(candidates) != 1:
            names = [name for name, _ in candidates]
            raise ValueError(
                f"Could not uniquely resolve the final linear head for task "
                f"{task_name!r}; candidates={names}. Pass heads explicitly."
            )
        resolved[task_name] = candidates[0][1]

    return resolved


@torch.no_grad()
def head_latent_alignment(sae: nn.Module, head: nn.Module) -> torch.Tensor:
    """Return the exact per-unit latent effect on pre-interpolation logits."""

    decoder = getattr(sae, "decoder", None)
    decoder_weight = getattr(decoder, "weight", None)
    if not torch.is_tensor(decoder_weight) or decoder_weight.ndim != 2:
        raise TypeError("sae.decoder must be an nn.Linear-like module.")

    head_weight = _linear_head_weight(head)
    if head_weight.shape[1] != decoder_weight.shape[0]:
        raise ValueError(
            "Head input dimension and SAE decoder output dimension do not match: "
            f"{head_weight.shape[1]} versus {decoder_weight.shape[0]}."
        )
    return (head_weight @ decoder_weight).detach().float().cpu()


@torch.no_grad()
def decoder_column_norms(sae: nn.Module) -> torch.Tensor:
    decoder = getattr(sae, "decoder", None)
    weight = getattr(decoder, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise TypeError("sae.decoder must be an nn.Linear-like module.")
    return weight.norm(dim=0).detach().float().cpu()


def compute_task_attributions(
    *,
    result: AnalysisResult,
    task_specs: Mapping[str, TaskSpec],
    sae: nn.Module,
    heads: Mapping[str, nn.Module],
) -> dict[str, TaskAttribution]:
    attributions: dict[str, TaskAttribution] = {}

    for task_name, head in heads.items():
        if task_name not in result.tasks:
            raise KeyError(f"No collected statistics for task {task_name!r}.")
        if task_name not in task_specs:
            raise KeyError(f"No TaskSpec for task {task_name!r}.")

        alignment = head_latent_alignment(sae, head).double()
        expected_classes = task_specs[task_name].num_classes
        if alignment.shape != (expected_classes, result.num_latents):
            raise ValueError(
                f"Task {task_name!r}: expected alignment shape "
                f"({expected_classes}, {result.num_latents}), got "
                f"{tuple(alignment.shape)}."
            )

        if expected_classes == 1:
            contrast = alignment.clone()
        else:
            other_mean = (
                alignment.sum(dim=0, keepdim=True) - alignment
            ) / (expected_classes - 1)
            contrast = alignment - other_mean

        statistics = result.tasks[task_name]
        class_mass = statistics.class_mass.clamp_min(1e-12)
        mean_activation = (
            statistics.class_activation_sum / class_mass[None, :]
        ).T
        firing_rate = (
            statistics.class_firing_sum / class_mass[None, :]
        ).T
        mean_contribution = mean_activation * contrast

        attributions[task_name] = TaskAttribution(
            task_name=task_name,
            alignment=alignment,
            contrast=contrast,
            mean_activation=mean_activation,
            firing_rate=firing_rate,
            mean_contribution=mean_contribution,
        )

    return attributions


def select_relevant_latents(
    *,
    result: AnalysisResult,
    attributions: Mapping[str, TaskAttribution],
    per_class: int = 10,
    global_importance: int = 32,
    max_latents: int | None = 128,
) -> list[int]:
    """Select positive/negative task contributors plus globally important latents."""

    if per_class < 0 or global_importance < 0:
        raise ValueError("Selection counts cannot be negative.")

    selected: list[int] = []
    seen: set[int] = set()

    def add(indices: Sequence[int]) -> None:
        for latent_id in indices:
            latent_id = int(latent_id)
            if latent_id in seen:
                continue
            if max_latents is not None and len(selected) >= max_latents:
                return
            seen.add(latent_id)
            selected.append(latent_id)

    for attribution in attributions.values():
        for scores in attribution.mean_contribution:
            positive = torch.nonzero(scores > 0, as_tuple=False).flatten()
            if positive.numel() and per_class:
                count = min(per_class, positive.numel())
                order = torch.topk(scores[positive], k=count).indices
                add(positive[order].tolist())

            negative = torch.nonzero(scores < 0, as_tuple=False).flatten()
            if negative.numel() and per_class:
                count = min(per_class, negative.numel())
                order = torch.topk(-scores[negative], k=count).indices
                add(negative[order].tolist())

    if global_importance:
        count = min(global_importance, result.num_latents)
        add(torch.topk(result.importance, k=count).indices.tolist())

    return selected
