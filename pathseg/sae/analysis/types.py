from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Label-space information required by the analysis pass."""

    name: str
    num_classes: int
    class_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Task name cannot be empty.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"Task {self.name!r}: expected {self.num_classes} class names, "
                f"got {len(self.class_names)}."
            )

    def class_name(self, class_idx: int) -> str:
        if self.class_names:
            return self.class_names[class_idx]
        return str(class_idx)


@dataclass(slots=True)
class TaskStatistics:
    """Streaming statistics for one task's label space."""

    class_mass: torch.Tensor
    class_activation_sum: torch.Tensor
    class_firing_sum: torch.Tensor

    @classmethod
    def zeros(cls, num_latents: int, num_classes: int) -> TaskStatistics:
        return cls(
            class_mass=torch.zeros(num_classes, dtype=torch.float64),
            class_activation_sum=torch.zeros(
                num_latents,
                num_classes,
                dtype=torch.float64,
            ),
            class_firing_sum=torch.zeros(
                num_latents,
                num_classes,
                dtype=torch.float64,
            ),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "class_mass": self.class_mass,
            "class_activation_sum": self.class_activation_sum,
            "class_firing_sum": self.class_firing_sum,
        }


@dataclass(slots=True)
class DatasetStatistics:
    """Unlabelled activation statistics for one source dataset."""

    token_mass: float
    activation_sum: torch.Tensor
    firing_sum: torch.Tensor

    @classmethod
    def zeros(cls, num_latents: int) -> DatasetStatistics:
        return cls(
            token_mass=0.0,
            activation_sum=torch.zeros(num_latents, dtype=torch.float64),
            firing_sum=torch.zeros(num_latents, dtype=torch.float64),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "token_mass": self.token_mass,
            "activation_sum": self.activation_sum,
            "firing_sum": self.firing_sum,
        }


@dataclass(frozen=True, slots=True)
class TopActivation:
    latent_id: int
    activation: float
    dataset_name: str
    task_name: str
    image_id: str
    sample_id: str
    token_y: int
    token_x: int
    grid_height: int
    grid_width: int
    target_class: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "activation": self.activation,
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
            "image_id": self.image_id,
            "sample_id": self.sample_id,
            "token_y": self.token_y,
            "token_x": self.token_x,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "target_class": self.target_class,
        }


@dataclass(slots=True)
class AnalysisResult:
    num_latents: int
    token_mass: float
    activation_sum: torch.Tensor
    activation_sq_sum: torch.Tensor
    firing_sum: torch.Tensor
    tasks: dict[str, TaskStatistics]
    datasets: dict[str, DatasetStatistics]
    decoder_norms: torch.Tensor | None = None
    top_activations: dict[int, list[TopActivation]] = field(default_factory=dict)

    @property
    def density(self) -> torch.Tensor:
        return self.firing_sum / max(self.token_mass, 1e-12)

    @property
    def mean_positive_activation(self) -> torch.Tensor:
        return self.activation_sum / self.firing_sum.clamp_min(1e-12)

    @property
    def importance(self) -> torch.Tensor:
        """Expected squared activation.

        This is directly comparable between latents when decoder columns have
        unit norm.  Otherwise multiply by ``decoder_norms.square()``.
        """

        importance = self.activation_sq_sum / max(self.token_mass, 1e-12)
        if self.decoder_norms is not None:
            importance = importance * self.decoder_norms.square()
        return importance

    @property
    def actual_l0(self) -> float:
        return float(self.firing_sum.sum().item() / max(self.token_mass, 1e-12))

    @property
    def dead_latents(self) -> torch.Tensor:
        return torch.nonzero(self.firing_sum == 0, as_tuple=False).flatten()

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_latents": self.num_latents,
            "token_mass": self.token_mass,
            "activation_sum": self.activation_sum,
            "activation_sq_sum": self.activation_sq_sum,
            "firing_sum": self.firing_sum,
            "tasks": {
                name: statistics.state_dict()
                for name, statistics in self.tasks.items()
            },
            "datasets": {
                name: statistics.state_dict()
                for name, statistics in self.datasets.items()
            },
            "decoder_norms": self.decoder_norms,
            "top_activations": {
                latent_id: [example.as_dict() for example in examples]
                for latent_id, examples in self.top_activations.items()
            },
        }
