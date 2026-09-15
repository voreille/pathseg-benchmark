from __future__ import annotations

import heapq
import itertools
from collections.abc import Mapping, Sequence

import torch

from pathseg.sae.analysis.targets import class_fractions
from pathseg.sae.analysis.types import (
    AnalysisResult,
    DatasetStatistics,
    TaskSpec,
    TaskStatistics,
    TopActivation,
)


def _latent_maps(
    latents: torch.Tensor,
    spatial_size: tuple[int, int],
) -> torch.Tensor:
    if latents.ndim != 3:
        raise ValueError(
            f"latents must have shape [B, N, L], got {tuple(latents.shape)}."
        )
    height, width = spatial_size
    if latents.shape[1] != height * width:
        raise ValueError(
            f"Received {latents.shape[1]} tokens but spatial size "
            f"{spatial_size} contains {height * width}."
        )
    return latents.transpose(1, 2).reshape(
        latents.shape[0],
        latents.shape[-1],
        height,
        width,
    )


class StreamingSAECollector:
    """Accumulate SAE statistics without retaining per-token activations."""

    def __init__(
        self,
        *,
        num_latents: int,
        tasks: Mapping[str, TaskSpec],
        ignore_idx: int,
    ) -> None:
        if num_latents <= 0:
            raise ValueError("num_latents must be positive.")

        self.num_latents = int(num_latents)
        self.tasks = dict(tasks)
        self.ignore_idx = int(ignore_idx)

        self.token_mass = 0.0
        self.activation_sum = torch.zeros(num_latents, dtype=torch.float64)
        self.activation_sq_sum = torch.zeros(num_latents, dtype=torch.float64)
        self.firing_sum = torch.zeros(num_latents, dtype=torch.float64)
        self.task_statistics = {
            name: TaskStatistics.zeros(num_latents, spec.num_classes)
            for name, spec in self.tasks.items()
        }
        self.dataset_statistics: dict[str, DatasetStatistics] = {}

    def update(
        self,
        *,
        latents: torch.Tensor,
        targets: torch.Tensor,
        task_names: Sequence[str],
        dataset_name: str,
        spatial_size: tuple[int, int],
    ) -> None:
        if latents.shape[0] != len(task_names):
            raise ValueError("Number of task names does not match latent batch size.")
        if targets.shape[0] != latents.shape[0]:
            raise ValueError("Target and latent batch sizes do not match.")
        if latents.shape[-1] != self.num_latents:
            raise ValueError(
                f"Expected {self.num_latents} latents, got {latents.shape[-1]}."
            )

        maps = _latent_maps(latents, spatial_size).float()
        dataset_stats = self.dataset_statistics.setdefault(
            str(dataset_name),
            DatasetStatistics.zeros(self.num_latents),
        )

        for task_name in dict.fromkeys(task_names):
            if task_name not in self.tasks:
                raise KeyError(
                    f"Unknown task {task_name!r}; configured tasks are "
                    f"{sorted(self.tasks)}."
                )

            indices = [
                index
                for index, sample_task in enumerate(task_names)
                if sample_task == task_name
            ]
            index_tensor = torch.as_tensor(indices, device=maps.device)
            task_maps = maps.index_select(0, index_tensor)
            task_targets = targets.index_select(0, index_tensor)
            spec = self.tasks[task_name]

            fractions, valid_fraction = class_fractions(
                task_targets,
                num_classes=spec.num_classes,
                output_size=spatial_size,
                ignore_idx=self.ignore_idx,
            )
            valid_weights = valid_fraction[:, 0]
            class_weights = fractions * valid_fraction
            firing = (task_maps > 0).float()

            activation_sum = torch.einsum(
                "blhw,bhw->l", task_maps, valid_weights
            )
            activation_sq_sum = torch.einsum(
                "blhw,bhw->l", task_maps.square(), valid_weights
            )
            firing_sum = torch.einsum("blhw,bhw->l", firing, valid_weights)
            token_mass = float(valid_weights.sum().item())

            self.activation_sum += activation_sum.detach().double().cpu()
            self.activation_sq_sum += activation_sq_sum.detach().double().cpu()
            self.firing_sum += firing_sum.detach().double().cpu()
            self.token_mass += token_mass

            dataset_stats.activation_sum += activation_sum.detach().double().cpu()
            dataset_stats.firing_sum += firing_sum.detach().double().cpu()
            dataset_stats.token_mass += token_mass

            task_stats = self.task_statistics[task_name]
            task_stats.class_mass += (
                class_weights.sum(dim=(0, 2, 3)).detach().double().cpu()
            )
            task_stats.class_activation_sum += torch.einsum(
                "blhw,bchw->lc",
                task_maps,
                class_weights,
            ).detach().double().cpu()
            task_stats.class_firing_sum += torch.einsum(
                "blhw,bchw->lc",
                firing,
                class_weights,
            ).detach().double().cpu()

    def finalize(self) -> AnalysisResult:
        if self.token_mass <= 0:
            raise RuntimeError("No valid target tokens were collected.")
        return AnalysisResult(
            num_latents=self.num_latents,
            token_mass=self.token_mass,
            activation_sum=self.activation_sum,
            activation_sq_sum=self.activation_sq_sum,
            firing_sum=self.firing_sum,
            tasks=self.task_statistics,
            datasets=self.dataset_statistics,
        )


class TopActivationCollector:
    """Keep diverse top activation locations for selected latents only."""

    def __init__(
        self,
        *,
        latent_ids: Sequence[int],
        tasks: Mapping[str, TaskSpec],
        ignore_idx: int,
        examples_per_latent: int = 16,
        max_per_image: int = 2,
        candidate_multiplier: int = 8,
    ) -> None:
        latent_ids = tuple(dict.fromkeys(int(value) for value in latent_ids))
        if not latent_ids:
            raise ValueError("latent_ids cannot be empty.")
        if examples_per_latent <= 0:
            raise ValueError("examples_per_latent must be positive.")
        if max_per_image <= 0:
            raise ValueError("max_per_image must be positive.")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive.")

        self.latent_ids = latent_ids
        self.tasks = dict(tasks)
        self.ignore_idx = int(ignore_idx)
        self.examples_per_latent = int(examples_per_latent)
        self.max_per_image = int(max_per_image)
        self.heap_capacity = int(examples_per_latent * candidate_multiplier)
        self._counter = itertools.count()
        self._heaps: dict[int, list[tuple[float, int, TopActivation]]] = {
            latent_id: [] for latent_id in latent_ids
        }

    def update(
        self,
        *,
        latents: torch.Tensor,
        targets: torch.Tensor,
        task_names: Sequence[str],
        image_ids: Sequence[str],
        sample_ids: Sequence[str] | None = None,
        dataset_name: str,
        spatial_size: tuple[int, int],
    ) -> None:
        maps = _latent_maps(latents, spatial_size).float()
        batch_size, num_latents, height, width = maps.shape
        if sample_ids is None:
            sample_ids = image_ids
        if (
            len(task_names) != batch_size
            or len(image_ids) != batch_size
            or len(sample_ids) != batch_size
        ):
            raise ValueError("Batch metadata does not match latent batch size.")
        if any(
            latent_id < 0 or latent_id >= num_latents
            for latent_id in self.latent_ids
        ):
            raise IndexError("At least one selected latent ID is out of bounds.")

        valid_grid = torch.zeros(
            batch_size,
            height,
            width,
            dtype=torch.bool,
            device=maps.device,
        )
        target_grid = torch.full(
            (batch_size, height, width),
            self.ignore_idx,
            dtype=torch.long,
            device=maps.device,
        )

        for task_name in dict.fromkeys(task_names):
            if task_name not in self.tasks:
                raise KeyError(f"Unknown task {task_name!r}.")
            indices = [
                index
                for index, sample_task in enumerate(task_names)
                if sample_task == task_name
            ]
            index_tensor = torch.as_tensor(indices, device=maps.device)
            fractions, valid_fraction = class_fractions(
                targets.index_select(0, index_tensor),
                num_classes=self.tasks[task_name].num_classes,
                output_size=spatial_size,
                ignore_idx=self.ignore_idx,
            )
            local_valid = valid_fraction[:, 0] > 0
            local_targets = fractions.argmax(dim=1)
            valid_grid.index_copy_(0, index_tensor, local_valid)
            target_grid.index_copy_(0, index_tensor, local_targets)

        selected = torch.as_tensor(self.latent_ids, device=maps.device)
        selected_maps = maps.index_select(1, selected)
        flat = selected_maps.permute(0, 2, 3, 1).reshape(-1, len(self.latent_ids))
        flat = flat.masked_fill(~valid_grid.reshape(-1, 1), float("-inf"))

        candidate_count = min(self.heap_capacity, flat.shape[0])
        values, positions = torch.topk(flat, k=candidate_count, dim=0)
        values = values.detach().cpu()
        positions = positions.detach().cpu()
        target_grid_cpu = target_grid.detach().cpu()
        tokens_per_image = height * width

        for selected_index, latent_id in enumerate(self.latent_ids):
            heap = self._heaps[latent_id]
            for candidate_index in range(candidate_count):
                activation = float(values[candidate_index, selected_index].item())
                if not torch.isfinite(values[candidate_index, selected_index]):
                    continue
                if activation <= 0:
                    continue

                flat_position = int(
                    positions[candidate_index, selected_index].item()
                )
                batch_index, token_index = divmod(
                    flat_position,
                    tokens_per_image,
                )
                token_y, token_x = divmod(token_index, width)
                target_class = int(
                    target_grid_cpu[batch_index, token_y, token_x].item()
                )
                if target_class == self.ignore_idx:
                    target_class = None

                example = TopActivation(
                    latent_id=latent_id,
                    activation=activation,
                    dataset_name=str(dataset_name),
                    task_name=str(task_names[batch_index]),
                    image_id=str(image_ids[batch_index]),
                    sample_id=str(sample_ids[batch_index]),
                    token_y=token_y,
                    token_x=token_x,
                    grid_height=height,
                    grid_width=width,
                    target_class=target_class,
                )
                entry = (activation, next(self._counter), example)
                if len(heap) < self.heap_capacity:
                    heapq.heappush(heap, entry)
                elif activation > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    def finalize(self) -> dict[int, list[TopActivation]]:
        result: dict[int, list[TopActivation]] = {}
        for latent_id, heap in self._heaps.items():
            candidates = [entry[2] for entry in sorted(heap, reverse=True)]
            image_counts: dict[tuple[str, str], int] = {}
            examples: list[TopActivation] = []

            for example in candidates:
                image_key = (example.dataset_name, example.image_id)
                if image_counts.get(image_key, 0) >= self.max_per_image:
                    continue
                examples.append(example)
                image_counts[image_key] = image_counts.get(image_key, 0) + 1
                if len(examples) == self.examples_per_latent:
                    break

            result[latent_id] = examples
        return result
