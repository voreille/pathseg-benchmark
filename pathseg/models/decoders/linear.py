from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class LinearSemanticDecoder(nn.Module):
    """One independent 1x1-convolution head per semantic task.

    The decoder consumes the feature-map tuple produced by an encoder. It uses
    the final/deepest feature map and returns named logits for a single task or
    all configured tasks.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes_by_task: Mapping[str, int],
        bias: bool = True,
    ) -> None:
        super().__init__()

        if not num_classes_by_task:
            raise ValueError("At least one semantic task is required.")

        self.in_channels = int(in_channels)
        self.num_classes_by_task = {
            str(name): int(num_classes)
            for name, num_classes in num_classes_by_task.items()
        }

        self.heads = nn.ModuleDict(
            {
                name: nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=bias,
                )
                for name, num_classes in self.num_classes_by_task.items()
            }
        )

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(self.heads.keys())

    @staticmethod
    def _final_feature_map(
        feature_maps: Sequence[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        # Accept a tensor as a convenience, but the canonical boundary is a
        # shallow-to-deep sequence of NCHW feature maps.
        if torch.is_tensor(feature_maps):
            final_map = feature_maps
        else:
            if len(feature_maps) == 0:
                raise ValueError("feature_maps cannot be empty.")
            final_map = feature_maps[-1]

        if final_map.ndim != 4:
            raise ValueError(
                "LinearSemanticDecoder expects a BxCxHxW feature map, "
                f"got shape {tuple(final_map.shape)}."
            )

        return final_map

    def forward(
        self,
        feature_maps: Sequence[torch.Tensor] | torch.Tensor,
        task: str | None = None,
    ) -> dict[str, torch.Tensor]:
        final_map = self._final_feature_map(feature_maps)

        if final_map.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, "
                f"got {final_map.shape[1]}."
            )

        if task is not None:
            if task not in self.heads:
                raise KeyError(
                    f"Unknown task {task!r}; available tasks: {self.task_names}."
                )
            return {task: self.heads[task](final_map)}

        return {name: head(final_map) for name, head in self.heads.items()}

