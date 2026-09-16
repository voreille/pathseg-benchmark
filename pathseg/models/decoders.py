from __future__ import annotations

import torch
from torch import nn

from pathseg.models.semantic_segmenter import FeatureMaps, SemanticLogits


class LinearSemanticDecoder(nn.Module):
    def __init__(self, embed_dim, heads):
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, feature_maps, task=None):
        x = feature_maps[-1]

        if task is not None:
            return {task: self.heads[task](x)}

        return {name: head(x) for name, head in self.heads.items()}


class LinearDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.num_classes = out_channels
        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        feature_maps: FeatureMaps,
    ) -> torch.Tensor:
        return self.head(feature_maps[-1])


class MultiTaskDecoder(nn.Module):
    def __init__(
        self,
        heads: dict[str, nn.Module],
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    @property
    def num_classes_by_task(self) -> dict[str, int]:
        return {task: head.num_classes for task, head in self.heads.items()}

    def forward(
        self,
        feature_maps: FeatureMaps,
        task: str | None = None,
    ) -> SemanticLogits:
        if task is not None:
            if task not in self.heads:
                raise KeyError(
                    f"Unknown task {task!r}. Available tasks: {list(self.heads)}."
                )

            return {
                task: self.heads[task](feature_maps),
            }

        return {name: head(feature_maps) for name, head in self.heads.items()}
