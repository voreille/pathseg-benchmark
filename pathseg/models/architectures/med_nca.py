"""Med-NCA (Kalkhof et al., IPMI 2023) as a pathseg semantic segmenter.

Port of the backbone NCA and the multi-level forward of
https://github.com/MECLabTUDA/Med-NCA (commit a844a72, MIT License,
Copyright (c) 2020 Ming). Only the model is ported; the upstream training
loop, datasets and losses are not used. Deviations from upstream are logged in
``experiments/mednca/README.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BackboneNCA(nn.Module):
    """One NCA level: upstream ``BackboneNCA`` (perception) + ``BasicNCA`` (update).

    The state is channels-last ``B x H x W x channel_n`` with layout
    ``[image | outputs | hidden]``. The first ``input_channels`` channels are
    re-injected after every step and are therefore never updated.
    """

    def __init__(
        self,
        channel_n: int,
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.channel_n = int(channel_n)
        self.input_channels = int(input_channels)
        self.fire_rate = float(fire_rate)

        # Same parameter names and creation order as upstream, so state dicts
        # are interchangeable and seeded initialization matches.
        self.fc0 = nn.Linear(self.channel_n * 3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, self.channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.p0 = nn.Conv2d(
            self.channel_n,
            self.channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.p1 = nn.Conv2d(
            self.channel_n,
            self.channel_n,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.p0(x), self.p1(x)), 1)

    def update(self, x: torch.Tensor, fire_rate: float | None = None) -> torch.Tensor:
        if fire_rate is None:
            fire_rate = self.fire_rate

        # Upstream transposes dims 1 and 3 (B x H x W x C -> B x C x W x H),
        # so the perception convs run on the H/W-transposed grid. Kept as is
        # for weight compatibility.
        dx = self.perceive(x.transpose(1, 3)).transpose(1, 3)
        dx = self.fc1(F.relu(self.fc0(dx)))

        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) > fire_rate
        dx = dx * stochastic.float().to(dx.device)

        return x + dx

    def forward(
        self,
        x: torch.Tensor,
        steps: int = 64,
        fire_rate: float | None = None,
    ) -> torch.Tensor:
        for _ in range(steps):
            x_next = self.update(x, fire_rate)
            x = torch.cat(
                (x[..., : self.input_channels], x_next[..., self.input_channels :]),
                3,
            )
        return x
