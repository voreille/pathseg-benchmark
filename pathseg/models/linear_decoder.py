from pathlib import Path

import torch
import torch.nn as nn

from pathseg.leace.leace import LeaceEraser
from pathseg.models.encoder import Encoder


class LinearDecoder(Encoder):
    def __init__(
        self,
        encoder_id,
        num_classes,
        img_size,
        sub_norm=False,
        ckpt_path="",
        discard_last_mlp=False,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.head(x)
        x = x.transpose(1, 2)

        return x.reshape(x.shape[0], -1, *self.grid_size)


class LinearDecoderBackbone(Encoder):
    def __init__(
        self,
        encoder_id,
        num_classes,
        img_size,
        sub_norm=False,
        ckpt_path="",
        discard_last_mlp=False,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class LinearDecoderLEACE(Encoder):
    def __init__(
        self,
        encoder_id: str,
        num_classes: int,
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
        leace_path: str | Path = "",
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        if leace_path:
            self.leace_eraser = LeaceEraser.load(leace_path, map_location="cpu")
        else:
            self.leace_eraser = None

        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] image; encoder outputs [B, Q, D]
        Returns: [B, num_classes, Ht, Wt]
        """
        # 1) encode -> [B, Q, D]
        x = super().forward(x)

        # 2) apply LEACE on token dim if available
        if self.leace_eraser is not None:
            B, Q, D = x.shape
            x_flat = x.reshape(B * Q, D)
            x_flat = self.leace_eraser(x_flat)  # [B*Q, D]
            x = x_flat.reshape(B, Q, D)

        # 3) linear head on each token: [B, Q, num_classes]
        x = self.head(x)

        # 4) [B, Q, C] -> [B, C, Q] -> [B, C, Ht, Wt]
        x = x.transpose(1, 2)
        Ht, Wt = self.grid_size  # inherited from Encoder
        return x.reshape(x.shape[0], -1, *self.grid_size)
