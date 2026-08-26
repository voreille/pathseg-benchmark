import torch

from pathseg.models.encoder import Encoder
from pathseg.sae.topk_sae import TopKSAE


class EncoderWithTopKSAE(Encoder):
    def __init__(
        self,
        encoder_id: str,
        num_latents: int,
        k: int,
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        self.sae = TopKSAE(
            input_dim=self.embed_dim,
            num_latents=num_latents,
            k=k,
        )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C] -> [B, C, H, W]
        """
        gh, gw = self.grid_size

        if x.shape[1] != gh * gw:
            raise ValueError(f"Expected {gh * gw} tokens, received {x.shape[1]}")

        return x.transpose(1, 2).reshape(x.shape[0], -1, gh, gw)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = super().forward(x)
        reconstructed_tokens, latents = self.sae(tokens)

        return {
            "tokens": tokens,
            "reconstructed_tokens": reconstructed_tokens,
            "latents": latents,
        }

    def forward_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.forward(x)

        return {
            "features": self._tokens_to_map(output["tokens"]),
            "reconstructed_features": self._tokens_to_map(
                output["reconstructed_tokens"]
            ),
            "latent_features": self._tokens_to_map(output["latents"]),
        }

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        # decoder.weight: [input_dim, num_latents]
        self.decoder.weight.div_(
            self.decoder.weight.norm(
                dim=0,
                keepdim=True,
            ).clamp_min(1e-8)
        )
