import torch

from pathseg.models.encoder import Encoder
from pathseg.sae.topk_sae import TopKSAE


class EncoderWithTopKSAE(Encoder):
    """
    ViT encoder followed by a Top-k sparse autoencoder.

    The ViT produces patch tokens:
        tokens: [B, N, D]

    The SAE produces:
        reconstructed_tokens: [B, N, D]
        latents:               [B, N, M]

    where:
        D is the ViT embedding dimension;
        M is the SAE dictionary size.
    """

    def __init__(
        self,
        encoder_id: str,
        num_latents: int,
        k: int,
        img_size: tuple[int, int],
        sub_norm: bool = False,
        ckpt_path: str = "",
        discard_last_mlp: bool = False,
        freeze_encoder: bool = True,
    ):
        super().__init__(
            encoder_id=encoder_id,
            img_size=img_size,
            sub_norm=sub_norm,
            ckpt_path=ckpt_path,
            discard_last_mlp=discard_last_mlp,
        )

        self.num_latents = num_latents
        self.k = k
        self.freeze_encoder = freeze_encoder

        self.sae = TopKSAE(
            input_dim=self.embed_dim,
            num_latents=num_latents,
            k=k,
        )

        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        """
        Keep the ViT in evaluation mode when it is frozen.

        This matters for stochastic depth, dropout, and similar layers:
        the SAE should see deterministic activations.
        """
        super().train(mode)

        if self.freeze_encoder:
            self.encoder.eval()

        return self

    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return frozen or trainable ViT patch tokens.

        Output:
            [B, N, D]
        """
        if self.freeze_encoder:
            with torch.no_grad():
                return super().forward(x)

        return super().forward(x)

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token representations into spatial feature maps.

        Input:
            x: [B, N, C]

        Output:
            [B, C, H, W]
        """
        gh, gw = self.grid_size
        batch_size, num_tokens, channels = x.shape

        if num_tokens != gh * gw:
            raise ValueError(
                f"Expected {gh * gw} patch tokens for grid "
                f"{self.grid_size}, but received {num_tokens}"
            )

        return x.transpose(1, 2).reshape(
            batch_size,
            channels,
            gh,
            gw,
        )

    def forward_tokens(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Apply the ViT and SAE while preserving token layout.
        """
        tokens = self._forward_encoder(x)

        reconstructed_tokens, latents = self.sae(tokens)

        return {
            "tokens": tokens,
            "reconstructed_tokens": reconstructed_tokens,
            "latents": latents,
        }

    def forward_feature_maps(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Apply the ViT and SAE and return spatial feature maps.

        Shapes:
            features:               [B, D, H, W]
            reconstructed_features: [B, D, H, W]
            latent_features:        [B, M, H, W]
        """
        output = self.forward_tokens(x)

        return {
            "features": self._tokens_to_map(output["tokens"]),
            "reconstructed_features": self._tokens_to_map(
                output["reconstructed_tokens"]
            ),
            "latent_features": self._tokens_to_map(output["latents"]),
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.forward_tokens(x)
