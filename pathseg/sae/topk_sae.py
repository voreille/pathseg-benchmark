import torch
import torch.nn.functional as F
from torch import nn


class TopKSAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_latents: int,
        k: int,
    ):
        super().__init__()

        if not 0 < k <= num_latents:
            raise ValueError("k must be in [1, num_latents]")

        self.input_dim = input_dim
        self.num_latents = num_latents
        self.k = k

        self.encoder = nn.Linear(input_dim, num_latents, bias=False)
        self.decoder = nn.Linear(num_latents, input_dim, bias=False)

        # Shared input/output centering bias.
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.decoder.weight,
            std=1.0 / self.input_dim**0.5,
        )
        self.normalize_decoder_()

        # Tied only at initialization.
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        # decoder.weight: [input_dim, num_latents]
        self.decoder.weight.div_(
            self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        preactivations = self.encoder(x - self.pre_bias)

        values, indices = torch.topk(
            preactivations,
            k=self.k,
            dim=-1,
        )

        # Standard TopK SAE uses nonnegative activations.
        values = F.relu(values)

        latents = torch.zeros_like(preactivations)
        latents.scatter_(-1, indices, values)

        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents

    def forward_with_aux(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        reconstruction, latents = self(x)

        return {
            "reconstructed_tokens": reconstruction,
            "latents": latents,
        }
