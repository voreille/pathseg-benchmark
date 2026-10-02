from __future__ import annotations

import torch

from pathseg.models.architectures.med_nca import MedNCAEncoder


def randomize_(module: torch.nn.Module, seed: int = 0) -> None:
    """fc1 starts at zero; randomize so NCA updates are non-trivial."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator) * 0.05
            )


def make_encoder(**kwargs) -> MedNCAEncoder:
    config = {"channel_n": 24, "hidden_size": 32, "steps": 6}
    config.update(kwargs)
    encoder = MedNCAEncoder(output_channels=5, **config)
    randomize_(encoder)
    return encoder


def test_stage_composition_matches_forward_feature_maps():
    encoder = make_encoder().train()
    x = torch.rand(2, 3, 32, 40)

    torch.manual_seed(0)
    (expected,) = encoder.forward_feature_maps(x)

    torch.manual_seed(0)
    xd = encoder.downscale(x)
    state = encoder.run_level(
        "fine",
        encoder.upscale_state(
            encoder.run_level("coarse", encoder.init_state(xd), xd),
            x.shape[-2:],
        ),
        x,
    )

    assert expected.shape == (2, 24, 32, 40)
    torch.testing.assert_close(state.permute(0, 3, 1, 2), expected, rtol=0.0, atol=0.0)


def test_run_level_never_updates_image_channels():
    encoder = make_encoder().train()
    x = torch.rand(2, 3, 16, 16)
    state = torch.randn(2, 16, 16, 24)

    output = encoder.run_level("fine", state, x)

    torch.testing.assert_close(output[..., :3], x.permute(0, 2, 3, 1), rtol=0.0, atol=0.0)
    assert not torch.equal(output[..., 3:], state[..., 3:])


def test_downscale_and_upscale_sizes():
    encoder = make_encoder(scale_factor=4)
    x = torch.rand(1, 3, 448, 448)

    xd = encoder.downscale(x)
    assert xd.shape == (1, 3, 112, 112)

    state = encoder.init_state(xd)
    assert state.shape == (1, 112, 112, 24)
    assert encoder.upscale_state(state, x.shape[-2:]).shape == (1, 448, 448, 24)
