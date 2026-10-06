from __future__ import annotations

import pytest
import torch

from pathseg.models.architectures.med_nca import (
    MedNCAEncoder,
    MedNCASegmenter,
    StateSliceDecoder,
)


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


def test_eval_mode_averages_n_eval_runs_passes():
    encoder = make_encoder(n_eval_runs=3)
    x = torch.rand(2, 3, 16, 24)

    encoder.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        (averaged,) = encoder.forward_feature_maps(x)

    torch.manual_seed(0)
    with torch.no_grad():
        passes = [encoder.forward_state(x) for _ in range(3)]
    expected = torch.stack(passes).mean(dim=0).permute(0, 3, 1, 2)

    torch.testing.assert_close(averaged, expected, rtol=1e-6, atol=1e-6)
    assert not torch.equal(passes[0], passes[1])


@pytest.mark.parametrize("training", [False, True])
def test_max_batch_size_does_not_change_outputs(training):
    # fire_rate=0 makes every update fire, so chunking can't change fire masks.
    x = torch.rand(5, 3, 16, 24)
    encoder = make_encoder(fire_rate=0.0, n_eval_runs=2).train(training)
    (expected,) = encoder.forward_feature_maps(x)

    encoder.max_batch_size = 2
    (chunked,) = encoder.forward_feature_maps(x)

    torch.testing.assert_close(chunked, expected, rtol=1e-6, atol=1e-6)


def test_max_batch_size_splits_the_forward(monkeypatch):
    encoder = make_encoder(max_batch_size=2).eval()
    sizes = []
    forward_state = encoder.forward_state
    monkeypatch.setattr(
        encoder,
        "forward_state",
        lambda imgs: sizes.append(imgs.shape[0]) or forward_state(imgs),
    )
    with torch.no_grad():
        (state,) = encoder.forward_feature_maps(torch.rand(5, 3, 16, 24))

    assert sizes == [2, 2, 1]
    assert state.shape == (5, 24, 16, 24)


def test_train_mode_runs_a_single_pass():
    encoder = make_encoder(n_eval_runs=3).train()
    x = torch.rand(1, 3, 16, 16)

    torch.manual_seed(0)
    (output,) = encoder.forward_feature_maps(x)
    torch.manual_seed(0)
    expected = encoder.forward_state(x).permute(0, 3, 1, 2)

    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def count_updates(encoder: MedNCAEncoder) -> list[int]:
    calls = [0]

    for level in encoder.levels.values():
        update = level.update

        def counted(*args, _update=update, **kwargs):
            calls[0] += 1
            return _update(*args, **kwargs)

        level.update = counted
    return calls


def forward_backward(encoder: MedNCAEncoder, x: torch.Tensor):
    encoder.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    (state,) = encoder.forward_feature_maps(x)
    state.square().mean().backward()
    grads = {name: p.grad.clone() for name, p in encoder.named_parameters()}
    return state.detach(), grads


def test_grad_checkpointing_preserves_outputs_and_gradients():
    encoder = make_encoder(steps=7).train()
    calls = count_updates(encoder)
    x = torch.rand(2, 3, 16, 24)

    expected_state, expected_grads = forward_backward(encoder, x)
    assert calls[0] == 2 * 7

    calls[0] = 0
    encoder.set_grad_checkpointing(3)  # chunks of 3, 3, 1 steps
    state, grads = forward_backward(encoder, x)
    assert calls[0] == 2 * 2 * 7  # every step is recomputed in backward

    torch.testing.assert_close(state, expected_state, rtol=0.0, atol=0.0)
    assert grads.keys() == expected_grads.keys()
    for name, grad in grads.items():
        assert grad.abs().sum() > 0, name
        torch.testing.assert_close(grad, expected_grads[name], rtol=1e-6, atol=1e-7)


def test_grad_checkpointing_inactive_in_eval_and_no_grad():
    encoder = make_encoder(steps=4, grad_checkpointing_every=1)
    calls = count_updates(encoder)
    x = torch.rand(1, 3, 16, 16)

    encoder.eval()
    encoder.forward_feature_maps(x)
    assert calls[0] == 2 * 4

    calls[0] = 0
    encoder.train()
    with torch.no_grad():
        encoder.forward_feature_maps(x)
    assert calls[0] == 2 * 4


def test_segmenter_single_task_output_and_param_count():
    network = MedNCASegmenter({"ignite": 16}, steps=2)
    num_parameters = sum(parameter.numel() for parameter in network.parameters())
    print(f"MedNCASegmenter(channel_n=64, hidden_size=128) parameters: {num_parameters}")

    # 2 levels x (2 * (64*64*9 + 64) + 192*128 + 128 + 128*64)
    assert num_parameters == 213_504
    assert sum(p.numel() for p in network.decoder.parameters()) == 0
    assert network.upsample_logits is False
    assert network.num_classes_by_task == {"ignite": 16}

    for mode in (network.train, network.eval):
        mode()
        with torch.no_grad():
            output = network(torch.rand(2, 3, 32, 32))
        assert set(output) == {"ignite"}
        assert output["ignite"].shape == (2, 16, 32, 32)

    with torch.no_grad():
        assert set(network(torch.rand(1, 3, 32, 32), task="ignite")) == {"ignite"}


def test_decoder_task_selection_mirrors_multitask_decoder():
    decoder = StateSliceDecoder({"ignite": 16, "anorak": 7})
    state = torch.arange(30.0).view(1, 30, 1, 1).expand(2, 30, 4, 5)

    all_tasks = decoder((state,))
    assert list(all_tasks) == ["ignite", "anorak"]
    torch.testing.assert_close(all_tasks["ignite"], state[:, 3:19])
    torch.testing.assert_close(all_tasks["anorak"], state[:, 19:26])

    selected = decoder((state,), task="anorak")
    assert list(selected) == ["anorak"]
    torch.testing.assert_close(selected["anorak"], state[:, 19:26])

    with pytest.raises(KeyError, match="Unknown task"):
        decoder((state,), task="bcss")


def test_segmenter_multitask_channels():
    network = MedNCASegmenter({"ignite": 16, "anorak": 7}, channel_n=96, steps=1)
    assert network.encoder.channel_n == 96
    assert network.encoder.output_channels == 23

    with torch.no_grad():
        output = network(torch.rand(1, 3, 16, 16))
    assert {task: tuple(logits.shape) for task, logits in output.items()} == {
        "ignite": (1, 16, 16, 16),
        "anorak": (1, 7, 16, 16),
    }


def test_segmenter_rejects_too_few_channels():
    with pytest.raises(ValueError, match="channel_n"):
        MedNCASegmenter({"ignite": 16}, channel_n=18)
