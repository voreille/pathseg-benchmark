from __future__ import annotations

import pytest
import torch

from pathseg.models.architectures.octree_nca import (
    OctreeNCAEncoder,
    OctreeNCASegmenter,
)


def randomize_(module: torch.nn.Module, seed: int = 0) -> None:
    """fc1 starts at zero; randomize so NCA updates are non-trivial."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator) * 0.05
            )


def make_encoder(**kwargs) -> OctreeNCAEncoder:
    config = {"channel_n": 24, "hidden_size": 32, "steps": (3, 2, 4)}
    config.update(kwargs)
    encoder = OctreeNCAEncoder(output_channels=5, **config)
    randomize_(encoder)
    return encoder


def test_stage_composition_matches_forward_feature_maps():
    encoder = make_encoder().train()
    x = torch.rand(2, 3, 32, 40)

    torch.manual_seed(0)
    (expected,) = encoder.forward_feature_maps(x)

    torch.manual_seed(0)
    images = encoder.pyramid(x)
    state = encoder.run_level(2, encoder.init_state(images[2]), images[2])
    state = encoder.run_level(1, encoder.upscale_state(state, images[1].shape[-2:]), images[1])
    state = encoder.run_level(0, encoder.upscale_state(state, images[0].shape[-2:]), images[0])

    assert expected.shape == (2, 24, 32, 40)
    torch.testing.assert_close(state.permute(0, 3, 1, 2), expected, rtol=0.0, atol=0.0)


def test_run_level_never_updates_image_channels():
    encoder = make_encoder().train()
    x = torch.rand(2, 3, 16, 16)
    state = torch.randn(2, 16, 16, 24)

    output = encoder.run_level(0, state, x)

    torch.testing.assert_close(output[..., :3], x.permute(0, 2, 3, 1), rtol=0.0, atol=0.0)
    assert not torch.equal(output[..., 3:], state[..., 3:])


def test_run_level_rejects_unknown_level():
    encoder = make_encoder()
    with pytest.raises(KeyError, match="Unknown level"):
        encoder.run_level(3, torch.zeros(1, 4, 4, 24), torch.zeros(1, 3, 4, 4))


def test_level_sizes_and_pyramid():
    encoder = make_encoder(steps=(10, 10, 10, 10, 28))
    assert encoder.level_sizes(448, 448) == [
        (448, 448),
        (224, 224),
        (112, 112),
        (56, 56),
        (28, 28),
    ]
    # Upstream's rule for other sizes: ceil halving.
    assert encoder.level_sizes(30, 45) == [(30, 45), (15, 23), (8, 12), (4, 6), (2, 3)]

    x = torch.rand(1, 3, 32, 32)
    images = encoder.pyramid(x)
    assert images[0] is x
    # Nearest downscaling by 2^l picks the top-left pixel of each block.
    torch.testing.assert_close(images[1], x[..., ::2, ::2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(images[3], x[..., ::8, ::8], rtol=0.0, atol=0.0)


def test_odd_sizes_run():
    encoder = make_encoder().eval()
    with torch.no_grad():
        (state,) = encoder.forward_feature_maps(torch.rand(1, 3, 30, 45))
    assert state.shape == (1, 24, 30, 45)


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
    # fire_rate=1 makes every cell fire, so chunking can't change fire masks.
    x = torch.rand(5, 3, 16, 24)
    encoder = make_encoder(fire_rate=1.0, n_eval_runs=2).train(training)
    (expected,) = encoder.forward_feature_maps(x)

    encoder.max_batch_size = 2
    (chunked,) = encoder.forward_feature_maps(x)

    torch.testing.assert_close(chunked, expected, rtol=1e-6, atol=1e-6)


def test_fire_rate_is_the_update_probability():
    level = make_encoder(fire_rate=0.25).backbone_ncas[0]
    mask = level.fire_mask(torch.zeros(4, 64, 64, 24), level.fire_rate)
    assert mask.shape == (4, 64, 64, 1)
    assert abs(mask.float().mean().item() - 0.25) < 0.02


def test_train_mode_runs_a_single_pass():
    encoder = make_encoder(n_eval_runs=3).train()
    x = torch.rand(1, 3, 16, 16)

    torch.manual_seed(0)
    (output,) = encoder.forward_feature_maps(x)
    torch.manual_seed(0)
    expected = encoder.forward_state(x).permute(0, 3, 1, 2)

    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def count_updates(encoder: OctreeNCAEncoder) -> list[int]:
    calls = [0]

    for level in encoder.backbone_ncas:
        update = level.update

        def counted(*args, _update=update, **kwargs):
            calls[0] += 1
            return _update(*args, **kwargs)

        level.update = counted
    return calls


def forward_backward(encoder: OctreeNCAEncoder, x: torch.Tensor):
    encoder.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    (state,) = encoder.forward_feature_maps(x)
    state.square().mean().backward()
    grads = {name: p.grad.clone() for name, p in encoder.named_parameters()}
    return state.detach(), grads


def test_grad_checkpointing_preserves_outputs_and_gradients():
    encoder = make_encoder(steps=(7, 5, 3)).train()
    calls = count_updates(encoder)
    x = torch.rand(2, 3, 16, 24)

    expected_state, expected_grads = forward_backward(encoder, x)
    assert calls[0] == 15

    calls[0] = 0
    encoder.set_grad_checkpointing(3)
    state, grads = forward_backward(encoder, x)
    assert calls[0] == 2 * 15  # every step is recomputed in backward

    torch.testing.assert_close(state, expected_state, rtol=0.0, atol=0.0)
    assert grads.keys() == expected_grads.keys()
    for name, grad in grads.items():
        assert grad.abs().sum() > 0, name
        torch.testing.assert_close(grad, expected_grads[name], rtol=1e-6, atol=1e-7)


def test_grad_checkpointing_inactive_in_eval_and_no_grad():
    encoder = make_encoder(grad_checkpointing_every=1)
    calls = count_updates(encoder)
    x = torch.rand(1, 3, 16, 16)

    encoder.eval()
    encoder.forward_feature_maps(x)
    assert calls[0] == 9

    calls[0] = 0
    encoder.train()
    with torch.no_grad():
        encoder.forward_feature_maps(x)
    assert calls[0] == 9


def test_segmenter_single_task_output_and_param_count():
    network = OctreeNCASegmenter({"ignite": 16}, channel_n=48, steps=(1, 1, 1, 1, 2))
    num_parameters = sum(parameter.numel() for parameter in network.parameters())
    print(f"OctreeNCASegmenter(channel_n=48, hidden_size=64) parameters: {num_parameters}")

    # 5 levels x ((48*9 + 48) + (96*64 + 64) + 64*45)
    assert num_parameters == 47_840
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


def test_upstream_peso_param_count():
    # Upstream PESO: channel_n 16 (3 in + 1 out), hidden 64, 5 levels -> the
    # paper's 15,520 parameters.
    network = OctreeNCASegmenter({"peso": 1})
    assert sum(p.numel() for p in network.parameters()) == 15_520


def test_segmenter_rejects_bad_config():
    with pytest.raises(ValueError, match="channel_n"):
        OctreeNCASegmenter({"ignite": 16}, channel_n=18)
    with pytest.raises(ValueError, match="steps"):
        OctreeNCASegmenter({"ignite": 16}, channel_n=48, steps=(10, 0))
    with pytest.raises(ValueError, match="steps"):
        OctreeNCASegmenter({"ignite": 16}, channel_n=48, steps=())


def test_plain_step_matches_fused_step():
    from pathseg.models import nca

    level = make_encoder().backbone_ncas[0]
    x = torch.randn(2, 9, 7, 24, requires_grad=True)
    stochastic = torch.rand(2, 9, 7, 1) > 0.5

    outputs, grads = [], []
    for step in (nca._nca_step, nca._nca_step_plain):
        level.zero_grad(set_to_none=True)
        x.grad = None
        out = step(x, *level.folded_weights(), stochastic, torch.float32)
        out.square().sum().backward()
        outputs.append(out.detach())
        grads.append([x.grad.clone()] + [p.grad.clone() for p in level.parameters()])

    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-6, atol=1e-6)
    for actual, expected in zip(grads[1], grads[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def use_compiled_steps(monkeypatch) -> list[int]:
    """Replace the compiled step by a counting eager stand-in."""
    from pathseg.models import nca

    calls = [0]

    def fake_compiled(*args):
        calls[0] += 1
        return nca._nca_step_plain(*args)

    monkeypatch.setattr(nca, "_get_compiled_nca_step", lambda: fake_compiled)
    return calls


def test_compile_step_is_off_by_default_and_cuda_training_only(monkeypatch):
    calls = use_compiled_steps(monkeypatch)
    assert OctreeNCASegmenter({"ignite": 16}, channel_n=48).encoder.compile_step is False

    encoder = make_encoder(compile_step=True)
    assert encoder.compile_step is True
    x = torch.rand(1, 3, 16, 16)
    encoder.train()
    encoder.forward_feature_maps(x)
    assert calls[0] == 0  # CPU: eager step

    if not torch.cuda.is_available():
        return
    encoder.cuda()
    x = x.cuda()
    encoder.forward_feature_maps(x)
    assert calls[0] == 9
    with torch.no_grad():
        encoder.forward_feature_maps(x)
    encoder.eval()
    encoder.forward_feature_maps(x)
    assert calls[0] == 9  # no_grad and eval: eager step


@pytest.fixture
def no_tf32():
    """True fp32 convs/matmuls: TF32 rounding (eager cuDNN default) exceeds the tolerance."""
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled step is CUDA-only")
def test_compiled_step_matches_eager_on_cuda(no_tf32):
    encoder = make_encoder(steps=(5, 3, 4), grad_checkpointing_every=2).cuda().train()
    x = torch.rand(2, 3, 32, 40, device="cuda")

    expected_state, expected_grads = forward_backward(encoder, x)
    encoder.set_compile_step(True)
    state, grads = forward_backward(encoder, x)

    torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-5)
    for name, grad in grads.items():
        torch.testing.assert_close(grad, expected_grads[name], rtol=1e-3, atol=1e-5)
