"""Variant B (MedNCATraining): upstream training recipe for Med-NCA."""

from __future__ import annotations

import pytest
import torch

from pathseg.models.architectures.med_nca import MedNCASegmenter
from pathseg.training.med_nca import MedNCATraining

IGNORE = 255
NUM_CLASSES = 5


def test_random_crop_uses_one_position_for_state_image_and_target():
    batch, height, width = 4, 20, 28
    rows = torch.arange(height).view(1, height, 1).expand(batch, height, width)
    cols = torch.arange(width).view(1, 1, width).expand(batch, height, width)
    sample = torch.arange(batch).view(batch, 1, 1).expand(batch, height, width)
    coords = torch.stack((rows, cols, sample), dim=-1).float()  # B x H x W x 3

    torch.manual_seed(0)
    state, imgs, targets = MedNCATraining.random_crop(
        coords,
        coords.permute(0, 3, 1, 2),
        rows * width + cols,
        (5, 7),
    )

    assert state.shape == (batch, 5, 7, 3)
    assert imgs.shape == (batch, 3, 5, 7)
    assert targets.shape == (batch, 5, 7)
    torch.testing.assert_close(imgs, state.permute(0, 3, 1, 2))
    torch.testing.assert_close(
        targets, (state[..., 0] * width + state[..., 1]).long()
    )
    torch.testing.assert_close(state[..., 2], sample[:, :5, :7].float())
    # Contiguous windows inside the tile.
    assert torch.all(state[:, :, 1:, 1] - state[:, :, :-1, 1] == 1)
    assert torch.all(state[:, 1:, :, 0] - state[:, :-1, :, 0] == 1)
    assert int(state[..., 0].max()) < height and int(state[..., 1].max()) < width


def make_module(loss_name: str = "upstream_dice_bce", **kwargs) -> MedNCATraining:
    network = MedNCASegmenter(
        {"ignite": NUM_CLASSES}, channel_n=16, hidden_size=16, steps=3
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return MedNCATraining(
        network=network,
        tasks={"ignite": {"loss_name": loss_name}},
        ignore_idx=IGNORE,
        img_size=(32, 32),
        tiler=None,
        **kwargs,
    )


def make_batch(batch: int = 2, size: int = 32):
    generator = torch.Generator().manual_seed(3)
    imgs = torch.randint(0, 256, (batch, 3, size, size), generator=generator).float()
    targets = []
    for _ in range(batch):
        class_map = torch.randint(0, NUM_CLASSES, (size, size), generator=generator)
        labels = class_map.unique()
        targets.append(
            {"masks": class_map[None] == labels[:, None, None], "labels": labels}
        )
    return imgs, targets


@pytest.mark.parametrize("loss_name", ["upstream_dice_bce", "cross_entropy_dice"])
def test_training_step_trains_both_levels_on_a_crop(loss_name):
    module = make_module(loss_name).train()
    assert module.fine_crop_size(32, 32) == (8, 8)  # coarse size, as upstream

    output = module.training_step(make_batch(), 0)
    assert output.ndim == 0 and torch.isfinite(output)
    output.backward()
    for name, parameter in module.network.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("loss_name", ["upstream_dice_bce", "cross_entropy_dice"])
def test_training_step_skips_batches_without_labelled_pixels(loss_name):
    module = make_module(loss_name).train()
    imgs, targets = make_batch()
    for target in targets:
        target["masks"] = target["masks"][:0]
        target["labels"] = target["labels"][:0]
    assert module.training_step((imgs, targets), 0) is None


def test_training_step_uses_the_task_criterion(monkeypatch):
    module = make_module("upstream_dice_bce").train()
    calls = []
    criterion = module.criteria["ignite"]
    original = criterion.forward

    def spy(logits, targets):
        calls.append((logits.shape, targets.shape))
        return original(logits, targets)

    monkeypatch.setattr(criterion, "forward", spy)
    module.training_step(make_batch(), 0)
    assert calls == [((2, NUM_CLASSES, 8, 8), (2, 8, 8))]  # on the 8 x 8 crop


def test_upstream_optimizer_settings():
    module = make_module()
    config = module.configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]

    assert isinstance(optimizer, torch.optim.Adam)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(16e-4)
    assert group["betas"] == (0.5, 0.5)
    assert group["weight_decay"] == 0
    assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    assert scheduler.gamma == pytest.approx(0.9999)
    assert config["lr_scheduler"]["interval"] == "step"
    assert sum(p.numel() for g in optimizer.param_groups for p in g["params"]) == sum(
        p.numel() for p in module.network.parameters()
    )
