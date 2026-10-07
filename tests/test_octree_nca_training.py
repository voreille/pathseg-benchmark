"""OctreeNCATraining and the UpstreamEMA callback: the OctreeNCA authors' recipe."""

from __future__ import annotations

import lightning.pytorch as pl
import pytest
import torch

from pathseg.models.architectures.octree_nca import OctreeNCASegmenter
from pathseg.training.callbacks.ema import UpstreamEMA
from pathseg.training.octree_nca import OctreeNCATraining

IGNORE = 255
NUM_CLASSES = 5


def make_module(loss_name: str = "upstream_dice_bce", **kwargs) -> OctreeNCATraining:
    network = OctreeNCASegmenter(
        {"ignite": NUM_CLASSES}, channel_n=16, hidden_size=16, steps=(2, 2, 3)
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return OctreeNCATraining(
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
def test_training_step_trains_every_level_on_the_full_tile(loss_name, monkeypatch):
    module = make_module(loss_name).train()
    calls = []
    criterion = module.criteria["ignite"]
    original = criterion.forward

    def spy(logits, targets):
        calls.append((tuple(logits.shape), tuple(targets.shape)))
        return original(logits, targets)

    monkeypatch.setattr(criterion, "forward", spy)
    output = module.training_step(make_batch(), 0)

    assert calls == [((2, NUM_CLASSES, 32, 32), (2, 32, 32))]  # whole tile, no crop
    assert output.ndim == 0 and torch.isfinite(output)
    output.backward()
    for name, parameter in module.network.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


def test_upstream_optimizer_settings():
    module = make_module()
    config = module.configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]

    assert isinstance(optimizer, torch.optim.Adam)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(16e-4)
    assert group["betas"] == (0.9, 0.99)
    assert group["weight_decay"] == 0
    assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    assert scheduler.gamma == pytest.approx(0.9999**8)
    assert config["lr_scheduler"]["interval"] == "step"
    assert config["lr_scheduler"]["frequency"] == 200
    assert sum(p.numel() for g in optimizer.param_groups for p in g["params"]) == sum(
        p.numel() for p in module.network.parameters()
    )


def test_benchmark_optimizer_when_upstream_optimizer_is_off():
    module = make_module(upstream_optimizer=False, weight_decay=0.05)
    module.trainer = type("T", (), {"estimated_stepping_batches": 10})()
    optimizer = module.configure_optimizers()["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)


def test_ema_callback_comes_from_the_module():
    (callback,) = make_module().configure_callbacks()
    assert isinstance(callback, UpstreamEMA)
    assert callback.decay == 0.99 and callback.every_n_steps == 200
    assert make_module(ema_decay=None).configure_callbacks() == []


class _Scalar(pl.LightningModule):
    """One parameter, SGD with lr 1 on loss = w: w drops by 1 every step."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))
        self.history = [0.0]
        self.validated = []

    def training_step(self, batch, batch_idx):
        return self.w * 1.0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.history.append(self.w.item())

    def validation_step(self, batch, batch_idx):
        self.validated.append(self.w.item())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1.0)


def test_upstream_ema_timing_init_blend_and_checkpoint(tmp_path):
    decay, every, steps = 0.5, 3, 7
    module = _Scalar()
    data = torch.utils.data.DataLoader(torch.zeros(steps, 1), batch_size=1)
    trainer = pl.Trainer(
        max_steps=steps,
        callbacks=[UpstreamEMA(decay=decay, every_n_steps=every)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        val_check_interval=steps,
        limit_val_batches=1,
        accelerator="cpu",
    )
    trainer.fit(module, data, data)

    # Upstream: shadow starts at the initial weights, then blends after steps
    # 3 and 6 (one "epoch" = 3 steps); the weights after step k are -k.
    assert module.history == [float(-k) for k in range(steps + 1)]
    shadow = 0.0
    for k in (3, 6):
        shadow = decay * shadow + (1 - decay) * module.history[k]
    # Validation (after step 7) sees the EMA weights.
    assert module.validated == [pytest.approx(shadow)]

    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    # Training ended: Lightning copied the EMA into the model, and the
    # checkpoint's state_dict holds the EMA weights.
    assert module.w.item() == pytest.approx(shadow)
    assert checkpoint["state_dict"]["w"].item() == pytest.approx(shadow)
