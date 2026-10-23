"""Parity of the OctreeNCA port against the upstream implementation.

Upstream (https://github.com/MECLabTUDA/OctreeNCA) is imported from
``$OCTREENCA_REPO`` (default ``/home/valentin/external-repos/OctreeNCA``). The
tests are skipped when that checkout (or ``einops``, which the upstream backbone
imports) is absent; ``pathseg`` itself never imports it. Only the backbone
module is imported: ``Model_OctreeNCA_2d_patching2`` pulls in torchio,
matplotlib and the ViTCA code, so its multi-level chain (``forward_train`` with
``patch_sizes = None``, which is also ``forward_eval``) is replicated below.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pathseg.models.architectures.octree_nca import (
    OctreeBackboneNCA,
    OctreeNCAEncoder,
)

OCTREENCA_REPO = Path(
    os.environ.get("OCTREENCA_REPO", "/home/valentin/external-repos/OctreeNCA")
)

INPUT_CHANNELS = 3
CHANNEL_N = 16
HIDDEN_SIZE = 32
FIRE_RATE = 0.5


@pytest.fixture(scope="module")
def upstream_backbone_cls():
    if not OCTREENCA_REPO.is_dir():
        pytest.skip(f"Upstream OctreeNCA checkout not found at {OCTREENCA_REPO}.")
    pytest.importorskip("einops")

    # Upstream Med-NCA's checkout also has a top-level ``src`` package; set any
    # cached one aside so ``src`` resolves to OctreeNCA here, and restore it.
    cached = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "src" or name.startswith("src.")
    }
    sys.path.insert(0, str(OCTREENCA_REPO))
    try:
        module = importlib.import_module("src.models.Model_BasicNCA2D_fast")
    finally:
        sys.path.remove(str(OCTREENCA_REPO))
        for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
            del sys.modules[name]
        sys.modules.update(cached)
    return module.BasicNCA2DFast


def make_upstream(upstream_backbone_cls, seed: int):
    torch.manual_seed(seed)
    model = upstream_backbone_cls(
        CHANNEL_N,
        FIRE_RATE,
        torch.device("cpu"),
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
        kernel_size=3,
        normalization="none",
    )
    # fc1 is zero-initialized; randomize everything so the update is non-trivial.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    return model


def random_state(batch: int, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.rand(batch, height, width, CHANNEL_N, generator=generator)


def test_backbone_seeded_init_matches_upstream(upstream_backbone_cls):
    torch.manual_seed(3)
    upstream = upstream_backbone_cls(
        CHANNEL_N,
        FIRE_RATE,
        torch.device("cpu"),
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
        kernel_size=3,
        normalization="none",
    )
    torch.manual_seed(3)
    port = OctreeBackboneNCA(
        CHANNEL_N,
        fire_rate=FIRE_RATE,
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
    )

    upstream_state = upstream.state_dict()
    port_state = port.state_dict()
    assert upstream_state.keys() == port_state.keys()
    for name, value in upstream_state.items():
        torch.testing.assert_close(port_state[name], value, rtol=0.0, atol=0.0)


def test_backbone_matches_upstream(upstream_backbone_cls):
    upstream = make_upstream(upstream_backbone_cls, seed=0)
    port = OctreeBackboneNCA(
        CHANNEL_N,
        fire_rate=FIRE_RATE,
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
    )
    port.load_state_dict(upstream.state_dict())
    # Non-square input catches H/W transposition mistakes.
    state = random_state(2, 12, 20)

    torch.manual_seed(7)
    expected = upstream(state, steps=10, fire_rate=FIRE_RATE)
    torch.manual_seed(7)
    actual = port(state, steps=10)

    # Float tolerance: the port folds conv + fc0 into one conv, so the
    # summation order (not the function) differs from upstream.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        actual[..., :INPUT_CHANNELS], state[..., :INPUT_CHANNELS], rtol=0.0, atol=0.0
    )


def upstream_octree_forward(models, imgs, octree_res, steps) -> torch.Tensor:
    """Upstream ``OctreeNCA2DPatch2.forward_train`` with ``patch_sizes = [None] * L``.

    ``models[l]`` is ``backbone_ncas[l]`` (0 = finest), ``octree_res[l]`` its
    resolution and ``steps[l]`` its step count. ``imgs``: ``B x C x H x W``.
    Returns the channels-last level-0 state.
    """
    original = imgs
    coarsest = len(models) - 1
    state = torch.zeros(imgs.shape[0], *octree_res[-1], CHANNEL_N)
    small = F.interpolate(original, size=octree_res[-1])
    state[..., :INPUT_CHANNELS] = small.permute(0, 2, 3, 1)

    for level in range(coarsest, -1, -1):
        state = models[level](state, steps=steps[level], fire_rate=FIRE_RATE)
        if level > 0:
            scale = tuple(
                octree_res[level - 1][c] // octree_res[level][c] for c in range(2)
            )
            x = torch.nn.Upsample(scale_factor=scale, mode="nearest")(
                state.permute(0, 3, 1, 2)
            )
            x[:, :INPUT_CHANNELS] = F.interpolate(original, size=octree_res[level - 1])
            state = x.permute(0, 2, 3, 1)
    return state


@pytest.mark.parametrize("train", [True, False])
def test_multi_level_forward_matches_upstream(upstream_backbone_cls, train: bool):
    steps = (3, 4, 2, 5, 6)
    upstream_models = [
        make_upstream(upstream_backbone_cls, seed=level) for level in range(len(steps))
    ]
    encoder = OctreeNCAEncoder(
        output_channels=2,
        channel_n=CHANNEL_N,
        hidden_size=HIDDEN_SIZE,
        steps=steps,
        fire_rate=FIRE_RATE,
        input_channels=INPUT_CHANNELS,
    )
    for level, model in enumerate(upstream_models):
        encoder.backbone_ncas[level].load_state_dict(model.state_dict())
    encoder.train(train)

    imgs = torch.rand(2, INPUT_CHANNELS, 64, 96, generator=torch.Generator().manual_seed(5))
    octree_res = encoder.level_sizes(64, 96)
    assert octree_res == [(64, 96), (32, 48), (16, 24), (8, 12), (4, 6)]

    torch.manual_seed(11)
    with torch.no_grad():
        expected = upstream_octree_forward(upstream_models, imgs, octree_res, steps)
    torch.manual_seed(11)
    with torch.no_grad():
        (actual,) = encoder.forward_feature_maps(imgs)

    torch.testing.assert_close(actual, expected.permute(0, 3, 1, 2), rtol=1e-5, atol=1e-5)


def test_upstream_state_dict_loads_into_encoder(upstream_backbone_cls):
    upstream_models = [make_upstream(upstream_backbone_cls, seed=s) for s in range(5)]
    state_dict = torch.nn.ModuleList(upstream_models).state_dict()
    encoder = OctreeNCAEncoder(
        output_channels=1,
        channel_n=CHANNEL_N,
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
    )
    # Upstream stores the levels as OctreeNCA2DPatch2.backbone_ncas.
    encoder.backbone_ncas.load_state_dict(state_dict)
