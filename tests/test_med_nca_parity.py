"""Parity of the Med-NCA port against the upstream implementation.

Upstream (https://github.com/MECLabTUDA/Med-NCA, MIT) is imported from
``$MEDNCA_REPO`` (default ``/home/valentin/external-repos/Med-NCA``). The tests
are skipped when that checkout is absent; ``pathseg`` itself never imports it.
Only ``src.models`` is imported: the upstream agents pull in torchio, nibabel,
seaborn and cv2, so the agent's multi-level chain is replicated below.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

from pathseg.models.architectures.med_nca import BackboneNCA

MEDNCA_REPO = Path(
    os.environ.get("MEDNCA_REPO", "/home/valentin/external-repos/Med-NCA")
)

INPUT_CHANNELS = 3
CHANNEL_N = 16
HIDDEN_SIZE = 32
FIRE_RATE = 0.5


@pytest.fixture(scope="module")
def upstream_backbone_cls():
    if not MEDNCA_REPO.is_dir():
        pytest.skip(f"Upstream Med-NCA checkout not found at {MEDNCA_REPO}.")

    sys.path.insert(0, str(MEDNCA_REPO))
    try:
        module = importlib.import_module("src.models.Model_BackboneNCA")
    finally:
        sys.path.remove(str(MEDNCA_REPO))
    return module.BackboneNCA


def make_upstream(upstream_backbone_cls, seed: int):
    torch.manual_seed(seed)
    model = upstream_backbone_cls(
        CHANNEL_N,
        FIRE_RATE,
        torch.device("cpu"),
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
    )
    # Upstream BackboneNCA does not forward input_channels to BasicNCA, which
    # then always freezes a single channel. Patch it to the intended value.
    model.input_channels = INPUT_CHANNELS
    # fc1 is zero-initialized; randomize everything so the update is non-trivial.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    return model


def make_port(upstream_model) -> BackboneNCA:
    port = BackboneNCA(
        CHANNEL_N,
        fire_rate=FIRE_RATE,
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
    )
    port.load_state_dict(upstream_model.state_dict())
    return port


def random_state(batch: int, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1234)
    return torch.rand(batch, height, width, CHANNEL_N, generator=generator)


def test_backbone_matches_upstream(upstream_backbone_cls):
    upstream = make_upstream(upstream_backbone_cls, seed=0)
    port = make_port(upstream)
    # Non-square input catches H/W transposition mistakes.
    state = random_state(2, 12, 20)

    torch.manual_seed(7)
    expected = upstream(state, steps=10, fire_rate=FIRE_RATE)
    torch.manual_seed(7)
    actual = port(state, steps=10)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        actual[..., :INPUT_CHANNELS], state[..., :INPUT_CHANNELS], rtol=0.0, atol=0.0
    )


def test_backbone_seeded_init_matches_upstream(upstream_backbone_cls):
    torch.manual_seed(3)
    upstream = upstream_backbone_cls(
        CHANNEL_N,
        FIRE_RATE,
        torch.device("cpu"),
        hidden_size=HIDDEN_SIZE,
        input_channels=INPUT_CHANNELS,
    )
    torch.manual_seed(3)
    port = BackboneNCA(
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
