"""UpstreamDiceBCELoss: Med-NCA's training loss, registered as loss_name "upstream_dice_bce"."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch

from pathseg.training.histo_loss import UpstreamDiceBCELoss
from pathseg.training.semantic_common import SemanticTaskSpec, build_criterion

MEDNCA_REPO = Path(
    os.environ.get("MEDNCA_REPO", "/home/valentin/external-repos/Med-NCA")
)
IGNORE = 255
NUM_CLASSES = 5


def random_logits_and_targets(seed: int = 0, batch: int = 3, size: int = 12):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, NUM_CLASSES, size, size, generator=generator)
    # Class 4 is absent, so the "present classes only" rule is exercised.
    targets = torch.randint(0, NUM_CLASSES - 1, (batch, size, size), generator=generator)
    return logits, targets


def test_matches_upstream_dice_bce():
    if not MEDNCA_REPO.is_dir():
        pytest.skip(f"Upstream Med-NCA checkout not found at {MEDNCA_REPO}.")
    sys.path.insert(0, str(MEDNCA_REPO))
    try:
        losses = importlib.import_module("src.losses.LossFunctions")
    finally:
        sys.path.remove(str(MEDNCA_REPO))

    logits, targets = random_logits_and_targets()
    # Agent_Multi_NCA.batch_step: sum DiceBCE over output channels whose
    # target contains a positive pixel.
    upstream = losses.DiceBCELoss()
    expected = sum(
        upstream(logits[:, k], (targets == k).float())
        for k in range(NUM_CLASSES)
        if (targets == k).any()
    )

    actual = UpstreamDiceBCELoss(IGNORE)(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_leaves_out_ignored_pixels():
    logits, targets = random_logits_and_targets(seed=1)
    targets[0, :6] = IGNORE
    targets[2, :, 8:] = IGNORE
    valid = targets != IGNORE
    loss = UpstreamDiceBCELoss(IGNORE)

    # Same loss on only the valid pixels, packed into a 1 x K x N x 1 batch.
    packed_logits = logits.permute(1, 0, 2, 3)[:, valid].view(1, NUM_CLASSES, -1, 1)
    packed_targets = targets[valid].view(1, -1, 1)

    torch.testing.assert_close(
        loss(logits, targets), loss(packed_logits, packed_targets), rtol=1e-5, atol=1e-6
    )


def test_absent_classes_get_no_gradient():
    logits, targets = random_logits_and_targets(seed=2)
    logits.requires_grad_(True)
    UpstreamDiceBCELoss(IGNORE)(logits, targets).backward()
    assert logits.grad[:, NUM_CLASSES - 1].abs().sum() == 0
    assert logits.grad[:, : NUM_CLASSES - 1].abs().sum() > 0


def test_is_zero_without_labelled_pixels():
    logits, targets = random_logits_and_targets()
    logits.requires_grad_(True)
    loss = UpstreamDiceBCELoss(IGNORE)(logits, torch.full_like(targets, IGNORE))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad.abs().sum() == 0


def test_registered_in_build_criterion():
    criterion = build_criterion(
        SemanticTaskSpec(loss_name="upstream_dice_bce"),
        num_classes=NUM_CLASSES,
        ignore_idx=IGNORE,
    )
    assert isinstance(criterion, UpstreamDiceBCELoss)
    assert criterion.ignore_index == IGNORE

    with pytest.raises(ValueError, match="class weights"):
        build_criterion(
            SemanticTaskSpec(
                loss_name="upstream_dice_bce", class_weights=(1.0,) * NUM_CLASSES
            ),
            num_classes=NUM_CLASSES,
            ignore_idx=IGNORE,
        )
