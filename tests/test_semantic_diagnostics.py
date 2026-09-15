from types import SimpleNamespace

import pytest
import torch

from pathseg.training.semantic_common import SemanticLightningModule


class PlotHarness:
    task_specs = {
        "ignite": SimpleNamespace(num_classes=3),
        "anorak": SimpleNamespace(num_classes=2),
    }
    ignore_idx = 255


def test_all_head_plot_has_one_gt_and_every_prediction():
    harness = PlotHarness()
    image = torch.rand(3, 8, 6)
    target = torch.randint(0, 3, (8, 6))
    logits = {
        "ignite": torch.randn(3, 8, 6),
        "anorak": torch.randn(2, 8, 6),
    }

    plot = SemanticLightningModule.plot_semantic_heads(
        harness,
        image,
        target,
        logits,
        target_task="ignite",
    )

    assert plot.width > 0
    assert plot.height > 0


def test_all_head_plot_crashes_when_a_head_is_missing():
    harness = PlotHarness()

    with pytest.raises(ValueError, match="missing=.*anorak"):
        SemanticLightningModule.plot_semantic_heads(
            harness,
            torch.rand(3, 8, 6),
            torch.zeros(8, 6, dtype=torch.long),
            {"ignite": torch.randn(3, 8, 6)},
            target_task="ignite",
        )
