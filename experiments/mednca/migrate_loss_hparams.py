"""Migrate Variant B checkpoints saved with the removed ``MedNCATraining(loss=...)`` arg.

``MedNCATraining`` used to pick its training loss with a ``loss`` init arg
(``"upstream_dice_bce"`` or ``"benchmark"``) that bypassed ``tasks.<task>.loss_name``.
The loss now comes from ``loss_name`` only. ``LightningCLI`` applies the hyperparameters
stored in ``--ckpt_path`` on ``validate``/``test``, so an old checkpoint fails there on the
unknown ``loss`` key.

For each checkpoint this writes ``<name>.migrated.ckpt`` next to it (the original is not
modified), with ``loss`` removed from ``hyper_parameters`` and, if it was
``"upstream_dice_bce"``, every task's ``loss_name`` set to ``"upstream_dice_bce"`` (what
the run trained with). ``"benchmark"`` already meant the configured ``loss_name``.
Weights, optimizer and scheduler states are copied unchanged.

Example:

    python experiments/mednca/migrate_loss_hparams.py runs/checkpoints/7ro3wqzo/*.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def migrate(path: Path) -> Path | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if "loss" not in hparams:
        print(f"{path}: no `loss` hparam, nothing to migrate")
        return None

    loss = hparams.pop("loss")
    if loss == "upstream_dice_bce":
        for task_config in hparams["tasks"].values():
            task_config["loss_name"] = "upstream_dice_bce"
    elif loss != "benchmark":
        raise ValueError(f"{path}: unknown `loss` hparam {loss!r}")

    out = path.with_name(f"{path.stem}.migrated.ckpt")
    torch.save(checkpoint, out)
    loss_names = {task: c["loss_name"] for task, c in hparams["tasks"].items()}
    print(f"{path} -> {out}: loss={loss!r} -> loss_name {loss_names}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.checkpoints:
        if path.name.endswith(".migrated.ckpt"):
            continue
        migrate(path)


if __name__ == "__main__":
    main()
