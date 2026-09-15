from copy import deepcopy

import pytest

from pathseg.training.sae_launcher import (
    SAEConfigError,
    compose_sae_cli_config,
)


def semantic_config():
    return {
        "seed_everything": 0,
        "trainer": {"max_steps": 100},
        "model": {
            "class_path": "pathseg.training.semantic.SemanticTraining",
            "init_args": {
                "encoder_class_path": "example.Encoder",
                "encoder_init_args": {"encoder_id": "h0-mini"},
                "decoder_name": "linear",
                "decoder_init_args": {"bias": True},
                "tasks": {
                    "ignite": {"num_classes": 16, "source_id": 0},
                    "anorak": {"num_classes": 7, "source_id": 1},
                },
                "eval_task_names": ["ignite", "anorak"],
                "tiler_name": "grid_pad_tiler",
                "tiler_init_args": {"tile": 896},
                "upsample_logits": True,
                "interpolation_mode": "bilinear",
            },
        },
        "data": {
            "class_path": "example.MultiTaskDataModule",
            "init_args": {
                "datasets": [{"name": "IGNITE"}, {"name": "ANORAK"}],
                "img_size": [896, 896],
                "ignore_idx": 255,
            },
        },
    }


def minimal_sae_config():
    return {
        "semantic_run": {
            "config_path": "/runs/semantic/config.yaml",
            "checkpoint_path": "/runs/semantic/best.ckpt",
        },
        "trainer": {
            "precision": "16-mixed",
            "max_steps": 40_000,
        },
        "sae": {
            "class_path": "example.TopKSAE",
            "init_args": {
                "input_dim": 768,
                "num_latents": 3072,
                "k": 32,
            },
            "lr": 3e-4,
            "weight_decay": 0.0,
            "normalize_decoder": True,
        },
    }


def test_compose_sae_config_reuses_semantic_model_and_data():
    base = semantic_config()
    sae = minimal_sae_config()
    base_before = deepcopy(base)

    resolved = compose_sae_cli_config(sae, base)
    model_args = resolved["model"]["init_args"]

    assert resolved["data"] == base["data"]
    assert resolved["trainer"] == sae["trainer"]
    assert resolved["trainer"] != base["trainer"]
    assert model_args["encoder_class_path"] == "example.Encoder"
    assert model_args["tasks"] == base["model"]["init_args"]["tasks"]
    assert model_args["img_size"] == [896, 896]
    assert model_args["ignore_idx"] == 255
    assert model_args["upsample_logits"] is True
    assert model_args["semantic_init_checkpoint_path"].endswith("best.ckpt")
    assert model_args["sae_class_path"] == "example.TopKSAE"
    assert base == base_before


def test_compose_requires_linked_img_size():
    base = semantic_config()
    del base["data"]["init_args"]["img_size"]

    with pytest.raises(SAEConfigError, match="img_size"):
        compose_sae_cli_config(minimal_sae_config(), base)
