import pytest
import torch
import torch.nn as nn

from pathseg.models.decoders.linear import LinearSemanticDecoder
from pathseg.models.sae_semantic_segmenter import SAESemanticSegmenter
from pathseg.models.semantic_segmenter import SemanticSegmenter


class DummyEncoder(nn.Module):
    embed_dim = 8

    def forward_feature_maps(self, imgs):
        batch_size = imgs.shape[0]
        return (torch.ones(batch_size, self.embed_dim, 4, 3, device=imgs.device),)


class IdentitySAE(nn.Module):
    def forward_with_aux(self, tokens):
        return {
            "reconstructed_tokens": tokens,
            "latents": tokens,
        }


def test_linear_decoder_single_selected_task():
    decoder = LinearSemanticDecoder(
        in_channels=8,
        num_classes_by_task={"a": 3, "b": 5},
    )
    output = decoder((torch.randn(2, 8, 4, 3),), task="b")

    assert set(output) == {"b"}
    assert output["b"].shape == (2, 5, 4, 3)


def test_semantic_segmenter_two_heads():
    network = SemanticSegmenter(
        encoder=DummyEncoder(),
        decoder=LinearSemanticDecoder(
            in_channels=8,
            num_classes_by_task={"a": 3, "b": 5},
        ),
        upsample_logits=True,
    )
    output = network(torch.randn(2, 3, 56, 42))

    assert output["a"].shape == (2, 3, 56, 42)
    assert output["b"].shape == (2, 5, 56, 42)


def test_segmenter_crashes_on_implicit_resize():
    network = SemanticSegmenter(
        encoder=DummyEncoder(),
        decoder=LinearSemanticDecoder(
            in_channels=8,
            num_classes_by_task={"semantic": 3},
        ),
        upsample_logits=False,
    )

    with pytest.raises(ValueError, match="upsample_logits=True"):
        network(torch.randn(2, 3, 56, 42))


def test_identity_sae_preserves_semantic_logits():
    encoder = DummyEncoder()
    decoder = LinearSemanticDecoder(
        in_channels=8,
        num_classes_by_task={"semantic": 3},
    )
    baseline = SemanticSegmenter(
        encoder=encoder,
        decoder=decoder,
        upsample_logits=True,
    )
    with_sae = SAESemanticSegmenter(
        encoder=encoder,
        decoder=decoder,
        sae=IdentitySAE(),
        upsample_logits=True,
    )
    imgs = torch.randn(2, 3, 56, 42)

    expected = baseline(imgs)["semantic"]
    output = with_sae.forward_with_aux(
        imgs,
        include_original_logits=True,
    )

    torch.testing.assert_close(output["logits"]["semantic"], expected)
    torch.testing.assert_close(output["original_logits"]["semantic"], expected)
