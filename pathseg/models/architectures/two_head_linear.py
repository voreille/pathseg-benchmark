from __future__ import annotations

from pathseg.models.decoders import LinearDecoder, MultiTaskDecoder
from pathseg.models.encoder import Encoder
from pathseg.models.semantic_segmenter import SemanticSegmenter


class IgniteAnorakLinearSegmenter(SemanticSegmenter):
    """Shared encoder with linear semantic heads for IGNITE and ANORAK."""

    def __init__(
        self,
        encoder_id: str,
        ignite_num_classes: int,
        anorak_num_classes: int,
        *,
        img_size: tuple[int, int] = (448, 448),
        ckpt_path: str = "",
        sub_norm: bool = False,
        discard_last_mlp: bool = False,
        discard_last_block: bool = False,
        random_weights: bool = False,
        upsample_logits: bool = False,
        interpolation_mode: str = "bilinear",
    ) -> None:
        encoder = Encoder(
            encoder_id=encoder_id,
            img_size=img_size,
            ckpt_path=ckpt_path,
            sub_norm=sub_norm,
            discard_last_mlp=discard_last_mlp,
            discard_last_block=discard_last_block,
            random_weights=random_weights,
        )

        decoder = MultiTaskDecoder(
            heads={
                "ignite": LinearDecoder(
                    in_channels=encoder.embed_dim,
                    out_channels=ignite_num_classes,
                ),
                "anorak": LinearDecoder(
                    in_channels=encoder.embed_dim,
                    out_channels=anorak_num_classes,
                ),
            }
        )

        super().__init__(
            encoder=encoder,
            decoder=decoder,
            upsample_logits=upsample_logits,
            interpolation_mode=interpolation_mode,
        )
