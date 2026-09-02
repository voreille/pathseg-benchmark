from pathseg.models.semantic_segmenter import SemanticSegmenter


def map_to_tokens(feature_map):
    return feature_map.flatten(2).transpose(1, 2)


def tokens_to_map(tokens, reference_map):
    batch_size, _, embed_dim = tokens.shape
    height, width = reference_map.shape[-2:]

    return tokens.transpose(1, 2).reshape(
        batch_size,
        embed_dim,
        height,
        width,
    )


class SAESemanticSegmenter(SemanticSegmenter):
    def feature_maps_from_latents(
        self,
        latents,
        feature_maps,
    ):
        final_map = feature_maps[-1]

        reconstructed_map = tokens_to_map(
            self.sae.decode(latents),
            final_map,
        )

        return (
            *feature_maps[:-1],
            reconstructed_map,
        )

    def decode_latents(
        self,
        latents,
        feature_maps,
        task=None,
    ):
        reconstructed_feature_maps = self.feature_maps_from_latents(
            latents,
            feature_maps,
        )

        return self.decode(
            reconstructed_feature_maps,
            task=task,
        )

    def forward_sae(self, imgs):
        feature_maps = self.encode(imgs)
        tokens = map_to_tokens(feature_maps[-1])
        sae_output = self.sae.forward_with_aux(tokens)

        return {
            **sae_output,
            "feature_maps": feature_maps,
        }

    def forward(self, imgs, task=None):
        output = self.forward_sae(imgs)

        return self.decode_latents(
            output["latents"],
            output["feature_maps"],
            task=task,
        )