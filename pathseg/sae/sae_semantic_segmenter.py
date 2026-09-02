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
    def forward_sae(self, imgs):
        feature_maps = self.encode(imgs)
        final_map = feature_maps[-1]

        tokens = map_to_tokens(final_map)
        sae_output = self.sae.forward_with_aux(tokens)

        reconstructed_map = tokens_to_map(
            sae_output["reconstructed_tokens"],
            final_map,
        )

        reconstructed_feature_maps = (
            *feature_maps[:-1],
            reconstructed_map,
        )

        return {
            **sae_output,
            "feature_maps": feature_maps,
            "reconstructed_feature_maps": reconstructed_feature_maps,
        }

    def forward(self, imgs, task=None):
        output = self.forward_sae(imgs)

        return self.decode(
            output["reconstructed_feature_maps"],
            task=task,
        )
