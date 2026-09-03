"""Post-training analysis utilities for semantic sparse autoencoders.

The public entry point is :func:`analyze_sae` and accepts an already constructed
semantic segmenter and data module. The package CLI additionally reconstructs
those objects from minimal SAE and semantic configuration files.
"""

from pathseg.sae.analysis.attribution import (
    TaskAttribution,
    compute_task_attributions,
    decoder_column_norms,
    head_latent_alignment,
    resolve_task_heads,
    select_relevant_latents,
)
from pathseg.sae.analysis.collector import (
    StreamingSAECollector,
    TopActivationCollector,
)
from pathseg.sae.analysis.interventions import (
    ablate_latents,
    ablate_reconstructed_tokens,
    decode_latents,
    decode_reconstructed_tokens,
    linear_logit_delta,
)
from pathseg.sae.analysis.latent_selection import (
    LatentSpec,
    load_latent_manifest,
    parse_latent_spec,
)
from pathseg.sae.analysis.runner import analyze_sae
from pathseg.sae.analysis.types import (
    AnalysisResult,
    TaskSpec,
    TopActivation,
)

__all__ = [
    "AnalysisResult",
    "LatentSpec",
    "StreamingSAECollector",
    "TaskAttribution",
    "TaskSpec",
    "TopActivation",
    "TopActivationCollector",
    "ablate_reconstructed_tokens",
    "ablate_latents",
    "analyze_sae",
    "compute_task_attributions",
    "decode_latents",
    "decode_reconstructed_tokens",
    "decoder_column_norms",
    "head_latent_alignment",
    "linear_logit_delta",
    "load_latent_manifest",
    "parse_latent_spec",
    "resolve_task_heads",
    "select_relevant_latents",
]
