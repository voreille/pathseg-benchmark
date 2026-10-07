# NCA segmentation: literature notes (2026-10-07)

Collected to choose modelling changes after Med-NCA Variant B (`7ro3wqzo`, val mIoU 0.407,
below the U-Net baseline). The figures below come from the papers; **verify them in the
code** before relying on them for a port.

## OctreeNCA (closest to our setting)

Lemke et al., "OctreeNCA: Single-Pass 184 MP Segmentation on Consumer Hardware",
BMVC 2025, [arXiv 2508.06993](https://arxiv.org/html/2508.06993v1),
code: https://github.com/MECLabTUDA/OctreeNCA (same lab as Med-NCA / M3D-NCA).

- Pathology: PESO (30 H&E prostatectomy WSIs), downsampled to **0.48 µm/px** (ours: 0.5).
  Binary task (epithelium). Dice: OctreeNCA 86.31 (15,520 params), Med-NCA 84.41,
  UNet 82.79 (36.9M), SegFormer 86.31 (3.7M), SAM 41.12.
- Levels: 5 for pathology, factor 2 each: 320 → 160 → 80 → 40 → 20. The coarsest level is
  deliberately ~20×20.
- Image pyramid: an octree "by averaging pixels in each node" (area downscaling, i.e.
  antialiased; Med-NCA uses ×4 bilinear without antialiasing).
- Steps: 10 per level. The coarsest level runs α₀·max(H, W) steps, α₀ = 1.0 (ablated
  1.0/1.5/2.0: little difference). Steps 5/10/20 ablated.
- State upsampling between levels: nearest neighbour (same as Med-NCA).
- Backbone: 16 state channels, hidden size 64, 3×3 kernel, fire rate 0.5. No
  normalisation layer: the ablation (none / batch / layer) favours none.
- Training: full-resolution 320×320 patches (no fine-level crop), patches without
  foreground excluded. Batch 3. Loss (2 − λ)·BCE + λ·Dice, λ = 1. Adam (0.9, 0.99),
  lr 1.6e-3, exponential decay 0.9992, EMA of weights 0.99 (ablated 0 / 0.9 / 0.99).
- More levels help OctreeNCA but made no significant difference for M3D-NCA (their
  Table 4).
- A CUDA inference kernel gives the same results as the PyTorch model (seeded); it is
  only an inference speed/VRAM optimisation.

## Other relevant work

- **Review**: "Applications of Neural Cellular Automata: State of the Art, Challenges and
  Opportunities", [arXiv 2609.24595](https://arxiv.org/html/2609.24595v1) (2026). Most NCA
  segmentation is binary. The review states that no NCA segments more than 10 classes at
  once: IGNITE's 16 classes are beyond published NCA work. Fixes it lists for global
  context: multi-scale cascades (Med-NCA, M3D-NCA, OctreeNCA), positional encoding,
  multiplicative conditioning. For stability: batch duplication, random step counts.
- **M3D-NCA** (Kalkhof & Mukhopadhyay, MICCAI 2023, [arXiv 2309.02954](https://arxiv.org/pdf/2309.02954)):
  n-level patchification, batch duplication, BatchNorm in the update MLP, variance over
  stochastic passes as a quality metric (NQM).
- **FourierDiff-NCA** (Kalkhof et al. 2024, [arXiv 2401.06291](https://arxiv.org/pdf/2401.06291)):
  diffusion; global context via Fourier space; GroupNorm + multiplicative conditioning.
  Generative, not segmentation.
- **ViTCA** (Tesfaldet et al., NeurIPS 2022): localised self-attention NCA. Heavier, less
  "local NCA".
- **NCAtorch** (Spitznagel & Keuper 2026, [arXiv 2604.24990](https://arxiv.org/abs/2604.24990)):
  review + reference library, https://www.neural-cellular-automata.org/.

## Candidate changes for our improved model (after the OctreeNCA comparison)

In order, one ablation each:
1. Octree-style pyramid (area downscaling, ~28 px coarsest level, coarsest steps = side,
   10 steps elsewhere), full-tile training: no crops needed at this step count.
2. A learned 1×1 readout head on the hidden channels (multiclass readout is the gap the
   review names; it also stops raw logits from feeding back into the dynamics), with a
   larger `channel_n`.
3. Training only: EMA of weights, OctreeNCA's Adam settings.
4. Optional: batch duplication, random step counts.

Diagnostics so far: `experiments/mednca/README.md` → "Level ablations at evaluation".
The clean test of "what does the coarse state know" is a probe (1×1 conv trained on the
frozen coarse / fine states).
