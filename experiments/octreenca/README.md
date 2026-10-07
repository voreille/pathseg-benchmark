# OctreeNCA baseline

OctreeNCA (Lemke et al., "OctreeNCA: Single-Pass 184 MP Segmentation on Consumer
Hardware", BMVC 2025, [arXiv 2508.06993](https://arxiv.org/abs/2508.06993)) ported as a
`SemanticSegmenter` for the pathseg benchmark, as a second NCA baseline next to Med-NCA
(`experiments/mednca/README.md`). Same lab as Med-NCA.

The port mirrors the Med-NCA port wherever possible (user decision, 2026-10-07): same
config layout and loss switch (`tasks.ignite.loss_name`), same data, tiles, batch,
precision, step budget and model knobs as Med-NCA Variant B. Only what defines OctreeNCA
(backbone, levels, steps, hidden size, optimizer, EMA) comes from upstream.

## Upstream reference

- Repo: https://github.com/MECLabTUDA/OctreeNCA
- Commit: `70f774232305b76330a75edef9fddda8ec960cab`
- Local clone (read-only, never imported by `pathseg`): `/home/valentin/external-repos/OctreeNCA`
  (override with `OCTREENCA_REPO` for the parity test).
- The pathology run is `train_peso.py`. It merges `configs/models/peso.py`,
  `configs/trainers/nca.py`, `configs/datasets/peso.py`, `configs/tasks/segmentation.py`
  and `configs/default.py`, then overrides some keys. It builds the model with
  `src/utils/BaselineConfigs.py::EXP_OctreeNCA` and trains it with
  `src/agents/Agent_MedNCA_Simple.py::MedNCAAgent` (→ `Agent_UNet` → `Agent`).

### What upstream does (read before porting)

| Item | Upstream (PESO run) |
|---|---|
| Model class | `src/models/Model_OctreeNCA_2d_patching2.py::OctreeNCA2DPatch2` (`EXP_OctreeNCA` uses it, not `Model_OctreeNCA.py`) |
| Backbone class | `src/models/Model_BasicNCA2D_fast.py::BasicNCA2DFast` (`model.backbone_class`). One per level (`model.octree.separate_models: True`), stored in `backbone_ncas` (`nn.ModuleList`, index 0 = finest) |
| Perception | `cat(state, conv(state))` with `conv = Conv2d(C, C, 3, padding=1, padding_mode="reflect", groups=C)`: **one depthwise** 3×3 conv with bias (`model.kernel_size: [3]*5`) |
| Update rule | `fc0: Conv2d(2C, hidden, 1)` → `bn` → ReLU → `fc1: Conv2d(hidden, C − in, 1, bias=False)` (zero-initialised). `bn` is `Identity`, since `train_peso.py` sets `model.normalization: "none"` (the code default is BatchNorm). New state = `cat(image, state[:, in:] + dx · mask)` |
| Fire mask | `zeros(B, 1, H, W, device).bernoulli_(p=self.fire_rate)`: a cell **updates with probability `fire_rate`**. The opposite convention to Med-NCA (`rand > fire_rate`); the two are equal at 0.5. Drawn on the state's device, redrawn every step, shared by all channels. The `fire_rate` argument of `update`/`forward` is ignored. |
| Layout | The level's state is channels-last `B×H×W×C` between levels. `BasicNCA2DFast.forward` rearranges it to BCHW and runs the convs there, **without** Med-NCA's H/W transpose |
| Frozen input | The image channels are concatenated back after every step (`const_inputs`) |
| State channels | `[input | output | hidden]`. Logits = `state[..., in : in + out]` |
| Octree / pyramid | `model.octree.res_and_steps`. PESO: 5 levels, 320 → 160 → 80 → 40 → 20 (×2 per level). The image for level *l* is `F.interpolate(original, size=res_l)` with the **default mode, i.e. nearest**, always from the full-resolution image |
| Steps per level | `train_peso.py`: `steps = 10` for levels 0–3 and `int(alpha * 20)` with `alpha = 1.0` for the 20×20 coarsest level, i.e. 20. That is the paper's α₀·max(H, W) rule, hard-coded for 20 px |
| Seed | Zeros at the coarsest resolution, image in channels `0:in` |
| Between levels | `nn.Upsample(scale_factor=2, mode="nearest")` of the whole state, then channels `0:in` are overwritten by the next level's image. Output and hidden channels are carried over |
| `channel_n` / `hidden_size` | 16 (= 3 in + 1 out + **12 hidden**) / 64 (update-MLP width) |
| `fire_rate` | 0.5 |
| Parameters | 3,104 per level, 15,520 in total, which matches the paper |
| Inference path | `forward_eval`: the training path with `patch_sizes = None`, under `no_grad`, one stochastic pass, same fire rate. For an input of another size, the level sizes become `ceil(prev / 2)`. `MedNCAAgent.test` evaluates the EMA weights (`ema.apply_shadow`) |
| Training patches | `model.train.patch_sizes: [None] * 5`, so **every level runs on the whole 320 px patch; there is no crop**. `loss_weighted_patching: False` |
| Loss | `DiceLoss` + `BCELoss`, weights 1 / 1 (`dice_loss_weight = 1.0`). Both are sigmoid, `smooth = 1` for Dice, computed over the flattened batch, **per output channel whose target has a positive pixel**, summed. That is the same function as Med-NCA's `DiceBCELoss` and our `UpstreamDiceBCELoss` |
| Optimizer | `Adam(lr=1.6e-3, betas=(0.9, 0.99))`, no weight decay, over all levels at once |
| LR schedule | `ExponentialLR(γ = 0.9999⁸)`, stepped **once per epoch** (`trainer.update_lr_per_epoch: True`) |
| EMA | `src/utils/EMA.py`, decay 0.99, updated **once per epoch** (`trainer.ema.update_per: "epoch"`). The shadow is initialised from the initial weights. Evaluation swaps the shadow weights in |
| Batch / epoch | Batch 3, 200 steps per epoch (`trainer.num_steps_per_epoch`), 2000 epochs (400k steps, 1.2M samples), `batch_duplication` 1 |
| Gradients | No clipping or normalisation (`trainer.normalize_gradients: None`); the norm is only logged |
| Data | PESO 320 px patches at `img_level` 1 (0.48 µm/px), patches with epithelium only. Input `/255` then **ImageNet mean/std**. batchgenerators augmentations (`get_transform_arr`) |

### Upstream discrepancies noticed

- **Pyramid downscaling.** The paper builds the octree "by averaging pixels in each node",
  i.e. area downscaling (also in `experiments/nca_literature.md`). The code uses
  `F.interpolate` with the default `mode="nearest"`, for the seed and for every
  re-injected image. The code wins: the port uses nearest.
- **Steps rule.** The paper states α₀·max(H, W) for the coarsest level. The code
  hard-codes `int(alpha * 20)`, which equals that rule only for 20 px. For our 448 px
  tiles the coarsest level is 28 px, so the rule gives 28 steps.
- **`fire_rate` argument ignored.** `BasicNCA2DFast.update` uses `self.fire_rate` for the
  mask, whatever `fire_rate` it receives. They are equal in all upstream configs.
- **Whole-slide demo script.** `inference2_peso_oct.py` (the 184 MP demo) runs level 2
  twice and skips the separate upsampling to level 2's size. The model's own
  `forward_eval`, which the training and evaluation code use, does not, and it is what
  we port.
- **Inference at other sizes** uses `ceil(prev / 2)` sizes but upsamples by a fixed ×2,
  which only fits for sizes divisible by 2⁴. Our tiles (448) are, so this does not arise;
  the port upsamples to the next level's size explicitly.

## Deviations from upstream

_Filled in with the port._
