from typing import Optional
import torch
from torch.nn.functional import normalize, one_hot, pad
from tqdm import tqdm


def _infer_device_from_encoder(encoder: torch.nn.Module) -> torch.device:
    for p in encoder.parameters():
        return p.device
    for b in encoder.buffers():
        return b.device
    # Fallback if encoder has no params/buffers (rare)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProtoNetLayer(torch.nn.Module):
    """
    Sklearn-like class for SimpleShot.

    Implementation adapted from: https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py#L9C3-L9C3
    """

    def __init__(self,
                 metric: str = 'L2',
                 center_feats: bool = True,
                 normalize_feats: bool = True,
                 num_classes: int = 7,
                 patch_size: int = 14,
                 img_size: int = 224) -> None:

        super().__init__()
        self.metric = metric
        self.center_feats = center_feats
        self.normalize_feats = normalize_feats

        self.register_buffer("prototype_embeddings_",
                             torch.empty(0, 0),
                             persistent=True)
        self.register_buffer("mean_", torch.empty(0),
                             persistent=True)  # [1 x D] or empty
        self.register_buffer("support_counts_",
                             torch.empty(0, dtype=torch.long),
                             persistent=True)

        self.num_classes = num_classes
        self.patch_size = patch_size
        self.img_size = img_size

        self.pool_target = torch.nn.AvgPool2d(kernel_size=self.patch_size,
                                              stride=self.patch_size)

    @torch.no_grad()
    def fit(
        self,
        support_dataloader: torch.utils.data.DataLoader,
        encoder: torch.nn.Module,
    ) -> None:
        """
        support_dataloader must output X and y, where y is a one-hot encoding so NxC
        and X is a feature matrix of shape NxD
        """

        device = _infer_device_from_encoder(encoder)
        self.to(device)

        self.fit_mean(support_dataloader, encoder, device)
        self.fit_prototype(support_dataloader, encoder, device)

    def fit_mean(self, support_dataloader: torch.utils.data.DataLoader,
                 encoder: torch.nn.Module, device: torch.device) -> None:

        running_sum: torch.Tensor = torch.empty(0, device=device)
        D = None
        total = 0

        for batch in tqdm(support_dataloader, desc="Fitting mean"):
            imgs, targets = batch
            targets = self.to_per_pixel_targets_semantic(targets)

            for img, target in zip(imgs, targets):
                crops = self.tile_image(img, padding_value=0)
                target_crops = self.tile_image(target)
                n_crops = crops.shape[0]
                target_crops = self.pool_target(
                    target_crops)  # (n_crops, n_class, h, w)
                mask = target_crops.any(dim=1).view(n_crops,
                                                    -1)  # (n_crops, n_tokens)

                with torch.inference_mode():
                    X = encoder(crops.to(device))  # (n_crops, n_tokens, D)

                if D is None:
                    D = X.shape[2]
                    running_sum = torch.zeros(D, device=device)

                X = X.reshape(-1, D)
                mask = mask.reshape(-1)  # (n_crops * n_tokens,)
                X = X[mask]  # (n_valid, D)

                running_sum += X.sum(dim=0)
                total += X.shape[0]

        mu = running_sum / total
        self.mean_ = mu[None, :]  # [1 x D]

    def fit_prototype(
        self,
        support_dataloader: torch.utils.data.DataLoader,
        encoder: torch.nn.Module,
        device: torch.device,
    ) -> None:

        running_sum: Optional[torch.Tensor] = None
        total = 0
        D = self.mean_.shape[-1]
        for batch in tqdm(support_dataloader, desc="Fitting prototype"):
            imgs, targets = batch
            targets = self.to_per_pixel_targets_semantic(targets)

            for img, target in zip(imgs, targets):
                crops, target_crops = self.tile_image(img, target)
                n_crops = crops.shape[0]
                target_crops = self.pool_target(
                    target_crops)  # (n_crops, n_class, h, w)
                mask = target_crops.any(dim=1).view(n_crops,
                                                    -1)  # (n_crops, n_tokens)
                target_crops = target_crops.view(n_crops, self.num_classes, -1)
                target_crops = torch.transpose(
                    target_crops, 1, 2)  # (n_crops, n_tokens, num_classes)
                target_crops = target_crops.reshape(
                    -1, self.num_classes)  # (n_crops * n_tokens, num_classes)

                with torch.inference_mode():
                    X = encoder(crops.to(device))

                if running_sum is None:
                    running_sum = torch.zeros(
                        self.num_classes,
                        D,
                        device=device,
                    )

                X = X.reshape(-1, D)
                mask = mask.reshape(-1)  # (n_crops * n_tokens,)
                X = X[mask]  # (n_valid, D)
                target_crops = target_crops[mask]  # (n_valid, num_classes)
                target_crops = target_crops.to(device)

                running_sum = self.aggregate_prototypes(
                    X, target_crops, running_sum)
                total += X.shape[0]

        self.prototype_embeddings_ = running_sum / total

    def aggregate_prototypes(self, X: torch.Tensor, y: torch.Tensor,
                             running_sum: torch.Tensor) -> torch.Tensor:

        ### Apply centering and normalization (if set)
        if self.center_feats:
            X = X - self.mean_

        if self.normalize_feats:
            X = normalize(X, dim=-1, p=2)

        running_sum += y.T @ X
        return running_sum

    @torch.compiler.disable
    def tile_image(
        self,
        img: torch.Tensor,
        padding_value: int = 0,
    ) -> torch.Tensor:
        """
        img:     (C, H, W)
        target:  (num_classes, H, W)  -- e.g., one-hot planes
        returns:
          crops_img:    (N, C, Th, Tw)
          crops_target: (N, num_classes, Th, Tw)
        """
        C, H, W = img.shape
        # minimal pad (right, bottom) to make divisible by (Th, Tw)
        pad_h = (-H) % self.img_size
        pad_w = (-W) % self.img_size

        img_b = img.unsqueeze(0)  # (1,C,H,W)

        # pad=(left, right, top, bottom) for 2D in F.pad
        if pad_h or pad_w:
            img_b = pad(img_b, (0, pad_w, 0, pad_h), mode="constant", value=padding_value)

        img_tiles = (img_b.unfold(2, self.img_size, self.img_size).unfold(
            3, self.img_size,
            self.img_size).permute(0, 2, 3, 1, 4,
                                   5).reshape(-1, C, self.img_size,
                                              self.img_size))

        return img_tiles

    def predict(self, imgs: list[torch.Tensor],
                encoder: torch.nn.Module) -> torch.Tensor:
        """

        """

        ### Apply centering and normalization (if set)
        for img in imgs:

            X = encoder(img)
            if self.center_feats:
                X = X - self.mean_

        if self.normalize_feats:
            X = normalize(X, dim=-1, p=2)

        ### Compute distances, and get the closest prototype for each query as the label
        X = X[:, None]  # [N x 1 x D]
        prototype_embeddings = self.prototype_embeddings[
            None, :]  # [1 x C x D]
        pw_dist = (X - prototype_embeddings).norm(
            dim=-1, p=2)  # [N x C x D] distance/sim matrix
        labels_pred = self.prototype_labels[pw_dist.min(
            dim=1).indices]  # [N,] label vector
        return labels_pred

    def to_per_pixel_targets_semantic(self, targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            h, w = target["masks"].shape[-2:]
            per_pixel_target = torch.zeros(
                (self.num_classes, h, w),
                dtype=torch.float,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                class_id = target["labels"][i].item()
                per_pixel_target[class_id] += mask.float()

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def to_per_patch_target_semantic(
        self,
        targets: list[dict],
    ) -> list[torch.Tensor]:

        per_pixel_targets = self.to_per_pixel_targets_semantic(targets)
        per_patch_targets = []
        for per_pixel_target in per_pixel_targets:
            per_patch_target = self.pool_target(
                per_pixel_target[None, ...]).squeeze(0)
            per_patch_targets.append(per_patch_target)
        return per_patch_targets
