import torch
from torch import nn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode


class CustomTransformsExtended(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        scale_range: tuple[float, float],
        max_brightness_delta: float = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: float = 18,
        # new augmentation knobs:
        p_vflip: float = 0.5,
        p_rot90: float = 0.75,
        p_blur: float = 0.3,
        p_noise: float = 0.3,
        blur_kernel_size: int = 3,
        blur_sigma_min: float = 0.1,
        blur_sigma_max: float = 1.0,
        noise_sigma: float = 0.03,
    ):
        super().__init__()

        self.img_size = img_size

        # color jitter params
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.scale_range = scale_range

        # flips
        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.random_vertical_flip = T.RandomVerticalFlip(p=p_vflip)

        # final random crop to output size
        self.random_crop = T.RandomCrop(img_size)

        # extra geom
        self.p_rot90 = p_rot90

        # blur / noise params
        self.p_blur = p_blur
        self.p_noise = p_noise
        self.blur_kernel_size = (
            blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        )
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.noise_sigma = noise_sigma

    # --------- color jitter helpers ---------
    def random_factor(self, factor, center=1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img,
                brightness_factor=self.random_factor(self.max_brightness_factor),
            )
        return img

    def contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(
                img,
                contrast_factor=self.random_factor(self.max_contrast_factor),
            )
        return img

    def saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img,
                saturation_factor=self.random_factor(self.max_saturation_factor),
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(
                img,
                hue_factor=self.random_factor(self.max_hue_delta, center=0.0),
            )

        return img

    def color_jitter(self, img):
        img = self.brightness(img)

        if torch.rand(()) < 0.5:
            img = self.contrast(img)
            img = self.saturation_and_hue(img)
        else:
            img = self.saturation_and_hue(img)
            img = self.contrast(img)

        return img

    # --------- new image-only augs ---------
    def random_gaussian_blur(self, img):
        if torch.rand(()) >= self.p_blur:
            return img
        sigma = float(torch.empty(1).uniform_(self.blur_sigma_min, self.blur_sigma_max))
        blur = T.GaussianBlur(kernel_size=self.blur_kernel_size, sigma=sigma)
        return blur(img)

    def random_gaussian_noise(self, img):
        if torch.rand(()) >= self.p_noise:
            return img

        # Convert to float32 but keep values in [0, 255]
        img_f = F.to_dtype(img, torch.float32, scale=False)

        # Interpret noise_sigma as relative to [0, 1] → scale by 255
        sigma_intensity = self.noise_sigma * 255.0
        noise = sigma_intensity * torch.randn_like(img_f)

        img_f = (img_f + noise).clamp(0.0, 255.0)

        # Convert back to original dtype (uint8) with no rescaling
        img_noisy = F.to_dtype(img_f, img.dtype, scale=False)
        return img_noisy

    # --------- geom helpers ---------
    def random_rotate_90(self, img, target: dict):
        """
        Rotate image and masks by k * 90° with prob p_rot90.
        Uses F.rotate with proper interpolation:
          - BILINEAR for image
          - NEAREST for masks
        """
        if torch.rand(()) >= self.p_rot90:
            return img, target

        k = int(torch.randint(0, 4, (1,)))
        if k == 0:
            return img, target

        angle = float(k * 90)

        img_r = F.rotate(
            img,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
        )

        tgt_r = target
        if "masks" in target and target["masks"] is not None:
            tgt_r = dict(target)
            tgt_r["masks"] = F.rotate(
                target["masks"],
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
                expand=False,
            )

        return img_r, tgt_r

    # --------- pad + crop (unchanged) ---------
    def pad(self, img, target: dict):
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding, padding_mode="edge")
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def crop(self, img, target: dict):
        img_crop, target_crop = self.random_crop(img, target)

        mask_sums = target_crop["masks"].sum(dim=[-2, -1])
        non_empty_mask = mask_sums > 0

        if non_empty_mask.sum() == 0:
            return self.crop(img, target)

        target_crop["masks"] = target_crop["masks"][non_empty_mask]
        target_crop["labels"] = target_crop["labels"][non_empty_mask]

        return img_crop, target_crop

    # --------- main ---------
    def forward(self, img, target: dict):
        # 1) color jitter
        img = self.color_jitter(img)

        # 2) blur + noise
        img = self.random_gaussian_blur(img)
        img = self.random_gaussian_noise(img)

        # 3) flips
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.random_vertical_flip(img, target)

        # 4) 90° rotations
        img, target = self.random_rotate_90(img, target)

        # 5) scale jitter
        c, h, w = img.shape
        target_size = max(h, w)
        img, target = T.ScaleJitter(
            target_size=(target_size, target_size),
            scale_range=self.scale_range,
            antialias=True,
        )(img, target)

        # 6) pad + crop to final size
        img, target = self.pad(img, target)
        return self.crop(img, target)
