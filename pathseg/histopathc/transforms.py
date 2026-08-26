"""Histopathology artefact transforms with explicit pixel annotations.

Every transform is a plain callable object.  It returns an :class:`ArtifactResult`
containing the corrupted RGB image, a binary affected-pixel mask, a continuous
strength map, and serializable metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import color

from .corruptions import corrupt, defocus_blur, get_corruption_names

RGBArray = NDArray[np.uint8]
BinaryMask = NDArray[np.bool_]
SoftMask = NDArray[np.float32]


@dataclass(slots=True)
class ArtifactResult:
    """Output shared by all artefact transforms."""

    image: Image.Image
    mask: BinaryMask
    soft_mask: SoftMask
    metadata: dict[str, Any] = field(default_factory=dict)
    target_mask: NDArray[np.generic] | None = None

    def __post_init__(self) -> None:
        self.image = self.image.convert("RGB")
        width, height = self.image.size
        expected_shape = (height, width)

        self.mask = np.asarray(self.mask, dtype=bool)
        self.soft_mask = np.asarray(self.soft_mask, dtype=np.float32)

        if self.mask.shape != expected_shape:
            raise ValueError(
                f"mask shape {self.mask.shape} does not match image shape {expected_shape}"
            )
        if self.soft_mask.shape != expected_shape:
            raise ValueError(
                "soft_mask shape "
                f"{self.soft_mask.shape} does not match image shape {expected_shape}"
            )
        if not np.isfinite(self.soft_mask).all():
            raise ValueError("soft_mask contains NaN or infinite values")

        self.soft_mask = np.clip(self.soft_mask, 0.0, 1.0).astype(np.float32)

        if self.target_mask is not None:
            self.target_mask = np.asarray(self.target_mask)
            if self.target_mask.shape[:2] != expected_shape:
                raise ValueError(
                    "target_mask shape "
                    f"{self.target_mask.shape[:2]} does not match image shape {expected_shape}"
                )


class ArtifactTransform(Protocol):
    """Structural type implemented by all transforms in this module."""

    name: str
    localized: bool

    def __call__(
        self,
        image: Image.Image | RGBArray,
        *,
        rng: np.random.Generator,
        tissue_mask: NDArray[np.generic] | Image.Image | None = None,
        semantic_mask: NDArray[np.generic] | Image.Image | None = None,
    ) -> ArtifactResult: ...


def _as_rgb_array(image: Image.Image | RGBArray) -> RGBArray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if array.dtype != np.uint8:
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_tissue_mask(
    tissue_mask: NDArray[np.generic] | Image.Image | None,
    shape: tuple[int, int],
) -> BinaryMask:
    if tissue_mask is None:
        return np.ones(shape, dtype=bool)

    array = np.asarray(tissue_mask)
    if array.ndim == 3:
        array = np.any(array != 0, axis=-1)
    if array.shape != shape:
        raise ValueError(
            f"tissue_mask shape {array.shape} does not match image shape {shape}"
        )
    return array.astype(bool)


def _full_result(
    image: RGBArray,
    *,
    name: str,
    metadata: dict[str, Any],
) -> ArtifactResult:
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=bool)
    return ArtifactResult(
        image=Image.fromarray(image),
        mask=mask,
        soft_mask=mask.astype(np.float32),
        metadata={"name": name, "localized": False, **metadata},
    )


def _rgb_to_od(image: RGBArray) -> NDArray[np.float32]:
    intensity = (image.astype(np.float32) + 1.0) / 256.0
    return -np.log(np.clip(intensity, 1.0 / 256.0, 1.0))


def _od_to_rgb(optical_density: NDArray[np.float32]) -> RGBArray:
    intensity = np.exp(-np.clip(optical_density, 0.0, 6.0))
    return np.clip(np.rint(intensity * 256.0 - 1.0), 0, 255).astype(np.uint8)


class CorruptTransform:
    """Wrapper for a global Histopath-C corruption."""

    localized = False

    def __init__(self, corruption_name: str, severity: int = 5):
        if corruption_name not in get_corruption_names():
            raise ValueError(
                f"Unknown corruption {corruption_name!r}; "
                f"choose from {get_corruption_names()}"
            )
        if severity not in {1, 2, 3, 4, 5}:
            raise ValueError("severity must be an integer in [1, 5]")
        self.name = corruption_name
        self.severity = severity

    def __call__(
        self,
        image: Image.Image | RGBArray,
        *,
        rng: np.random.Generator,
        tissue_mask: NDArray[np.generic] | Image.Image | None = None,
        semantic_mask: NDArray[np.generic] | Image.Image | None = None,
    ) -> ArtifactResult:
        del tissue_mask, semantic_mask
        image_np = _as_rgb_array(image)
        output = corrupt(
            image_np,
            severity=self.severity,
            corruption_name=self.name,
            rng=rng,
        )
        return _full_result(
            output,
            name=self.name,
            metadata={"severity": self.severity},
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, severity={self.severity})"


class Staining:
    """Perturb H&E stain appearance in HED space.

    Parameters are sampled for every image from the supplied RNG.  This fixes a
    subtle issue in the original implementation, where a fixed seed in
    ``__init__`` made every image receive exactly the same stain transform.
    """

    localized = False

    def __init__(self, theta: float = 0.0, *, name: str = "staining"):
        if not isinstance(theta, Real) or theta < 0:
            raise ValueError("theta must be a non-negative number")
        self.theta = float(theta)
        self.name = name

    def __call__(
        self,
        image: Image.Image | RGBArray,
        *,
        rng: np.random.Generator,
        tissue_mask: NDArray[np.generic] | Image.Image | None = None,
        semantic_mask: NDArray[np.generic] | Image.Image | None = None,
    ) -> ArtifactResult:
        del tissue_mask, semantic_mask
        image_np = _as_rgb_array(image)
        alpha = rng.uniform(1.0 - self.theta, 1.0 + self.theta, size=3)
        beta = rng.uniform(-self.theta, self.theta, size=3)

        hed = color.rgb2hed(image_np)
        perturbed_hed = hed * alpha.reshape(1, 1, 3) + beta.reshape(1, 1, 3)
        output = np.clip(color.hed2rgb(perturbed_hed), 0.0, 1.0)
        output_np = np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)

        return _full_result(
            output_np,
            name=self.name,
            metadata={
                "theta": self.theta,
                "alpha": alpha.tolist(),
                "beta": beta.tolist(),
            },
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(theta={self.theta}, name={self.name!r})"


class AddDust:
    """Add localized dark dust smudges and return their pixel support."""

    name = "dust"
    localized = True

    def __init__(
        self,
        *,
        intensity: float = 0.7,
        min_smudges: int = 3,
        max_smudges: int = 10,
        rectangle_probability: float = 0.8,
        blur_sigma: float = 15.0,
        mask_threshold: float = 0.02,
    ):
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("intensity must be in [0, 1]")
        if not 0 <= min_smudges <= max_smudges:
            raise ValueError("require 0 <= min_smudges <= max_smudges")
        if not 0.0 <= rectangle_probability <= 1.0:
            raise ValueError("rectangle_probability must be in [0, 1]")
        if blur_sigma < 0:
            raise ValueError("blur_sigma must be non-negative")

        self.intensity = float(intensity)
        self.min_smudges = min_smudges
        self.max_smudges = max_smudges
        self.rectangle_probability = float(rectangle_probability)
        self.blur_sigma = float(blur_sigma)
        self.mask_threshold = float(mask_threshold)

    def _make_mask(
        self,
        height: int,
        width: int,
        rng: np.random.Generator,
    ) -> tuple[SoftMask, int]:
        mask = np.zeros((height, width), dtype=np.float32)
        count = int(rng.integers(self.min_smudges, self.max_smudges + 1))

        for _ in range(count):
            if rng.random() < self.rectangle_probability:
                rect_width = int(rng.integers(max(2, width // 6), max(3, width // 2 + 1)))
                rect_height = int(
                    rng.integers(max(2, height // 6), max(3, height // 2 + 1))
                )
                x0 = int(rng.integers(0, max(1, width - rect_width + 1)))
                y0 = int(rng.integers(0, max(1, height - rect_height + 1)))
                gradient = np.linspace(0.0, 1.0, rect_height, dtype=np.float32)[:, None]
                region = mask[y0 : y0 + rect_height, x0 : x0 + rect_width]
                np.maximum(region, gradient, out=region)
            else:
                vertical = bool(rng.integers(0, 2))
                max_thickness = max(4, min(height, width) // 20)
                thickness = int(rng.integers(3, max_thickness + 1))
                if vertical:
                    x0 = int(rng.integers(0, max(1, width - thickness + 1)))
                    mask[:, x0 : x0 + thickness] = 1.0
                else:
                    y0 = int(rng.integers(0, max(1, height - thickness + 1)))
                    mask[y0 : y0 + thickness, :] = 1.0

        if self.blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=self.blur_sigma)

        maximum = float(mask.max())
        if maximum > 0:
            mask /= maximum
        return np.clip(mask * self.intensity, 0.0, 1.0).astype(np.float32), count

    def __call__(
        self,
        image: Image.Image | RGBArray,
        *,
        rng: np.random.Generator,
        tissue_mask: NDArray[np.generic] | Image.Image | None = None,
        semantic_mask: NDArray[np.generic] | Image.Image | None = None,
    ) -> ArtifactResult:
        del tissue_mask, semantic_mask
        image_np = _as_rgb_array(image)
        height, width = image_np.shape[:2]
        soft_mask, count = self._make_mask(height, width, rng)

        output = image_np.astype(np.float32) / 255.0
        output *= 1.0 - soft_mask[..., None]
        output_np = np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8)

        return ArtifactResult(
            image=Image.fromarray(output_np),
            mask=soft_mask >= self.mask_threshold,
            soft_mask=soft_mask,
            metadata={
                "name": self.name,
                "localized": True,
                "smudge_count": count,
                "intensity": self.intensity,
                "blur_sigma": self.blur_sigma,
            },
        )


class AddAirBubble:
    """Add localized air bubbles with blur, tint, and highlights."""

    name = "air_bubble"
    localized = True

    def __init__(
        self,
        *,
        min_bubbles: int = 2,
        max_bubbles: int = 10,
        transparency: float = 0.3,
        blur_severity: int = 3,
        min_radius_fraction: float = 0.03,
        max_radius_fraction: float = 0.18,
        edge_blur_sigma: float = 2.0,
        mask_threshold: float = 0.02,
    ):
        if not 0 <= min_bubbles <= max_bubbles:
            raise ValueError("require 0 <= min_bubbles <= max_bubbles")
        if not 0.0 <= transparency <= 1.0:
            raise ValueError("transparency must be in [0, 1]")
        if blur_severity not in {1, 2, 3, 4, 5}:
            raise ValueError("blur_severity must be in [1, 5]")
        if not 0 < min_radius_fraction <= max_radius_fraction:
            raise ValueError("invalid radius fractions")

        self.min_bubbles = min_bubbles
        self.max_bubbles = max_bubbles
        self.transparency = float(transparency)
        self.blur_severity = blur_severity
        self.min_radius_fraction = float(min_radius_fraction)
        self.max_radius_fraction = float(max_radius_fraction)
        self.edge_blur_sigma = float(edge_blur_sigma)
        self.mask_threshold = float(mask_threshold)

    def __call__(
        self,
        image: Image.Image | RGBArray,
        *,
        rng: np.random.Generator,
        tissue_mask: NDArray[np.generic] | Image.Image | None = None,
        semantic_mask: NDArray[np.generic] | Image.Image | None = None,
    ) -> ArtifactResult:
        del tissue_mask, semantic_mask
        image_np = _as_rgb_array(image)
        height, width = image_np.shape[:2]
        short_side = min(height, width)

        hard_mask = np.zeros((height, width), dtype=np.float32)
        highlight_mask = np.zeros((height, width), dtype=np.float32)
        count = int(rng.integers(self.min_bubbles, self.max_bubbles + 1))
        radii: list[int] = []

        min_radius = max(2, int(round(short_side * self.min_radius_fraction)))
        max_radius = max(min_radius, int(round(short_side * self.max_radius_fraction)))

        for _ in range(count):
            radius = int(rng.integers(min_radius, max_radius + 1))
            center_x = int(rng.integers(0, width))
            center_y = int(rng.integers(0, height))
            radii.append(radius)

            cv2.circle(hard_mask, (center_x, center_y), radius, 1.0, -1, cv2.LINE_AA)

            highlight_radius = max(1, int(round(radius * 0.35)))
            highlight_center = (
                int(round(center_x - 0.28 * radius)),
                int(round(center_y - 0.28 * radius)),
            )
            cv2.circle(
                highlight_mask,
                highlight_center,
                highlight_radius,
                1.0,
                -1,
                cv2.LINE_AA,
            )

        soft_mask = hard_mask
        if self.edge_blur_sigma > 0:
            soft_mask = cv2.GaussianBlur(
                hard_mask, (0, 0), sigmaX=self.edge_blur_sigma
            )
            highlight_mask = cv2.GaussianBlur(
                highlight_mask, (0, 0), sigmaX=self.edge_blur_sigma
            )
        soft_mask = np.clip(soft_mask, 0.0, 1.0).astype(np.float32)
        highlight_mask = np.clip(highlight_mask, 0.0, 1.0).astype(np.float32)

        blurred = defocus_blur(image_np, self.blur_severity, rng)
        blur_alpha = soft_mask[..., None]
        output = (
            image_np.astype(np.float32) * (1.0 - blur_alpha)
            + blurred.astype(np.float32) * blur_alpha
        )

        bubble_color = np.array([200.0, 220.0, 255.0], dtype=np.float32)
        tint_alpha = soft_mask[..., None] * self.transparency
        output = output * (1.0 - tint_alpha) + bubble_color * tint_alpha

        highlight_alpha = (
            highlight_mask[..., None] * min(1.0, 1.5 * self.transparency)
        )
        output = output * (1.0 - highlight_alpha) + 255.0 * highlight_alpha
        output_np = np.clip(np.rint(output), 0, 255).astype(np.uint8)

        affected = np.maximum(soft_mask, highlight_mask)
        return ArtifactResult(
            image=Image.fromarray(output_np),
            mask=affected >= self.mask_threshold,
            soft_mask=affected,
            metadata={
                "name": self.name,
                "localized": True,
                "bubble_count": count,
                "radii_pixels": radii,
                "transparency": self.transparency,
                "blur_severity": self.blur_severity,
            },
        )


class AddTissueFold:
    """Simulate a fold by cutting, translating, and overlapping one image side.

    A smooth curve divides the image into two non-overlapping pieces. One piece
    is translated toward the other. The overlap is rendered as a reflected,
    mildly deformed flap and fused with the stationary tissue in optical-density
    space. The invalid outer border created by the translation is cropped.

    When ``semantic_mask`` is supplied, it is transformed with the same
    geometry. Pixels in the actual overlap are set to ``ignore_label`` because
    two tissue layers occupy the same image location there.
    """

    name = "tissue_fold"
    localized = True

    def __init__(
        self,
        *,
        displacement_fraction: tuple[float, float] = (0.07, 0.18),
        curve_amplitude_fraction: tuple[float, float] = (0.005, 0.045),
        curve_cycles: tuple[float, float] = (0.25, 0.85),
        curve_offset_fraction: tuple[float, float] = (-0.08, 0.08),
        direction_jitter_degrees: tuple[float, float] = (-8.0, 8.0),
        flap_compression: tuple[float, float] = (0.85, 1.15),
        elastic_fraction: tuple[float, float] = (0.0, 0.08),
        elastic_cycles: tuple[float, float] = (0.5, 1.5),
        overlap_strength: tuple[float, float] = (0.45, 1.0),
        crease_strength: tuple[float, float] = (0.08, 0.35),
        crease_width_fraction: tuple[float, float] = (0.015, 0.05),
        blur_sigma: tuple[float, float] = (0.3, 1.5),
        blur_mix: tuple[float, float] = (0.1, 0.45),
        edge_feather_sigma: tuple[float, float] = (0.6, 1.8),
        mask_threshold: float = 0.03,
        minimum_tissue_overlap_fraction: float = 0.002,
        maximum_gap_fraction: float = 0.002,
        crop_padding: int = 1,
        ignore_label: int = 255,
        max_placement_attempts: int = 30,
    ):
        self.displacement_fraction = displacement_fraction
        self.curve_amplitude_fraction = curve_amplitude_fraction
        self.curve_cycles = curve_cycles
        self.curve_offset_fraction = curve_offset_fraction
        self.direction_jitter_degrees = direction_jitter_degrees
        self.flap_compression = flap_compression
        self.elastic_fraction = elastic_fraction
        self.elastic_cycles = elastic_cycles
        self.overlap_strength = overlap_strength
        self.crease_strength = crease_strength
        self.crease_width_fraction = crease_width_fraction
        self.blur_sigma = blur_sigma
        self.blur_mix = blur_mix
        self.edge_feather_sigma = edge_feather_sigma
        self.mask_threshold = float(mask_threshold)
        self.minimum_tissue_overlap_fraction = float(
            minimum_tissue_overlap_fraction
        )
        self.maximum_gap_fraction = float(maximum_gap_fraction)
        self.crop_padding = int(crop_padding)
        self.ignore_label = int(ignore_label)
        self.max_placement_attempts = int(max_placement_attempts)

        if self.crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")

    @staticmethod
    def _warp_translation(
        array: NDArray[np.generic],
        *,
        dx: float,
        dy: float,
        interpolation: int,
        border_value: int | tuple[int, int, int],
    ) -> NDArray[np.generic]:
        height, width = array.shape[:2]
        matrix = np.array(
            [[1.0, 0.0, dx], [0.0, 1.0, dy]],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            array,
            matrix,
            dsize=(width, height),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )

    @staticmethod
    def _crop_box_for_translation(
        *,
        width: int,
        height: int,
        dx: float,
        dy: float,
        padding: int,
    ) -> tuple[int, int, int, int]:
        # Region for which both the original and translated canvases have a
        # valid source coordinate.  right/bottom are exclusive.
        left = int(np.ceil(max(dx, 0.0))) + padding
        right = width - int(np.ceil(max(-dx, 0.0))) - padding
        top = int(np.ceil(max(dy, 0.0))) + padding
        bottom = height - int(np.ceil(max(-dy, 0.0))) - padding

        if right - left < 32 or bottom - top < 32:
            raise RuntimeError(
                "fold displacement leaves a crop smaller than 32 pixels"
            )
        return left, top, right, bottom

    @staticmethod
    def _prepare_semantic_mask(
        semantic_mask: NDArray[np.generic] | Image.Image | None,
        shape: tuple[int, int],
    ) -> NDArray[np.generic] | None:
        if semantic_mask is None:
            return None
        array = np.asarray(semantic_mask)
        if array.shape[:2] != shape:
            raise ValueError(
                f"semantic_mask shape {array.shape[:2]} does not match {shape}"
            )
        if array.ndim != 2:
            raise ValueError(
                "semantic_mask must be a 2-D label image for geometric folding"
            )
        return np.ascontiguousarray(array)

    def __call__(
        self,
        image: Image.Image | RGBArray,
        *,
        rng: np.random.Generator,
        tissue_mask: NDArray[np.generic] | Image.Image | None = None,
        semantic_mask: NDArray[np.generic] | Image.Image | None = None,
    ) -> ArtifactResult:
        image_np = _as_rgb_array(image)
        height, width = image_np.shape[:2]
        shape = (height, width)
        tissue = _as_tissue_mask(tissue_mask, shape)
        labels = self._prepare_semantic_mask(semantic_mask, shape)

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        short_side = float(min(height, width))
        tissue_pixels = max(int(tissue.sum()), 1)
        minimum_overlap_pixels = max(
            16,
            int(round(tissue_pixels * self.minimum_tissue_overlap_fraction)),
        )

        tissue_y, tissue_x = np.nonzero(tissue)
        if tissue_x.size:
            anchor_x = float(np.mean(tissue_x))
            anchor_y = float(np.mean(tissue_y))
        else:
            anchor_x = (width - 1) / 2.0
            anchor_y = (height - 1) / 2.0

        placement: dict[str, Any] | None = None

        for _ in range(self.max_placement_attempts):
            # n is the average normal of the cutting curve; t runs along it.
            normal_angle = float(rng.uniform(0.0, np.pi))
            normal = np.array(
                [np.cos(normal_angle), np.sin(normal_angle)],
                dtype=np.float32,
            )
            tangent = np.array([-normal[1], normal[0]], dtype=np.float32)

            center = np.array([anchor_x, anchor_y], dtype=np.float32)
            center += normal * float(
                rng.uniform(*self.curve_offset_fraction) * short_side
            )

            rel_x = xx - center[0]
            rel_y = yy - center[1]
            s = tangent[0] * rel_x + tangent[1] * rel_y
            q = normal[0] * rel_x + normal[1] * rel_y
            s_min = float(s.min())
            s_max = float(s.max())
            s_span = max(s_max - s_min, 1.0)
            s01 = (s - s_min) / s_span

            amplitude = float(
                rng.uniform(*self.curve_amplitude_fraction) * short_side
            )
            cycles = float(rng.uniform(*self.curve_cycles))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            phase_argument = 2.0 * np.pi * cycles * s01 + phase
            curve_q = amplitude * np.sin(phase_argument)
            dcurve_ds = (
                amplitude
                * np.cos(phase_argument)
                * (2.0 * np.pi * cycles / s_span)
            )
            signed_distance = q - curve_q

            # Randomly choose which side is moved, then move it primarily
            # toward the stationary side. This creates overlap, not a gap.
            moving_side_sign = 1.0 if rng.random() < 0.5 else -1.0
            moving_source_mask = moving_side_sign * signed_distance > 0.0
            stationary_mask = ~moving_source_mask

            # Reject curves that leave almost all tissue on one side.
            tissue_a = int((tissue & stationary_mask).sum())
            tissue_b = int((tissue & moving_source_mask).sum())
            if min(tissue_a, tissue_b) < minimum_overlap_pixels:
                continue

            displacement = float(
                rng.uniform(*self.displacement_fraction) * short_side
            )
            jitter = np.deg2rad(
                float(rng.uniform(*self.direction_jitter_degrees))
            )
            movement_direction = (
                -moving_side_sign * np.cos(jitter) * normal
                + np.sin(jitter) * tangent
            )
            movement_direction /= max(
                float(np.linalg.norm(movement_direction)), 1e-8
            )
            delta = movement_direction * displacement
            dx, dy = float(delta[0]), float(delta[1])

            crop_box = self._crop_box_for_translation(
                width=width,
                height=height,
                dx=dx,
                dy=dy,
                padding=self.crop_padding,
            )
            left, top, right, bottom = crop_box

            moved_mask = self._warp_translation(
                moving_source_mask.astype(np.uint8),
                dx=dx,
                dy=dy,
                interpolation=cv2.INTER_NEAREST,
                border_value=0,
            ).astype(bool)

            overlap = stationary_mask & moved_mask
            coverage = stationary_mask | moved_mask
            coverage_crop = coverage[top:bottom, left:right]
            gap_fraction = 1.0 - float(coverage_crop.mean())
            if gap_fraction > self.maximum_gap_fraction:
                continue

            moved_tissue = self._warp_translation(
                tissue.astype(np.uint8),
                dx=dx,
                dy=dy,
                interpolation=cv2.INTER_NEAREST,
                border_value=0,
            ).astype(bool)
            tissue_overlap = overlap & tissue & moved_tissue
            if int(tissue_overlap.sum()) < minimum_overlap_pixels:
                continue

            placement = {
                "normal_angle": normal_angle,
                "normal": normal,
                "tangent": tangent,
                "center": center,
                "s": s,
                "q": q,
                "s01": s01,
                "curve_q": curve_q,
                "dcurve_ds": dcurve_ds,
                "signed_distance": signed_distance,
                "moving_side_sign": moving_side_sign,
                "moving_source_mask": moving_source_mask,
                "stationary_mask": stationary_mask,
                "moved_mask": moved_mask,
                "overlap": overlap,
                "coverage": coverage,
                "tissue_overlap": tissue_overlap,
                "amplitude": amplitude,
                "cycles": cycles,
                "phase": phase,
                "displacement": displacement,
                "dx": dx,
                "dy": dy,
                "crop_box": crop_box,
                "gap_fraction": gap_fraction,
            }
            break

        if placement is None:
            raise RuntimeError(
                "failed to generate a valid piecewise tissue fold"
            )

        dx = placement["dx"]
        dy = placement["dy"]
        stationary_mask = placement["stationary_mask"]
        moved_mask = placement["moved_mask"]
        overlap = placement["overlap"]
        coverage = placement["coverage"]

        moved_image = self._warp_translation(
            image_np,
            dx=dx,
            dy=dy,
            interpolation=cv2.INTER_LINEAR,
            border_value=(255, 255, 255),
        )

        # Compose the two cut pieces. The translated piece is not used in the
        # overlap itself; that area is replaced by the reflected flap below.
        output = np.full_like(image_np, 255)
        output[stationary_mask] = image_np[stationary_mask]
        moved_only = moved_mask & ~stationary_mask
        output[moved_only] = moved_image[moved_only]

        # Approximate the local normal of the smooth curve. The curve is a graph
        # q=f(s), so n_local is proportional to n-f'(s)t.
        normal = placement["normal"]
        tangent = placement["tangent"]
        slope = placement["dcurve_ds"]
        norm = np.sqrt(1.0 + slope * slope)
        local_nx = (normal[0] - slope * tangent[0]) / norm
        local_ny = (normal[1] - slope * tangent[1]) / norm
        local_tx = -local_ny
        local_ty = local_nx

        # Signed distance is approximate but accurate for the deliberately
        # smooth, low-slope curves used here.
        local_distance = placement["signed_distance"] / norm
        compression = float(rng.uniform(*self.flap_compression))
        elastic_amplitude = float(
            rng.uniform(*self.elastic_fraction) * placement["displacement"]
        )
        elastic_cycles = float(rng.uniform(*self.elastic_cycles))
        elastic_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        elastic_shift = elastic_amplitude * np.sin(
            2.0 * np.pi * elastic_cycles * placement["s01"]
            + elastic_phase
        )

        # x - (1+c)d*n reflects across the curve when c=1. Mild compression and
        # tangential elastic displacement prevent a perfectly rigid mirror.
        map_x = (
            xx
            - (1.0 + compression) * local_distance * local_nx
            + elastic_shift * local_tx
        ).astype(np.float32)
        map_y = (
            yy
            - (1.0 + compression) * local_distance * local_ny
            + elastic_shift * local_ty
        ).astype(np.float32)

        flap = cv2.remap(
            image_np,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        reflected_tissue = cv2.remap(
            tissue.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

        hard_fold = overlap & tissue & reflected_tissue
        feather_sigma = float(rng.uniform(*self.edge_feather_sigma))
        soft_fold = cv2.GaussianBlur(
            hard_fold.astype(np.float32),
            (0, 0),
            sigmaX=max(feather_sigma, 0.01),
        )
        soft_fold = np.clip(soft_fold, 0.0, 1.0)

        overlap_strength = float(rng.uniform(*self.overlap_strength))
        output_od = _rgb_to_od(output)
        flap_od = _rgb_to_od(flap)
        output_od += soft_fold[..., None] * overlap_strength * flap_od

        crease_width = float(
            rng.uniform(*self.crease_width_fraction)
            * placement["displacement"]
        )
        crease_width = max(0.7, crease_width)
        crease_profile = np.exp(
            -0.5 * (local_distance / crease_width) ** 2
        ).astype(np.float32)
        crease_profile *= tissue.astype(np.float32)
        crease_strength = float(rng.uniform(*self.crease_strength))
        output_od += crease_profile[..., None] * crease_strength
        output_np = _od_to_rgb(output_od)

        artifact_soft = np.maximum(soft_fold, crease_profile)

        # Tiny discretization gaps are filled from the original image and are
        # included in the artefact/ignore support.
        gap = ~coverage
        if np.any(gap):
            output_np[gap] = image_np[gap]
            artifact_soft = np.maximum(
                artifact_soft, gap.astype(np.float32)
            )

        sigma = float(rng.uniform(*self.blur_sigma))
        blur_mix = float(rng.uniform(*self.blur_mix))
        if sigma > 0.0 and blur_mix > 0.0:
            blurred = cv2.GaussianBlur(output_np, (0, 0), sigmaX=sigma)
            blur_alpha = artifact_soft[..., None] * blur_mix
            output_np = np.clip(
                np.rint(
                    output_np.astype(np.float32) * (1.0 - blur_alpha)
                    + blurred.astype(np.float32) * blur_alpha
                ),
                0,
                255,
            ).astype(np.uint8)

        transformed_labels: NDArray[np.generic] | None = None
        if labels is not None:
            label_border = self.ignore_label
            moved_labels = self._warp_translation(
                labels,
                dx=dx,
                dy=dy,
                interpolation=cv2.INTER_NEAREST,
                border_value=label_border,
            )
            transformed_labels = np.full(
                shape,
                self.ignore_label,
                dtype=labels.dtype,
            )
            transformed_labels[stationary_mask] = labels[stationary_mask]
            transformed_labels[moved_only] = moved_labels[moved_only]
            transformed_labels[overlap] = np.asarray(
                self.ignore_label, dtype=labels.dtype
            )
            transformed_labels[gap] = np.asarray(
                self.ignore_label, dtype=labels.dtype
            )

        left, top, right, bottom = placement["crop_box"]
        crop = np.s_[top:bottom, left:right]
        output_crop = np.ascontiguousarray(output_np[crop])
        soft_crop = np.ascontiguousarray(
            np.clip(artifact_soft[crop], 0.0, 1.0).astype(np.float32)
        )
        binary_crop = soft_crop >= self.mask_threshold
        labels_crop = (
            None
            if transformed_labels is None
            else np.ascontiguousarray(transformed_labels[crop])
        )

        return ArtifactResult(
            image=Image.fromarray(output_crop),
            mask=binary_crop,
            soft_mask=soft_crop,
            target_mask=labels_crop,
            metadata={
                "name": self.name,
                "localized": True,
                "geometry": "piecewise_cut_translate_reflect",
                "normal_angle_radians": placement["normal_angle"],
                "moving_side": int(placement["moving_side_sign"]),
                "curve_amplitude_pixels": placement["amplitude"],
                "curve_cycles": placement["cycles"],
                "displacement_pixels": placement["displacement"],
                "translation_xy": [dx, dy],
                "crop_box_xyxy": [left, top, right, bottom],
                "original_size_wh": [width, height],
                "output_size_wh": [right - left, bottom - top],
                "flap_compression": compression,
                "elastic_amplitude_pixels": elastic_amplitude,
                "elastic_cycles": elastic_cycles,
                "overlap_strength": overlap_strength,
                "crease_strength": crease_strength,
                "crease_width_pixels": crease_width,
                "blur_sigma": sigma,
                "blur_mix": blur_mix,
                "gap_fraction_before_fill": placement["gap_fraction"],
                "semantic_overlap_label": (
                    self.ignore_label if labels is not None else None
                ),
            },
        )

def build_default_transforms() -> dict[str, ArtifactTransform]:
    """Build the transform registry used by the dataset generator."""

    return {
        "gaussian_noise": CorruptTransform("gaussian_noise", 3),
        "shot_noise": CorruptTransform("shot_noise", 3),
        "defocus_blur": CorruptTransform("defocus_blur", 4),
        "motion_blur": CorruptTransform("motion_blur", 5),
        "brightness": CorruptTransform("brightness", 2),
        "contrast": CorruptTransform("contrast", 1),
        "dust": AddDust(),
        "air_bubble": AddAirBubble(
            min_bubbles=10,
            max_bubbles=20,
            transparency=0.3,
            blur_severity=2,
        ),
        "stain_light": Staining(0.15, name="stain_light"),
        "stain_heavy": Staining(0.25, name="stain_heavy"),
        "tissue_fold": AddTissueFold(),
    }
