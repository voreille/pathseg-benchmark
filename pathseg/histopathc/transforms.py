import random
import numbers
import numpy as np

import cv2
from skimage import color
from PIL import Image, ImageDraw, ImageFilter

from .corruptions import corrupt, defocus_blur, get_corruption_names


import torch
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class CorruptTransform(object):
    def __init__(self, corruption_name: str, corruption_severity: int = 5):
        super().__init__()
        self.corruption_name = corruption_name
        self.corruption_severity = corruption_severity

        if self.corruption_name not in get_corruption_names():
            raise ValueError(f"Corruption name {self.corruption_name} is not valid. \nchoose from {get_corruption_names()}")
        
        else:
            print(f"\n\nCorruptTransform: {self.corruption_name} with severity {self.corruption_severity}")

    def __call__(self, img):
        """ 
        Args:
            results (dict): The input data dictionary.
        Returns:
            dict: The corrupted data dictionary.

        Note that the input image should be numpy array (bgr or rgb).
        """

        # Convert the image to numpy array
        img = np.array(img)
        # Corrupt the image
        img = corrupt(img, severity=self.corruption_severity, corruption_name=self.corruption_name)
        # Convert the image back to PIL
        img = Image.fromarray(img)
        
        
        return img
    

class Staining(object):
    def __init__(self, theta=0.): # light: theta=0.05; heavy: theta=0.2
        assert isinstance(theta, numbers.Number), "theta should be a single number."
        self.theta = theta

        # using a fixed seed
        rng_state = np.random.get_state()
        np.random.seed(42)
        self.alpha = np.random.uniform(1-theta, 1+theta, (1, 3))
        self.betti = np.random.uniform(-theta, theta, (1, 3))

        # Restore the original RNG state
        np.random.set_state(rng_state)

    @staticmethod
    def adjust_HED(img, alpha, betti):
        img = np.array(img)

        s = np.reshape(color.rgb2hed(img), (-1, 3))
        ns = alpha * s + betti  # perturbations on HED color space
        nimg = color.hed2rgb(np.reshape(ns, img.shape))

        imin = nimg.min()
        imax = nimg.max()
        rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]

        return Image.fromarray(rsimg)

    def __call__(self, img):
        return self.adjust_HED(img, self.alpha, self.betti)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'theta={0}'.format(self.theta)
        format_string += ',alpha={0}'.format(self.alpha)
        format_string += ',betti={0}'.format(self.betti)
        return format_string


class AddAirBubble(torch.nn.Module):
    def __init__(self, min_bubbles=2, max_bubbles=10, transparency=0.3, blur_severity=3):
        super().__init__()
        self.min_bubbles = min_bubbles
        self.max_bubbles = max_bubbles
        self.transparency = transparency
        self.blur_severity = blur_severity

    def forward(self, org_img):

        img = org_img.convert("RGB")  # Convert input image to RGB
        img_np = np.array(img)  # Convert to NumPy for processing

        width, height = img.size
        num_bubbles = random.randint(self.min_bubbles, self.max_bubbles)
        max_radius = min(width, height) // 2
        min_radius = max_radius // 7  


        # Create bubble mask
        bubble_mask = np.zeros((height, width), dtype=np.uint8)
        bubble_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(bubble_layer)



        bubble_positions = []
        for _ in range(num_bubbles):
            radius = random.randint(min_radius, max_radius)
            x, y = random.randint(0, width - radius), random.randint(0, height - radius)
            bubble_positions.append((x, y, radius))
            
            draw.ellipse([x, y, x + radius, y + radius], fill=(200, 220, 255, int(255 * self.transparency)))
            cv2.circle(bubble_mask, (x + radius // 2, y + radius // 2), radius // 2, 255, -1)



        # Apply defocus blur inside bubble regions only
        blurred_img = defocus_blur(img_np, severity=self.blur_severity).astype(np.uint8)
        blurred_img = cv2.bitwise_and(blurred_img, blurred_img, mask=bubble_mask)
        img_np = cv2.bitwise_and(img_np, img_np, mask=255 - bubble_mask) + blurred_img
        
        img = Image.fromarray(img_np).convert("RGBA")

        # Add highlights
        draw = ImageDraw.Draw(bubble_layer)
        for x, y, radius in bubble_positions:
            highlight_radius = int(radius * 0.4)
            highlight_x, highlight_y = x + int(radius * 0.3), y + int(radius * 0.3)
            draw.ellipse(
                [highlight_x, highlight_y, highlight_x + highlight_radius, highlight_y + highlight_radius],
                fill=(255, 255, 255, int(255 * (self.transparency * 1.5)))
            )

        # Apply Gaussian blur to bubble overlay
        bubble_layer = bubble_layer.filter(ImageFilter.GaussianBlur(radius=2))
        img = Image.alpha_composite(img, bubble_layer)
        img = img.convert("RGB")  

        return img


class AddDust(torch.nn.Module):
    def __init__(self, dust_intensity=0.7, min_smudges=3, max_smudges=10):
        """
        PyTorch Transform to add multiple dust smudges to histopathology images.

        Parameters:
        - dust_intensity: Controls the strength of dust smudges (0 to 1).
        - min_smudges: Minimum number of smudges applied per image.
        - max_smudges: Maximum number of smudges applied per image.
        """
        super().__init__()
        self.dust_intensity = dust_intensity
        self.min_smudges = min_smudges
        self.max_smudges = max_smudges

    def generate_dust_overlay(self, img_size):
        """Generates multiple dust smudges, with rectangles being more frequent."""
        dust_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)

        num_smudges = np.random.randint(self.min_smudges, self.max_smudges + 1)

        for _ in range(num_smudges):
            shape_type = np.random.choice(["rectangle", "thin_line"], p=[0.8, 0.2])  # 80% rectangles, 20% thin lines

            if shape_type == "rectangle":
                # Generate a random rectangular smudge with variable size
                rect_width = np.random.randint(img_size[0] // 6, img_size[0] // 2)
                rect_height = np.random.randint(img_size[1] // 6, img_size[1] // 2)

                x_start = np.random.randint(0, img_size[0] - rect_width)
                y_start = np.random.randint(0, img_size[1] - rect_height)
                x_end = x_start + rect_width
                y_end = y_start + rect_height

                for i in range(y_start, y_end):
                    alpha = (i - y_start) / (y_end - y_start)  # Gradient effect
                    dust_mask[i, x_start:x_end] = int(alpha * 255)

            else:  # "thin_line"
                # Choose vertical or horizontal thin line
                is_vertical = np.random.rand() > 0.5
                thickness = np.random.randint(3, 10)  # Thin line width

                if is_vertical:
                    x_start = np.random.randint(0, img_size[0] - thickness)
                    x_end = x_start + thickness
                    y_start = 0
                    y_end = img_size[1]
                    dust_mask[y_start:y_end, x_start:x_end] = 255  # Solid line
                else:
                    y_start = np.random.randint(0, img_size[1] - thickness)
                    y_end = y_start + thickness
                    x_start = 0
                    x_end = img_size[0]
                    dust_mask[y_start:y_end, x_start:x_end] = 255  # Solid line

        # Apply Gaussian blur for a more natural dust effect
        dust_mask = cv2.GaussianBlur(dust_mask, (35, 35), 15)

        # Normalize and adjust intensity
        dust_mask = (dust_mask / 255.0) * self.dust_intensity
        return dust_mask

    def forward(self, img_pil):
        """Applies the dust augmentation to the input PIL image."""
        org_img = img_pil.copy()
        img_np = np.array(org_img).astype(np.float32) / 255.0  # Normalize image

        img_size = (img_np.shape[1], img_np.shape[0])  # Width, Height

        # Generate and apply dust overlay
        dust_mask = self.generate_dust_overlay(img_size)
        img_np = img_np * (1 - dust_mask[..., None])  # Apply dust (darken)

        # Convert back to PIL image
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        return img_pil
