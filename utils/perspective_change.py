"""
    Reimplementation of torchvision.transforms.RandomPerspective augmentation
    with cropping black pixels and resizing image to original shape.
"""

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class RandomPerspectiveCropResize(torch.nn.Module):
    def __init__(self,
                 scale_factor: float,
                 dist_prob: float,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_factor = scale_factor
        self.dist_prob = dist_prob

    def forward(self,
                img: torch.Tensor) -> torch.Tensor:
        if not (torch.rand(1) < self.dist_prob):
            return img

        channels, height, width = F.get_dimensions(img)

        src, dst = transforms.RandomPerspective().get_params(width, height, self.scale_factor)
        src = np.array(src)
        dst = np.array(dst)

        res = F.perspective(img, src, dst)

        # Find ROI
        x1, x2 = np.sort(dst[:, 0])[1:3]
        y1, y2 = np.sort(dst[:, 1])[1:3]

        resize_transform = transforms.Resize((width, height))

        cropped_res = res[:, :, y1: y2, x1: x2]  # cut ROI
        cropped_resized_res = resize_transform(cropped_res)  # resize to original shape

        return cropped_resized_res
