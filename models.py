import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim

class DirtyNet(nn.Module):
    """
    Simple model to classify dirty/clean and estimate difficulty.
    """
    def __init__(self):
        super().__init__()

    def forward(self, img_tensor):
        # Convert tensor to numpy image
        img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # Convert to grayscale
        gray = np.mean(img, axis=2)
        # Heuristic: mean brightness and stddev
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        # If image is dark or high variance, call it dirty
        is_dirty = mean_brightness < 0.7 or std_brightness > 0.15
        # Difficulty: scale stddev to 1-100
        difficulty = min(max((std_brightness / 0.3) * 100, 1), 100) if is_dirty else 0
        return torch.tensor([float(is_dirty)]), torch.tensor([difficulty])

class CleanQualityNet(nn.Module):
    """
    Model to rate cleaning quality using SSIM.
    """
    def __init__(self):
        super().__init__()

    def forward(self, before_tensor, after_tensor):
        # Convert tensors to numpy images
        before = before_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        after = after_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # Convert to grayscale
        before_gray = np.mean(before, axis=2)
        after_gray = np.mean(after, axis=2)
        # Compute SSIM (higher means more similar), specify data_range=1.0 for normalized images
        score = ssim(before_gray, after_gray, data_range=1.0)
        # Points awarded: the more different, the higher the score
        points = (1 - score) * 100
        return torch.tensor([points]) 