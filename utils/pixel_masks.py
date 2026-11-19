"""Pixel-wise confidence masks for a rendered image and its reference.

This is a photometric confidence mask, not a semantic segmentation mask.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, reference: torch.Tensor) -> torch.Tensor:
    coords = torch.arange(size, device=reference.device, dtype=reference.dtype)
    coords = coords - (size - 1) * 0.5
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel).view(1, 1, size, size)


def _gaussian_blur(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    size = kernel.shape[-1]
    pad = size // 2
    # Reflect padding is undefined when the image is smaller than the pad.
    mode = "reflect" if image.shape[-2] > pad and image.shape[-1] > pad else "replicate"
    image = F.pad(image, (pad, pad, pad, pad), mode=mode)
    return F.conv2d(image, kernel.expand(image.shape[1], 1, size, size), groups=image.shape[1])


def _ssim_map(a: torch.Tensor, b: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    channels = a.shape[1]
    kernel = _gaussian_kernel(size, sigma, a).expand(channels, 1, size, size)
    pad = size // 2
    mode = "reflect" if a.shape[-2] > pad and a.shape[-1] > pad else "replicate"
    a = F.pad(a, (pad, pad, pad, pad), mode=mode)
    b = F.pad(b, (pad, pad, pad, pad), mode=mode)
    mu_a = F.conv2d(a, kernel, groups=channels)
    mu_b = F.conv2d(b, kernel, groups=channels)
    va = (F.conv2d(a * a, kernel, groups=channels) - mu_a * mu_a).clamp_min(0.0)
    vb = (F.conv2d(b * b, kernel, groups=channels) - mu_b * mu_b).clamp_min(0.0)
    cov = F.conv2d(a * b, kernel, groups=channels) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a.square() + mu_b.square() + c1) * (va + vb + c2))
    return score.mean(dim=1, keepdim=True).clamp(0.0, 1.0)


def pixel_confidence_mask(
    rendered: torch.Tensor,
    target: torch.Tensor,
    ssim_threshold: float = 0.8,
    analysis_width: Optional[int] = 640,
    *,
    analysis_size: Optional[Tuple[int, int]] = None,
    rgb_threshold: float = 0.12,
    rgb_tolerance: Optional[float] = None,
    neighborhood_radius: int = 2,
    brightness_tolerance: float = 0.08,
    return_score: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Return an HxW uint8 confidence mask (1 keeps supervision, 0 rejects it).

    ``ssim_threshold`` is the cutoff for the combined confidence score: the
    geometric mean of the multi-scale Gaussian-window SSIM score and the
    neighborhood-tolerant RGB score. It is not a threshold on SSIM alone.
    Analysis is downscaled to ``analysis_width`` (aspect ratio preserved)
    unless an explicit ``analysis_size=(height, width)`` is supplied. With
    ``return_score=True``, return a continuous HxW combined confidence map on
    the input device alongside the binary mask.
    """
    if rgb_tolerance is not None:
        rgb_threshold = rgb_tolerance
    if rendered.dim() != 3 or target.dim() != 3 or rendered.shape != target.shape:
        raise ValueError("rendered and target must have identical CxHxW shapes")
    if rendered.shape[0] not in (1, 3, 4) or min(rendered.shape[-2:]) < 1:
        raise ValueError("expected 1, 3, or 4 channels and non-empty spatial dimensions")
    if rendered.device != target.device:
        raise ValueError("rendered and target must be on the same device")
    if not rendered.is_floating_point() or not target.is_floating_point():
        raise TypeError("rendered and target must be floating point tensors")
    if not (0.0 <= ssim_threshold <= 1.0 and 0.0 <= rgb_threshold <= 1.0):
        raise ValueError("thresholds must be in [0, 1]")
    if neighborhood_radius < 0 or brightness_tolerance < 0:
        raise ValueError("neighborhood_radius and brightness_tolerance must be non-negative")

    height, width = rendered.shape[-2:]
    if analysis_size is None:
        if analysis_width is None or analysis_width <= 0:
            raise ValueError("analysis_width must be positive, or specify analysis_size")
        scale = min(1.0, float(analysis_width) / width)
        out_h, out_w = max(1, int(round(height * scale))), max(1, int(round(width * scale)))
    else:
        out_h, out_w = analysis_size
        if out_h <= 0 or out_w <= 0:
            raise ValueError("analysis_size dimensions must be positive")

    # Float32 keeps convolution stable for half precision render buffers.
    a = rendered.unsqueeze(0).float()
    b = target.unsqueeze(0).float()
    if (out_h, out_w) != (height, width):
        a = F.interpolate(a, size=(out_h, out_w), mode="bilinear", align_corners=False)
        b = F.interpolate(b, size=(out_h, out_w), mode="bilinear", align_corners=False)
    a, b = a.clamp(0.0, 1.0), b.clamp(0.0, 1.0)

    # Remove a robust global per-channel exposure offset before local tests.
    offset = (b - a).flatten(2).median(dim=2).values.view(1, -1, 1, 1)
    offset = offset.clamp(-brightness_tolerance, brightness_tolerance)
    corrected_a = (a + offset).clamp(0.0, 1.0)

    short_side = min(out_h, out_w)
    windows = [size for size in (11, 7, 3) if size <= short_side]
    if not windows:
        windows = [1]
    ssim_maps = [_ssim_map(corrected_a, b, size, max(0.5, size / 6.0)) for size in windows]
    ssim_score = torch.stack(ssim_maps, dim=0).mean(dim=0)

    # Allow a nearby matching RGB sample to account for tiny registration shifts.
    residual = (corrected_a - b).abs().mean(dim=1, keepdim=True)
    radius = min(neighborhood_radius, max(out_h, out_w) - 1)
    if radius:
        k = 2 * radius + 1
        local_error = -F.max_pool2d(-residual, kernel_size=k, stride=1, padding=radius)
    else:
        local_error = residual
    rgb_score = (1.0 - local_error / max(rgb_threshold, 1e-6)).clamp(0.0, 1.0)

    # Geometric mean requires both structure and color evidence to agree.
    score = torch.sqrt(ssim_score * rgb_score).clamp(0.0, 1.0)
    if (out_h, out_w) != (height, width):
        score = F.interpolate(score, size=(height, width), mode="bilinear", align_corners=False)
    score = score[0, 0].to(dtype=torch.float32)
    mask = (score >= ssim_threshold).to(dtype=torch.uint8)
    return (mask, score) if return_score else mask


__all__ = ["pixel_confidence_mask"]
