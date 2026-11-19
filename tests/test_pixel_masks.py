import unittest
import torch

from utils.pixel_masks import pixel_confidence_mask


def _image(height=97, width=151, device="cpu"):
    y = torch.linspace(0.0, 1.0, height, device=device).view(1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, width)
    pattern = (0.35 + 0.25 * x + 0.20 * y + 0.12 * torch.sin(17 * x + 9 * y)).clamp(0, 1)
    return torch.cat((pattern, pattern * 0.8, pattern * 0.6), dim=0)


class PixelMaskTests(unittest.TestCase):
    def test_identical_images_are_trusted_at_non_divisible_size(self):
        image = _image()
        mask, score = pixel_confidence_mask(image, image, return_score=True)
        assert mask.shape == image.shape[-2:]
        assert mask.dtype == torch.uint8
        assert mask.device == image.device
        assert torch.all(mask == 1)
        assert torch.allclose(score, torch.ones_like(score), atol=1e-5)
    
    
    def test_local_change_is_rejected_with_coherent_boundary(self):
        image = _image()
        changed = image.clone()
        changed[:, 30:65, 60:100] = 1.0 - changed[:, 30:65, 60:100]
        mask = pixel_confidence_mask(image, changed, analysis_width=113, ssim_threshold=0.8)
        assert mask[45:50, 75:85].sum() == 0
        assert mask[:, :35].float().mean() > 0.95
        # The score is continuous at source resolution, rather than replicated blocks.
        _, score = pixel_confidence_mask(image, changed, analysis_width=113, return_score=True)
        assert score[45:50, 75:85].mean() < 0.5
        assert score.shape == image.shape[-2:]
    
    
    def test_small_translation_and_global_brightness_are_tolerated(self):
        image = _image(64, 91)
        shifted = torch.roll(image, shifts=(1, 1), dims=(-2, -1))
        shifted = (shifted + 0.025).clamp(0, 1)
        mask = pixel_confidence_mask(image, shifted, analysis_width=640)
        assert mask[8:-8, 8:-8].float().mean() > 0.90
    
    
    def test_small_image_and_explicit_analysis_size(self):
        image = _image(3, 5)
        mask = pixel_confidence_mask(image, image, analysis_size=(2, 3))
        assert mask.shape == (3, 5)
        assert mask.dtype == torch.uint8
        assert mask.min().item() == 1 and mask.max().item() == 1
    
    
    def test_value_semantics_and_device(self):
        image = _image(16, 23)
        changed = image.clone()
        changed[:, 5:11, 7:16] = 1.0 - changed[:, 5:11, 7:16]
        mask = pixel_confidence_mask(image, changed, ssim_threshold=0.8)
        assert set(mask.unique().tolist()).issubset({0, 1})
        assert mask.device == image.device
        assert mask[7, 10].item() == 0
        if torch.cuda.is_available():
            cuda_mask = pixel_confidence_mask(image.cuda(), image.cuda())
            assert cuda_mask.device.type == "cuda"
            assert cuda_mask.dtype == torch.uint8
    
    
    def test_invalid_range_and_shape_are_rejected(self):
        image = _image(8, 9)
        with self.assertRaises(ValueError):
            pixel_confidence_mask(image, image, ssim_threshold=1.1)
        with self.assertRaises(ValueError):
            pixel_confidence_mask(image, image[:, :, :-1])
