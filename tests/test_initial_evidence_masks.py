import numpy as np
import torch
import unittest

from utils.initial_evidence_masks import InitialEvidenceMask


class _Model:
    def __init__(self):
        self.embedding_inputs = []

    def get_image_embeddings(self, image):
        self.embedding_inputs.append(image.detach().clone())
        # Deterministic, nonzero Cx4x4 embeddings, independent of model weights.
        pooled = torch.nn.functional.adaptive_avg_pool2d(image, (4, 4))
        return torch.cat((pooled, torch.ones_like(pooled[:, :1])), dim=1)


class _Proposals:
    def __init__(self, first, second):
        self.proposals = [first, second]
        self.calls = []

    def predict_multiscale(self, model, torch_module, image, domain, **kwargs):
        self.calls.append((image.detach().clone(), domain.copy(), kwargs))
        masks = self.proposals[len(self.calls) - 1]
        return masks, None, None, None, None


class _SSIM:
    @staticmethod
    def ssim_evidence(before, after, valid_masks):
        return {"candidate_masks": np.zeros((1,) + valid_masks[0].shape, dtype=bool)}


def _confidence(target, rendered, **kwargs):
    score = torch.ones(target.shape[-2:], dtype=torch.float32, device=target.device)
    score[3, 0] = 0.1  # low-confidence unknown, even though it is far from change
    return (torch.ones_like(score, dtype=torch.uint8), score)


class _LocalEvidence:
    def __init__(self):
        self.arguments = None

    def build_local_change_evidence(self, before, after, feature, ssim, valid,
                                    pre_masks=None, post_masks=None,
                                    alignment_radius=None, external_pose_verified=None):
        self.arguments = {
            "pre_masks": pre_masks.copy(), "post_masks": post_masks.copy(),
            "external_pose_verified": external_pose_verified,
        }
        change = np.zeros(valid.shape, dtype=bool)
        unknown = np.ones(valid.shape, dtype=bool)
        change[2, 3] = True
        return {
            "change_mask": change,
            "unknown_mask": unknown,
            "residual_map": np.zeros(valid.shape, dtype=np.float32),
            "diagnostics": {"test_stub": True},
        }


class InitialEvidenceMaskTest(unittest.TestCase):
    def test_h0_proposals_use_conservative_fusion_and_preserve_polarity(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = InitialEvidenceMask.__new__(InitialEvidenceMask)
        generator.device = device
        generator.analysis_width = 8
        generator.grid_sizes = (2,)
        generator.proposal_batch_size = 4
        generator.alignment_radius = 0
        generator.refine_unknown = True
        generator.fuse_h0_proposals = True
        generator._torch = torch
        generator.model = _Model()
        from scipy.ndimage import distance_transform_edt
        generator._distance_transform_edt = distance_transform_edt
        generator._pixel_confidence_mask = _confidence

        pre = np.zeros((1, 4, 5), dtype=bool)
        post = np.zeros((1, 4, 5), dtype=bool)
        pre[0, 1:3, 1:3] = True
        post[0, 2:4, 3:5] = True
        proposals = _Proposals(pre, post)
        evidence = _LocalEvidence()
        generator._proposals = proposals
        generator._ssim = _SSIM()
        generator._local = evidence

        rendered = torch.full((3, 4, 5), 0.7, dtype=torch.float32)
        target = torch.full((3, 4, 5), 0.2, dtype=torch.float32)
        mask, debug = generator(rendered, target, return_debug=True)

        self.assertEqual(tuple(mask.shape), (4, 5))
        self.assertEqual(mask.dtype, torch.uint8)
        self.assertEqual(mask.device.type, torch.device(device).type)
        self.assertEqual(mask[2, 3].item(), 0)  # raw change is always retained
        self.assertEqual(mask[0, 4].item(), 1)  # high-confidence distant unknown recovers stable
        self.assertEqual(mask[3, 0].item(), 0)  # low-confidence distant unknown remains rejected
        self.assertEqual(len(proposals.calls), 1)
        self.assertEqual(debug["analysis_hw"], (4, 5))
        self.assertLess(abs(generator.model.embedding_inputs[0].mean().item() - target.mean().item()), 1e-6)
        self.assertLess(abs(generator.model.embedding_inputs[1].mean().item() - rendered.mean().item()), 1e-6)
        self.assertLess(abs(proposals.calls[0][0].mean().item() - target.mean().item()), 1e-6)
        np.testing.assert_array_equal(evidence.arguments["pre_masks"], pre)
        self.assertEqual(evidence.arguments["post_masks"].shape, (0, 4, 5))
        self.assertIs(evidence.arguments["external_pose_verified"], False)
        self.assertEqual(debug["sam_mask_influence"], "explicit_conservative_h0_pre_proposal_fusion")
        self.assertEqual(debug["h0_proposals_accepted"], 0)  # 20% proposal is too large
        self.assertEqual(debug["post_proposals_generated"], False)
        self.assertEqual(tuple(debug["class_labels"].shape), (4, 5))
        self.assertEqual(debug["class_labels"][2, 3].item(), 2)
        self.assertEqual(debug["raw_unknown"][0, 4].item(), 1)
        self.assertEqual(debug["refined_unknown"][0, 4].item(), 0)
        self.assertEqual(debug["raw_unknown"][3, 0].item(), 1)
        self.assertEqual(debug["refined_unknown"][3, 0].item(), 1)

        generator.refine_unknown = False
        generator.fuse_h0_proposals = False
        generator._pixel_confidence_mask = lambda *args, **kwargs: self.fail(
            "confidence computation should be skipped when refinement is disabled")
        proposals.calls = []
        unrefined_mask = generator(rendered, target)
        self.assertEqual(unrefined_mask[0, 4].item(), 0)
        self.assertEqual(proposals.calls, [])

    def test_supported_proposal_expands_change_but_weak_or_large_do_not(self):
        shape = (100, 100)
        raw_change = np.zeros(shape, dtype=bool)
        raw_unknown = np.zeros(shape, dtype=bool)
        accepted = np.zeros(shape, dtype=bool)
        accepted[10:30, 10:30] = True  # 4% image area
        raw_change[10:22, 10:20] = True  # 30% of proposal
        raw_unknown[22:28, 10:20] = True  # 15%, change margin is 15 pp

        weak = np.zeros(shape, dtype=bool)
        weak[40:60, 10:30] = True  # 4%; support margin below 10 pp
        raw_change[40:52, 10:20] = True  # 30%
        raw_unknown[52:62, 10:20] = True  # 25%

        oversized = np.zeros(shape, dtype=bool)
        oversized[50:80, 50:100] = True  # 15% image area
        raw_change[50:70, 50:80] = True
        raw_unknown[70:80, 50:80] = True
        proposals = np.stack((accepted, weak, oversized))
        fused, added, accepted_count, audit = InitialEvidenceMask._merge_supported_pre_proposals(
            raw_change, raw_unknown, raw_unknown, proposals)

        self.assertEqual(accepted_count, 1)
        self.assertTrue(audit[0]["accepted"])
        self.assertFalse(audit[1]["accepted"])
        self.assertIn("change_unknown_margin", audit[1]["rejection_reasons"])
        self.assertFalse(audit[2]["accepted"])
        self.assertIn("area", audit[2]["rejection_reasons"])
        self.assertTrue(np.all(fused[raw_change]))
        self.assertEqual(int(added.sum()), 220)
        self.assertFalse(np.any(added & raw_unknown))
        # Final mask preserves both change and unknown, and rejects only the
        # newly supported stable pixels in the accepted proposal.
        final_mask = (~fused & ~raw_unknown).astype(np.uint8)
        self.assertEqual(final_mask[15, 15], 0)
        self.assertEqual(final_mask[23, 12], 0)
        self.assertEqual(final_mask[15, 25], 0)
        self.assertEqual(final_mask[45, 25], 1)
        self.assertEqual(final_mask[55, 55], 0)  # raw change in rejected large proposal stays rejected
        self.assertEqual(final_mask[55, 90], 1)  # stable pixels in rejected large proposal stay stable
