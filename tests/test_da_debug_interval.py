"""Exercise DA debug snapshots through the training loop with a tiny fake scene."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

import train_upda


class QuietProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


class DADebugIntervalTests(unittest.TestCase):
    def run_fake_training(self, start, interval, iterations, debug_interval,
                          checkpoint_iteration=None, debug=True, debug_names=None):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "mask").mkdir()
            (root / "label").mkdir()
            initial = np.array([[255, 0], [0, 0]], dtype=np.uint8)
            Image.fromarray(initial).save(str(root / "label" / "old_view.png"))
            camera = SimpleNamespace(
                image_name="old_view",
                original_image=torch.full((3, 2, 2), 0.5),
                mask=torch.zeros((2, 2), dtype=torch.uint8),
            )
            scene = SimpleNamespace(model_path=str(root / "model"),
                                    getTrainCameras=lambda: [camera])
            model = mock.Mock()
            dataset = SimpleNamespace(source_path=str(root), model_path=scene.model_path,
                                      sh_degree=0, white_background=False)
            opt = SimpleNamespace(iterations=iterations, random_background=False,
                                  lambda_dssim=0.2, densify_until_iter=0)
            candidate = torch.tensor([[0, 1], [0, 0]], dtype=torch.uint8)

            def fake_render(cam, *args):
                return {"render": cam.original_image.detach().clone().requires_grad_(),
                        "viewspace_points": None, "visibility_filter": None, "radii": None}

            original_tensor = torch.tensor

            def cpu_tensor(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return original_tensor(*args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(train_upda, "prepare_output_and_logger", return_value=None))
                stack.enter_context(mock.patch.object(train_upda, "GaussianModel", return_value=model))
                stack.enter_context(mock.patch.object(train_upda, "Scene", return_value=scene))
                stack.enter_context(mock.patch.object(train_upda, "render", side_effect=fake_render))
                stack.enter_context(mock.patch.object(train_upda, "pixel_confidence_mask", return_value=candidate))
                stack.enter_context(mock.patch.object(train_upda, "ssim", side_effect=lambda a, b: a.sum() * 0 + 1))
                stack.enter_context(mock.patch.object(train_upda, "training_report"))
                stack.enter_context(mock.patch.object(train_upda, "tqdm", QuietProgress))
                stack.enter_context(mock.patch.object(torch, "tensor", side_effect=cpu_tensor))
                stack.enter_context(mock.patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor))
                stack.enter_context(mock.patch.object(torch.cuda, "Event"))
                stack.enter_context(mock.patch.dict("os.environ", {"GS_SAVE_MASKED_IMAGES": "0"}))
                updates = stack.enter_context(mock.patch.object(
                    train_upda, "update_masks_ssim", wraps=train_upda.update_masks_ssim))
                checkpoint = None
                if checkpoint_iteration is not None:
                    checkpoint = "fake-checkpoint"
                    stack.enter_context(mock.patch.object(torch, "load", return_value=("params", checkpoint_iteration)))
                train_upda.training(
                    dataset, opt, SimpleNamespace(debug=False), [], [], [], checkpoint, -1,
                    mask_onn=True, mask_update_start=start, mask_update_interval=interval,
                    skip_final_eval=True,
                    debug_mask_names=(debug_names if debug_names is not None
                                      else (["old_view"] if debug else [])),
                    debug_mask_interval=debug_interval,
                )
                calls = [(call[1]["iteration"], call[1]["update_index"]) for call in updates.call_args_list]

            snapshot_dir = root / "model" / "mask_debug" / "old_view"
            snapshots = {}
            if snapshot_dir.is_dir():
                for path in snapshot_dir.glob("*.png"):
                    with Image.open(str(path)) as image:
                        snapshots[path.name] = np.array(image)
            return calls, snapshots, camera.mask.numpy().copy()

    def test_default_schedule_saves_tenth_actual_update_at_1943(self):
        calls, snapshots, final_mask = self.run_fake_training(900, 108, 1943, 10)
        self.assertEqual(calls, [(971 + 108 * i, i + 1) for i in range(10)])
        self.assertEqual(set(snapshots), {"initial.png", "update_001943.png", "final.png"})
        np.testing.assert_array_equal(snapshots["initial.png"], [[255, 0], [0, 0]])
        np.testing.assert_array_equal(snapshots["update_001943.png"], [[255, 255], [0, 0]])
        np.testing.assert_array_equal(snapshots["final.png"], final_mask * 255)

    def test_nondefault_start_counts_updates_and_debug_does_not_change_masks(self):
        calls, snapshots, final_mask = self.run_fake_training(23, 7, 49, 2)
        no_debug_calls, no_debug_snapshots, no_debug_final = self.run_fake_training(23, 7, 49, 2, debug=False)
        self.assertEqual(calls, [(27, 1), (34, 2), (41, 3), (48, 4)])
        self.assertEqual(no_debug_calls, calls)
        self.assertEqual(set(snapshots), {"initial.png", "update_000034.png", "update_000048.png", "final.png"})
        self.assertEqual(no_debug_snapshots, {})
        np.testing.assert_array_equal(final_mask, no_debug_final)

    def test_checkpoint_resume_starts_a_new_update_count(self):
        calls, snapshots, _ = self.run_fake_training(23, 7, 62, 2, checkpoint_iteration=35)
        self.assertEqual(calls, [(41, 1), (48, 2), (55, 3), (62, 4)])
        self.assertEqual(set(snapshots), {"initial.png", "update_000048.png", "update_000062.png", "final.png"})

    def test_debug_name_normalization_supports_extensions_paths_and_dotted_stems(self):
        self.assertEqual(train_upda.normalize_debug_mask_names([
            "t0_NewLevelSequence_0003.jpeg", "plain_stem", "frame.v2.png",
            r"folder\\with spaces\\frame.v2.JPEG", "  spaced_name.jpg  ",
        ]), {"t0_NewLevelSequence_0003", "plain_stem", "frame.v2", "spaced_name"})

    def test_debug_name_normalization_handles_empty_inputs(self):
        self.assertEqual(train_upda.normalize_debug_mask_names(None), set())
        self.assertEqual(train_upda.normalize_debug_mask_names([]), set())
        self.assertEqual(train_upda.normalize_debug_mask_names([None, "", "  "]), set())

    def test_snapshot_selector_matches_camera_stem_when_cli_name_has_extension(self):
        calls, snapshots, _ = self.run_fake_training(
            0, 2, 3, 1, debug_names=["folder/old_view.jpeg"])
        self.assertEqual(calls, [(1, 1), (3, 2)])
        self.assertTrue(snapshots)
        self.assertEqual(set(snapshots), {
            "initial.png", "update_000001.png", "update_000003.png", "final.png"})


if __name__ == "__main__":
    unittest.main()
