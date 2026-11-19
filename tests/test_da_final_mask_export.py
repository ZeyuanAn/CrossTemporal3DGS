"""Verify final DA mask export with stub cameras and no renderer."""
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

import test_da_debug_interval as debug_fixture
import train_upda


class FinalMaskExportTests(unittest.TestCase):
    def test_exports_only_old_views_with_binary_polarity_and_no_mutation(self):
        with TemporaryDirectory() as temp:
            first = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)
            second = torch.tensor([[1, 0, 1]], dtype=torch.uint8)
            new_view = torch.zeros((2, 2), dtype=torch.uint8)
            cameras = [SimpleNamespace(image_name="old_a", mask=first),
                       SimpleNamespace(image_name="new_a", mask=new_view),
                       SimpleNamespace(image_name="old_b", mask=second)]
            scene = SimpleNamespace(model_path=temp, getTrainCameras=lambda: cameras)
            before_masks = [camera.mask.clone() for camera in cameras]
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            before_torch = torch.random.get_rng_state().clone()

            train_upda.export_final_old_masks(scene, {"new_a"})

            folder = Path(temp) / "final_masks"
            self.assertEqual({p.name for p in folder.glob("*.png")}, {"old_a.png", "old_b.png"})
            with Image.open(str(folder / "old_a.png")) as image:
                np.testing.assert_array_equal(np.array(image), [[0, 255], [255, 0]])
            with Image.open(str(folder / "old_b.png")) as image:
                np.testing.assert_array_equal(np.array(image), [[255, 0, 255]])
            for camera, before in zip(cameras, before_masks):
                self.assertTrue(torch.equal(camera.mask, before))
            self.assertEqual(random.getstate(), before_python)
            after_numpy = np.random.get_state()
            self.assertEqual(before_numpy[0], after_numpy[0])
            np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
            self.assertEqual(before_numpy[2:], after_numpy[2:])
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_torch))

    def test_default_export_occurs_even_when_final_evaluation_is_skipped(self):
        original = train_upda.export_final_old_masks
        saved = []

        def record_export(scene, new_names):
            original(scene, new_names)
            with Image.open(str(Path(scene.model_path) / "final_masks" / "old_view.png")) as image:
                saved.append(np.array(image))

        fixture = debug_fixture.DADebugIntervalTests()
        with mock.patch.object(train_upda, "export_final_old_masks", side_effect=record_export):
            fixture.run_fake_training(23, 7, 49, 2)
        self.assertEqual(len(saved), 1)
        np.testing.assert_array_equal(saved[0], [[255, 255], [0, 0]])

    def test_explicit_disable_skips_export(self):
        original_training = train_upda.training

        def training_without_export(*args, **kwargs):
            kwargs["save_final_masks"] = False
            return original_training(*args, **kwargs)

        fixture = debug_fixture.DADebugIntervalTests()
        with mock.patch.object(train_upda, "training", side_effect=training_without_export):
            with mock.patch.object(train_upda, "export_final_old_masks") as export:
                fixture.run_fake_training(23, 7, 49, 2)
        export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
