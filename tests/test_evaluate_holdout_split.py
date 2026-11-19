import json
import tempfile
import unittest
from pathlib import Path

from holdout_split_validation import validate_training_split


class HoldoutSplitValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.scene = self.root / 'external-scene'
        self.scene.mkdir()
        self.trained = {'train_old': ['old0', 'old1'], 'train_new': ['new0', 'new1'],
                        'test_new': ['test0', 'test1']}
        self.write(self.model / 'run.json', {'source': 'some/other/absolute/path', 'split': self.trained})
        self.split_path = self.scene / 'split.json'

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_identical_split_passes(self):
        self.write(self.split_path, self.trained)
        self.assertTrue(validate_training_split(self.model, self.split_path))

    def test_reordered_views_pass(self):
        current = {key: list(reversed(names)) for key, names in self.trained.items()}
        self.write(self.split_path, current)
        self.assertTrue(validate_training_split(self.model, self.split_path))

    def test_test_view_used_for_training_fails(self):
        current = dict(self.trained)
        current['train_new'] = ['new0', 'test1']
        current['test_new'] = ['new1', 'test0']
        self.write(self.split_path, current)
        with self.assertRaisesRegex(ValueError, 'train_new views differ'):
            validate_training_split(self.model, self.split_path)

    def test_legacy_model_without_split_metadata_is_not_blocked(self):
        (self.model / 'run.json').unlink()
        self.write(self.split_path, self.trained)
        self.assertFalse(validate_training_split(self.model, self.split_path))


if __name__ == '__main__':
    unittest.main()
