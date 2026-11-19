import json
import struct
import tempfile
import unittest
from pathlib import Path

from prepare_scene import build_split, prepare_scene, read_image_names_binary
from run_pipeline import check_scene
from scene_split import validate_split


class PreparedSceneTests(unittest.TestCase):
    def make_source(self, root):
        source = Path(root) / 'original'
        (source / 'images').mkdir(parents=True)
        (source / 'mask').mkdir()
        (source / 'label').mkdir()
        sparse = source / 'sparse' / '0'
        sparse.mkdir(parents=True)
        names = ['old0.jpg', 'old1.jpg'] + ['new{}.jpg'.format(i) for i in range(6)]
        for name in names:
            (source / 'images' / name).write_bytes(b'image')
            if name.startswith('new'):
                (source / 'mask' / name).write_bytes(b'marker')
        (source / 'label' / 'old0.png').write_bytes(b'original label')
        with (sparse / 'images.bin').open('wb') as f:
            f.write(struct.pack('<Q', len(names)))
            for index, name in enumerate(names):
                f.write(struct.pack('<i7di', index + 1, 1, 0, 0, 0, 0, 0, 0, 1))
                f.write(name.encode('utf8') + b'\0')
                f.write(struct.pack('<Q', 0))
        (sparse / 'cameras.bin').write_bytes(b'camera placeholder')
        (sparse / 'points3D.ply').write_bytes(b'ply placeholder')
        return source, names

    def test_copy_is_isolated_and_split_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, names = self.make_source(tmp)
            destination = Path(tmp) / 'work' / 'scene'
            self.assertEqual(read_image_names_binary(source / 'sparse/0/images.bin'), names)
            split = prepare_scene(source, destination)
            self.assertEqual([len(split[k]) for k in ('train_old', 'train_new', 'test_new')], [2, 4, 2])
            self.assertEqual(split['test_new'], ['new0', 'new3'])
            self.assertEqual(list((destination / 'label').iterdir()), [])
            self.assertEqual((source / 'label/old0.png').read_bytes(), b'original label')
            self.assertEqual(check_scene(destination)[1], split)
            with self.assertRaises(FileExistsError):
                prepare_scene(source, destination)

    def test_refuses_overlap_or_mismatched_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, names = self.make_source(tmp)
            split = build_split(source)
            split['test_new'].append(split['train_new'][0])
            with self.assertRaisesRegex(ValueError, 'overlapping'):
                validate_split(split, [Path(n).stem for n in names])
            (source / 'mask' / 'unknown.jpg').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'without COLMAP'):
                build_split(source)

    def test_refuses_unmarked_or_reused_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = self.make_source(tmp)
            with self.assertRaisesRegex(ValueError, 'unmarked'):
                check_scene(source)
            destination = Path(tmp) / 'work'
            prepare_scene(source, destination)
            (destination / 'label' / 'stale.png').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'already contains labels'):
                check_scene(destination)

    def test_custom_test_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = self.make_source(tmp)
            test_list = Path(tmp) / 'test.txt'
            test_list.write_text('new1.jpg\nnew4\n', encoding='utf8')
            split = build_split(source, test_list)
            self.assertEqual(split['test_new'], ['new1', 'new4'])
            self.assertNotIn('new1', split['train_new'])


if __name__ == '__main__':
    unittest.main()
