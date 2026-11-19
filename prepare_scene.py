"""Copy a prepared COLMAP scene to an isolated working directory."""
import argparse
import json
import shutil
import struct
from pathlib import Path
from scene_split import validate_split


def read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise ValueError('Truncated COLMAP images.bin')
    return data


def read_image_names_binary(path):
    names = []
    with path.open('rb') as f:
        count, = struct.unpack('<Q', read_exact(f, 8))
        for _ in range(count):
            read_exact(f, 64)
            chars = bytearray()
            while True:
                char = read_exact(f, 1)
                if char == b'\0':
                    break
                chars.extend(char)
                if len(chars) > 16384:
                    raise ValueError('Invalid image name in COLMAP images.bin')
            names.append(chars.decode('utf-8'))
            count_2d, = struct.unpack('<Q', read_exact(f, 8))
            remaining = count_2d * 24
            while remaining:
                remaining -= len(read_exact(f, min(remaining, 1024 * 1024)))
    return names


def read_image_names_text(path):
    names = []
    with path.open('r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError('Malformed image entry in ' + str(path))
            names.append(fields[9].strip())
            if f.readline() == '':
                raise ValueError('Missing points line in ' + str(path))
    return names


def build_split(source, test_list=None):
    source = Path(source)
    sparse = source / 'sparse' / '0'
    if (sparse / 'images.bin').is_file() and (sparse / 'cameras.bin').is_file():
        names = read_image_names_binary(sparse / 'images.bin')
    elif (sparse / 'images.txt').is_file() and (sparse / 'cameras.txt').is_file():
        names = read_image_names_text(sparse / 'images.txt')
    else:
        raise ValueError('A matching pair of COLMAP images/cameras files is required')
    if not (sparse / 'points3D.ply').is_file():
        raise ValueError('A prepared sparse/0/points3D.ply is required')
    if not (source / 'mask').is_dir():
        raise ValueError('mask/ must contain marker files for new-time views')
    filenames = [Path(name).name for name in names]
    stems = [Path(name).stem for name in filenames]
    if len(set(x.casefold() for x in stems)) != len(stems):
        raise ValueError('Image stems must be unique, including case')
    missing = [name for name in filenames if not (source / 'images' / name).is_file()]
    if missing:
        raise ValueError('COLMAP images missing from images/: ' + ', '.join(missing[:10]))
    markers = {p.stem for p in (source / 'mask').iterdir() if p.is_file()}
    unknown = markers - set(stems)
    if unknown:
        raise ValueError('mask/ contains markers without COLMAP cameras: ' + ', '.join(sorted(unknown)[:10]))
    new, old = sorted(markers), sorted(set(stems) - markers)
    if test_list is None:
        test = new[::3]
        policy = 'sorted new views, every third starting at index 0 (candidate split)'
    else:
        test = [Path(line.strip()).stem for line in Path(test_list).read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
        if len(test) != len(set(test)) or not set(test).issubset(new):
            raise ValueError('The test view list must contain distinct new-view stems')
        policy = 'explicit test list'
    test_set = set(test)
    split = {'version': 1, 'policy': policy, 'train_old': old,
             'train_new': [name for name in new if name not in test_set], 'test_new': test}
    validate_split(split, stems)
    return split


def prepare_scene(source, destination, test_list=None):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate directories')
    split = build_split(source, test_list)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / 'images', destination / 'images')
    shutil.copytree(source / 'mask', destination / 'mask')
    out_sparse = destination / 'sparse' / '0'
    out_sparse.mkdir(parents=True)
    for name in ('images.bin', 'cameras.bin', 'images.txt', 'cameras.txt', 'points3D.ply'):
        path = source / 'sparse' / '0' / name
        if path.is_file():
            shutil.copy2(path, out_sparse / name)
    (destination / 'label').mkdir()
    (destination / '.cross_temporal_working_copy').write_text('Prepared copy; safe for generated labels.\n', encoding='utf-8')
    (destination / 'split.json').write_text(json.dumps(split, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Prepared input scene (never modified)')
    parser.add_argument('--destination', required=True, help='New directory for the working copy')
    parser.add_argument('--test-list', help='UTF-8 file listing one new test image name/stem per line')
    args = parser.parse_args()
    split = prepare_scene(args.source, args.destination, args.test_list)
    print('Prepared {} old train / {} new train / {} new test views'.format(
        len(split['train_old']), len(split['train_new']), len(split['test_new'])))
    print('Split:', Path(args.destination).resolve() / 'split.json')
    print('Point-cloud provenance must be checked before reporting test metrics.')


if __name__ == '__main__':
    main()
