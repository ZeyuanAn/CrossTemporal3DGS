"""Run the candidate two-stage update on an isolated prepared scene."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_scene import read_image_names_binary, read_image_names_text
from scene_split import read_split

ROOT = Path(__file__).resolve().parent


def check_scene(scene):
    scene = Path(scene).resolve()
    if not (scene / '.cross_temporal_working_copy').is_file():
        raise ValueError('Use prepare_scene.py first; refusing to modify an unmarked source dataset')
    sparse = scene / 'sparse' / '0'
    if (sparse / 'images.bin').is_file() and (sparse / 'cameras.bin').is_file():
        names = read_image_names_binary(sparse / 'images.bin')
    else:
        names = read_image_names_text(sparse / 'images.txt')
    split = read_split(scene / 'split.json', [Path(name).stem for name in names])
    marker_stems = {p.stem for p in (scene / 'mask').iterdir() if p.is_file()}
    if marker_stems != set(split['train_new'] + split['test_new']):
        raise ValueError('mask/ marker files do not match split.json new-time views')
    labels = list((scene / 'label').iterdir())
    if labels:
        raise ValueError('Working copy already contains labels; prepare a fresh copy for a new run')
    return scene, split


def launch(command, env):
    print('\nRunning:', ' '.join(str(x) for x in command), flush=True)
    subprocess.run([str(x) for x in command], cwd=str(ROOT), env=env, check=True)


def launch_stage(command, env, stage, settings, save):
    """Launch one stage and persist subprocess failures before re-raising them."""
    try:
        launch(command, env)
    except subprocess.CalledProcessError as exc:
        settings['status'] = 'failed'
        settings['failed_stage'] = stage
        failure = {
            'returncode': exc.returncode,
            'command': [str(value) for value in command],
            'error': str(exc),
        }
        for key, value in (('stdout', exc.stdout), ('stderr', exc.stderr)):
            if value is not None:
                failure[key] = value.decode('utf-8', errors='replace') if isinstance(value, bytes) else str(value)
        settings['failure'] = failure
        save()
        raise


def check_environment(env):
    probe = (
        "import torch, torchvision, numpy, PIL, plyfile, pytorch_msssim, lpips; "
        "from simple_knn._C import distCUDA2; "
        "from diff_gaussian_rasterization import GaussianRasterizer; "
        "assert torch.cuda.is_available(), 'CUDA is not available'; "
        "print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, "
        "'GPU', torch.cuda.get_device_name(0))"
    )
    print("Checking Python, CUDA, and compiled extensions...", flush=True)
    try:
        subprocess.run([sys.executable, "-c", probe], env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Environment check failed; see the traceback above and README.md") from exc


def build_stage_commands(args, scene, output):
    """Build BU/DA argv lists without launching either training process."""
    bu_switch, bu_end = (1, 3) if args.smoke else (3900, 5900)
    da_iters = 3 if args.smoke else args.stage2_iterations
    resolution = 8 if args.smoke and args.resolution == -1 else args.resolution
    common = ['-s', scene, '--eval', '--mask', 'True', '-r', str(resolution)]
    bu = [sys.executable, ROOT / 'train_upbu.py', *common, '-m', output / 'bu',
          '--iterations', str(bu_end), '--phase_switch_iter', str(bu_switch),
          '--mask_update_iter', str(bu_end), '--threshold', str(args.bu_threshold),
          '--block_size', str(args.bu_block), '--mask_method', args.initial_mask_method,
          '--mask_analysis_width', str(args.mask_analysis_width),
          '--sam_grid_size', str(args.sam_grid_size), '--skip_final_eval']
    if args.sam_checkpoint:
        bu += ['--sam_checkpoint', str(args.sam_checkpoint)]
    da = [sys.executable, ROOT / 'train_upda.py', *common, '-m', output / 'da',
          '--iterations', str(da_iters), '--threshold', str(args.da_threshold),
          '--block_size', str(args.da_block), '--mask_method', args.mask_method,
          '--mask_analysis_width', str(args.mask_analysis_width),
          '--debug_mask_interval', str(args.debug_mask_interval)]
    if args.debug_mask_names:
        da += ['--debug_mask_names', *args.debug_mask_names]
    if args.smoke:
        da += ['--mask_update_start', '0', '--mask_update_interval', '2']
    if args.smoke or not args.evaluate:
        da += ['--skip_final_eval']
    return bu, da, da_iters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True, help='Working copy produced by prepare_scene.py')
    p.add_argument('--output', required=True, help='New output directory')
    p.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES setting')
    p.add_argument('--resolution', type=int, default=-1)
    p.add_argument('--bu-threshold', type=float, default=0.8,
                   help='BU confidence cutoff; applies to pixel/block masks, not adaptive evidence masks')
    p.add_argument('--bu-block', type=int, default=16)
    p.add_argument('--initial-mask-method', choices=('evidence', 'pixel', 'block'), default='pixel')
    p.add_argument('--mask-method', choices=('pixel', 'block'), default='pixel',
                   help='DA iterative mask method')
    p.add_argument('--sam-checkpoint', type=Path, default=None)
    p.add_argument('--sam-grid-size', type=int, default=8)
    p.add_argument('--mask-analysis-width', type=int, default=640)
    p.add_argument('--da-threshold', type=float, default=0.92)
    p.add_argument('--da-block', type=int, default=32)
    p.add_argument('--debug-mask-names', nargs='+', default=[])
    p.add_argument('--debug-mask-interval', type=int, default=1,
                   help='Save a DA debug mask every N actual mask updates')
    p.add_argument('--stage2-iterations', type=int, default=7000)
    p.add_argument('--evaluate', action='store_true', help='Compute final DA LPIPS/SSIM/PSNR; LPIPS may download weights')
    p.add_argument('--smoke', action='store_true', help='Exercise both stages and label update in a few iterations')
    args = p.parse_args()
    if (args.stage2_iterations < 1 or args.bu_block < 1 or args.da_block < 1
            or args.sam_grid_size < 1 or args.mask_analysis_width < 1
            or args.debug_mask_interval < 1):
        raise ValueError('Iterations, block sizes, mask dimensions, and debug interval must be positive')
    if args.initial_mask_method == 'evidence':
        if args.sam_checkpoint is None or not args.sam_checkpoint.is_file():
            raise FileNotFoundError('--initial-mask-method evidence requires --sam-checkpoint pointing to an existing weights file')
        args.sam_checkpoint = args.sam_checkpoint.resolve()
    scene, split = check_scene(args.scene)
    output = Path(args.output).resolve()
    if output == scene or scene in output.parents or output in scene.parents:
        raise ValueError('Output and scene must be separate directories')
    if output.exists():
        raise FileExistsError('Output directory already exists: ' + str(output))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['GS_SAVE_MASKED_IMAGES'] = '0'
    # Honour the source default for -r -1; explicit -r values still resize.
    env.pop('GS_KEEP_NATIVE_RES', None)
    check_environment(env)
    if args.initial_mask_method == 'evidence':
        adapter_probe = 'from utils.initial_evidence_masks import InitialEvidenceMask'
        subprocess.run([sys.executable, '-c', adapter_probe], cwd=str(ROOT), env=env, check=True)
    output.mkdir(parents=True, exist_ok=False)
    settings = {'source': str(scene), 'gpu': args.gpu, 'smoke': args.smoke,
                'split': split, 'point_cloud_provenance': 'unverified prepared input',
                'bu_switch_iteration': 1 if args.smoke else 3900,
                'bu_mask_iteration': 3 if args.smoke else 5900,
                'resolution': 8 if args.smoke and args.resolution == -1 else args.resolution,
                'evaluate': args.evaluate and not args.smoke,
                'bu_threshold': args.bu_threshold, 'bu_block': args.bu_block,
                'bu_threshold_effective': args.initial_mask_method in ('pixel', 'block'),
                'bu_threshold_applies_to': ['pixel', 'block'],
                'bu_threshold_note': ('Used as the BU confidence cutoff for pixel/block masks; '
                                      'evidence masks use the reference algorithm adaptive threshold.'),
                'initial_mask_method': args.initial_mask_method,
                'mask_method': args.mask_method,
                'sam_checkpoint': str(args.sam_checkpoint) if args.sam_checkpoint else None,
                'sam_grid_size': args.sam_grid_size,
                'mask_analysis_width': args.mask_analysis_width,
                'da_debug_mask_names': args.debug_mask_names,
                'da_debug_mask_interval': args.debug_mask_interval,
                'da_threshold': args.da_threshold, 'da_block': args.da_block,
                'stage2_iterations': 3 if args.smoke else args.stage2_iterations,
                'status': 'started'}
    record = output / 'run.json'
    def save():
        record.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save()
    bu, da, da_iters = build_stage_commands(args, scene, output)
    launch_stage(bu, env, 'bu', settings, save)
    expected = split['train_old'] + split['train_new']
    missing = [name for name in expected if not (scene / 'label' / (name + '.png')).is_file()
               or (scene / 'label' / (name + '.png')).stat().st_size == 0]
    if missing:
        raise RuntimeError('BU did not produce all labels: ' + ', '.join(missing[:10]))
    settings['status'] = 'bu_complete'
    save()
    launch_stage(da, env, 'da', settings, save)
    ply = output / 'da' / 'point_cloud' / ('iteration_' + str(da_iters)) / 'point_cloud.ply'
    if not ply.is_file() or not ply.stat().st_size:
        raise RuntimeError('DA completed without its expected point-cloud output: ' + str(ply))
    settings['status'] = 'complete'
    settings['output_ply'] = str(ply)
    save()
    print('\nTwo-stage candidate run completed:', ply)
    print('This alone does not establish paper fidelity or test-set independence.')


if __name__ == '__main__':
    main()
