"""Render held-out views of a trained candidate model and report image metrics."""
import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import lpips
from PIL import Image, ImageDraw
import torch
import torchvision
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.loss_utils import ssim
from holdout_split_validation import validate_training_split


def read_config(path):
    node = ast.parse(Path(path).read_text(encoding='utf-8-sig'), mode='eval').body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'Namespace' and not node.args):
        raise ValueError('Unsupported cfg_args format')
    values = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError('Unsupported cfg_args expansion')
        values[kw.arg] = ast.literal_eval(kw.value)
    return SimpleNamespace(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='DA model directory containing cfg_args')
    parser.add_argument('--scene', required=True, help='Prepared scene used during training')
    parser.add_argument('--output', required=True, help='New evaluation output directory')
    parser.add_argument('--iteration', type=int, default=-1, help='Saved iteration, or -1 for latest')
    args = parser.parse_args()
    model_path, scene_path, out = [Path(x).resolve() for x in (args.model, args.scene, args.output)]
    if out.exists():
        raise FileExistsError('Output already exists: ' + str(out))
    config = read_config(model_path / 'cfg_args')
    config.model_path = str(model_path)
    config.source_path = str(scene_path)
    config.eval = True
    split_verified = validate_training_split(model_path, scene_path / 'split.json')
    gaussians = GaussianModel(config.sh_degree)
    scene = Scene(config, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTestCameras()
    if not cameras:
        raise ValueError('No held-out new-time cameras found in split.json')
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1., 1., 1.] if config.white_background else [0., 0., 0.], device='cuda')
    perceptual = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
    for subdir in ('renders', 'reference', 'comparisons'):
        (out / subdir).mkdir(parents=True, exist_ok=False)
    results = []
    with torch.no_grad():
        for cam in cameras:
            pred = render(cam, gaussians, pipeline, background)['render'].clamp(0, 1)
            target = cam.original_image.cuda().clamp(0, 1)
            mse = torch.mean((pred - target) ** 2).item()
            psnr_value = -10 * math.log10(mse) if mse > 0 else float('inf')
            ssim_value = ssim(pred, target).item()
            lpips_value = perceptual(pred.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
            row = {'image': cam.image_name, 'psnr_db': psnr_value, 'ssim': ssim_value, 'lpips_alex': lpips_value}
            results.append(row)
            render_path = out / 'renders' / (cam.image_name + '.png')
            target_path = out / 'reference' / (cam.image_name + '.png')
            torchvision.utils.save_image(pred, str(render_path))
            torchvision.utils.save_image(target, str(target_path))
            with Image.open(target_path) as left, Image.open(render_path) as right:
                panel = Image.new('RGB', (left.width + right.width, max(left.height, right.height) + 35), 'white')
                panel.paste(left.convert('RGB'), (0, 35))
                panel.paste(right.convert('RGB'), (left.width, 35))
                draw = ImageDraw.Draw(panel)
                draw.text((12, 10), 'Reference', fill='black')
                draw.text((left.width + 12, 10), 'Rendered', fill='black')
                panel.save(out / 'comparisons' / (cam.image_name + '.png'))
            print('{}: PSNR={:.3f} dB, SSIM={:.5f}, LPIPS-Alex={:.5f}'.format(
                cam.image_name, psnr_value, ssim_value, lpips_value), flush=True)
    keys = ('psnr_db', 'ssim', 'lpips_alex')
    averages = {k: sum(row[k] for row in results) / len(results) for k in keys}
    report = {'iteration': scene.loaded_iter, 'scene': str(scene_path), 'model': str(model_path),
              'split_file': str(scene_path / 'split.json'), 'views': results, 'mean': averages,
              'training_split_verified': split_verified,
              'point_cloud_provenance': 'prepared input; test-view independence not established'}
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Mean:', averages, flush=True)
    print('Comparisons and metrics:', out, flush=True)


if __name__ == '__main__':
    main()
