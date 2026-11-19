#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
try:
    from pytorch_msssim import ssim as ssim2
except ImportError:
    ssim2 = None
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from cli_utils import parse_bool
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.pixel_masks import pixel_confidence_mask
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
import numpy as np
try:
    import lpips
except ImportError:
    lpips = None
import torchvision.utils as vutils


_DEBUG_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_debug_mask_name(name):
    """Normalize a debug image name to its path-free camera-name stem."""
    if name is None:
        return ""
    name = str(name).strip().replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name:
        return ""
    stem, extension = os.path.splitext(name)
    if extension.lower() in _DEBUG_IMAGE_EXTENSIONS:
        name = stem
    return name.strip()


def normalize_debug_mask_names(names):
    """Normalize optional CLI/API debug names, dropping empty values."""
    if names is None:
        return set()
    if isinstance(names, (str, bytes)):
        names = [names]
    normalized_names = set()
    for name in names:
        normalized = normalize_debug_mask_name(name)
        if normalized:
            normalized_names.add(normalized)
    return normalized_names

def compute_ssim_batch(rendered_blocks, gt_blocks, window_size=11):
    """ 计算一批图像块的 SSIM，使用 pytorch_msssim，在 GPU 上计算 """
    if ssim2 is None:
        raise ImportError("pytorch_msssim is required for --mask_method block")
    return ssim2(rendered_blocks, gt_blocks, data_range=1.0, size_average=False)

def pad_to_size(tensor, target_size):
    """ 将张量填充到指定大小 """
    C, H, W = tensor.shape
    pad_h = target_size - H
    pad_w = target_size - W
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

def update_masks_ssim(scene, gaussians, pipe, background, source_path, threshold=0.8, block_size=24,
                      mask_method="pixel", analysis_width=640, rgb_threshold=0.12,
                      debug_dir=None, debug_names=None, debug_interval=1, update_index=1,
                      iteration=0):
    if mask_method == "block" and block_size <= 0:
        raise ValueError("block_size must be positive for block masks")
    with torch.no_grad():
        train_cameras = scene.getTrainCameras()

        for viewpoint_cam in tqdm(train_cameras, desc="Updating masks"):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0).cuda()  # (C, H, W)
            gt_image = torch.clamp(viewpoint_cam.original_image.cuda(), 0.0, 1.0)
            mask_tensor = viewpoint_cam.mask.cuda()

            C, H, W = rendered_image.shape

            if mask_method == "pixel":
                candidate_mask = pixel_confidence_mask(
                    rendered_image, gt_image, ssim_threshold=threshold,
                    analysis_width=analysis_width, rgb_threshold=rgb_threshold)
            else:

                # 计算填充量，使尺寸为 block_size 的整数倍
                pad_h = (block_size - H % block_size) % block_size
                pad_w = (block_size - W % block_size) % block_size
                rendered_padded = F.pad(rendered_image, (0, pad_w, 0, pad_h))
                gt_padded = F.pad(gt_image, (0, pad_w, 0, pad_h))
                H_pad, W_pad = H + pad_h, W + pad_w

            # 分块处理
                nh, nw = H_pad // block_size, W_pad // block_size
                rendered_blocks = rendered_padded.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
                rendered_blocks = rendered_blocks.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, block_size, block_size)

                gt_blocks = gt_padded.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
                gt_blocks = gt_blocks.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, block_size, block_size)

            # 批量计算 SSIM
                ssim_values = compute_ssim_batch(rendered_blocks, gt_blocks, window_size=11)

            # 生成块级 mask 矩阵
                ssim_matrix = ssim_values.view(nh, nw)
                mask_matrix = (ssim_matrix > threshold).float()

            # 修复维度排列问题
                mask_expanded = mask_matrix.unsqueeze(-1).unsqueeze(-1).expand(nh, nw, block_size, block_size)
                mask_padded = mask_expanded.permute(0, 2, 1, 3).contiguous().view(H_pad, W_pad)

            # 裁剪到原始尺寸
                candidate_mask = mask_padded[:H, :W].to(torch.uint8)

            # 仅在原始 mask 为 0 的地方更新
            viewpoint_cam.mask = torch.where(mask_tensor == 0, candidate_mask, mask_tensor)
            if (debug_dir and normalize_debug_mask_name(viewpoint_cam.image_name) in debug_names
                    and update_index % debug_interval == 0):
                mask_path = os.path.join(debug_dir, viewpoint_cam.image_name,
                                         "update_{:06d}.png".format(iteration))
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                Image.fromarray((viewpoint_cam.mask.cpu().numpy() * 255).astype(np.uint8)).save(mask_path)

def export_final_old_masks(scene, new_view_names):
    """Save stored masks for old training views after optimization has ended."""
    output_dir = os.path.join(scene.model_path, "final_masks")
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for cam in scene.getTrainCameras():
            if cam.image_name in new_view_names:
                continue
            mask = (cam.mask.detach().cpu().numpy() > 0).astype(np.uint8) * 255
            Image.fromarray(mask).save(os.path.join(output_dir, cam.image_name + ".png"))


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mask_onn=False, threshold=0.8, block_size=24, mask_update_start=900, mask_update_interval=108, skip_final_eval=False, mask_method="pixel", analysis_width=640, rgb_threshold=0.12, debug_mask_names=None, debug_mask_interval=1, save_final_masks=True):
    if mask_method == "block" and block_size <= 0:
        raise ValueError("block_size must be positive for block masks")
    if not skip_final_eval and lpips is None:
        raise ImportError("lpips is required for final evaluation; install lpips or pass --skip_final_eval")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    debug_mask_names = normalize_debug_mask_names(debug_mask_names)
    debug_mask_dir = os.path.join(scene.model_path, "mask_debug") if debug_mask_names else None
    if debug_mask_dir:
        os.makedirs(debug_mask_dir, exist_ok=True)
    # 定义可能的文件后缀
    possible_extensions = ["", ".png", ".jpg", ".jpeg",".JPG"]

    with torch.no_grad():
        if mask_onn:
            train_cameras = scene.getTrainCameras()
            missing_labels = []
            for viewpoint_cam in tqdm(train_cameras, desc="Updating masks"):
                mask_found = False
                for ext in possible_extensions:
                    mask_path = os.path.join(dataset.source_path, "label", viewpoint_cam.image_name + ext)
                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path).convert('L')
                        mask_found = True
                        break
                if mask_found:
                    image_size = viewpoint_cam.original_image.shape[-2:]  # (height, width)
                    mask = mask.resize(image_size[::-1], Image.NEAREST)  # PIL 的 resize 需要 (width, height) 顺序
                    mask = torch.tensor(np.array(mask) > 0, dtype=torch.uint8, device=viewpoint_cam.mask.device)
                    viewpoint_cam.mask = mask
                else:
                    missing_labels.append(viewpoint_cam.image_name)
            if missing_labels:
                raise FileNotFoundError("BU labels missing for training views: " + ", ".join(missing_labels[:10]))

        if debug_mask_dir:
            for cam in scene.getTrainCameras():
                if normalize_debug_mask_name(cam.image_name) in debug_mask_names:
                    snapshot_path = os.path.join(debug_mask_dir, cam.image_name, "initial.png")
                    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                    Image.fromarray((cam.mask.cpu().numpy() * 255).astype(np.uint8)).save(snapshot_path)

    mask_dir = os.path.join(dataset.source_path, "mask")
    mask_set = {os.path.splitext(name)[0] for name in os.listdir(mask_dir)
                if os.path.isfile(os.path.join(mask_dir, name))}
    if mask_update_interval <= 0:
        raise ValueError("mask_update_interval must be positive")
    if mask_method not in ("pixel", "block"):
        raise ValueError("mask_method must be pixel or block")
    if debug_mask_interval <= 0:
        raise ValueError("debug_mask_interval must be positive")
    if analysis_width <= 0 or not 0.0 <= threshold <= 1.0 or not 0.0 <= rgb_threshold <= 1.0:
        raise ValueError("mask analysis width must be positive and thresholds must be in [0, 1]")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    mask_on = False
    mask_update_index = 0  # Count mask updates in this process, including after checkpoint restore.
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        # 先整体训练多次
        if mask_onn:
            if iteration > 0:
                mask_on = True
                mask_onn = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        # Load the mask and apply if mask_on is enabled
        if mask_on:
            # Load corresponding mask (assumes same filename as image in 'mask' folder)
            mask_found = viewpoint_cam.image_name in mask_set

            if mask_found:
                mask = torch.ones(viewpoint_cam.original_image.shape[1], viewpoint_cam.original_image.shape[2],
                                  dtype=torch.uint8, device=viewpoint_cam.original_image.device)
            else:
                mask = viewpoint_cam.mask


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # These four PNGs are diagnostics only.  They are very expensive when
        # the working directory is a network filesystem, so runners can turn
        # them off without altering rendering, masking, or optimisation.
        save_masked_images = os.getenv("GS_SAVE_MASKED_IMAGES", "0").lower() not in {"0", "false", "no", "off"}

        if mask_on:
            # 保存时应用遮罩，但保持原始图像用于计算损失
            masked_image = image * mask
            masked_gt_image = gt_image * mask

            if save_masked_images and iteration % 10 == 0:  # 控制保存频率
                save_path = "./masked_images"
                os.makedirs(save_path, exist_ok=True)
                image_folder = os.path.join(save_path, viewpoint_cam.image_name)
                os.makedirs(image_folder, exist_ok=True)

                # 保存 Mask
                mask_expanded = mask.expand_as(image).float()
                vutils.save_image(mask_expanded, os.path.join(image_folder, f"mask_{iteration}.png"))
                
                # 直接在 GPU 上保存 image 和 masked_image
                vutils.save_image(masked_image, os.path.join(image_folder, f"masked_rendered_{iteration}.png"))
                vutils.save_image(image, os.path.join(image_folder, f"rendered_{iteration}.png"))
                vutils.save_image(gt_image, os.path.join(image_folder, f"original_gt_{iteration}.png"))  # 保存原始gt

        # 计算L1损失时应用遮罩
        if mask_on:
            # 应用遮罩到渲染图像，保持gt_image不变
            rendered_masked = image * mask

            # 计算L1损失时仅考虑遮罩区域
            loss_per_pixel = torch.abs(rendered_masked - gt_image) * mask  # 差异乘以遮罩
            Ll1 = loss_per_pixel.sum() / (mask.sum() + 1e-6)  # 避免除以零

            # 计算SSIM损失（若需要处理遮罩，需自定义SSIM）
            # 此处假设使用原SSIM，可能不够精确，但保持结构不变
            ssim_loss = 1.0 - ssim(rendered_masked, gt_image * mask)  # 对gt_image应用遮罩仅用于损失计算
        else:
            Ll1 = l1_loss(image, gt_image)
            ssim_loss = 1.0 - ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss.backward()

        iter_end.record()

        # Keep the existing schedule; the debug interval counts actual updates in this process.
        if iteration > mask_update_start and (iteration+1) % mask_update_interval == 0 and mask_on:
            mask_update_index += 1
            update_masks_ssim(scene, gaussians, pipe, background, dataset.source_path,
                              threshold=threshold, block_size=block_size, mask_method=mask_method,
                              analysis_width=analysis_width, rgb_threshold=rgb_threshold,
                              debug_dir=debug_mask_dir, debug_names=debug_mask_names,
                              debug_interval=debug_mask_interval, update_index=mask_update_index,
                              iteration=iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if debug_mask_dir:
        for cam in scene.getTrainCameras():
            if normalize_debug_mask_name(cam.image_name) in debug_mask_names:
                snapshot_path = os.path.join(debug_mask_dir, cam.image_name, "final.png")
                os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
                Image.fromarray((cam.mask.cpu().numpy() * 255).astype(np.uint8)).save(snapshot_path)

    if save_final_masks:
        export_final_old_masks(scene, mask_set)

    if skip_final_eval:
        return

    print("\n训练完成，开始评估...")
    # 获取训练和测试相机组（假设测试相机可用）
    train_cameras = scene.getTrainCameras().copy()
    test_cameras = scene.getTestCameras().copy()

    # 初始化评估指标存储
    metrics = {
        'train': {'gs': {'l1': [], 'psnr': [], 'ssim': [], 'lpips': []}},
        'test': {'gs': {'l1': [], 'psnr': [], 'ssim': [], 'lpips': []}}
    }

    # 初始化LPIPS模型
    if lpips is None:
        raise ImportError("lpips is required for final evaluation; use --skip_final_eval to skip it")
    lpips_loss = lpips.LPIPS(net='alex').cuda()

    def evaluate_group(cameras, group_name):
        for cam in tqdm(cameras, desc=f"评估{group_name}组"):
            # 3DGS渲染
            gs_render = render(cam, gaussians, pipe, bg)["render"]
            gt_image = cam.original_image.cuda()

            # 计算指标
            l1 = l1_loss(gs_render, gt_image).item()
            psnr = 10 * torch.log10(1. / torch.mean((gs_render - gt_image) ** 2)).item()
            ssim_val = ssim(gs_render, gt_image).item()
            lpips_val = lpips_loss(gs_render.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item()

            # 存储
            metrics[group_name]['gs']['l1'].append(l1)
            metrics[group_name]['gs']['psnr'].append(psnr)
            metrics[group_name]['gs']['ssim'].append(ssim_val)
            metrics[group_name]['gs']['lpips'].append(lpips_val)

    # 执行评估
    evaluate_group(train_cameras, 'train')
    if test_cameras:
        evaluate_group(test_cameras, 'test')

    # 打印和保存每组平均结果
    for group in ['train', 'test']:
        values = metrics[group]['gs']
        if not values['l1']:
            continue

        avg_l1 = sum(values['l1']) / len(values['l1'])
        avg_psnr = sum(values['psnr']) / len(values['psnr'])
        avg_ssim = sum(values['ssim']) / len(values['ssim'])
        avg_lpips = sum(values['lpips']) / len(values['lpips'])

        print(f"\n== {group.upper()} 组平均结果 ==")
        print(f"3DGS 模型: L1: {avg_l1:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")

        # 保存到日志
        with open(os.path.join(scene.model_path, "metrics.log"), "a") as tlog:
            tlog.write(f"\n da, {mask_method}, {threshold}, {analysis_width}, {rgb_threshold}, {block_size}, {group}, avg, {avg_l1:.4f}, {avg_psnr:.2f}, {avg_ssim:.4f}, {avg_lpips:.4f}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,10000,30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000,10000,30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mask", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--mask_update_start", type=int, default=900)
    parser.add_argument("--mask_update_interval", type=int, default=108)
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--block_size", type=int, default=24)
    parser.add_argument("--mask_method", choices=("pixel", "block"), default="pixel")
    parser.add_argument("--mask_analysis_width", type=int, default=640)
    parser.add_argument("--mask_rgb_threshold", type=float, default=0.12)
    parser.add_argument("--debug_mask_names", nargs="+", default=[])
    parser.add_argument("--debug_mask_interval", type=int, default=1,
                        help="Save selected masks every N updates in this process; resets after checkpoint restore")
    parser.add_argument("--skip_final_mask_export", action="store_true",
                        help="Do not export final masks for old training views")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.mask, args.threshold, args.block_size, args.mask_update_start, args.mask_update_interval, args.skip_final_eval, args.mask_method, args.mask_analysis_width, args.mask_rgb_threshold, args.debug_mask_names, args.debug_mask_interval, not args.skip_final_mask_export)

    # All done
    print("\nTraining complete.")
