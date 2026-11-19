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
from utils.pixel_masks import pixel_confidence_mask

# def compute_ssim_batch(rendered_blocks, gt_blocks, window_size=11):
#     """批量计算SSIM，输入形状为(B, C, H, W)，返回形状为(B,)的SSIM值"""
#     # 假设ssim函数已支持批量计算，例如使用pytorch_msssim库
#     return ssim2(rendered_blocks, gt_blocks, data_range=1.0, win_size=window_size, size_average=False)
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

def update_masks_ssim(scene, gaussians, pipe, background, source_path, threshold=0.8, block_size=32,
                      mask_method="pixel", evidence_generator=None, analysis_width=640):
    with torch.no_grad():
        train_cameras = scene.getTrainCameras()
        save_path = os.path.join(source_path, "label")
        os.makedirs(save_path, exist_ok=True)

        for viewpoint_cam in tqdm(train_cameras, desc="Updating masks"):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0).cuda()  # (C, H, W)
            gt_image = torch.clamp(viewpoint_cam.original_image.cuda(), 0.0, 1.0)

            C, H, W = rendered_image.shape
            if mask_method == "pixel":
                mask = pixel_confidence_mask(rendered_image, gt_image,
                                             ssim_threshold=threshold,
                                             analysis_width=analysis_width)
            elif mask_method == "evidence":
                if evidence_generator is None:
                    raise ValueError("evidence_generator is required for evidence masks")
                # Adapter contract: rendered image first, target image second.
                mask = evidence_generator(rendered_image, gt_image)
                if not torch.is_tensor(mask):
                    mask = torch.as_tensor(mask, device=rendered_image.device)
                mask = mask.to(device=rendered_image.device, dtype=torch.uint8)
                if tuple(mask.shape) != (H, W):
                    raise ValueError("InitialEvidenceMask must return an HxW mask")
            elif mask_method == "block":
                if block_size <= 0:
                    raise ValueError("block_size must be positive for block masks")
                # 计算填充量，使尺寸为 block_size 的整数倍
                pad_h = (block_size - H % block_size) % block_size
                pad_w = (block_size - W % block_size) % block_size
                rendered_padded = F.pad(rendered_image, (0, pad_w, 0, pad_h))
                gt_padded = F.pad(gt_image, (0, pad_w, 0, pad_h))
                H_pad, W_pad = H + pad_h, W + pad_w
                nh, nw = H_pad // block_size, W_pad // block_size
                rendered_blocks = rendered_padded.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
                rendered_blocks = rendered_blocks.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, block_size, block_size)
                gt_blocks = gt_padded.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
                gt_blocks = gt_blocks.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, block_size, block_size)
                ssim_values = compute_ssim_batch(rendered_blocks, gt_blocks, window_size=11)
                ssim_matrix = ssim_values.view(nh, nw)
                mask_matrix = (ssim_matrix > threshold).float()
                mask_expanded = mask_matrix.unsqueeze(-1).unsqueeze(-1).expand(nh, nw, block_size, block_size)
                mask_padded = mask_expanded.permute(0, 2, 1, 3).contiguous().view(H_pad, W_pad)
                mask = mask_padded[:H, :W].to(torch.uint8)
            else:
                raise ValueError("mask_method must be evidence, pixel, or block")

            # 处理 mask 并保存
            mask_to_save = mask * 255  # 变成 0-255 之间的 uint8 格式
            mask_to_save = mask_to_save.cpu().numpy().astype(np.uint8)  # 转换为 NumPy 数组
            mask_image = Image.fromarray(mask_to_save)  # 转换为 PIL 图像
            filename = f"{viewpoint_cam.image_name}.png"  # 确保文件名正确
            save_path_full = os.path.join(save_path, filename)
            mask_image.save(save_path_full)  # 保存

            # 仅在原始 mask 为 0 的地方更新
            viewpoint_cam.mask = mask

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mask_onn=False, threshold=0.8, block_size=32, phase_switch_iter=3900, mask_update_iter=5900, skip_final_eval=False, mask_method="pixel", sam_checkpoint=None, sam_grid_size=8, mask_analysis_width=640):
    if not skip_final_eval and lpips is None:
        raise ImportError("lpips is required for final evaluation; install lpips or pass --skip_final_eval")
    evidence_generator = None
    if mask_method == "evidence":
        if not sam_checkpoint or not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError("--sam_checkpoint must point to an existing EfficientSAM checkpoint for evidence masks")
        try:
            from utils.initial_evidence_masks import InitialEvidenceMask
        except ImportError as exc:
            raise ImportError("--mask_method evidence requires utils.initial_evidence_masks.InitialEvidenceMask and its dependencies") from exc
        evidence_generator = InitialEvidenceMask(
            sam_checkpoint, device="cuda", analysis_width=mask_analysis_width,
            grid_sizes=(sam_grid_size,))
    elif mask_method not in ("pixel", "block"):
        raise ValueError("mask_method must be evidence, pixel, or block")
    if mask_method == "block" and block_size <= 0:
        raise ValueError("block_size must be positive for block masks")
    if sam_grid_size <= 0 or mask_analysis_width <= 0:
        raise ValueError("sam_grid_size and mask_analysis_width must be positive")
    if not skip_final_eval and lpips is None:
        raise ImportError("lpips is required for final evaluation; install lpips or pass --skip_final_eval")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    mask_set = set()
    mask_dir = os.path.join(dataset.source_path, "mask")

    if os.path.exists(mask_dir):
        for f in os.listdir(mask_dir):
            name, ext = os.path.splitext(f)
            mask_set.add(name)
    # 定义可能的文件后缀
    train_names = {cam.image_name for cam in scene.getTrainCameras()}
    if not (train_names & mask_set) or not (train_names - mask_set):
        raise ValueError("BU requires both old and new training cameras; check mask/ and split.json")
    if not 0 < phase_switch_iter < mask_update_iter <= opt.iterations:
        raise ValueError("Expected 0 < phase_switch_iter < mask_update_iter <= iterations")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    mask_on = True
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 先整体训练多次
        if iteration > phase_switch_iter:
            mask_on = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        while(True):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            # Load corresponding mask (assumes same filename as image in 'mask' folder)
            mask_found = viewpoint_cam.image_name in mask_set

            if mask_found:
                mask = torch.ones(viewpoint_cam.original_image.shape[1], viewpoint_cam.original_image.shape[2],
                                  dtype=torch.uint8, device=viewpoint_cam.original_image.device)
            else:
                mask = viewpoint_cam.mask
            if mask_on:
                # **判断 mask 是否全 0**
                if torch.all(mask == 0):
                    break
                else:
                    continue
            else:
                # **判断 mask 是否全 0**
                if torch.all(mask == 0):
                    continue
                else:
                    break


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # Loss with mask
        # Save masked images during the first iteration
        save_path = "./masked_images"
        os.makedirs(save_path, exist_ok=True)

        if mask_on:
            mask = 1-mask

        # 应用遮罩到渲染图像，保持gt_image不变
        rendered_masked = image * mask

        # 计算L1损失时仅考虑遮罩区域
        loss_per_pixel = torch.abs(rendered_masked - gt_image) * mask  # 差异乘以遮罩
        Ll1 = loss_per_pixel.sum() / (mask.sum() + 1e-6)  # 避免除以零

        # 计算SSIM损失（若需要处理遮罩，需自定义SSIM）
        # 此处假设使用原SSIM，可能不够精确，但保持结构不变
        ssim_loss = 1.0 - ssim(rendered_masked, gt_image * mask)  # 对gt_image应用遮罩仅用于损失计算
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss.backward()

        iter_end.record()

        # Mask更新逻辑（每隔50次迭代，从第50次开始）
        if iteration == mask_update_iter:
            update_masks_ssim(scene, gaussians, pipe, background, dataset.source_path,
                              threshold=threshold, block_size=block_size, mask_method=mask_method,
                              evidence_generator=evidence_generator, analysis_width=mask_analysis_width)
            break


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
            tlog.write(f"\n bu, {mask_method}, {threshold}, {block_size}, {sam_grid_size}, {mask_analysis_width}, {group}, avg, {avg_l1:.4f}, {avg_psnr:.2f}, {avg_ssim:.4f}, {avg_lpips:.4f}")

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[4999])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[4999,5800])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--mask", type=parse_bool, nargs="?", const=True, default=False)
    parser.add_argument("--phase_switch_iter", type=int, default=3900)
    parser.add_argument("--mask_update_iter", type=int, default=5900)
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.92)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--mask_method", choices=("evidence", "pixel", "block"), default="pixel")
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--sam_grid_size", type=int, default=8)
    parser.add_argument("--mask_analysis_width", type=int, default=640)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.mask, args.threshold, args.block_size, args.phase_switch_iter, args.mask_update_iter, args.skip_final_eval, args.mask_method, args.sam_checkpoint, args.sam_grid_size, args.mask_analysis_width)

    # All done
    print("\nTraining complete.")
