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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, is_train=False, iteration=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # DropGaussian
    if is_train:
        # Create initial compensation factor (1 for each Gaussian)
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

        # Apply DropGaussian with compensation
        drop_rate = 0.2 * (iteration/10000)
        d = torch.nn.Dropout(p=drop_rate)
        compensation = d(compensation)

        # Apply to opacity
        opacity = opacity * compensation[:, None]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_depth(viewpoint_camera, pc: GaussianModel, pipe, bg_value: torch.Tensor, scaling_modifier=1.0,
                 near_plane=0.01, far_plane=100.0):
    """
    Render depth map of the scene.

    Args:
        viewpoint_camera: Camera viewpoint
        pc: GaussianModel containing the 3D Gaussians
        pipe: Pipeline configuration
        bg_depth: Background depth value tensor (must be on GPU!)
        scaling_modifier: Scale modifier for Gaussians
        near_plane: Near clipping plane
        far_plane: Far clipping plane

    Returns:
        Dictionary containing rendered depth map and auxiliary information
    """

    # Create zero tensor for screen space points gradients
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_value,  # Background depth instead of color
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,  # No need for spherical harmonics for depth
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 修正：计算点到相机中心的真实距离
    # 方法1：直接计算欧几里得距离（推荐）
    camera_center = viewpoint_camera.camera_center  # [3]
    # 计算每个3D点到相机中心的距离
    distances = torch.norm(means3D - camera_center.unsqueeze(0), dim=1)  # [N]
    depths = distances

    # 方法2：如果需要相机空间距离（备选方案）
    # camera_space_points = torch.matmul(viewpoint_camera.world_view_transform[:3, :3],
    #                                    means3D.T) + viewpoint_camera.world_view_transform[:3, 3:4]
    # # 计算相机空间中的欧几里得距离
    # depths = torch.norm(camera_space_points, dim=0)  # [N]

    # 直接使用原始深度值，不进行预归一化
    # 让rasterizer使用原始深度值进行渲染
    depth_colors = depths.unsqueeze(-1).repeat(1, 3)

    # Handle covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Use depth-based colors instead of SH features
    colors_precomp = depth_colors
    shs = None

    # Rasterize visible Gaussians to depth image
    rendered_depth, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # 渲染后进行全局归一化
    # 提取单通道深度图（所有通道都相同）
    raw_depth_map = rendered_depth[0, :, :]  # [H, W]

    # 创建有效深度mask（排除背景）
    # 需要确保bg_depth是一个较大的值来表示背景

    valid_mask = raw_depth_map != bg_value

    # 如果bg_depth设置得不合理，用另一种方式检测背景
    if valid_mask.sum() == 0 or valid_mask.sum() == raw_depth_map.numel():
        # 使用统计方法检测异常值作为背景
        flattened = raw_depth_map.flatten()
        sorted_values, _ = torch.sort(flattened)
        # 假设背景值是最大值或最小值
        if sorted_values[-1] - sorted_values[-2] > sorted_values[1] - sorted_values[0]:
            # 最大值可能是背景
            bg_value = sorted_values[-1]
        else:
            # 最小值可能是背景
            bg_value = sorted_values[0]
        valid_mask = raw_depth_map != bg_value

    if valid_mask.sum() > 0:
        # 只对有效区域进行归一化
        valid_depths = raw_depth_map[valid_mask]
        min_depth = valid_depths.min()
        max_depth = valid_depths.max()

        # 避免除零
        if max_depth > min_depth:
            # 对有效区域归一化到[0, 1]，然后反转：近处为1（白色），远处为0（黑色）
            normalized_depth_map = torch.zeros_like(raw_depth_map)
            depth_normalized = (valid_depths - min_depth) / (max_depth - min_depth)
            normalized_depth_map[valid_mask] = 1.0 - depth_normalized  # 反转深度值
            # 背景设为0（黑色）表示无效区域
            normalized_depth_map[~valid_mask] = 0.0
        else:
            # 所有有效深度都相同的情况
            normalized_depth_map = torch.zeros_like(raw_depth_map)
            normalized_depth_map[valid_mask] = 0.5  # 设为中等灰度
            normalized_depth_map[~valid_mask] = 0.0  # 背景为黑色
    else:
        # 没有有效深度的情况，全部设为黑色
        normalized_depth_map = torch.zeros_like(raw_depth_map)

    # 创建归一化后的RGB深度图用于可视化
    normalized_rgb_depth = normalized_depth_map.unsqueeze(0).repeat(3, 1, 1)

    return {"render": normalized_rgb_depth,  # 归一化后的RGB深度图
            "raw_render": rendered_depth,  # 原始渲染结果
            "depth_map": normalized_depth_map.unsqueeze(0),  # 单通道归一化深度[0,1]
            "raw_depth": raw_depth_map.unsqueeze(0),  # 单通道原始深度值
            "min_depth": min_depth if valid_mask.sum() > 0 else torch.tensor(0.0),
            "max_depth": max_depth if valid_mask.sum() > 0 else torch.tensor(1.0),
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def render_depth_alternative(viewpoint_camera, pc: GaussianModel, pipe, bg_value: torch.Tensor, scaling_modifier=1.0,
                 near_plane=0.01, far_plane=100.0):
    """
    Render depth map of the scene.

    Args:
        viewpoint_camera: Camera viewpoint
        pc: GaussianModel containing the 3D Gaussians
        pipe: Pipeline configuration
        bg_depth: Background depth value tensor (must be on GPU!)
        scaling_modifier: Scale modifier for Gaussians
        near_plane: Near clipping plane
        far_plane: Far clipping plane

    Returns:
        Dictionary containing rendered depth map and auxiliary information
    """

    # Create zero tensor for screen space points gradients
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_value,  # Background depth instead of color
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=0,  # No need for spherical harmonics for depth
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 修正：计算点到相机中心的真实距离
    # 方法1：直接计算欧几里得距离（推荐）
    camera_center = viewpoint_camera.camera_center  # [3]
    # 计算每个3D点到相机中心的距离
    distances = torch.norm(means3D - camera_center.unsqueeze(0), dim=1)  # [N]
    depths = distances

    # 直接使用原始深度值，不进行预归一化
    # 让rasterizer使用原始深度值进行渲染
    depth_colors = depths.unsqueeze(-1).repeat(1, 3)

    # Handle covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Use depth-based colors instead of SH features
    colors_precomp = depth_colors
    shs = None

    # Rasterize visible Gaussians to depth image
    rendered_depth, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # 渲染后进行全局归一化
    # 提取单通道深度图（所有通道都相同）
    raw_depth_map = rendered_depth[0, :, :]  # [H, W]

    # 创建有效深度mask（排除背景）
    # 需要确保bg_depth是一个较大的值来表示背景

    valid_mask = raw_depth_map != bg_value

    # 如果bg_depth设置得不合理，用另一种方式检测背景
    if valid_mask.sum() == 0 or valid_mask.sum() == raw_depth_map.numel():
        # 使用统计方法检测异常值作为背景
        flattened = raw_depth_map.flatten()
        sorted_values, _ = torch.sort(flattened)
        # 假设背景值是最大值或最小值
        if sorted_values[-1] - sorted_values[-2] > sorted_values[1] - sorted_values[0]:
            # 最大值可能是背景
            bg_value = sorted_values[-1]
        else:
            # 最小值可能是背景
            bg_value = sorted_values[0]
        valid_mask = raw_depth_map != bg_value

    if valid_mask.sum() > 0:
        # 只对有效区域进行归一化
        valid_depths = raw_depth_map[valid_mask]
        min_depth = valid_depths.min()
        max_depth = valid_depths.max()

        # 避免除零
        if max_depth > min_depth:
            # 对有效区域归一化到[0, 1]，然后反转：近处为1（白色），远处为0（黑色）
            normalized_depth_map = torch.zeros_like(raw_depth_map)
            depth_normalized = (valid_depths - min_depth) / (max_depth - min_depth)
            normalized_depth_map[valid_mask] = 1.0 - depth_normalized  # 反转深度值
            # 背景设为0（黑色）表示无效区域
            normalized_depth_map[~valid_mask] = 0.0
        else:
            # 所有有效深度都相同的情况
            normalized_depth_map = torch.zeros_like(raw_depth_map)
            normalized_depth_map[valid_mask] = 0.5  # 设为中等灰度
            normalized_depth_map[~valid_mask] = 0.0  # 背景为黑色
    else:
        # 没有有效深度的情况，全部设为黑色
        normalized_depth_map = torch.zeros_like(raw_depth_map)

    return {"depth_map": normalized_depth_map.unsqueeze(0)}