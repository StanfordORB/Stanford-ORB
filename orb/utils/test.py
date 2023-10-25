from .ppp import list_of_dicts__to__dict_of_lists
import os
from pathlib import Path
import pyexr
import numpy as np
from .metrics import calc_PSNR as _psnr, calc_depth_distance, calc_normal_distance, erode_mask, calc_depth_distance_per_scene
from .preprocess import load_rgb_png, load_rgb_exr, srgb_to_rgb, load_hdr_rgba, rgb_to_srgb, load_mask_png, cv2_downsize
from .eval_mesh import compute_shape_score
import logging
import torch
from lpips import LPIPS
from kornia.losses import ssim_loss
from orb.constant import BENCHMARK_RESOLUTION
_lpips = None


logger = logging.getLogger(__name__)


def assert_inputs_target(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    # inputs and targets have range [0, 1]
    assert inputs.dtype == np.float32, inputs.dtype
    assert target.dtype == np.float32, target.dtype
    assert mask.dtype == np.float32, mask.dtype
    assert inputs.shape == target.shape == (BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), (inputs.shape, target.shape)
    assert mask.shape == (BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION), mask.shape


def lpips(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if LPIPS is None:
        return np.nan
    global _lpips
    if _lpips is None:
        _lpips = LPIPS(net='vgg', verbose=False).cuda()
    inputs = rgb_to_srgb(inputs)
    target = rgb_to_srgb(target)

    mask = erode_mask(mask, None)
    inputs = inputs * mask[:, :, None]
    target = target * mask[:, :, None]

    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    return _lpips(inputs, target, normalize=True).item()


def ssim(inputs: np.ndarray, target: np.ndarray, mask: np.ndarray):
    if ssim_loss is None:
        return np.nan

    mask = erode_mask(mask, None)
    inputs = inputs * mask[:, :, None]
    target = target * mask[:, :, None]

    # image_pred and image_gt: (1, 3, H, W) in range [0, 1]
    inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(target, dtype=torch.float32, device='cuda').permute(2, 0, 1).unsqueeze(0)
    dssim_ = ssim_loss(inputs, target, 3).item()  # dissimilarity in [0, 1]
    return 1 - 2 * dssim_  # in [-1, 1]


def compute_similarity(input_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray,
                       scale_invariant=True):
    assert_inputs_target(input_rgb, target_rgb, mask)
    mask = (mask > .5).astype(np.float32)
    out = {}
    out['psnr_hdr'], _, _ = _psnr(input_rgb, target_rgb, mask, max_value=4, use_gt_median=True, tonemapping=False, divide_mask=False, scale_invariant=scale_invariant)
    out['psnr_ldr'], input_srgb, target_srgb = _psnr(input_rgb, target_rgb, mask, max_value=1, use_gt_median=False, tonemapping=True, divide_mask=False, scale_invariant=scale_invariant)
    out['lpips'] = lpips(input_srgb, target_srgb, mask)
    out['ssim'] = ssim(input_srgb, target_srgb, mask)
    return out


def compute_metrics_image_similarity(results: list, scale_invariant=True) -> dict:
    ret = []
    for item in results:
        target_rgba = load_hdr_rgba(item['target_image'], downsize_factor=4)
        if item['output_image'] is None:
            input_rgb_hdr = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), dtype=np.float32)
        elif item['output_image'].endswith('.exr'):
            input_rgb_hdr = load_rgb_exr(item['output_image'])
        elif item['output_image'].endswith('.png'):
            input_rgb_ldr = load_rgb_png(item['output_image'])
            input_rgb_hdr = srgb_to_rgb(input_rgb_ldr)
        else:
            raise NotImplementedError(item['output_image'])

        ret.append(compute_similarity(input_rgb_hdr, target_rgba[:, :, :3], target_rgba[:, :, 3], scale_invariant=scale_invariant))

    ret = list_of_dicts__to__dict_of_lists(ret)
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def compute_metrics_shape(results: dict) -> dict:
    return compute_shape_score(**results)


def compute_similarity_albedo(input_rgb: np.ndarray, target_rgb: np.ndarray, mask: np.ndarray):
    assert_inputs_target(input_rgb, target_rgb, mask)
    mask = (mask > .5).astype(np.float32)
    out = {}
    # it's actually psnr in linear rgb space
    out['psnr_ldr'], input_rgb, target_rgb = _psnr(input_rgb, target_rgb, mask, max_value=1, use_gt_median=False, tonemapping=False, divide_mask=False, scale_invariant=True)
    out['lpips'] = lpips(input_rgb, target_rgb, mask)
    out['ssim'] = ssim(input_rgb, target_rgb, mask)
    return out


def compute_metrics_material(results: list) -> dict:
    ret = []
    for item in results:
        target_srgb = load_rgb_png(item['target_image'], downsize_factor=4)
        target_rgb = srgb_to_rgb(target_srgb)
        target_alpha = load_mask_png(item['target_mask'], downsize_factor=4).astype(np.float32)
        if item['output_image'] is None:
            input_rgb = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), dtype=np.float32)
        elif item['output_image'].endswith('.exr'):
            input_rgb = load_rgb_exr(item['output_image'])
        elif item['output_image'].endswith('.png'):
            input_srgb = load_rgb_png(item['output_image'])
            input_rgb = srgb_to_rgb(input_srgb)
        else:
            raise NotImplementedError(item['output_image'])
        ret.append(compute_similarity_albedo(input_rgb, target_rgb, target_alpha))

    ret = list_of_dicts__to__dict_of_lists(ret)
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def compute_metrics_geometry(results: list) -> dict:
    ret = []
    input_depth_all = []
    target_depth_all = []
    mask_all = []
    for item in results:
        ret.append(dict())
        target_normal = cv2_downsize(np.load(item['target_normal']), downsize_factor=4)
        input_normal = load_rgb_exr(item['output_normal'])
        mask = load_mask_png(item['target_mask'], downsize_factor=4).astype(np.float32)
        ret[-1].update({
            'normal_angle': calc_normal_distance(input_normal, target_normal, mask),
        })
        if pyexr is not None and item['output_depth'] is not None:
            target_depth = cv2_downsize(np.load(item['target_depth']), downsize_factor=4)
            input_depth = pyexr.open(item['output_depth']).get().squeeze()

            input_depth_all.append(input_depth)
            target_depth_all.append(target_depth)
            mask_all.append(mask)

    ret = list_of_dicts__to__dict_of_lists(ret)
    ret = {k: np.mean(v) for k, v in ret.items()}

    if len(input_depth_all) > 0:
        ret['depth_mse_scene'] = calc_depth_distance_per_scene(input_depth_all, target_depth_all, mask_all)

    return ret
