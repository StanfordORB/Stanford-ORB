import os
from functools import partial
import numpy as np
from pathlib import Path
from PIL import Image
from orb.utils.preprocess import load_rgb_exr, load_hdr_rgba, rgb_to_srgb, cv2_downsize, load_rgb_png, load_mask_png
import pyexr
from orb.utils.paper import load_scores
from orb.constant import PROJ_ROOT, DOWNSIZE_FACTOR, BENCHMARK_RESOLUTION
import cv2


ALL_METHODS = [
    'nvdiffrecmc_pseudo_gt',
    'nvdiffrec_pseudo_gt',
    'idr', 'nerf',
    'neuralpil',
    'physg',
    'nvdiffrec',
    'nerd',
    'nerfactor',
    'invrender',
    'nvdiffrecmc',
    'singleimage',
    'sirfs',
]

OUT_DIR = os.path.join(PROJ_ROOT, 'logs/paper/qualitative')
os.makedirs(OUT_DIR, exist_ok=True)


NA_IMAGE = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), dtype=np.float32)


def scale_image(img_pred, img_gt, mask_gt):
    if mask_gt.ndim == 3:
        mask_gt = mask_gt[..., 0]
    if mask_gt.dtype == np.float32:
        mask_gt = (mask_gt * 255).clip(0, 255).astype(np.uint8)

    img_pred_orig = img_pred
    img_gt_orig = img_gt
    # mask_gt_erode = cv2.erode(mask_gt, np.ones((5, 5), np.uint8))
    # mask_gt_erode = (mask_gt_erode > 127).astype(np.float32)
    mask_gt_orig = (mask_gt > 127).astype(np.float32)

    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask_gt = cv2.erode(mask_gt, kernel)
    mask_gt = (mask_gt > 127).astype(np.float32)
    img_pred = img_pred * mask_gt[..., None]
    img_gt = img_gt * mask_gt[..., None]

    if img_pred.shape[-1] == 1:
        print('min max', img_pred[mask_gt > .5].min(), img_pred[mask_gt > .5].max())

    invalid = False
    for c in range(img_pred.shape[-1]):
        pred_median = np.median(img_pred[..., c][np.where(mask_gt > 0.5)])
        if pred_median <= 1e-6 or np.isnan(pred_median) or np.isinf(pred_median):
            invalid = True
    if invalid and pred_median >= -1e-6:  # FIXME very hacky
        print('invalid prediction from calc_PSNR', pred_median)
        img_pred = np.ones_like(img_pred)

    if pred_median < 0:
        print('flipping prediction signs')
        img_pred = -img_pred  # FIXME: ?? rerun depth benchmark
        img_pred_orig = -img_pred_orig

    scale = []
    for c in range(img_pred.shape[-1]):
        gt_median = np.median(img_gt[..., c][np.where(mask_gt > 0.5)])
        pred_median = np.median(img_pred[..., c][np.where(mask_gt > 0.5)])
        scale.append(gt_median / pred_median)
    scale = np.array(scale)
    if img_pred.shape[-1] == 1:
        print(scale, img_pred[mask_gt > .5].min(), img_pred[mask_gt > .5].max())
    img_pred = (img_pred_orig * mask_gt_orig[...,  None]) * scale
    img_gt = img_gt_orig * mask_gt_orig[..., None]
    return img_pred, img_gt


def load_normal(item):
    target_normal = cv2_downsize(np.load(item['target_normal']), downsize_factor=DOWNSIZE_FACTOR)
    target_alpha = load_mask_png(Path(item['target_normal']).parent.parent.parent / 'blender_format_LDR/test_mask' / Path(item['target_normal']).with_suffix('.png').name, downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)
    if item['output_normal'] is not None:
        input_normal = load_rgb_exr(item['output_normal'])
    else:
        input_normal = NA_IMAGE
    input_normal = (input_normal * .5 + .5).clip(0, 1)
    target_normal = (target_normal * .5 + .5).clip(0, 1)

    # input_normal = (input_normal * target_alpha[..., None]) + (1 - target_alpha[..., None])
    # target_normal = (target_normal * target_alpha[..., None]) + (1 - target_alpha[..., None])
    return input_normal, target_normal


def load_depth(item):
    if item['target_depth'] is not None:
        target_depth = cv2_downsize(np.load(item['target_depth']), downsize_factor=DOWNSIZE_FACTOR)
    else:
        target_depth = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION))
    target_alpha = load_mask_png(Path(item['target_normal']).parent.parent.parent / 'blender_format_LDR/test_mask' / Path(item['target_normal']).with_suffix('.png').name, downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)
    if item['output_depth'] is not None:
        print(item['output_depth'])
        input_depth = pyexr.open(item['output_depth']).get().squeeze()
    else:
        print('no depth')
        input_depth = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION))

    input_depth, target_depth = scale_image(input_depth[:, :, None], target_depth[:, :, None], target_alpha)
    input_depth = input_depth[:, :, 0]
    target_depth = target_depth[:, :, 0]

    valid_pixels = target_depth[target_alpha > .8]
    valid_pixels_min = np.quantile(valid_pixels, .05)
    valid_pixels_max = np.quantile(valid_pixels, .95) * 1.02
    input_depth = ((input_depth - valid_pixels_min) / (valid_pixels_max - valid_pixels_min)).clip(0, 1)
    target_depth = ((target_depth - valid_pixels_min) / (valid_pixels_max - valid_pixels_min)).clip(0, 1)

    input_depth = np.stack([input_depth, input_depth, input_depth], axis=-1)
    target_depth = np.stack([target_depth, target_depth, target_depth], axis=-1)

    # input_depth = input_depth * target_alpha[:, :, None] + (1 - target_alpha[:, :, None])
    # target_depth = target_depth * target_alpha[:, :, None] + (1 - target_alpha[:, :, None])
    return input_depth, target_depth


def load_image(item):
    if item['target_image'].endswith('.exr'):
        target_rgba = load_hdr_rgba(item['target_image'], downsize_factor=DOWNSIZE_FACTOR)
        target_rgb = target_rgba[..., :3]
        target_rgb_ldr = rgb_to_srgb(target_rgb)
        target_alpha = target_rgba[..., 3]
    else:
        target_rgb_ldr = load_rgb_png(item['target_image'], downsize_factor=DOWNSIZE_FACTOR)
        target_alpha = load_mask_png(os.path.join(os.path.dirname(item['target_image']), '../../blender_format_LDR/test_mask', os.path.basename(item['target_image'])), downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)

    if item['output_image'] is None:
        input_rgb_ldr = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), dtype=np.float32)
    elif item['output_image'].endswith('.exr'):
        input_rgb_hdr = load_rgb_exr(item['output_image'])
        input_rgb_ldr = rgb_to_srgb(input_rgb_hdr)
    elif item['output_image'].endswith('.png'):
        input_rgb_ldr = load_rgb_png(item['output_image'])
    else:
        raise NotImplementedError()
    #
    # if np.isnan(input_rgb_ldr).any():
    #     print(f"nan in {item['output_image']}")
    #     input_rgb_ldr[np.isnan(input_rgb_ldr)] = 1

    input_rgb_ldr, target_rgb_ldr = scale_image(input_rgb_ldr, target_rgb_ldr, target_alpha)
    # input_rgb_ldr = input_rgb_ldr * target_alpha[:, :, None] + (1 - target_alpha[:, :, None])
    # target_rgb_ldr = target_rgb_ldr * target_alpha[:, :, None] + (1 - target_alpha[:, :, None])
    return input_rgb_ldr, target_rgb_ldr


def crop_image_with_mask(image, mask, first_compose=True):
    if first_compose:
        image = compose_image_with_mask(image, mask)
    coordinates = cv2.boundingRect((mask * 255).astype(np.uint8))
    x, y, w, h = coordinates
    buffer = 40
    size = max(w, h) + 2 * buffer
    if size > image.shape[0]:
        size = image.shape[0]
    center_x = x + w // 2
    center_y = y + h // 2
    crop_x = max(center_x - size // 2, 0)
    crop_y = max(center_y - size // 2, 0)
    image = image[crop_y:crop_y + size, crop_x:crop_x + size]
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return image


def compose_image_with_mask(image, mask):
    mask = (mask * 255).clip(0, 255).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    mask = (mask > 127).astype(np.float32)
    return image * mask[:, :, None] + (1 - mask[:, :, None])


def plot_scene(scene, test_ind, novel_ind):
    data = load_scores('invrender')
    crop_test = partial(crop_image_with_mask, mask=load_hdr_rgba(data['info'][scene]['view'][test_ind]['target_image'], downsize_factor=DOWNSIZE_FACTOR)[..., 3])
    crop_novel = partial(crop_image_with_mask, mask=load_hdr_rgba(data['info'][scene]['light'][novel_ind]['target_image'], downsize_factor=DOWNSIZE_FACTOR)[..., 3])
    rows = []

    cols = []
    cols.append(load_normal(data['info'][scene]['geometry'][test_ind])[1])
    cols.append(load_depth(data['info'][scene]['geometry'][test_ind])[1])
    cols.append(load_image(data['info'][scene]['view'][test_ind])[1])
    cols.append(load_image(data['info'][scene]['light'][novel_ind])[1])
    cols.append(load_image(data['info'][scene]['material'][test_ind])[1])
    # new_cols = []
    # for col in cols:
    #     new_cols.append(cv2_downsize(col, 4))
    # cols = new_cols
    cols = [crop_test(cols[0]), crop_test(cols[1]), crop_test(cols[2]), crop_novel(cols[3]), crop_test(cols[4])]
    row = np.concatenate(cols, axis=1)
    rows.append(row)
    Image.fromarray((row.clip(0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, f'{scene}_gt.png'))

    for method in ALL_METHODS:
        data = load_scores(method)
        if 'material' not in data['info'][scene]:
            data['info'][scene]['material'] = []

        cols = []
        if len(data['info'][scene]['geometry']) == 0:
            cols.append(NA_IMAGE)
            cols.append(NA_IMAGE)
        else:
            cols.append(load_normal(data['info'][scene]['geometry'][test_ind])[0])
            cols.append(load_depth(data['info'][scene]['geometry'][test_ind])[0])

        if len(data['info'][scene]['view']) == 0:
            cols.append(NA_IMAGE)
        else:
            cols.append(load_image(data['info'][scene]['view'][test_ind])[0])

        if len(data['info'][scene]['light']) == 0:
            cols.append(NA_IMAGE)
        else:
            cols.append(load_image(data['info'][scene]['light'][novel_ind])[0])

        if len(data['info'][scene]['material']) == 0:
            cols.append(NA_IMAGE)
        else:
            cols.append(load_image(data['info'][scene]['material'][test_ind])[0])

        # new_cols = []
        # for col in cols:
        #     new_cols.append(cv2_downsize(col, 4))
        # cols = new_cols

        cols = [crop_test(cols[0]), crop_test(cols[1]), crop_test(cols[2]), crop_novel(cols[3]), crop_test(cols[4])]
        row = np.concatenate(cols, axis=1)
        rows.append(row)
        Image.fromarray((row.clip(0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, f'{scene}_{method}.png'))

    row_sep = np.ones_like(rows[0][:20, :, :])
    new_rows = []
    for ind in range(len(rows)):
        new_rows.append(rows[ind])
        if ind < len(rows) - 1:
            new_rows.append(row_sep)
    rows = new_rows

    image = np.concatenate(rows, axis=0)
    Image.fromarray((image.clip(0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, f'{scene}.png'))
    return image


if __name__ == "__main__":
    plot_scene('scene002_obj019_blocks', 1, 1)
    # plot_scene('scene003_obj008_grogu', 3, 0)
