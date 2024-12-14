from .ppp import list_of_dicts__to__dict_of_lists
import os
from pathlib import Path
import json
import datetime
import glob
import numpy as np
from tu2.config import build_from_config

from typing import Dict, List
from tu2.config import overwrite_cfg
from orb.constant import PROJ_ROOT, VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, REVISION_SCENES, RELEASE_SCENES, BENCHMARK_RESOLUTION, DOWNSIZE_FACTOR
from orb.utils.metrics import calc_PSNR as _psnr, calc_depth_distance, calc_normal_distance, erode_mask, calc_depth_distance_per_scene
from orb.utils.preprocess import load_rgb_png, load_rgb_exr, srgb_to_rgb, load_hdr_rgba, rgb_to_srgb, load_mask_png, cv2_downsize
import logging

try:
    import torch
    import pyexr
    from lpips import LPIPS
    from kornia.losses import ssim_loss
    from orb.utils.eval_mesh import compute_shape_score
    _lpips = None
except Exception as _:
    LPIPS = None
    pyexr = None
    ssim_loss = None
    compute_shape_score = None


logger = logging.getLogger(__name__)


LEADERBOARD_DIR = os.path.join(PROJ_ROOT, 'logs/leaderboard')
DEBUG = os.getenv('DEBUG') == '1'
OVERWRITE = os.getenv('OVERWRITE') == '1'
OVERWRITE_VIEW = os.getenv('OVERWRITE_VIEW') == '1'
OVERWRITE_LIGHT = os.getenv('OVERWRITE_LIGHT') == '1'
OVERWRITE_GEOMETRY = os.getenv('OVERWRITE_GEOMETRY') == '1'
OVERWRITE_MATERIAL = os.getenv('OVERWRITE_MATERIAL') == '1'
OVERWRITE_SHAPE = os.getenv('OVERWRITE_SHAPE') == '1'
NO_SCORE_VIEW = os.getenv('NO_SCORE_VIEW') == '1'
NO_SCORE_LIGHT = os.getenv('NO_SCORE_LIGHT') == '1'
NO_SCORE_GEOMETRY = os.getenv('NO_SCORE_GEOMETRY') == '1'
NO_SCORE_MATERIAL = os.getenv('NO_SCORE_MATERIAL') == '1'
NO_SCORE_SHAPE = os.getenv('NO_SCORE_SHAPE') == '1'
UPDATE_FROM_JSON = False

# NO_SCORE_VIEW = True
# NO_SCORE_LIGHT = True
# NO_SCORE_GEOMETRY = True
# NO_SCORE_MATERIAL = True
# NO_SCORE_SHAPE = True
# UPDATE_FROM_JSON = True

logger.info(f'DEBUG={DEBUG}, OVERWRITE={OVERWRITE}, OVERWRITE_VIEW={OVERWRITE_VIEW}, OVERWRITE_LIGHT={OVERWRITE_LIGHT}, OVERWRITE_GEOMETRY={OVERWRITE_GEOMETRY}, OVERWRITE_SHAPE={OVERWRITE_SHAPE}, '
            f'NO_SCORE_VIEW={NO_SCORE_VIEW}, NO_SCORE_LIGHT={NO_SCORE_LIGHT}, NO_SCORE_GEOMETRY={NO_SCORE_GEOMETRY}, NO_SCORE_MATERIAL={NO_SCORE_MATERIAL}, NO_SCORE_SHAPE={NO_SCORE_SHAPE}')

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


def compute_metrics_image_similarity(results: List, scale_invariant=True) -> Dict:
    ret = []
    for item in results:
        target_rgba = load_hdr_rgba(item['target_image'], downsize_factor=DOWNSIZE_FACTOR)
        if item['output_image'] is None:
            input_rgb_hdr = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), dtype=np.float32)
        elif item['output_image'].endswith('.exr'):
            input_rgb_hdr = load_rgb_exr(item['output_image'])
        elif item['output_image'].endswith('.png'):
            input_rgb_ldr = load_rgb_png(item['output_image'])
            input_rgb_hdr = srgb_to_rgb(input_rgb_ldr)
        else:
            raise NotImplementedError(item['output_image'])

        if item['output_image'] and ('nerd' in item['output_image'] or 'neuralpil' in item['output_image']):
            input_rgb_hdr = input_rgb_hdr * 1 / (1.2 * 2 ** 8)

        ret.append(compute_similarity(input_rgb_hdr, target_rgba[:, :, :3], target_rgba[:, :, 3], scale_invariant=scale_invariant))

    ret = list_of_dicts__to__dict_of_lists(ret)
    # print(ret)
    if LPIPS is not None:
        for k, v in ret.items():
            if np.isnan(v).any():
                logger.error(f'NAN in {k}, {v}, {np.asarray(results)[np.isnan(v)]}')
                import ipdb; ipdb.set_trace()  # FIXME for numbers in the paper this SHOULD NOT HAPPEN
                # ret[k] = np.asarray(v)[~np.isnan(v)]
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def compute_metrics_shape(results: dict) -> dict:
    if compute_shape_score is None:
        return {}
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


def compute_metrics_material(results: List) -> Dict:
    ret = []
    for item in results:
        target_srgb = load_rgb_png(item['target_image'], downsize_factor=DOWNSIZE_FACTOR)
        target_rgb = srgb_to_rgb(target_srgb)
        if 'target_mask' in item:
            target_alpha = load_mask_png(item['target_mask'], downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)
        else:
            target_alpha_path = os.path.join(os.path.dirname(item['target_image']), '../../blender_format_LDR/test_mask', os.path.basename(item['target_image']))
            if not os.path.exists(target_alpha_path):
                raise RuntimeError(f'[ERROR] specify the file path for {item["target_image"]} based on your data format')
            target_alpha = load_mask_png(target_alpha_path, downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)
        if item['output_image'] is None:
            input_rgb = np.ones((BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3), dtype=np.float32)
        elif item['output_image'].endswith('.exr'):
            input_rgb = load_rgb_exr(item['output_image'])
        elif item['output_image'].endswith('.png'):
            input_srgb = load_rgb_png(item['output_image'])
            input_rgb = srgb_to_rgb(input_srgb)
        else:
            raise NotImplementedError(item['output_image'])
        if np.isnan(input_rgb).any():
            logger.error(f'NAN in input_rgb_ldr {str(item)}')
            import ipdb; ipdb.set_trace()
        ret.append(compute_similarity_albedo(input_rgb, target_rgb, target_alpha))

    ret = list_of_dicts__to__dict_of_lists(ret)
    # print(ret)
    if LPIPS is not None:
        for k, v in ret.items():
            if np.isnan(v).any():
                logger.error(f'NAN in {k}, {v}, {np.asarray(results)[np.isnan(v)]}')
                import ipdb; ipdb.set_trace()  # FIXME for numbers in the paper this SHOULD NOT HAPPEN
                # ret[k] = np.asarray(v)[~np.isnan(v)]
    ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def compute_metrics_geometry(results: List) -> Dict:
    ret = []
    input_depth_all = []
    target_depth_all = []
    mask_all = []
    for item in results:
        ret.append(dict())
        target_normal = cv2_downsize(np.load(item['target_normal']), downsize_factor=DOWNSIZE_FACTOR)
        input_normal = load_rgb_exr(item['output_normal'])

        if 'target_mask' in item:
            mask = load_mask_png(item['target_mask'], downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)
        else:
            mask_path = Path(item['target_normal']).parent.parent.parent / 'blender_format_LDR/test_mask' / Path(item['target_normal']).with_suffix('.png').name
            if not mask_path.exists():
                raise RuntimeError(f'[ERROR] specify the file path for {item["target_image"]} based on your data format')
            mask = load_mask_png(mask_path, downsize_factor=DOWNSIZE_FACTOR).astype(np.float32)

        ret[-1].update({
            'normal_angle': calc_normal_distance(input_normal, target_normal, mask),
        })
        if pyexr is not None and item['output_depth'] is not None:
            target_depth = cv2_downsize(np.load(item['target_depth']), downsize_factor=DOWNSIZE_FACTOR)
            input_depth = pyexr.open(item['output_depth']).get().squeeze()

            if 'nerfactor' in item['output_depth'] or 'invrender' in item['output_depth']:
                input_depth = -input_depth

            ret[-1].update({
                'depth_mse': calc_depth_distance(input_depth, target_depth, mask),
            })
            input_depth_all.append(input_depth)
            target_depth_all.append(target_depth)
            mask_all.append(mask)

    ret = list_of_dicts__to__dict_of_lists(ret)
    for k, v in ret.items():
        if np.isnan(v).any():
            logger.error(f'NAN in {k}, {v}, {np.asarray(results)[np.isnan(v)]}')
            # import ipdb; ipdb.set_trace()  # FIXME for numbers in the paper this SHOULD NOT HAPPEN
            # ret[k] = np.asarray(v)[~np.isnan(v)]
    ret = {k: np.mean(v) for k, v in ret.items()}

    if len(input_depth_all) > 0:
        ret['depth_mse_scene'] = calc_depth_distance_per_scene(input_depth_all, target_depth_all, mask_all)

    return ret


def load_latest_leaderboard() -> Dict:
    if len(glob.glob(os.path.join(LEADERBOARD_DIR, '*.json'))) == 0:
        os.makedirs(LEADERBOARD_DIR, exist_ok=True)
        return dict()
    path = sorted(glob.glob(os.path.join(LEADERBOARD_DIR, '*.json')), key=os.path.getmtime)[-1]
    logger.info(f'Loading from {path}')
    with open(path) as f:
        return json.load(f)


def write_new_leaderboard(data):
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
    path = os.path.join(LEADERBOARD_DIR, f'{timestamp}.json')
    if os.path.exists(path):
        raise RuntimeError(path)

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    with open(os.path.join(LEADERBOARD_DIR, 'latest.json'), 'w') as f:
        json.dump(data, f, indent=4)


def update_leaderboard(new_data):
    data = load_latest_leaderboard()
    for k, v in new_data.items():
        overwrite_cfg(data, k, v, check_exists=False, recursive=True)
    write_new_leaderboard(data)


def load_method_leaderboard(method):
    os.makedirs(os.path.join(LEADERBOARD_DIR, 'baselines'), exist_ok=True)
    path = os.path.join(LEADERBOARD_DIR, f'baselines/{method}.json')
    if not os.path.exists(path):
        return dict()
    logger.info(f'Loading from {path}')
    with open(path) as f:
        return json.load(f)


def write_method_leaderboard(method, data):
    with open(os.path.join(LEADERBOARD_DIR, f'baselines/{method}.json'), 'w') as f:
        json.dump(data, f, indent=4)


def update_method_leaderboard(method, new_data):
    data = load_method_leaderboard(method)
    for k, v in new_data.items():
        overwrite_cfg(data, k, v, check_exists=False, recursive=True)
    write_method_leaderboard(method, data)


def compute_metrics(method):
    if UPDATE_FROM_JSON:
        data = load_method_leaderboard(method)
        update_leaderboard({'scores': {method: {'scores_stats': data['scores_stats'],
                                                'scores': data['scores']}},
                            'info': {method: data['info']}})
        logger.info('Done!')
        exit()

    logger.info(f'Computing metrics for {method}')
    pipeline = build_from_config(f'orb.pipelines.{method}.Pipeline')()

    if DEBUG:
        results = pipeline.test_shape('scene007_obj021_cactus', overwrite=True)
        # results = pipeline.test_shape('scene001_obj003_baking', overwrite=True)
        # results = pipeline.test_new_view('scene001_obj003_baking', overwrite=True)
        # results = pipeline.test_shape('scene007_obj007_gnome', overwrite=True)
        # results = pipeline.test_shape('scene002_obj019_blocks', overwrite=True)
        compute_metrics_shape(results)
        import ipdb; ipdb.set_trace()

    if DEBUG:  # TODO DEBUG
        results = pipeline.test_new_view('scene001_obj003_baking', overwrite=True)
        results = pipeline.test_geometry('scene001_obj003_baking', overwrite=True)
        compute_metrics_geometry(results)
        # results = pipeline.test_new_light('scene003_obj007_gnome', overwrite=True)
        # results = pipeline.test_new_view('scene001_obj018_teapot', overwrite=True)
        # FIXME if inconsistent with training logs, there might be something wrong in data preprocessing
        # results = pipeline.test_inverse_rendering('scene005_obj016_pitcher', overwrite=True)
        import ipdb; ipdb.set_trace()
    if DEBUG:  # TODO DEBUG
        # results = pipeline.test_inverse_rendering('scene007_obj016_pitcher', overwrite=True)
        # results = pipeline.test_new_view('scene007_obj016_pitcher', overwrite=True)
        # results = pipeline.test_new_light('scene002_obj008_grogu', overwrite=True)
        # results = pipeline.test_new_light('scene005_obj016_pitcher', overwrite=True)
        results = pipeline.test_new_light('scene003_obj008_grogu', overwrite=True)
        compute_metrics_image_similarity(results)
        # results = pipeline.test_inverse_rendering('scene001_obj003_baking')
        # compute_metrics_image_similarity(results)
        import ipdb; ipdb.set_trace()

    info = dict()
    ret_new_view = dict()
    ret_new_light = dict()
    ret_geometry = dict()
    ret_material = dict()
    ret_shape = dict()

    if VERSION == 'submission':
        scenes = SUBMISSION_SCENES
    elif VERSION == 'rebuttal':
        scenes = REBUTTAL_SCENES
    elif VERSION == 'revision':
        scenes = REVISION_SCENES
    elif VERSION == 'release':
        scenes = RELEASE_SCENES
    else:
        raise NotImplementedError()
    for scene in scenes:
        info[scene] = dict()
        if not NO_SCORE_VIEW:
            results = pipeline.test_new_view(scene, overwrite=OVERWRITE or OVERWRITE_VIEW)  # FIXME
            ret_new_view[scene] = compute_metrics_image_similarity(results, scale_invariant=False)
            print('novel view', scene, ret_new_view[scene])
            info[scene].update({'view': results})

        if not NO_SCORE_LIGHT:
            results = pipeline.test_new_light(scene, overwrite=OVERWRITE or OVERWRITE_LIGHT)  # FIXME
            # only test relighting in held-out scenes
            new_results = []
            for item in results:
                if scene in item['target_image']:
                    continue
                new_results.append(item)
            logger.info(f'HACK: relighting scenes # from {len(results)} to {len(new_results)}')
            results = new_results

            ret_new_light[scene] = compute_metrics_image_similarity(results, scale_invariant=True)
            print('novel light', scene, ret_new_light[scene])
            info[scene].update({'light': results})

        if not NO_SCORE_GEOMETRY:
            results = pipeline.test_geometry(scene, overwrite=OVERWRITE or OVERWRITE_GEOMETRY)
            ret_geometry[scene] = compute_metrics_geometry(results)
            print('geometry', scene, ret_geometry[scene], len(results))
            info[scene].update({'geometry': results})

        if not NO_SCORE_MATERIAL:
            results = pipeline.test_material(scene, overwrite=OVERWRITE or OVERWRITE_MATERIAL)
            ret_material[scene] = compute_metrics_material(results)
            print('material', scene, ret_material[scene], len(results))
            info[scene].update({'material': results})

        if not NO_SCORE_SHAPE:
            results = pipeline.test_shape(scene, overwrite=OVERWRITE or OVERWRITE_SHAPE)
            ret_shape[scene] = compute_metrics_shape(results)
            print('shape', scene, ret_shape[scene], len(results))
            info[scene].update({'shape': results})

    scores = {'view_all': ret_new_view, 'light_all': ret_new_light, 'geometry_all': ret_geometry, 'material_all': ret_material,
              'shape_all': ret_shape}

    ret_new_view = list_of_dicts__to__dict_of_lists(list(ret_new_view.values()))
    print(ret_new_view)
    ret_new_view = {k: (np.mean(v), np.std(v)) for k, v in ret_new_view.items()}
    ret_new_view['scene_count'] = len(scores['view_all'])

    ret_new_light = list_of_dicts__to__dict_of_lists(list(ret_new_light.values()))
    print(ret_new_light)
    ret_new_light = {k: (np.mean(v), np.std(v)) for k, v in ret_new_light.items()}
    ret_new_light['scene_count'] = len(scores['light_all'])

    ret_geometry = list_of_dicts__to__dict_of_lists(list(ret_geometry.values()))
    print(ret_geometry)
    ret_geometry = {k: (np.mean(v), np.std(v)) for k, v in ret_geometry.items()}
    ret_geometry['scene_count'] = len(scores['geometry_all'])

    ret_material = list_of_dicts__to__dict_of_lists(list(ret_material.values()))
    print(ret_material)
    ret_material = {k: (np.mean(v), np.std(v)) for k, v in ret_material.items()}
    ret_material['scene_count'] = len(scores['material_all'])

    ret_shape = list_of_dicts__to__dict_of_lists(list(ret_shape.values()))
    print(ret_shape)
    ret_shape = {k: (np.mean(v), np.std(v)) for k, v in ret_shape.items()}
    ret_shape['scene_count'] = len(scores['shape_all'])

    scores_stats = {'view': ret_new_view, 'light': ret_new_light, 'geometry': ret_geometry, 'material': ret_material,
                    'shape': ret_shape}
    if os.getenv('IMAEGINT_PSEUDO_GT') == '1':
        method = method + '_pseudo_gt'
    update_method_leaderboard(method, {'scores_stats': scores_stats, 'scores': scores, 'info': info})
    update_leaderboard({'scores': {method: {'scores_stats': scores_stats, 'scores': scores}},
                        'info': {method: info}})
    logger.info('Done!')
