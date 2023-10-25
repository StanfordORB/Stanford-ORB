import json
from tqdm import tqdm
import os
import argparse
import numpy as np
from orb.utils.ppp import list_of_dicts__to__dict_of_lists
from orb.utils.test import compute_metrics_image_similarity, compute_metrics_geometry, compute_metrics_shape, compute_metrics_material
from orb.constant import get_scenes_from_id

import logging

logger = logging.getLogger(__name__)


VERBOSE = False


def process(input_path: str, output_path: str, scenes: list[str]):
    with open(input_path, 'r') as f:
        info = json.load(f)['info']

    ret_new_view = dict()
    ret_new_light = dict()
    ret_geometry = dict()
    ret_material = dict()
    ret_shape = dict()

    logger.info(f'Processing {len(scenes)} scenes...')
    for scene in tqdm(scenes):
        results = info[scene]['view']
        ret_new_view[scene] = compute_metrics_image_similarity(results, scale_invariant=False)
        if VERBOSE:
            print('novel view', scene, ret_new_view[scene])

        results = info[scene]['light']
        ret_new_light[scene] = compute_metrics_image_similarity(results, scale_invariant=True)
        if VERBOSE:
            print('novel light', scene, ret_new_light[scene])

        results = info[scene]['geometry']
        ret_geometry[scene] = compute_metrics_geometry(results)
        if VERBOSE:
            print('geometry', scene, ret_geometry[scene], len(results))

        results = info[scene]['material']
        ret_material[scene] = compute_metrics_material(results)
        if VERBOSE:
            print('material', scene, ret_material[scene], len(results))

        results = info[scene]['shape']
        ret_shape[scene] = compute_metrics_shape(results)
        if VERBOSE:
            print('shape', scene, ret_shape[scene], len(results))

    scores = {'view_all': ret_new_view, 'light_all': ret_new_light, 'geometry_all': ret_geometry, 'material_all': ret_material,
              'shape_all': ret_shape}

    ret_new_view = list_of_dicts__to__dict_of_lists(list(ret_new_view.values()))
    if VERBOSE:
        print(ret_new_view)
    ret_new_view = {k: (np.mean(v), np.std(v)) for k, v in ret_new_view.items()}
    ret_new_view['scene_count'] = len(scores['view_all'])

    ret_new_light = list_of_dicts__to__dict_of_lists(list(ret_new_light.values()))
    if VERBOSE:
        print(ret_new_light)
    ret_new_light = {k: (np.mean(v), np.std(v)) for k, v in ret_new_light.items()}
    ret_new_light['scene_count'] = len(scores['light_all'])

    ret_geometry = list_of_dicts__to__dict_of_lists(list(ret_geometry.values()))
    if VERBOSE:
        print(ret_geometry)
    ret_geometry = {k: (np.mean(v), np.std(v)) for k, v in ret_geometry.items()}
    ret_geometry['scene_count'] = len(scores['geometry_all'])

    ret_material = list_of_dicts__to__dict_of_lists(list(ret_material.values()))
    if VERBOSE:
        print(ret_material)
    ret_material = {k: (np.mean(v), np.std(v)) for k, v in ret_material.items()}
    ret_material['scene_count'] = len(scores['material_all'])

    ret_shape = list_of_dicts__to__dict_of_lists(list(ret_shape.values()))
    if VERBOSE:
        print(ret_shape)
    ret_shape = {k: (np.mean(v), np.std(v)) for k, v in ret_shape.items()}
    ret_shape['scene_count'] = len(scores['shape_all'])

    scores_stats = {'view': ret_new_view, 'light': ret_new_light, 'geometry': ret_geometry, 'material': ret_material,
                    'shape': ret_shape}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'scores_stats': scores_stats,
                   'scores': scores,
                   'info': info}, f, indent=4)
    logger.info(f'Results saved to {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, required=True)
    parser.add_argument('-o', '--output-path', type=str, required=True)
    parser.add_argument('-s', '--scenes', type=str, default='auto', help='full | light | example | auto')
    args = parser.parse_args()

    if args.scenes == 'auto':
        with open(args.input_path, 'r') as f:
            info = json.load(f)['info']
        scenes = list(info.keys())
    else:
        scenes = get_scenes_from_id(args.scenes)
    process(args.input_path, args.output_path, scenes)


if __name__ == "__main__":
    main()
