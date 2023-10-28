import argparse
import importlib
import os
from orb.constant import PROJ_ROOT
from orb.constant import get_scenes_from_id
import json
import logging
_lpips = None


logger = logging.getLogger(__name__)


OVERWRITE = False
OVERWRITE_VIEW = False
OVERWRITE_LIGHT = False
OVERWRITE_GEOMETRY = False
OVERWRITE_MATERIAL = False
OVERWRITE_SHAPE = False


def process(method, output_path, scenes):
    logger.info(f'Computing metrics for {method}')
    pipeline = getattr(importlib.import_module(f'orb.pipelines.{method}', package=None), 'Pipeline')()

    info = dict()
    for scene in scenes:
        info[scene] = dict()
        results = pipeline.test_new_view(scene, overwrite=OVERWRITE or OVERWRITE_VIEW)
        info[scene].update({'view': results})

        results = pipeline.test_new_light(scene, overwrite=OVERWRITE or OVERWRITE_LIGHT)
        info[scene].update({'light': results})

        results = pipeline.test_geometry(scene, overwrite=OVERWRITE or OVERWRITE_GEOMETRY)
        info[scene].update({'geometry': results})

        results = pipeline.test_material(scene, overwrite=OVERWRITE or OVERWRITE_MATERIAL)
        info[scene].update({'material': results})

        results = pipeline.test_shape(scene, overwrite=OVERWRITE or OVERWRITE_SHAPE)
        info[scene].update({'shape': results})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'info': info}, f, indent=4)

    logger.info(f'Outputs saved to: {output_path}')
    logger.info('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, required=False, default='mymethod')
    parser.add_argument('-o', '--output-path', type=str, required=False, default=None)
    parser.add_argument('-s', '--scenes', type=str, default='example', help='full | light | example')
    args = parser.parse_args()

    scenes = get_scenes_from_id(args.scenes)
    if args.output_path is None:
        output_path = os.path.join(PROJ_ROOT, f'logs/test/{args.method}.json')
    else:
        output_path = args.output_path
    process(args.method, output_path, scenes)


if __name__ == "__main__":
    main()
