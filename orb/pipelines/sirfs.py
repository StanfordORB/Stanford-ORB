import sys
import numpy as np
from PIL import Image
import glob
import os
import json
import pyexr
from orb.constant import PROJ_ROOT, VERSION
from orb.utils.preprocess import load_rgb_png
from orb.pipelines.base import BasePipeline
import logging
logger = logging.getLogger(__name__)


SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
SIRFS_ROOT = os.path.join(PROJ_ROOT, 'imageint/third_party/sirfs')
os.makedirs(SIRFS_ROOT, exist_ok=True)


class Pipeline(BasePipeline):
    def test_new_view(self, scene: str, overwrite=False):
        return []

    def test_new_light(self, scene: str, overwrite=False):
        return []

    def test_shape(self, scene: str, overwrite: bool):
        return {'output_mesh': None, 'target_mesh': None}

    def test_geometry(self, scene: str, overwrite=False):
        output_pattern = os.path.join(SIRFS_ROOT, 'evals/test', scene, '*_normal_map_processed.exr')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Running SIRFS on {scene}, output to {output_pattern}')
            self.test_geometry_execute(scene)
        else:
            logger.info(f'Found {len(glob.glob(output_pattern))} SIRFS outputs for {scene}, output {output_pattern}')
        output_paths = list(sorted(glob.glob(output_pattern)))
        target_paths = list(sorted(glob.glob(os.path.join('/viscam/projects/imageint/capture_scene_data/data/', scene, 'final_output/geometry_outputs/normal_maps/*.npy'))),)
        if len(output_paths) != len(target_paths):
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        return [{
            'output_depth': None, 'target_depth': None,
            'output_normal': f1, 'target_normal': f2
        } for f1, f2 in zip(output_paths, target_paths)]

    def test_geometry_execute(self, scene):
        test_out_dir = os.path.join(SIRFS_ROOT, 'evals/test', scene)
        os.makedirs(test_out_dir, exist_ok=True)
        for path in glob.glob(os.path.join(SIRFS_ROOT, 'out/sirfs_all_test_images_512', scene, 'results/normal_map/*_normal_map.png')):
            # convert .png to .exr
            output_path = os.path.join(test_out_dir, os.path.basename(path).replace('.png', '_processed.exr'))
            normal = load_rgb_png(path)
            normal = normal * 2 - 1
            pyexr.write(output_path, normal)
            Image.fromarray((normal * 127.5 + 127.5).astype(np.uint8)).save(output_path.replace('_processed.exr', '.png'))

    def test_material(self, scene: str, overwrite=False):
        output_paths = list(sorted(glob.glob(os.path.join(SIRFS_ROOT, 'out/sirfs_all_test_images_512', scene, 'results/reflectance_map_div2/*_reflectance_map_div2.png'))))
        target_paths = list(sorted(glob.glob(os.path.join('/viscam/projects/imageint/capture_scene_data/data/', scene, 'final_output/geometry_outputs/albedo_maps/*.png'))))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
