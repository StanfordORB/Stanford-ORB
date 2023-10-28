from .base import BasePipeline
import json
from orb.constant import PROJ_ROOT
import glob
import os


import logging


SCENE_DATA_DIR = os.path.join(PROJ_ROOT, 'data/stanfordorb/')  # TODO: update with your data path
logger = logging.getLogger(__name__)


class Pipeline(BasePipeline):
    def test_inverse_rendering(self, scene: str, overwrite: bool = False):
        return self.test_core(scene, 'train', overwrite)

    def test_new_view(self, scene: str, overwrite: bool = False):
        return self.test_core(scene, 'test', overwrite)

    def test_new_light(self, scene: str, overwrite: bool = False):
        scene1, scene2 = [scene for scene in os.listdir(os.path.join(SCENE_DATA_DIR, f'llff_colmap_HDR/{scene}/sparse')) if scene != '0']
        return (self.test_core(scene, split=scene1, overwrite=overwrite) +
                self.test_core(scene, split=scene2, overwrite=overwrite))

    def test_geometry(self, scene: str, overwrite: bool = False):
        depth_target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, f'ground_truth/{scene}/z_depth/*.npy'))))
        depth_output_paths = [None] * len(depth_target_paths)  # TODO: update with prediction results
        normal_target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, f'ground_truth/{scene}/surface_normal/*.npy'))))
        normal_output_paths = [None] * len(normal_target_paths)  # TODO: update with prediction results
        target_mask_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, f'blender_HDR/{scene}/test_mask/*.png'))))
        return [
            {'output_depth': f1, 'target_depth': f2,
             'output_normal': f3, 'target_normal': f4,
             'target_mask': f5}
            for f1, f2, f3, f4, f5 in zip(
                depth_output_paths,
                depth_target_paths,
                normal_output_paths,
                normal_target_paths,
                target_mask_paths,
            )
        ]

    def test_material(self, scene: str, overwrite: bool = False):
        target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, f'ground_truth/{scene}/pseudo_gt_albedo/*.png'))))
        output_paths = [None] * len(target_paths)  # TODO: update with prediction results
        target_mask_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, f'blender_HDR/{scene}/test_mask/*.png'))))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind],
                 'target_mask': target_mask_paths[ind]}
                for ind in range(len(output_paths))]

    def test_core(self, scene: str, split: str, overwrite: bool):
        # `scene`: the name of the training scene
        # `split`: 'train', 'test', or the name of the novel scene
        target_paths = []
        with open(os.path.join(SCENE_DATA_DIR, f'blender_HDR/{scene}/transforms_{split if split in ["train", "test"] else "test" if split == scene else "novel"}.json')) as f:
            for frame in json.load(f)['frames']:
                if 'scene_name' in frame:
                    if frame['scene_name'] == split:
                        target_paths.append(os.path.join(SCENE_DATA_DIR, "blender_HDR", frame['scene_name'], frame['file_path'] + '.exr'))
                else:
                    target_paths.append(os.path.join(SCENE_DATA_DIR, f'blender_HDR', scene, frame['file_path'] + '.exr'))
        for target_path in target_paths:
            assert os.path.exists(target_path), target_path
        output_paths = [None] * len(target_paths)  # TODO: update with prediction results
        ret = [{'output_image': output_paths[ind], 'target_image': target_paths[ind]}
               for ind in range(len(output_paths))]
        return ret

    def test_shape(self, scene: str, overwrite: bool = False):
        output_mesh_path = None  # TODO: update with prediction results
        target_mesh_path = os.path.join(SCENE_DATA_DIR, f'ground_truth/{scene}/mesh_blender/mesh.obj')
        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}
