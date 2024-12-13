import sys
import os
from orb.constant import VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, DEFAULT_SCENE_DATA_DIR, INVRENDER_ROOT, DEBUG_SAVE_DIR
sys.path.insert(0, os.path.join(INVRENDER_ROOT, 'code'))
from orb.pipelines.base import BasePipeline
from typing import Dict, List
import json
from orb.third_party.invrender.code.scripts.relight import relight_obj
from orb.utils.load_data import get_novel_scenes
from pathlib import Path
import glob
import logging

logger = logging.getLogger(__name__)

# EXP_ID = '0525'
if VERSION == 'submission':
    EXP_ID = "0530_use_pyexr_fix"
elif VERSION == 'rebuttal':
    EXP_ID = '0813'
elif VERSION == 'revision':
    EXP_ID = {s: '0530_use_pyexr_fix' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES})
elif VERSION == 'release':
    EXP_ID = {s: '0530_use_pyexr_fix' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES + SUBMISSION_ADD_SCENES})
elif VERSION == 'extension':
    EXP_ID = '1109'
else:
    raise NotImplementedError()

if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    if EXP_ID == "0525":
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data'
    else:
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
logger.info(f'using scene data dir {SCENE_DATA_DIR}')
CHECKPOINT = 'latest'

os.makedirs(os.path.join(INVRENDER_ROOT, 'evals'), exist_ok=True)


class Pipeline(BasePipeline):
    def test_shape(self, scene: str, overwrite: bool) -> dict[str, str]:
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(INVRENDER_ROOT, 'exps', f'Mat-{scene}', exp_id)
        if len(os.listdir(exp_dir)) > 1:
            logger.error(f'{exp_dir} should have only one experiment, but got {os.listdir(exp_dir)}')

        timestamp = os.path.basename(sorted(glob.glob(os.path.join(exp_dir, '202*')), key=os.path.getmtime)[-1])
        if not os.path.exists(os.path.join(exp_dir, timestamp, 'checkpoints', 'ModelParameters', CHECKPOINT + '.pth')):
            raise RuntimeError(f'checkpoint {CHECKPOINT} does not exist')
        with open(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')) as f:
            kwargs = json.load(f)
        if not Path(kwargs['data_split_dir']).samefile(os.path.join(SCENE_DATA_DIR, scene)):
            import ipdb;
            ipdb.set_trace()
        kwargs['conf'] = os.path.join(exp_dir, timestamp, 'runconf.conf')

        evals_folder_name = 'evals/shape'
        output_mesh_path = os.path.join(exp_dir.replace("exps", evals_folder_name), 'output_mesh_world.obj')

        if overwrite or not os.path.exists(output_mesh_path):
            logger.info(f'Starting evaluation for {exp_dir}/{timestamp}')
            relight_obj(conf=kwargs['conf'],
                        relits_folder_name=evals_folder_name,
                        data_split_dir=kwargs['data_split_dir'],
                        expname=kwargs['expname'],
                        exps_folder_name=kwargs['exps_folder_name'],
                        timestamp=timestamp,
                        checkpoint=CHECKPOINT,
                        frame_skip=1,
                        split='test',
                        eval_light=False,
                        eval_shape=True,
                        )
            DEBUG = True
            if DEBUG:
                import trimesh
                debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'invrender', scene)
                os.makedirs(debug_save_dir, exist_ok=True)
                output_mesh = trimesh.load_mesh(output_mesh_path)
                output_mesh.export(os.path.join(debug_save_dir, 'world_invrender.obj'))

        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

    def test_inverse_rendering(self, scene: str, overwrite=False) -> List:
        return self.test_core(scene, 'train', overwrite)

    def test_new_view(self, scene: str, overwrite=False) -> List:
        return self.test_core(scene, 'test', overwrite)

    def test_new_light(self, scene: str, overwrite=False) -> List:
        # os.environ['INVRENDER_GT_ENV_MAP_DEBUG'] = '1'
        os.environ['INVRENDER_GT_ENV_MAP'] = '1'
        if VERSION == 'extension':
            novel_scenes = get_novel_scenes(scene, SCENE_DATA_DIR)
        else:
            scene1, scene2 = [scene for scene in os.listdir(os.path.join(SCENE_DATA_DIR, scene, 'final_output/llff_format_HDR/sparse')) if scene != '0']
            novel_scenes = [scene1, scene2]
        ret = []
        for light_scene in [scene, *novel_scenes]:
            os.environ['INVRENDER_LIGHT_SCENE'] = light_scene
            ret += self.test_core(scene, light_scene, overwrite)
            os.environ['INVRENDER_LIGHT_SCENE'] = ''
        os.environ['INVRENDER_GT_ENV_MAP'] = '0'
        return ret

    def test_core(self, scene: str, split: str, overwrite=False) -> List:
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(INVRENDER_ROOT, 'exps', f'Mat-{scene}', exp_id)
        if len(os.listdir(exp_dir)) > 1:
            logger.error(f'{exp_dir} should have only one experiment, but got {os.listdir(exp_dir)}')

        timestamp = os.path.basename(sorted(glob.glob(os.path.join(exp_dir, '202*')), key=os.path.getmtime)[-1])
        if not os.path.exists(os.path.join(exp_dir, timestamp, 'checkpoints', 'ModelParameters', CHECKPOINT + '.pth')):
            raise RuntimeError(f'checkpoint {CHECKPOINT} does not exist')
        if EXP_ID == '0525':
            raise NotImplementedError()
            expname = exp_dir.removeprefix('/viscam/projects/imageint/yzzhang/imageint/imageint/third_party/invrender/exps/Mat-')
            kwargs = {
                'conf': '/viscam/projects/imageint/yzzhang/imageint/configs/invrender.conf',
                'data_split_dir': os.path.join(SCENE_DATA_DIR, scene),
                'exps_folder_name': 'exps',
                'expname': expname,
            }
        else:
            with open(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')) as f:
                kwargs = json.load(f)
        if not Path(kwargs['data_split_dir']).samefile(os.path.join(SCENE_DATA_DIR, scene)):
            import ipdb; ipdb.set_trace()
        kwargs['conf'] = os.path.join(exp_dir, timestamp, 'runconf.conf')

        evals_folder_name = 'evals/' + split
        output_pattern = os.path.join(exp_dir.replace("exps", evals_folder_name), 'sg_rgb_*.exr')

        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Starting evaluation for {exp_dir}/{timestamp}')
            relight_obj(conf=kwargs['conf'],
                        relits_folder_name=evals_folder_name,
                        data_split_dir=kwargs['data_split_dir'],
                        expname=kwargs['expname'],
                        exps_folder_name=kwargs['exps_folder_name'],
                        timestamp=timestamp,
                        checkpoint=CHECKPOINT,
                        frame_skip=1,
                        split=split if split in ['test', 'train'] else 'test' if split == scene else 'novel',
                        eval_light=True,
                        eval_shape=False,
                        )
        else:
            logger.info(f'Found existing evaluation for {exp_dir}/{timestamp} {CHECKPOINT} with {len(glob.glob(output_pattern))} images')
        output_paths = list(sorted(glob.glob(output_pattern)))

        target_paths = []
        with open(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/transforms_{split if split in ["train", "test"] else "test" if split == scene else "novel"}.json')) as f:
            for frame in json.load(f)['frames']:
                if 'scene_name' in frame:
                    if frame['scene_name'] == split:
                        target_paths.append(os.path.join(SCENE_DATA_DIR, frame['scene_name'], f'final_output/blender_format_HDR', frame['file_path'] + '.exr'))
                else:
                    target_paths.append(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR', frame['file_path'] + '.exr'))

        if len(output_paths) != len(target_paths):  # FIXME
            # import ipdb; ipdb.set_trace()
            logger.error(str([len(output_paths), len(target_paths), output_pattern, output_paths, target_paths]))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_geometry(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir = os.path.join(INVRENDER_ROOT, 'evals', 'test', f'Mat-{scene}', exp_id)
        normal_output_pattern = os.path.join(test_out_dir, 'normal_*.exr')
        depth_output_pattern = os.path.join(test_out_dir, 'depth_*.exr')
        return [
            {'output_depth': f1, 'target_depth': f2,
             'output_normal': f3, 'target_normal': f4}
            for f1, f2, f3, f4 in zip(
                sorted(glob.glob(depth_output_pattern)),
                sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/z_maps/*.npy'))),
                sorted(glob.glob(normal_output_pattern)),
                sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/normal_maps/*.npy'))),
            )
        ]

    def test_material(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir = os.path.join(INVRENDER_ROOT, 'evals', 'test', f'Mat-{scene}', exp_id)
        output_pattern = os.path.join(test_out_dir, 'albedo_*.png')
        output_paths = sorted(glob.glob(output_pattern))
        target_paths = sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/albedo_maps/*.png')))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
