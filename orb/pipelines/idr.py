import sys
from orb.constant import VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, DEFAULT_SCENE_DATA_DIR, IDR_ROOT, DEBUG_SAVE_DIR
import os
import numpy as np
sys.path.insert(0, os.path.join(IDR_ROOT, 'code'))
import glob
import json
from orb.third_party.idr.code.evaluation.eval import evaluate
from orb.pipelines.base import BasePipeline
from typing import Dict, Any, List
import logging
os.environ['IDR_NO_CACHE'] = '1'

logger = logging.getLogger(__name__)


# EXP_ID = '0525'
# EXP_ID = '0530_fix_cam_key'
# EXP_ID = '0530_rm_old_data'
if VERSION == 'submission':
    EXP_ID = '0531'
elif VERSION == 'rebuttal':
    EXP_ID = '0813'
elif VERSION == 'revision':
    EXP_ID = {s: '0531' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES})
elif VERSION == 'release':
    EXP_ID = {s: '0531' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES + SUBMISSION_ADD_SCENES})
elif VERSION == 'extension':
    EXP_ID = '1108'
else:
    raise NotImplementedError

if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
    CONFIG_NAME = 'dtu_fixed_cameras'
else:
    if EXP_ID == "0525":
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data'
        CONFIG_NAME = 'dtu_fixed_cameras'
    elif EXP_ID == '0530_fix_cam_key':
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data'  # FIXME this is wrong
        CONFIG_NAME = 'dtu_fixed_cameras'
    else:
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
        CONFIG_NAME = 'data_0529_fixed_cameras'
logger.info(f'using scene data dir {SCENE_DATA_DIR}')
CHECKPOINT = 'latest'  # '2000'


os.makedirs(os.path.join(IDR_ROOT, 'evals'), exist_ok=True)


class Pipeline(BasePipeline):

    @staticmethod
    def test_shape(scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(IDR_ROOT, 'exps', f'{CONFIG_NAME}_{exp_id}_{scene}')
        if len(os.listdir(exp_dir)) > 1:
            logger.error(f'{exp_dir} should have only one experiment, but got {os.listdir(exp_dir)}')

        timestamp = os.path.basename(sorted(glob.glob(os.path.join(exp_dir, '202*')), key=os.path.getmtime)[-1])
        with open(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')) as f:
            kwargs = json.load(f)
        kwargs['conf'] = os.path.join(exp_dir, timestamp, 'runconf.conf')
        output_dir = os.path.join(IDR_ROOT, 'evals/shape', os.path.basename(exp_dir))
        output_pattern = os.path.join(output_dir, 'surface_world_coordinates_*.ply')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Running evaluation for {exp_dir}/{timestamp}')
            evaluate(conf=kwargs['conf'],
                     expname=kwargs['expname'],
                     exps_folder_name=kwargs['exps_folder_name'],
                     evals_folder_name=f'evals/shape',
                     timestamp=timestamp,
                     checkpoint=CHECKPOINT,
                     scan_id=kwargs['scan_id'],
                     data_dir=kwargs['data_dir'],
                     resolution=256,
                     eval_cameras=False,
                     eval_rendering=False,
                     eval_shape=True,
                     split='test',
                     )

            DEBUG = True
            if DEBUG:
                import trimesh
                output_mesh_path, = glob.glob(output_pattern)
                # debug_save_dir = f'/viscam/projects/imageint/yzzhang/tmp/debug_mesh/{scene}'
                debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'idr', 'test_shape', scene)
                os.makedirs(debug_save_dir, exist_ok=True)
                output_mesh = trimesh.load(output_mesh_path)
                output_mesh.export(os.path.join(debug_save_dir, 'world_idr.obj'))

                with open(os.path.join(SCENE_DATA_DIR, scene, 'final_output/blender_format_LDR/transforms_test.json'), 'r') as fp:
                    c2w = np.array(json.load(fp)['frames'][0]['transform_matrix'])
                w2c = np.linalg.inv(c2w)
                output_mesh.apply_transform(w2c)
                output_mesh.export(os.path.join(debug_save_dir, 'camera_idr.obj'))

        output_mesh_path, = glob.glob(output_pattern)
        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, f'{scene}/final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')

        return {
            'output_mesh': output_mesh_path,
            'target_mesh': target_mesh_path,
        }

    def test_inverse_rendering(self, scene: str, overwrite=False):
        return self.test_core(scene, split='train', overwrite=overwrite)

    def test_new_view(self, scene: str, overwrite=False):
        return self.test_core(scene, split='test', overwrite=overwrite)

    def test_core(self, scene, split, overwrite):
        assert split in ['train', 'test'], split
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(IDR_ROOT, 'exps', f'{CONFIG_NAME}_{exp_id}_{scene}')
        if len(os.listdir(exp_dir)) > 1:
            logger.error(f'{exp_dir} should have only one experiment, but got {os.listdir(exp_dir)}')

        timestamp = os.path.basename(sorted(glob.glob(os.path.join(exp_dir, '202*')), key=os.path.getmtime)[-1])
        with open(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')) as f:
            kwargs = json.load(f)
        kwargs['conf'] = os.path.join(exp_dir, timestamp, 'runconf.conf')

        # if not os.path.exists(os.path.join(exp_dir, timestamp, 'CHECKPOINTs', 'ModelParameters', CHECKPOINT + '.pth')):
        #     raise RuntimeError(f'CHECKPOINT {CHECKPOINT} does not exist')

        output_pattern = os.path.join(IDR_ROOT, 'evals', split, os.path.basename(exp_dir), 'rendering', 'eval_0*.png')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Running evaluation for {exp_dir}/{timestamp}')
            evaluate(conf=kwargs['conf'],
                     expname=kwargs['expname'],
                     exps_folder_name=kwargs['exps_folder_name'],
                     evals_folder_name=f'evals/{split}',
                     timestamp=timestamp,
                     checkpoint=CHECKPOINT,
                     scan_id=kwargs['scan_id'],
                     data_dir=kwargs['data_dir'],
                     resolution=512,
                     eval_cameras=False,
                     eval_rendering=True,
                     split=split,
                     )
        else:
            logger.info(f'Found existing evaluation for {exp_dir}/{timestamp} with {len(glob.glob(output_pattern))} images')
        output_paths = list(sorted(glob.glob(output_pattern)))
        target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/{split}/*.exr'))))
        if len(output_paths) != len(target_paths):  # FIXME
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        ret = [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
        return ret

    def test_new_light(self, scene: str, overwrite=True):
        return []

    def test_geometry(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir = os.path.join(IDR_ROOT, 'evals/test', f'{CONFIG_NAME}_{exp_id}_{scene}', 'rendering')
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
        return []
