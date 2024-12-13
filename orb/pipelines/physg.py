import sys
import os
from orb.constant import VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, PHYSG_ROOT, DEFAULT_SCENE_DATA_DIR, DEBUG_SAVE_DIR
sys.path.insert(0, os.path.join(PHYSG_ROOT, 'code'))
from orb.pipelines.base import BasePipeline
from typing import Dict, List
import json
from orb.third_party.physg.code.evaluation.eval import evaluate
from orb.utils.load_data import get_novel_scenes
import trimesh
import glob
import logging

logger = logging.getLogger(__name__)

# EXP_ID = "0524_fix_cam_id"
# EXP_ID = "0527_fix_gamma"
if VERSION == 'submission':
    EXP_ID = "0530_gamma1_exp2"
elif VERSION == 'rebuttal':
    EXP_ID = "0813"
elif VERSION == 'revision':
    EXP_ID = {s: '0530_gamma1_exp2' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES})
elif VERSION == 'release':
    EXP_ID = {s: '0530_gamma1_exp2' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES + SUBMISSION_ADD_SCENES})
elif VERSION == 'extension':
    EXP_ID = '1110'
else:
    raise NotImplementedError()
if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    if EXP_ID in ["0524_fix_cam_id", "0527_fix_gamma"]:
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data'
    else:
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
logger.info(f'using scene data dir {SCENE_DATA_DIR}')
CHECKPOINT = 'latest'  # '120000'


os.makedirs(os.path.join(PHYSG_ROOT, 'evals', 'test'), exist_ok=True)


class Pipeline(BasePipeline):
    def test_inverse_rendering(self, scene: str, overwrite=False) -> List:
        return self.test_core(scene, 'train', overwrite)

    def test_new_view(self, scene: str, overwrite=False) -> List:
        return self.test_core(scene, 'test', overwrite)

    def test_new_light(self, scene: str, overwrite=False) -> List:
        os.environ['PHYSG_GT_ENV_MAP'] = '1'
        if VERSION == 'extension':
            novel_scenes = get_novel_scenes(scene, SCENE_DATA_DIR)
        else:
            scene1, scene2 = [scene for scene in os.listdir(os.path.join(SCENE_DATA_DIR, scene, 'final_output/llff_format_HDR/sparse')) if scene != '0']
            novel_scenes = [scene1, scene2]
        ret = []
        for light_scene in [scene, *novel_scenes]:
            os.environ['PHYSG_LIGHT_SCENE'] = light_scene
            ret += self.test_core(scene, light_scene, overwrite)
            os.environ['PHYSG_LIGHT_SCENE'] = ''
        os.environ['PHYSG_GT_ENV_MAP'] = '0'
        return ret

    def test_shape(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        if scene in ['scene007_obj022_curry', 'scene002_obj012_cart', 'scene006_obj019_blocks', 'scene002_obj018_teapot']:
            empty_mesh = trimesh.Trimesh(vertices=[], faces=[])
            empty_mesh_path = '/viscam/projects/imageint/yzzhang/tmp/empty_mesh.obj'
            empty_mesh.export(empty_mesh_path)
            return {'output_mesh': empty_mesh_path, 'target_mesh': target_mesh_path}
        exp_dir = os.path.join(PHYSG_ROOT, 'exps', f'default-{scene}/{exp_id}')
        if len(os.listdir(exp_dir)) > 1:
            logger.error(f'{exp_dir} should have only one experiment, but got {os.listdir(exp_dir)}')

        timestamp = os.path.basename(sorted(glob.glob(os.path.join(exp_dir, '202*')), key=os.path.getmtime)[-1])
        if not os.path.exists(os.path.join(exp_dir, timestamp, 'checkpoints', 'ModelParameters', CHECKPOINT + '.pth')):
            raise RuntimeError(f'checkpoint {CHECKPOINT} does not exist')

        with open(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')) as f:
            kwargs = json.load(f)

        kwargs['conf'] = os.path.join(exp_dir, timestamp, 'runconf.conf')
        evals_folder_name = 'evals/shape'
        output_mesh_path = os.path.join(exp_dir.replace("exps", evals_folder_name), 'mesh.obj')
        if overwrite or not os.path.exists(output_mesh_path):
            logger.info(f'Starting evaluation for {exp_dir}/{timestamp} {CHECKPOINT}')
            evaluate(conf=kwargs['conf'],
                     write_idr=True,
                     gamma=kwargs['gamma'],
                     exposure=kwargs['exposure'],
                     data_split_dir=kwargs['data_split_dir'],
                     expname=kwargs['expname'],
                     exps_folder_name=kwargs['exps_folder_name'],
                     evals_folder_name=evals_folder_name,
                     timestamp=timestamp,
                     checkpoint=CHECKPOINT,
                     resolution=256,
                     save_exr=True,
                     save_png=True,
                     light_sg='',
                     geometry='',
                     view_name='',
                     diffuse_albedo='',
                     split='test',
                     eval_shape=True,
                     )
            DEBUG = True
            if DEBUG:
                output_mesh = trimesh.load(output_mesh_path)
                debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'physg', scene)
                os.makedirs(debug_save_dir, exist_ok=True)
                output_mesh.export(os.path.join(debug_save_dir, 'world_physg.obj'))

        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

    def test_core(self, scene: str, split: str, overwrite=False) -> List:
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(PHYSG_ROOT, 'exps', f'default-{scene}/{exp_id}')
        if len(os.listdir(exp_dir)) > 1:
            logger.error(f'{exp_dir} should have only one experiment, but got {os.listdir(exp_dir)}')

        timestamp = os.path.basename(sorted(glob.glob(os.path.join(exp_dir, '202*')), key=os.path.getmtime)[-1])
        if not os.path.exists(os.path.join(exp_dir, timestamp, 'checkpoints', 'ModelParameters', CHECKPOINT + '.pth')):
            raise RuntimeError(f'checkpoint {CHECKPOINT} does not exist')
        if exp_id == "0524_fix_cam_id": #not os.path.exists(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')):
            raise NotImplementedError()
            # hack
            expname = exp_dir.removeprefix('/viscam/projects/imageint/yzzhang/imageint/imageint/third_party/physg/exps/default-')
            kwargs = {
                'conf': '/viscam/projects/imageint/yzzhang/imageint/configs/physg.conf',
                'gamma': 0.4546,
                'exposure': 0,
                'data_split_dir': os.path.join(SCENE_DATA_DIR, scene),
                'exps_folder_name': 'exps',
                'expname': expname,
            }
        else:
            with open(os.path.join(exp_dir, timestamp, 'trainer_kwargs.json')) as f:
                kwargs = json.load(f)
            # if kwargs['data_split_dir'] == '/viscam/projects/imageint/data/capture_scene_data/data':
            #     print('changing data dir from', kwargs['data_split_dir'], 'to', os.path.join(SCENE_DATA_DIR, scene))
            #     kwargs['data_split_dir'] = os.path.join(SCENE_DATA_DIR, scene)
            # if not Path(kwargs['data_split_dir']).samefile(os.path.join(SCENE_DATA_DIR, scene)):
            #     import ipdb; ipdb.set_trace()
        kwargs['conf'] = os.path.join(exp_dir, timestamp, 'runconf.conf')

        evals_folder_name = 'evals/' + split
        output_pattern = os.path.join(exp_dir.replace("exps", evals_folder_name), 'sg_rgb_*.exr')
        if kwargs['gamma'] == 0.4546: #EXP_ID == '0524_fix_cam_id':  # FIXME hack
            kwargs['gamma'] = 1
            output_pattern = output_pattern.replace('.exr', '.png')

        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Starting evaluation for {exp_dir}/{timestamp} {CHECKPOINT}')
            evaluate(conf=kwargs['conf'],
                     write_idr=True,
                     gamma=kwargs['gamma'],
                     exposure=kwargs['exposure'],
                     data_split_dir=kwargs['data_split_dir'],
                     expname=kwargs['expname'],
                     exps_folder_name=kwargs['exps_folder_name'],
                     evals_folder_name=evals_folder_name,
                     timestamp=timestamp,
                     checkpoint=CHECKPOINT,
                     resolution=512,
                     save_exr=True,
                     save_png=True,
                     light_sg='',
                     # light_sg='' if split not in ['train', 'test'] else os.path.join(PROCESSED_SCENE_DATA_DIR, split, 'physg_format/env_map/'),   # FIXME
                     # light_sg='/viscam/projects/imageint/yzzhang/data/processed_data/scene001_obj003_baking/physg_format/env_map/0000/sg_128.npy',
                     geometry='',
                     view_name='',
                     diffuse_albedo='',
                     split=split,
                     eval_shape=False,
                     )
            # output_paths = list(glob.glob(output_pattern))
            # for output_path in output_paths:
            #     Image.fromarray((load_rgb_exr(output_path).clip(0, 1) * 255).astype(np.uint8)).save(output_path.replace('.exr', '.png'))
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
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        # assume consistent ordering  # TODO need to check for every method whether this is true
        ret = [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
        return ret

    def test_geometry(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir = os.path.join(PHYSG_ROOT, 'evals', 'test', f'default-{scene}/{exp_id}')
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
        test_out_dir = os.path.join(PHYSG_ROOT, 'evals', 'test', f'default-{scene}/{exp_id}')
        output_pattern = os.path.join(test_out_dir, 'sg_diffuse_albedo_*.exr')
        output_paths = sorted(glob.glob(output_pattern))
        target_paths = sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/albedo_maps/*.png')))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
