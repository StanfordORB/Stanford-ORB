import sys
from orb.constant import PROJ_ROOT, VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, DEFAULT_SCENE_DATA_DIR, NVDIFFREC_ROOT, DEBUG_SAVE_DIR
sys.path.insert(0, NVDIFFREC_ROOT)
import imageio
import logging
import numpy as np
import json
import os
import glob
from orb.utils.load_data import get_novel_scenes
# from tu.configs import get_attrdict
from tu2.ppp import get_attrdict
from orb.pipelines.base import BasePipeline
try:
    from orb.third_party.nvdiffrec.render import mesh, light
    from orb.third_party.nvdiffrec.geometry.dlmesh import DLMesh
    from orb.third_party.nvdiffrec.train import validate, initial_guess_material, prepare_batch
    from orb.third_party.nvdiffrec.render.util import rotate_x
    from orb.utils.extract_mesh import clean_mesh
    from orb.datasets.nvdiffrecmc import DatasetCapture
    import nvdiffrast.torch as dr
except:
    pass

try:
    import pyexr
except ImportError:
    pass
try:
    import trimesh
except ImportError:
    pass


from typing import List


logger = logging.getLogger(__name__)

if os.getenv('IMAEGINT_PSEUDO_GT') == '1':
    EXP_ID = 'scanned_geometry'
else:
    if VERSION == 'submission':
        EXP_ID = '0530'
    elif VERSION == 'rebuttal':
        EXP_ID = '0813_fix_import_fix'
    elif VERSION == 'revision':
        EXP_ID = {s: '0530' for s in SUBMISSION_SCENES}
        EXP_ID.update({s: '0813_fix_import_fix' for s in REBUTTAL_SCENES})
    elif VERSION == 'release':
        EXP_ID = {s: '0530' for s in SUBMISSION_SCENES}
        EXP_ID.update({s: '0813_fix_import_fix' for s in REBUTTAL_SCENES + SUBMISSION_ADD_SCENES})
    elif VERSION == 'extension':
        EXP_ID = '1110'
    else:
        raise NotImplementedError()
if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


class Pipeline(BasePipeline):
    def test_new_light(self, scene: str, overwrite: bool):
        os.environ['NVDIFFREC_GT_ENV_MAP'] = '1'
        if VERSION == 'extension':
            novel_scenes = get_novel_scenes(scene, SCENE_DATA_DIR)
        else:
            scene1, scene2 = [scene for scene in os.listdir(os.path.join(SCENE_DATA_DIR, scene, 'final_output/llff_format_HDR/sparse')) if scene != '0']
            novel_scenes = [scene1, scene2]
        ret = []
        for light_scene in [scene, *novel_scenes]:
            os.environ['NVDIFFREC_LIGHT_SCENE'] = light_scene
            ret += self.test_core(scene, light_scene, overwrite)
            os.environ['NVDIFFREC_LIGHT_SCENE'] = ''
        os.environ['NVDIFFREC_GT_ENV_MAP'] = '0'
        return ret

    def test_new_view(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        if exp_id == 'scanned_geometry':
            return []
        return self.test_core(scene, 'test', overwrite)

    def test_core(self, scene: str, split: str, overwrite=False) -> List:
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        if exp_id == 'scanned_geometry':
            if VERSION == 'submission' or (VERSION == 'revision' and scene in SUBMISSION_SCENES):
                # should actually use the one under geometry_outputs
                exp_dir = mesh_dir = f'/viscam/projects/imageint/capture_scene_data/data/{scene}/final_output/pseudo_gt_mesh'
            elif VERSION == 'rebuttal' or (VERSION == 'revision' and scene in REBUTTAL_SCENES):
                exp_dir = mesh_dir = f'/viscam/projects/imageint/capture_scene_data/data/{scene}/final_output/geometry_outputs/pseudo_gt_mesh'
            elif VERSION == 'release':
                exp_dir = mesh_dir = f'/viscam/projects/imageint/capture_scene_data/data/{scene}/final_output/geometry_outputs/pseudo_gt_mesh'
            else:
                raise NotImplementedError((VERSION, scene))
        else:
            exp_dir = os.path.join(NVDIFFREC_ROOT, f'out/{scene}/{exp_id}')
            mesh_dir = os.path.join(exp_dir, 'mesh')
        out_dir = os.path.join(NVDIFFREC_ROOT, f'evals/{split}/{scene}/{exp_id}')
        output_pattern = os.path.join(out_dir, '*_opt.png')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Running evaluation for {exp_dir}')
            logger.info(f'Output pattern: {output_pattern}')
            if not os.path.exists(os.path.join(exp_dir, 'config.json')):
                FLAGS = get_attrdict(
                    {
                        "config": os.path.join(PROJ_ROOT, 'configs/nvdiffrec.json'),
                        "iter": 5000,
                        "batch": 8,
                        "spp": 1,
                        "layers": 1,
                        "train_res": [
                            512,
                            512
                        ],
                        "display_res": [
                            512,
                            512
                        ],
                        "texture_res": [
                            1024,
                            1024
                        ],
                        "display_interval": 0,
                        "save_interval": 100,
                        "learning_rate": [
                            0.03,
                            0.01
                        ],
                        "min_roughness": 0.08,
                        "custom_mip": False,
                        "random_textures": True,
                        "background": "white",
                        "loss": "logl1",
                        # "out_dir": f"out/{scene}/{EXP_ID}",
                        # "out_dir": exp_dir,
                        "ref_mesh": os.path.join(SCENE_DATA_DIR, f"{scene}/final_output/blender_format_HDR"),
                        "base_mesh": None,
                        "validate": True,
                        "mtl_override": None,
                        "dmtet_grid": 128,
                        "mesh_scale": 2.4,
                        "env_scale": 1.0,
                        "envmap": None,
                        "display": [
                            {
                                "latlong": True
                            },
                            {
                                "bsdf": "kd"
                            },
                            {
                                "bsdf": "ks"
                            },
                            {
                                "bsdf": "normal"
                            }
                        ],
                        "camera_space_light": False,
                        "lock_light": False,
                        "lock_pos": False,
                        "sdf_regularizer": 0.2,
                        "laplace": "relative",
                        "laplace_scale": 3000,
                        "pre_load": True,
                        "kd_min": [
                            0.0,
                            0.0,
                            0.0,
                            0.0
                        ],
                        "kd_max": [
                            1.0,
                            1.0,
                            1.0,
                            1.0
                        ],
                        "ks_min": [
                            0,
                            0.25,
                            0.0
                        ],
                        "ks_max": [
                            1.0,
                            1.0,
                            1.0
                        ],
                        "nrm_min": [
                            -1.0,
                            -1.0,
                            0.0
                        ],
                        "nrm_max": [
                            1.0,
                            1.0,
                            1.0
                        ],
                        "cam_near_far": [
                            0.1,
                            1000.0
                        ],
                        "learn_light": True,
                        "local_rank": 0,
                        "multi_gpu": False
                    }
                )
            else:
                with open(os.path.join(exp_dir, 'config.json')) as f:
                    FLAGS = get_attrdict(json.load(f))
                FLAGS.out_dir = os.path.join(NVDIFFREC_ROOT, FLAGS.out_dir)
            FLAGS.random_textures = False

            glctx = dr.RasterizeGLContext()
            base_mesh = mesh.load_mesh(os.path.join(mesh_dir, 'mesh.obj'))
            geometry = DLMesh(base_mesh, FLAGS)
            for p in geometry.parameters():
                p.requires_grad_(False)
            mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)
            # mat = base_mesh.material
            dataset_validate = DatasetCapture(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/transforms_{split if split in ["train", "test"] else "test" if split == scene else "novel"}.json'), FLAGS)
            # dataset_validate.set_debug_mesh(base_mesh)  # hack
            if os.getenv('NVDIFFREC_LIGHT_SCENE', '') == '':
                lgt = light.load_env(os.path.join(mesh_dir, 'probe.hdr'))
            else:
                lgt = None
            validate(glctx, geometry, mat, lgt, dataset_validate, out_dir, FLAGS)
        else:
            logger.info(f'Found existing evaluation for {exp_dir} with {len(glob.glob(output_pattern))} images for {output_pattern}')

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

    def test_geometry(self, scene: str, overwrite=False) -> List:
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        if exp_id == 'scanned_geometry':
            return []
        out_dir = os.path.join(NVDIFFREC_ROOT, f'evals/geometry/{scene}/{exp_id}')
        normal_output_pattern = os.path.join(out_dir, 'normal_*.exr')
        depth_output_pattern = os.path.join(out_dir, 'depth_*.exr')

        if overwrite or (len(glob.glob(normal_output_pattern)) == 0 and len(glob.glob(depth_output_pattern)) == 0):
            logger.info(f'Running geometry evaluation for {scene} with {exp_id}, Output to {out_dir}')
            self.test_geometry_core(scene)
            return []  # hack: need a different conda environment
        else:
            logger.info(f'Found existing evaluation for {scene} with {exp_id}, Output to {out_dir}')
            assert len(glob.glob(normal_output_pattern)) == len(glob.glob(depth_output_pattern)), (normal_output_pattern, depth_output_pattern)

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

    def test_geometry_core(self, scene: str):
        from orb.utils.mesh_to_geometry import render_geometry
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NVDIFFREC_ROOT, f'out/{scene}/{exp_id}')
        mesh_dir = os.path.join(exp_dir, 'mesh')
        out_dir = os.path.join(NVDIFFREC_ROOT, f'evals/geometry/{scene}/{exp_id}')
        data_config = os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/transforms_test.json')
        render_geometry(data_config, mesh_dir, out_dir)

    def test_material(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        if exp_id == 'scanned_geometry':
            return []
        out_dir = os.path.join(NVDIFFREC_ROOT, f'evals/test/{scene}/{exp_id}')
        output_pattern = os.path.join(out_dir, 'val_*_kd.png')
        output_paths = list(sorted(glob.glob(output_pattern)))
        target_paths = list(sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/albedo_maps/*.png'))))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_shape(self, scene: str, overwrite: bool):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        if exp_id == 'scanned_geometry':
            return {'output_mesh': None, 'target_mesh': None}
        out_dir = os.path.join(NVDIFFREC_ROOT, f'evals/shape/{scene}/{exp_id}')
        os.makedirs(out_dir, exist_ok=True)
        output_mesh_path = os.path.join(out_dir, 'output_mesh_world.obj')
        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        if overwrite or not os.path.exists(output_mesh_path):
            exp_dir = os.path.join(NVDIFFREC_ROOT, f'out/{scene}/{exp_id}')
            mesh_dir = os.path.join(exp_dir, 'mesh')
            base_mesh = os.path.join(mesh_dir, 'mesh.obj')
            output_mesh = trimesh.load(base_mesh)
            output_mesh = trimesh.Trimesh(vertices=output_mesh.vertices, faces=output_mesh.faces)

            output_mesh.apply_transform(rotate_x(-np.pi / 2).numpy())

            output_mesh.export(output_mesh_path.replace('.obj', '_raw.obj'))
            if output_mesh.vertices.shape[0] > 0:
                output_mesh = clean_mesh(output_mesh)
            output_mesh.export(output_mesh_path)
            DEBUG = 1
            if DEBUG:
                debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'nvdiffrec', scene)
                os.makedirs(debug_save_dir, exist_ok=True)
                output_mesh.export(os.path.join(debug_save_dir, 'world_nvdiffrec.obj'))
        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}
