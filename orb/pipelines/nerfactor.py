import sys
from orb.constant import PROCESSED_SCENE_DATA_DIR, SUBMISSION_ADD_SCENES, SUBMISSION_SCENES, REBUTTAL_SCENES, VERSION, EXTENSION_SCENES, NERFACTOR_ROOT, DEFAULT_SCENE_DATA_DIR, DEBUG_SAVE_DIR
# sys.path.insert(0, '/viscam/projects/imageint/yzzhang/imageint/imageint/third_party/nerfactor/nerfactor')
# sys.path.insert(0, '/viscam/projects/imageint/yzzhang/imageint/imageint/third_party/nerfactor')
import os
sys.path.insert(0, os.path.join(NERFACTOR_ROOT, 'nerfactor'))
sys.path.insert(0, NERFACTOR_ROOT)
import logging
import glob
from orb.constant import PROJ_ROOT
import cv2
from orb.pipelines.base import BasePipeline
from orb.utils.extract_mesh import extract_mesh_from_nerf, clean_mesh
from tqdm import tqdm
import pyexr
from PIL import Image
import numpy as np
import json
try:
    import tensorflow as tf
    from orb.third_party.nerfactor.nerfactor import datasets
    from orb.third_party.nerfactor.nerfactor import models
    from orb.third_party.nerfactor.nerfactor.util import io as ioutil, config as configutil, light as lightutil
    from orb.third_party.nerfactor.third_party.xiuminglib import xiuminglib as xm
except ImportError:
    pass


logger = logging.getLogger(__name__)

if VERSION == 'release':
    EXP_ID = {s: '0609' for s in SUBMISSION_SCENES + REBUTTAL_SCENES}
    EXP_ID.update({s: '1001rerun4' for s in SUBMISSION_ADD_SCENES})
    SHAPE_EXP_ID = {s: '0609' for s in SUBMISSION_SCENES + REBUTTAL_SCENES + SUBMISSION_ADD_SCENES}
elif VERSION == 'extension':
    EXP_ID = {s: '1110' for s in EXTENSION_SCENES}
    SHAPE_EXP_ID = {s: '1110' for s in EXTENSION_SCENES}
else:
    raise NotImplementedError(VERSION)

if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
DEBUG = False


class Pipeline(BasePipeline):
    def test_new_light(self, scene: str, overwrite: bool):
        return self.test_core(scene, split='novel', overwrite=overwrite)

    def test_new_view(self, scene: str, overwrite=False):
        return self.test_core(scene, split='vali', overwrite=overwrite)

    def test_core(self, scene, split, overwrite):
        exp_id = EXP_ID[scene]
        target_paths = []
        with open(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/transforms_{"test" if split == "vali" else split}.json')) as f:
            for frame in json.load(f)['frames']:
                if 'scene_name' in frame:
                    target_paths.append(os.path.join(SCENE_DATA_DIR, frame['scene_name'], f'final_output/blender_format_HDR', frame['file_path'] + '.exr'))
                else:
                    target_paths.append(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR', frame['file_path'] + '.exr'))
        if scene in ['scene002_obj008_grogu', 'scene003_obj010_pepsi', 'scene004_obj010_pepsi', "scene007_obj016_pitcher", "scene002_obj012_cart"]:
            # all nans, should use all zeros
            return [{'output_image': None, 'target_image': target_paths[ind]} for ind in range(len(target_paths))]

        if split == 'vali':
            output_pattern = os.path.join(NERFACTOR_ROOT, f'output/train/{scene}_nerfactor/{exp_id}/lr5e-3/vis_{split}/ckpt-10/batch*/pred_rgb.png')
        else:
            output_pattern = os.path.join(NERFACTOR_ROOT, f'output/train/{scene}_nerfactor/{exp_id}/lr5e-3/vis_{split}/ckpt-10/batch*/pred_rgb_probes_custom.png')
        logger.info(f'found {len(glob.glob(output_pattern))} from {output_pattern}')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            self.test_core_execute(scene, split=split)
        output_paths = list(sorted(glob.glob(os.path.join(output_pattern))))
        if len(output_paths) != len(target_paths):  # FIXME
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_core_execute(self, scene, split):
        exp_id = EXP_ID[scene]
        ckpt = os.path.join(NERFACTOR_ROOT, f"output/train/{scene}_nerfactor/{exp_id}/lr5e-3/checkpoints/ckpt-10")
        config_ini = configutil.get_config_ini(ckpt)
        config = ioutil.read_config(config_ini)

        # Output directory
        outroot = os.path.join(config_ini[:-4], f'vis_{split}', os.path.basename(ckpt))
        # Make dataset
        dataset_name = config.get('DEFAULT', 'dataset')
        Dataset = datasets.get_dataset_class(dataset_name)
        dataset = Dataset(config, split, debug=DEBUG)
        n_views = dataset.get_n_views()
        no_batch = config.getboolean('DEFAULT', 'no_batch')
        datapipe = dataset.build_pipeline(no_batch=no_batch, no_shuffle=True)

        model_name = config.get('DEFAULT', 'model')
        Model = models.get_model_class(model_name)
        model = Model(config, debug=DEBUG)
        ioutil.restore_model(model, ckpt)

        # Optionally, color-correct the albedo
        albedo_scales = None
        # Optionally, edit BRDF
        brdf_z_override = None

        # For all test views
        for batch_i, batch in enumerate(tqdm(datapipe, desc="Inferring Views", total=n_views)):
            if split == 'novel':
                with open(dataset.files[batch_i]) as f:
                    filename = os.path.abspath(json.load(f)['original_path'])
                envmap = os.path.join(PROCESSED_SCENE_DATA_DIR, filename.split('/')[-5], 'invrender_format/env_map', filename.split('/')[-1].replace('.png', '/envmap_world_invrender.exr'))
                logger.info(f'Loading light for {filename} from {envmap}')
                # if envmap in [
                #     '/viscam/projects/imageint/yzzhang/data/processed_data/scene007_obj001_salt/invrender_format/env_map/0073/envmap_world_invrender.exr',
                #     '/viscam/projects/imageint/yzzhang/data/processed_data/scene007_obj007_gnome/invrender_format/env_map/0063/envmap_world_invrender.exr',
                # ]:
                #     logger.error('envmap contains nan')
                model.novel_probes['custom'] = model._load_light(envmap)
                model.novel_probes_uint['custom'] = lightutil.vis_light(model.novel_probes['custom'], h=model.embed_light_h)

            _, _, _, to_vis = model.call(
                batch, mode=split, relight_olat=False, relight_probes=split == 'novel',
                albedo_scales=albedo_scales, albedo_override=None,
                brdf_z_override=brdf_z_override)
            # Visualize
            outdir = os.path.join(outroot, 'batch{i:09d}'.format(i=batch_i))
            model.vis_batch(to_vis, outdir, mode=split, olat_vis=False)

            _, _, _, _, _, _, xyz, _, _ = batch
            depth = xyz.numpy().reshape(512, 512, 3)
            with open(dataset.files[batch_i]) as f:
                c2w = np.array([
                    float(x) for x in json.load(f)['cam_transform_mat'].split(',')
                ]).reshape(4, 4)
            depth = np.einsum('ij,hwj->hwi', np.linalg.inv(c2w), np.concatenate([depth, np.ones((512, 512, 1))], axis=2))[:, :, 2]

            alpha = to_vis['gt_alpha'] > .8
            if np.any(alpha):
                depth_vis = ((depth - depth[alpha].min()) / (depth[alpha].max() - depth[alpha].min())).clip(0, 1)
                Image.fromarray((depth_vis * 255).astype(np.uint8)).save(os.path.join(outdir, 'gt_depth.png'))

            normal = to_vis['pred_normal']
            normal = np.einsum('ij,hwj->hwi', c2w[:3, :3].T, normal)
            Image.fromarray(((normal.clip(-1, 1) + 1) * .5 * 255).astype(np.uint8)).save(os.path.join(outdir, 'pred_normal_cam.png'))

            pyexr.write(os.path.join(outdir, 'gt_depth.exr'), depth)
            pyexr.write(os.path.join(outdir, 'pred_normal_cam.exr'), normal)

            if split == 'novel':
                del model.novel_probes['custom']
                del model.novel_probes_uint['custom']
            if DEBUG:
                break

        # Compile all visualized batches into a consolidated view (e.g., an
        # HTML or a video)
        batch_vis_dirs = xm.os.sortglob(outroot, 'batch?????????')
        outpref = outroot # proper extension should be added in the function below
        view_at = model.compile_batch_vis(batch_vis_dirs, outpref, mode=split)
        logger.info("Compilation available for viewing at\n\t%s", view_at)

    def test_geometry(self, scene: str, overwrite=False):
        exp_id = EXP_ID[scene]
        target_depth_paths = list(sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/z_maps/*.npy'))))
        target_normal_paths = list(sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/normal_maps/*.npy'))))
        if scene in ['scene002_obj008_grogu', 'scene003_obj010_pepsi', 'scene004_obj010_pepsi', "scene007_obj016_pitcher"]:
            # all nans, should use all zeros
            zeros_normal = np.zeros((512, 512, 3), dtype=np.float32)
            zeros_normal_path = '/viscam/projects/imageint/yzzhang/tmp/zeros_normal.exr'
            pyexr.write(zeros_normal_path, zeros_normal)
            ones_depth = np.ones((512, 512), dtype=np.float32)
            ones_depth_path = '/viscam/projects/imageint/yzzhang/tmp/ones_depth.exr'
            pyexr.write(ones_depth_path, ones_depth)

            return [{'output_depth': ones_depth_path, 'target_depth': target_depth_paths[ind],
                     'output_normal': zeros_normal_path, 'target_normal': target_normal_paths[ind]}
                    for ind in range(len(target_depth_paths))]
        test_out_dir = os.path.join(NERFACTOR_ROOT, f'output/train/{scene}_nerfactor/{exp_id}/lr5e-3/vis_vali/ckpt-10/batch*')
        normal_output_pattern = os.path.join(test_out_dir, 'pred_normal_cam.exr')
        depth_output_pattern = os.path.join(test_out_dir, 'gt_depth.exr')
        return [
            {'output_depth': f1, 'target_depth': f2,
             'output_normal': f3, 'target_normal': f4}
            for f1, f2, f3, f4 in zip(
                sorted(glob.glob(depth_output_pattern)),
                target_depth_paths,
                sorted(glob.glob(normal_output_pattern)),
                target_normal_paths,
            )
        ]

    def test_material(self, scene: str, overwrite=False):
        exp_id = EXP_ID[scene]
        target_paths = list(sorted(glob.glob(os.path.join('/viscam/projects/imageint/capture_scene_data/data/', scene, 'final_output/geometry_outputs/albedo_maps/*.png'))))
        if scene in ['scene002_obj008_grogu', 'scene003_obj010_pepsi', 'scene004_obj010_pepsi', "scene007_obj016_pitcher", "scene002_obj012_cart"]:
            # all nans, should use all zeros
            return [{'output_image': None, 'target_image': target_paths[ind]} for ind in range(len(target_paths))]

        output_pattern = os.path.join(NERFACTOR_ROOT, f'output/train/{scene}_nerfactor/{exp_id}/lr5e-3/vis_vali/ckpt-10/batch*/pred_albedo.png')  #TODO FIXME
        logger.info(f'found {len(glob.glob(output_pattern))} from {output_pattern}')
        output_paths = list(sorted(glob.glob(os.path.join(output_pattern))))
        if len(output_paths) != len(target_paths):  # FIXME
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_shape(self, scene: str, overwrite: bool):
        exp_id = SHAPE_EXP_ID[scene]
        output_mesh_pattern = os.path.join(
            NERFACTOR_ROOT, f"output/train/{scene}_nerf/{exp_id}/lr1e-4/vis_shape/ckpt-*/output_mesh_world.obj"
        )
        # target_mesh_path = f'/viscam/projects/imageint/capture_scene_data/data/{scene}/final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj'
        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, f'{scene}/final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        if not overwrite and len(glob.glob(output_mesh_pattern)) > 0:
            output_mesh_path, = glob.glob(output_mesh_pattern)
            return {
                'output_mesh': output_mesh_path,
                'target_mesh': target_mesh_path,
            }
        ckpts = glob.glob(os.path.join(
            NERFACTOR_ROOT, f"output/train/{scene}_nerf/{exp_id}/lr1e-4/checkpoints/ckpt-*.index"))
        ckpt_ind = [
            int(os.path.basename(x)[len('ckpt-'):-len('.index')]) for x in ckpts]
        latest_ckpt = ckpts[np.argmax(ckpt_ind)]
        latest_ckpt = latest_ckpt[:-len('.index')]

        config_ini = configutil.get_config_ini(latest_ckpt)
        config = ioutil.read_config(config_ini)
        outroot = os.path.join(config_ini[:-4], f'vis_shape', os.path.basename(latest_ckpt))
        os.makedirs(outroot, exist_ok=True)
        output_mesh_path = os.path.join(outroot, 'output_mesh_world.obj')
        if overwrite or not os.path.exists(output_mesh_path):
            model_name = config.get('DEFAULT', 'model')
            Model = models.get_model_class(model_name)
            model = Model(config)
            ioutil.restore_model(model, latest_ckpt)

            embedder = model.embedder['xyz']
            fine_enc = model.net['fine_enc']
            fine_sigma_out = model.net.get(
                'fine_a_out', model.net['fine_sigma_out'])

            def fn(pts_flat):
                return tf.nn.relu(
                    fine_sigma_out(fine_enc(embedder(pts_flat))))

            output_mesh = extract_mesh_from_nerf(fn)
            output_mesh.export(output_mesh_path.replace('.obj', '_raw.obj'))

            if output_mesh.vertices.shape[0] > 0:
               output_mesh = clean_mesh(output_mesh)
            output_mesh.export(output_mesh_path)

            DEBUG = True
            if DEBUG:
                # debug_save_dir = f'/viscam/projects/imageint/yzzhang/tmp/debug_mesh/{scene}'
                debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'nerfactor', scene)
                os.makedirs(debug_save_dir, exist_ok=True)
                output_mesh.export(os.path.join(debug_save_dir, 'world_nerfactor.obj'))

        return {
            'output_mesh': output_mesh_path,
            'target_mesh': target_mesh_path,
        }
