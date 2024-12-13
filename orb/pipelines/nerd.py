import sys
from orb.constant import PROJ_ROOT, PROCESSED_SCENE_DATA_DIR, BENCHMARK_RESOLUTION, VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, NERD_ROOT, DEFAULT_SCENE_DATA_DIR, DEBUG_SAVE_DIR
sys.path.insert(0, NERD_ROOT)
import json
import glob
import os
import imageio
import pyexr
from orb.utils.extract_mesh import extract_mesh_from_nerf, clean_mesh
import numpy as np
# from tu.configs import get_attrdict
from tu2.ppp import get_attrdict
from orb.utils.postprocess import process_neuralpil_geometry
from PIL import Image
from typing import Dict, List
from orb.pipelines.base import BasePipeline
from orb.utils.load_data import get_novel_scenes
import logging
try:
    import tensorflow as tf
    import orb.third_party.nerd.dataflow.nerd as data
    import orb.third_party.nerd.nn_utils.math_utils as math_utils
    from orb.third_party.nerd.models.nerd_net import NerdModel
    from orb.third_party.nerd.train_nerd import eval_datasets
    from orb.third_party.nerd.nn_utils.tensorboard_visualization import to_8b
except:
    pass


logger = logging.getLogger(__name__)

NERD_ROOT = os.path.join(PROJ_ROOT, 'imageint/third_party/nerd')
# EXP_ID = '0525'
if VERSION == 'submission':
    EXP_ID = '0530'
elif VERSION == 'rebuttal':
    EXP_ID = '0813'
elif VERSION == 'revision':
    EXP_ID = {s: '0530' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES})
elif VERSION == 'release':
    EXP_ID = {s: '0530' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES + SUBMISSION_ADD_SCENES})
elif VERSION == 'extension':
    EXP_ID = '1110'
else:
    raise NotImplementedError()

if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    if EXP_ID == '0525':
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data'
    else:
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


class Pipeline(BasePipeline):
    def test_shape(self, scene: str, overwrite: bool):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NERD_ROOT, f'logs/{scene}/{exp_id}')
        with open(os.path.join(exp_dir, 'args.json')) as f:
            args = get_attrdict(json.load(f))
        output_pattern = os.path.join(NERD_ROOT, 'evals', 'shape', args.expname, '*/output_mesh_world.obj')
        target_mesh_path = os.path.join(SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        if not overwrite and len(glob.glob(output_pattern)) > 0:
            output_mesh_path, = glob.glob(output_pattern)
            return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

        args.mean_sgs_path = os.path.join(NERD_ROOT, args.mean_sgs_path)
        args.basedir = os.path.join(NERD_ROOT, args.basedir)
        # changing skips will mess up img_idx but since we use single_env lighting, it doesn't matter
        args.testskip = 1
        args.trainskip = 20  # FIXME??
        args.steps_per_epoch = 0  # hack to enforce repeats = 1 in dataloader
        batch_size = args.batch_size
        args.batch_size = 0
        args.jitter_coords = None

        strategy = tf.distribute.get_strategy()
        (
            hwf,
            near,
            far,
            render_poses,
            num_images,
            _,
            train_df,
            val_df,
            test_df,
        ) = data.create_dataflow(args)
        args.batch_size = batch_size  # hack
        with strategy.scope():
            nerd = NerdModel(num_images, args)
        start_step = nerd.restore()  # load the latest checkpoint available?
        testimgdir = os.path.join(
            NERD_ROOT, 'evals', 'shape',
            args.expname,
            "test_imgs_{:06d}".format(start_step),
        )
        os.makedirs(testimgdir, exist_ok=True)

        def fn(pts_flat):
            pts_embed = nerd.fine_model.pos_embedder(pts_flat)
            main_embd = nerd.fine_model.main_net_first(pts_embed)
            main_embd = nerd.fine_model.main_net_second(tf.concat([main_embd, pts_embed], -1))
            sigma_payload = nerd.fine_model.main_final_net(main_embd)
            sigma = sigma_payload[..., :1]
            return sigma

        output_mesh = extract_mesh_from_nerf(fn)
        output_mesh_path = os.path.join(testimgdir, 'output_mesh_world.obj')
        output_mesh.export(output_mesh_path.replace('.obj', '_raw.obj'))
        if output_mesh.vertices.shape[0] > 0:
            output_mesh = clean_mesh(output_mesh)
        output_mesh.export(output_mesh_path)

        DEBUG = True
        if DEBUG:
            debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'nerd', scene)
            os.makedirs(debug_save_dir, exist_ok=True)
            output_mesh.export(os.path.join(debug_save_dir, 'world_nerd.obj'))
        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

    def test_inverse_rendering(self, scene: str, overwrite=False) -> Dict:
        return self.test_core(scene, split='train', overwrite=overwrite)  # TODO
        ret = {'train_normal': [], 'train_depth': [], 'train_kd': [], 'train_ks': [], 'shape': None}
        return ret

    def test_new_light(self, scene: str, overwrite: bool) -> List[Dict[str, str]]:
        os.environ['NERD_GT_ENV_MAP'] = '1'
        if VERSION == 'extension':
            novel_scenes = get_novel_scenes(scene, SCENE_DATA_DIR)
        else:
            scene1, scene2 = [scene for scene in os.listdir(os.path.join(SCENE_DATA_DIR, scene, 'final_output/llff_format_HDR/sparse')) if scene != '0']
            novel_scenes = [scene1, scene2]
        ret = []
        for test_scene in [scene, *novel_scenes]:
            os.environ['NERD_LIGHT_SCENE'] = test_scene
            ret += self.test_core(scene, test_scene, overwrite)
            os.environ['NERD_LIGHT_SCENE'] = ''
        os.environ['NERD_GT_ENV_MAP'] = '0'
        return ret

    def test_new_view(self, scene: str, overwrite=False):
        return self.test_core(scene, split='test', overwrite=overwrite)

    def test_core(self, scene, split, overwrite):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        output_pattern = os.path.join(NERD_ROOT, 'evals', split, scene, exp_id, 'test_imgs_*/*_fine_hdr_rgb.exr')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            self.test_core_execute(scene, split=split)
        if len(os.listdir(os.path.join(NERD_ROOT, 'evals', split, scene, exp_id))) != 1:
            import ipdb; ipdb.set_trace()
        output_paths = list(sorted(glob.glob(os.path.join(output_pattern))))
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
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_core_execute(self, scene, split):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NERD_ROOT, f'logs/{scene}/{exp_id}')
        with open(os.path.join(exp_dir, 'args.json')) as f:
            args = get_attrdict(json.load(f))

        args.mean_sgs_path = os.path.join(NERD_ROOT, args.mean_sgs_path)
        args.basedir = os.path.join(NERD_ROOT, args.basedir)
        # changing skips will mess up img_idx but since we use single_env lighting, it doesn't matter
        args.testskip = 1
        args.trainskip = 20  # FIXME??
        args.steps_per_epoch = 0  # hack to enforce repeats = 1 in dataloader
        batch_size = args.batch_size
        args.batch_size = 0
        args.jitter_coords = None

        strategy = tf.distribute.get_strategy()
        (
            hwf,
            near,
            far,
            render_poses,
            num_images,
            _,
            train_df,
            val_df,
            test_df,
        ) = data.create_dataflow(args)
        args.batch_size = batch_size  # hack
        with strategy.scope():
            nerd = NerdModel(num_images, args)
        start_step = nerd.restore()  # load the latest checkpoint available?

        logger.info(f'starting step: {start_step}')
        ret, fine_ssim, fine_psnr = eval_datasets(
            strategy,
            {'train': train_df, 'test': test_df}.get(split, test_df),
            nerd,
            hwf,
            near,
            far,
            None,
            100,
            args.batch_size,
            args.single_env,
        )

        testimgdir = os.path.join(
            NERD_ROOT, 'evals', split,
            args.expname,
            "test_imgs_{:06d}".format(start_step),
        )
        print("Mean PSNR:", fine_psnr, "Mean SSIM:", fine_ssim)
        os.makedirs(testimgdir, exist_ok=True)
        alpha = ret["fine_acc_alpha"]
        for n, t in ret.items():
            print(n, t.shape)
            for b in range(t.shape[0]):
                to_save = t[b]
                if "normal" in n:
                    to_save = (t[b] * 0.5 + 0.5) * alpha[b] + (1 - alpha[b])

                if "env_map" in n:
                    imageio.imwrite(
                        os.path.join(testimgdir, "{:d}_{}.png".format(b, n)),
                        to_8b(
                            math_utils.linear_to_srgb(to_save / (1 + to_save))
                        )
                        .numpy()
                        .astype(np.uint8),
                        )
                    pyexr.write(
                        os.path.join(testimgdir, "{:d}_{}.exr".format(b, n)),
                        to_save.numpy(),
                    )
                elif "normal" in n or "depth" in n:
                    pyexr.write(
                        os.path.join(testimgdir, "{:d}_{}.exr".format(b, n)),
                        to_save.numpy(),
                    )
                    if "normal" in n:
                        imageio.imwrite(
                            os.path.join(
                                testimgdir, "{:d}_{}.png".format(b, n)
                            ),
                            to_8b(to_save).numpy(),
                        )
                elif 'hdr_rgb' in n or 'basecolor' in n:
                    hdr_rgb = to_save.numpy()
                    if np.isnan(hdr_rgb).any():
                        logger.info(f'NaN in {n}')
                        hdr_rgb[np.isnan(hdr_rgb)] = 0
                    pyexr.write(
                        os.path.join(testimgdir, "{:d}_{}.exr".format(b, n)),
                        to_save.numpy(),
                    )
                    # still use training time ev100  # FIXME
                    exp_val = 1 / (1.2 * 2 ** 8)
                    ldr_rgb = math_utils.linear_to_srgb(math_utils.saturate(to_save * exp_val)) * alpha[b] + (1 - alpha[b])
                    Image.fromarray((ldr_rgb.numpy().clip(0, 1) * 255).astype(np.uint8)).save(
                        os.path.join(testimgdir, "{:d}_{}.png".format(b, n)),
                    )
                    # {:d}_fine_rgb.png is not multiplied with exposure
                else:
                    Image.fromarray(to_8b(to_save).numpy().astype(np.uint8).squeeze()).save(
                        os.path.join(testimgdir, "{:d}_{}.png".format(b, n)),
                    )

    def test_geometry(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir, = glob.glob(os.path.join(NERD_ROOT, 'evals/test', scene, exp_id, 'test_imgs_*'))
        normal_output_pattern = os.path.join(test_out_dir, '*_fine_normal_processed.exr')
        depth_output_pattern = os.path.join(test_out_dir, '*_fine_depth_processed.exr')
        if overwrite or (len(glob.glob(normal_output_pattern)) == 0 and len(glob.glob(depth_output_pattern)) == 0):
            logger.info(f'Processing {scene}, Output {test_out_dir}')
            self.test_geometry_core(scene)
        else:
            logger.info(f'Found processed {scene}, Output {test_out_dir}')
        return [
            {'output_depth': f1, 'target_depth': f2,
             'output_normal': f3, 'target_normal': f4}
            for f1, f2, f3, f4 in zip(
                sorted(glob.glob(depth_output_pattern)),
                sorted(glob.glob(os.path.join(SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/z_maps/*.npy'))),
                sorted(glob.glob(normal_output_pattern)),
                sorted(glob.glob(os.path.join(SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/normal_maps/*.npy'))),
            )
        ]

    def test_geometry_core(self, scene: str):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir, = glob.glob(os.path.join(NERD_ROOT, 'evals/test', scene, exp_id, 'test_imgs_*'))
        alpha_maps = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_acc_alpha.png'))))
        normal_maps = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_normal.exr'))))
        depth_maps = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_depth.exr'))))
        process_neuralpil_geometry(scene, alpha_maps, normal_maps, depth_maps, test_out_dir)

    def test_material(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir, = glob.glob(os.path.join(NERD_ROOT, 'evals/test', scene, exp_id, 'test_imgs_*'))
        output_paths = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_basecolor.exr'))))
        target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/albedo_maps/*.png'))))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
