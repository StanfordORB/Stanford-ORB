import sys
import os
from orb.constant import VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, NEURALPIL_ROOT, DEFAULT_SCENE_DATA_DIR, DEBUG_SAVE_DIR
sys.path.insert(0, NEURALPIL_ROOT)
import json
import glob
import os
import imageio
import pyexr
import numpy as np
# from tu.configs import get_attrdict
from tu2.ppp import get_attrdict
from PIL import Image
from orb.pipelines.base import BasePipeline
from orb.utils.postprocess import process_neuralpil_geometry
from orb.utils.extract_mesh import extract_mesh_from_nerf, clean_mesh
import logging
try:
    import tensorflow as tf
    import orb.third_party.neuralpil.dataflow.nerd as data
    import orb.third_party.neuralpil.nn_utils.math_utils as math_utils
    from orb.third_party.neuralpil.models.neural_pil import NeuralPILModel
    from orb.third_party.neuralpil.train_neural_pil import eval_datasets
    from orb.third_party.neuralpil.nn_utils.tensorboard_visualization import to_8b
except:
    pass


logger = logging.getLogger(__name__)

# EXP_ID = '0523_after_rsync'
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
    raise NotImplementedError(VERSION)

if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    if EXP_ID == '0523_after_rsync':
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data'
    else:
        SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


class Pipeline(BasePipeline):
    def test_new_light(self, scene: str, overwrite: bool):
        return []

    def test_new_view(self, scene: str, overwrite=False):
        return self.test_core(scene, split='test', overwrite=overwrite)

    def test_core(self, scene, split, overwrite):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        output_pattern = os.path.join(NEURALPIL_ROOT, 'evals', split, scene, exp_id, 'test_imgs_*/*_fine_hdr_rgb.exr')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            self.test_core_execute(scene, split=split)
        # if len(os.listdir(os.path.join(NEURALPIL_ROOT, 'evals', split, scene, EXP_ID))) != 1:
        #     import ipdb; ipdb.set_trace()
        logger.info(f'output_pattern: {output_pattern}')
        output_paths = list(sorted(glob.glob(os.path.join(output_pattern))))
        target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/{split}/*.exr'))))
        if len(output_paths) != len(target_paths):  # FIXME
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_shape(self, scene: str, overwrite: bool):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NEURALPIL_ROOT, f'logs/{scene}/{exp_id}')
        logger.info(f'Evaluating for /{exp_dir}')
        with open(os.path.join(exp_dir, 'args.json')) as f:
            args = get_attrdict(json.load(f))
        output_pattern = os.path.join(NEURALPIL_ROOT, 'evals', 'shape', args.expname, '*/output_mesh_world.obj')
        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        if not overwrite and len(glob.glob(output_pattern)) > 0:
            output_mesh_path, = glob.glob(output_pattern)
            return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

        args.brdf_network_path = os.path.join(NEURALPIL_ROOT, args.brdf_network_path)
        args.brdf_preintegration_path = os.path.join(NEURALPIL_ROOT, args.brdf_preintegration_path)
        args.illumination_network_path = os.path.join(NEURALPIL_ROOT, args.illumination_network_path)
        args.basedir = os.path.join(NEURALPIL_ROOT, args.basedir)
        args.testskip = 1
        assert args.single_env
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            neuralpil = NeuralPILModel(num_images=None, args=args)
        start_step = neuralpil.restore()  # load the latest checkpoint available?
        testimgdir = os.path.join(
            # args.basedir,
            NEURALPIL_ROOT, 'evals', 'shape',
            args.expname,
            "test_imgs_{:06d}".format(start_step),
        )
        os.makedirs(testimgdir, exist_ok=True)

        def fn(pts_flat):
            pts_embed = neuralpil.fine_model.pos_embedder(pts_flat)
            main_embd = neuralpil.fine_model.main_net_first(pts_embed)
            main_embd = neuralpil.fine_model.main_net_second(tf.concat([main_embd, pts_embed], -1))
            sigma_payload = neuralpil.fine_model.main_final_net(main_embd)
            sigma = sigma_payload[..., :1]
            return sigma

        output_mesh_path = os.path.join(testimgdir, 'output_mesh_world.obj')
        output_mesh = extract_mesh_from_nerf(fn)
        output_mesh.export(output_mesh_path.replace('.obj', '_raw.obj'))

        if output_mesh.vertices.shape[0] > 0:
            output_mesh = clean_mesh(output_mesh)
        output_mesh.export(output_mesh_path)

        DEBUG = True
        if DEBUG:
            debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'neuralpil', scene)
            os.makedirs(debug_save_dir, exist_ok=True)
            output_mesh.export(os.path.join(debug_save_dir, 'world_neuralpil.obj'))
        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

    def test_core_execute(self, scene, split):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NEURALPIL_ROOT, f'logs/{scene}/{exp_id}')
        logger.info(f'Evaluating for /{exp_dir}')
        with open(os.path.join(exp_dir, 'args.json')) as f:
            args = get_attrdict(json.load(f))
        args.brdf_network_path = os.path.join(NEURALPIL_ROOT, args.brdf_network_path)
        args.brdf_preintegration_path = os.path.join(NEURALPIL_ROOT, args.brdf_preintegration_path)
        args.illumination_network_path = os.path.join(NEURALPIL_ROOT, args.illumination_network_path)
        args.basedir = os.path.join(NEURALPIL_ROOT, args.basedir)
        args.testskip = 1
        strategy = tf.distribute.get_strategy()
        (
            hwf,
            near,
            far,
            render_poses,
            num_images,
            mean_ev100,
            train_df,
            val_df,
            test_df,
        ) = data.create_dataflow(args)
        with strategy.scope():
            neuralpil = NeuralPILModel(num_images, args)
        neuralpil.call(
            ray_origins=tf.zeros((1, 3), tf.float32),
            ray_directions=tf.zeros((1, 3), tf.float32),
            camera_pose=tf.eye(3, dtype=tf.float32)[None, ...],
            near_bound=near,
            far_bound=far,
            illumination_idx=tf.zeros((1,), tf.int32),
            ev100=tf.zeros((1,), tf.float32),
            illumination_factor=tf.zeros((1,), tf.float32),
            training=False,
        )
        start_step = neuralpil.restore()  # load the latest checkpoint available?
        logger.info(f'starting step: {start_step}')
        illumination_factor = tf.stop_gradient(
            neuralpil.calculate_illumination_factor(
                tf.convert_to_tensor([[0, 1, 0]], tf.float32), mean_ev100
            )
        )
        assert args.single_env
        ret, fine_ssim, fine_psnr = eval_datasets(
            strategy,
            {'train': train_df, 'test': test_df}[split],
            neuralpil,
            hwf,
            near,
            far,
            None,
            100,
            args.batch_size,
            args.single_env,
            illumination_factor,
        )

        testimgdir = os.path.join(
            # args.basedir,
            NEURALPIL_ROOT, 'evals', split,
            args.expname,
            "test_imgs_{:06d}".format(start_step),
        )
        print("Mean PSNR:", fine_psnr, "Mean SSIM:", fine_ssim)
        os.makedirs(testimgdir, exist_ok=True)
        # Save all images in the test_dir
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
                elif 'hdr_rgb' in n:
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
        test_out_dir, = glob.glob(os.path.join(NEURALPIL_ROOT, 'evals/test', scene, exp_id, 'test_imgs_*'))
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
                sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/z_maps/*.npy'))),
                sorted(glob.glob(normal_output_pattern)),
                sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/normal_maps/*.npy'))),
            )
        ]

    def test_geometry_core(self, scene: str):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir, = glob.glob(os.path.join(NEURALPIL_ROOT, 'evals/test', scene, exp_id, 'test_imgs_*'))
        alpha_maps = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_acc_alpha.png'))))
        normal_maps = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_normal.exr'))))
        depth_maps = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_depth.exr'))))
        process_neuralpil_geometry(scene, alpha_maps, normal_maps, depth_maps, test_out_dir)

    def test_material(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        test_out_dir, = glob.glob(os.path.join(NEURALPIL_ROOT, 'evals/test', scene, exp_id, 'test_imgs_*'))
        output_paths = list(sorted(glob.glob(os.path.join(test_out_dir, '*_fine_diffuse.png'))))
        target_paths = sorted(glob.glob(os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/albedo_maps/*.png')))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]
