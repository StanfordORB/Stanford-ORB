import pyexr
import os
from orb.constant import PROJ_ROOT, VERSION, SUBMISSION_SCENES, REBUTTAL_SCENES, SUBMISSION_ADD_SCENES, DEFAULT_SCENE_DATA_DIR, NERF_ROOT, DEBUG_SAVE_DIR
import sys
sys.path.insert(0, NERF_ROOT)
import numpy as np
import torch
from PIL import Image
import glob
import os
import json
from orb.pipelines.base import BasePipeline
import logging
from orb.third_party.nerfpytorch.run_nerf import create_nerf, render
from orb.third_party.nerfpytorch.run_nerf_helpers import get_rays_np, get_rays
# from tu.configs import get_attrdict
from tu2.ppp import get_attrdict
from orb.datasets.nerf import load_capture_data
from orb.utils.extract_mesh import extract_mesh_from_nerf, clean_mesh
import trimesh

logger = logging.getLogger(__name__)


if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
if VERSION == 'submission':
    EXP_ID = "0605"
elif VERSION == 'rebuttal':
    EXP_ID = '0813'
elif VERSION == 'revision':
    EXP_ID = {s: '0605' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES})
elif VERSION == 'release':
    EXP_ID = {s: '0605' for s in SUBMISSION_SCENES}
    EXP_ID.update({s: '0813' for s in REBUTTAL_SCENES + SUBMISSION_ADD_SCENES})
elif VERSION == 'extension':
    EXP_ID = '1110'
else:
    raise NotImplementedError()


class Pipeline(BasePipeline):
    def test_new_view(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        output_pattern = os.path.join(NERF_ROOT, f'evals/test/{scene}/{exp_id}/renderonly_*/pred_rgb_*.png')
        if overwrite or len(glob.glob(output_pattern)) == 0:
            logger.info(f'Evaluating for {scene}, output to {output_pattern}')
            self.test_core_execute(scene)
        else:
            logger.info(f'Found {len(glob.glob(output_pattern))} for {output_pattern}')
        output_paths = list(sorted(glob.glob(os.path.join(output_pattern))))
        target_paths = list(sorted(glob.glob(os.path.join(SCENE_DATA_DIR, scene, f'final_output/blender_format_HDR/test/*.exr'))))
        if len(output_paths) != len(target_paths):  # FIXME
            logger.error(str([len(output_paths), len(target_paths), output_paths, target_paths]))
        return [{'output_image': output_paths[ind], 'target_image': target_paths[ind]} for ind in range(len(output_paths))]

    def test_core_execute(self, scene):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NERF_ROOT, 'logs', scene, exp_id)
        with open(os.path.join(exp_dir, 'args.json')) as f:
            args = get_attrdict(json.load(f))
        args.basedir = os.path.join(NERF_ROOT, args.basedir)
        images, poses, render_poses, hwf, i_split = load_capture_data(args.datadir, testskip=1)
        H, W, focal = hwf
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        _, rays_d = get_rays_np(H, W, K, np.eye(4))

        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        _, _, i_test = i_split

        _, render_kwargs_test, start, _, _ = create_nerf(args)
        testsavedir = os.path.join(NERF_ROOT, 'evals/test', scene, exp_id, f'renderonly_{start:06d}')
        os.makedirs(testsavedir, exist_ok=True)
        logger.info(f'Loading step {start}')
        render_kwargs_test.update(near=2., far=6.)

        for ind in range(len(i_test)):
            img_i = i_test[ind]
            # target = images[img_i]
            pose = poses[img_i, :3, :4]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, K, chunk=1024 * 8, c2w=torch.Tensor(pose), **render_kwargs_test)

            acc = acc.cpu().numpy()
            disp = disp.cpu().numpy()
            zvals = 1 / disp
            depth = np.linalg.norm(rays_d * zvals[:, :, None], axis=-1)
            if np.any(acc > .5):
                depth[np.isnan(depth)] = depth[acc > .5].min()
            else:
                depth = np.ones_like(depth)

            pyexr.write(os.path.join(testsavedir, f'depth_{img_i:06d}.exr'), depth)

            normal = extras['normal_map'].cpu().numpy()
            normal = np.einsum('ij,hwj->hwi', np.linalg.inv(pose[:3, :3]), normal)
            pyexr.write(os.path.join(testsavedir, f'normal_{img_i:06d}.exr'), normal)
            Image.fromarray(((normal.clip(-1, 1) * .5 + .5) * 255).astype(np.uint8)).save(os.path.join(testsavedir, f'normal_{img_i:06d}.png'))

            depth_vis = depth.copy()
            if np.any(acc > .5):
                depth_vis = (depth_vis - depth[acc > .5].min()) / (depth[acc > .5].max() - depth[acc > .5].min())
                depth_vis = np.clip(depth_vis, 0, 1)
                depth_vis[np.isnan(depth_vis)] = 0
            else:
                depth_vis = np.zeros_like(depth_vis)
            Image.fromarray((depth_vis * 255).astype(np.uint8)).save(os.path.join(testsavedir, f'depth_{img_i:06d}.png'))

            rgb = rgb.cpu().numpy()
            Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(testsavedir, f'pred_rgb_{img_i:06d}.png'))

    def test_geometry(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        normal_output_pattern = os.path.join(NERF_ROOT, f'evals/test/{scene}/{exp_id}/renderonly_*/normal_*.exr')
        depth_output_pattern = os.path.join(NERF_ROOT, f'evals/test/{scene}/{exp_id}/renderonly_*/depth_*.exr')
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

    def test_new_light(self, scene: str, overwrite=False):
        return []

    def test_material(self, scene: str, overwrite=False):
        return []

    def test_shape(self, scene: str, overwrite=False):
        exp_id = EXP_ID if isinstance(EXP_ID, str) else EXP_ID[scene]
        exp_dir = os.path.join(NERF_ROOT, 'logs', scene, exp_id)
        output_pattern = os.path.join(NERF_ROOT, 'evals/shape', scene, exp_id, f'*/output_mesh_world.obj')
        target_mesh_path = os.path.join(DEFAULT_SCENE_DATA_DIR, scene, 'final_output/geometry_outputs/pseudo_gt_mesh_spherify/mesh.obj')
        if not overwrite and len(glob.glob(output_pattern)) > 0:
            output_mesh_path, = glob.glob(output_pattern)
            return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}

        with open(os.path.join(exp_dir, 'args.json')) as f:
            args = get_attrdict(json.load(f))
        args.basedir = os.path.join(NERF_ROOT, args.basedir)
        _, render_kwargs_test, start, _, _ = create_nerf(args)
        render_kwargs_test.update(near=2., far=6.)
        net_fn = render_kwargs_test['network_query_fn']

        @torch.no_grad()
        def fn(pts_flat: np.ndarray):
            return net_fn(
                torch.from_numpy(pts_flat.reshape([-1, 1, 3])).cuda(),
                viewdirs=torch.zeros(pts_flat.shape, device='cuda', dtype=torch.float32),
                network_fn=render_kwargs_test['network_fine'])[..., -1:].cpu().numpy()

        testsavedir = os.path.join(NERF_ROOT, 'evals/shape', scene, exp_id, f'renderonly_{start:06d}')
        os.makedirs(testsavedir, exist_ok=True)
        output_mesh_path = os.path.join(testsavedir, 'output_mesh_world.obj')
        print(output_mesh_path)
        # mcubes.export_obj(vertices, triangles, mesh_path)
        if False:
            _, poses, _, (H, W, focal), i_split = load_capture_data(args.datadir, testskip=1)
            K = np.array([
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])

            _, _, i_test = i_split
            # with open(os.path.join(SCENE_DATA_DIR, scene, 'final_output/blender_format_LDR/transforms_test.json'), 'r') as fp:
            #     c2w = np.array(json.load(fp)['frames'][0]['transform_matrix'])
            # c2w = poses[i_test[0]]
            # w2c = np.linalg.inv(c2w)

            points = []
            for c2w in poses:
                rays_o, rays_d = get_rays_np(H, W, K, c2w)
                near, far = 2., 6.
                points_near = rays_o[..., None, :] + rays_d[..., None, :] * near  # [N_rays, N_samples, 3]
                points_far = rays_o[..., None, :] + rays_d[..., None, :] * far  # [N_rays, N_samples, 3]
                points.append(points_near)
                points.append(points_far)
            bounds = np.abs(np.stack(points)).reshape(-1, 3).max(0)
            print('bounds', bounds)

            output_mesh = extract_mesh_from_nerf(fn, bounds=bounds)
        else:
            output_mesh = extract_mesh_from_nerf(fn)
        output_mesh.export(output_mesh_path.replace('.obj', '_raw.obj'))

        if output_mesh.vertices.shape[0] > 0:
            output_mesh = clean_mesh(output_mesh)
        output_mesh.export(output_mesh_path)

        DEBUG = True
        if DEBUG:
            debug_save_dir = os.path.join(DEBUG_SAVE_DIR, 'nerf', scene)
            os.makedirs(debug_save_dir, exist_ok=True)
            output_mesh.export(os.path.join(debug_save_dir, 'world_nerf.obj'))

        # output_mesh.apply_transform(w2c)
        # output_mesh.export(os.path.join(testsavedir, 'output_mesh_camera.obj'))
        #
        # if DEBUG:
        #     output_mesh.export(os.path.join(debug_save_dir, 'camera_nerf.obj'))

        target_mesh = trimesh.load_mesh(target_mesh_path)
        target_mesh = trimesh.Trimesh(target_mesh.vertices, target_mesh.faces)
        target_mesh.export(os.path.join(testsavedir, 'target_mesh_world.obj'))
        if DEBUG:
            target_mesh.export(os.path.join(debug_save_dir, 'world_gt.obj'))
        # target_mesh.apply_transform(w2c)
        # target_mesh.export(os.path.join(testsavedir, 'target_mesh_camera.obj'))
        # if DEBUG:
        #     target_mesh.export(os.path.join(debug_save_dir, 'camera_gt.obj'))

        return {'output_mesh': output_mesh_path, 'target_mesh': target_mesh_path}
