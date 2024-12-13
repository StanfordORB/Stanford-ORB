import numpy as np
from pathlib import Path
import json
import os
from .preprocess import load_mask_png, load_rgb_png, load_rgb_exr
import pyexr
from PIL import Image
from orb.constant import BENCHMARK_RESOLUTION, DEFAULT_SCENE_DATA_DIR, VERSION


if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


def process_neuralpil_geometry(scene, alpha_maps, normal_maps, depth_maps, out_dir):
    with open(os.path.join(SCENE_DATA_DIR, scene, "final_output/blender_format_LDR/transforms_test.json"), "r") as f:
        transforms = json.load(f)
    frames = transforms['frames']
    camera_angle_x = transforms['camera_angle_x']
    H = W = BENCHMARK_RESOLUTION
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    for ind in range(len(frames)):
        frame = frames[ind]

        c2w = np.array(frame['transform_matrix'])
        normal_map = normal_maps[ind]
        normal_map = load_rgb_exr(normal_map)
        normal_map = normal_map * 2 - 1
        normal_map = normal_map / np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map = np.einsum('ij,hwj->hwi', np.linalg.inv(c2w[:3, :3]), normal_map)
        pyexr.write(os.path.join(out_dir, Path(normal_maps[ind]).stem + '_processed.exr'), normal_map)

        z_map = depth_maps[ind]
        z_map = pyexr.open(z_map).get()  # (H, W, 1)
        rays_o, rays_d = get_full_image_eval_grid(H, W, focal, np.eye(4))
        depth_map = np.linalg.norm(rays_d * z_map, axis=-1)
        pyexr.write(os.path.join(out_dir, Path(depth_maps[ind]).stem + '_processed.exr'), depth_map)

        alpha_map = alpha_maps[ind]
        alpha_map = load_mask_png(alpha_map)
        if alpha_map.any():
            depth_map_vis = depth_map.copy()
            depth_map_vis -= np.quantile(depth_map_vis[alpha_map], .01)
            depth_map_vis /= np.quantile(depth_map_vis[alpha_map], .99)
            depth_map_vis = depth_map_vis.clip(0, 1)
        else:
            depth_map_vis = np.zeros_like(depth_map)
        img = Image.fromarray((depth_map_vis * 255).astype(np.uint8))
        img.save(os.path.join(out_dir, Path(depth_maps[ind]).stem + '_processed.png'))

        img = Image.fromarray(((normal_map + 1) * .5 * 255).astype(np.uint8))
        img.save(os.path.join(out_dir, Path(normal_maps[ind]).stem + '_processed.png'))


def get_full_image_eval_grid(H: int, W: int, focal: float, c2w):
    """Get ray origins, directions from a pinhole camera.

    Args:
        H (int): the height of the image.
        W (int): the width of the image.
        focal (float): the focal length of the camera.
        c2w (tf.Tensor [3, 3]): the camera matrix.

    Returns:
        rays_origin (tf.Tensor, [H, W, 3]): the rays origin.
        rays_direction (tf.Tensor, [H, W, 3]): the rays direction.
    """
    i, j = np.meshgrid(np.arange(W) + .5, np.arange(H) + .5)
    dirs = np.stack(
        [
            (i - float(W) * float(0.5)) / float(focal),
            -(j - float(H) * float(0.5)) / float(focal),
            -np.ones_like(i),
            ],
        -1,
    )
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = c2w[:3, -1]
    # rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))
    return rays_o, rays_d
