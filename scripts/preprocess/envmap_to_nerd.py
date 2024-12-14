import sys
from orb.constant import NERD_ROOT
sys.path.insert(0, NERD_ROOT)
import argparse
from pathlib import Path
import glob
from orb.constant import PROCESSED_SCENE_DATA_DIR, DEFAULT_SCENE_DATA_DIR, VERSION
import os
import time
from typing import List, Tuple

import imageio
import numpy as np
import pyexr
import json
import tensorflow as tf
from orb.utils.env_map import env_map_to_cam_to_world_by_convention
from tqdm import tqdm

import orb.third_party.nerd.nn_utils.math_utils as math_utils
from orb.third_party.nerd.models.nerd_net.sgs_store import SgsStore
from orb.third_party.nerd.nn_utils.sg_rendering import SgRenderer
from skimage.transform import resize


if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
ENV_HEIGHT = 128


def optimize(
    sgs: tf.Tensor,
    directions: tf.Tensor,
    target: tf.Tensor,
    optimizer,
    sg_render: SgRenderer,
) -> Tuple[tf.Tensor]:
    """A single optimization step.

    Renders the color for each direction and compares it to the target.

    Returns:
        Loss and rendered envmap.
    """
    with tf.GradientTape() as tape:
        tape.watch(sgs)
        dir_flat = tf.reshape(directions, (-1, 1, 3))
        evaled_flat = sg_render._sg_evaluate(sgs, dir_flat)
        evaled_flat = tf.reduce_sum(evaled_flat, 1)
        evaled = tf.reshape(evaled_flat, target.shape)
        loss = tf.reduce_mean(
            tf.math.abs(tf.math.log(1 + target) - tf.math.log(1 + evaled))
        )
    grad_vars = (sgs,)
    gradients = tape.gradient(loss, grad_vars)
    optimizer.apply_gradients(zip(gradients, grad_vars))

    # optimization constraints
    ampl = tf.math.maximum(sgs[..., :3], 0.01)
    axis = math_utils.normalize(sgs[..., 3:6])
    sharpness = tf.math.maximum(sgs[..., 6:], 0.5)
    sgs.assign(tf.concat([ampl, axis, sharpness], -1))
    return loss, evaled


def fit_sgs(
        env_map: np.ndarray,
        directions: tf.Tensor,
        sg_render: SgRenderer,
        path: str,
        num_sgs: int = 24,
        steps: int = 2000,
) -> Tuple[List[np.ndarray]]:
    # env map to tensor
    env_map = tf.convert_to_tensor(env_map, dtype=tf.float32)

    start_time = time.time()
    sgs_axis_sharpness = SgsStore.setup_uniform_axis_sharpness(num_sgs)
    sgs_amplitude = np.ones_like(sgs_axis_sharpness[..., :3])
    sgs = np.concatenate([sgs_amplitude, sgs_axis_sharpness], -1)

    sgs_tf = tf.Variable(
        tf.convert_to_tensor(sgs[np.newaxis, ...], tf.float32),
        trainable=True,
        name="sgs_{}".format(num_sgs),
    )
    optimizer = tf.keras.optimizers.Adam(1e-2)
    bar = tqdm(range(steps), desc="lobes: {}".format(num_sgs))
    for _ in bar:
        loss, sg_envmap = optimize(sgs_tf, directions, env_map, optimizer, sg_render)
        bar.set_postfix({"loss": loss.numpy()})

    total_time = time.time() - start_time
    print("Fitting took", total_time)

    # write to disk
    sgs_path = os.path.join(path, "{0:03d}_sg.npy".format(num_sgs))
    env_path = os.path.join(path, "{0:03d}_sg_env.exr".format(num_sgs))
    np.save(sgs_path, sgs_tf.numpy())
    pyexr.write(env_path, sg_envmap.numpy())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, required=True)
    parser.add_argument('-o', '--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = os.path.join(SCENE_DATA_DIR, args.scene, 'final_output/llff_format_HDR/env_map/')
    output_dir = os.path.join(PROCESSED_SCENE_DATA_DIR, args.scene, 'nerd_format/env_map/')

    with open(os.path.join(SCENE_DATA_DIR, args.scene, 'final_output/blender_format_HDR/transforms_test.json')) as f:
        test_frames = json.load(f)['frames']

    for frame in test_frames:
        test_name = os.path.basename(frame['file_path']) + ".exr"
        input_path = os.path.join(input_dir, test_name)
        output_subdir = os.path.join(output_dir, Path(input_path).stem)
        os.makedirs(output_subdir, exist_ok=args.overwrite)

        # env_map = hdrio.imread(input_path)
        env_map = pyexr.open(input_path).get()[..., :3]
        env_map = resize(env_map, (ENV_HEIGHT, ENV_HEIGHT * 2))

        pyexr.write(os.path.join(output_subdir, 'envmap_cam.exr'), env_map)

        c2w = np.array(frame['transform_matrix'])
        env_map = env_map_to_cam_to_world_by_convention(env_map, c2w, convention='nerd')
        pyexr.write(os.path.join(output_subdir, 'envmap_world_nerd.exr'), env_map)

        directions = math_utils.uv_to_direction(
            math_utils.shape_to_uv(env_map.shape[0], env_map.shape[1])
        )
        sg_render = SgRenderer()

        fit_sgs(env_map=env_map, directions=directions, sg_render=sg_render, path=output_subdir)


if __name__ == "__main__":
    main()
