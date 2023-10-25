import os
import glob
import cv2
import imageio
import tensorflow as tf
import json
import numpy as np
from orb.constant import BENCHMARK_RESOLUTION
from orb.utils.preprocess import load_mask_png, load_rgb_png


def trans_t(t):
    return tf.convert_to_tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=tf.float32,
    )


def rot_phi(phi):
    return tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, tf.cos(phi), -tf.sin(phi), 0],
            [0, tf.sin(phi), tf.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def rot_theta(th):
    return tf.convert_to_tensor(
        [
            [tf.cos(th), 0, -tf.sin(th), 0],
            [0, 1, 0, 0],
            [tf.sin(th), 0, tf.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir, factor=None, width=None, height=None, trainskip=1, testskip=1, valskip=1):
    scene = os.path.basename(basedir)
    basedir = os.path.join(basedir, 'final_output/blender_format_LDR')
    splits = ["train", "val", "test"]
    splits = ['train', 'test']
    if os.getenv('NERD_GT_ENV_MAP') == '1' and scene != os.environ['NERD_LIGHT_SCENE']:
        splits[1] = 'novel'
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, "transforms_{}.json".format(s)), "r") as fp:
            metas[s] = json.load(fp)
    if 'novel' in splits:
        metas = {'train': metas['train'], 'test': metas['novel']}
    else:
        metas = {'train': metas['train'], 'test': metas['test']}

    metas['val'] = metas['test']
    splits = ['train', 'val', 'test']
    img0 = next(iter(glob.glob(os.path.join(basedir, "train", "*.png"))))
    sh = imageio.imread(img0).shape
    assert factor is not None and sh[0] / factor == BENCHMARK_RESOLUTION and sh[1] / factor == BENCHMARK_RESOLUTION

    all_imgs = []
    all_masks = []
    all_poses = []
    all_ev100 = []
    counts = [0]
    meta = None
    for s in splits:
        meta = metas[s]
        imgs = []
        masks = []
        poses = []
        if s == "train":
            skip = max(trainskip, 1)
        elif s == "val":
            skip = max(valskip, 1)
        else:
            skip = max(testskip, 1)

        for frame in meta["frames"][::skip]:
            if 'scene_name' in frame:
                if frame['scene_name'] != os.environ['NERD_LIGHT_SCENE']:
                    continue
                frame_basedir = basedir.replace(scene, frame['scene_name'])
                img_file = load_rgb_png(os.path.join(frame_basedir, frame["file_path"] + ".png"), downsize_factor=factor)
                mask_file = load_mask_png(os.path.join(frame_basedir, os.path.dirname(frame['file_path']) + '_mask', os.path.basename(frame['file_path']) + '.png'), downsize_factor=factor)
            else:
                img_file = load_rgb_png(os.path.join(basedir, frame["file_path"] + ".png"), downsize_factor=factor)
                mask_file = load_mask_png(os.path.join(basedir, os.path.dirname(frame['file_path']) + '_mask', os.path.basename(frame['file_path']) + '.png'), downsize_factor=factor)
            imgs.append(img_file)
            masks.append(mask_file[:, :, None])

            # Read the poses
            poses.append(np.array(frame["transform_matrix"]))

            all_ev100.append(8)
        imgs = np.array(imgs).astype(np.float32)
        # Continue with the masks.
        # They only require values to be between 0 and 1
        # Clip to be sure
        masks = np.clip(np.array(masks).astype(np.float32), 0, 1)

        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_masks.append(masks)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0).astype(np.float32)
    masks = np.concatenate(all_masks, 0).astype(np.float32)
    poses = np.concatenate(all_poses, 0)
    ev100s = np.stack(all_ev100, 0).astype(np.float32)

    H, W = imgs[0].shape[:2]
    if 'camera_angle_x' not in metas['test']:
        # hack
        for frame in metas['test']['frames']:
            if frame['scene_name'] == os.environ['NERD_LIGHT_SCENE']:
                camera_angle_x = float(frame['camera_angle_x'])
                break
    else:
        camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if os.getenv('NERD_GT_ENV_MAP_DEBUG') == '1':
        print('old focal', focal)
        focal = focal * .1
        print('new focal', focal)

    render_poses = tf.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    return imgs, masks, poses, ev100s, render_poses, [H, W, focal], i_split
