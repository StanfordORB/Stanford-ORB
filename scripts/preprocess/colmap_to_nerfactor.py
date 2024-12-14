import glob
from PIL import Image
import json
import argparse
from typing import List
import imageio
import os
import numpy as np
from tqdm import tqdm
from orb.utils.preprocess import load_rgb_png, load_mask_png
from orb.constant import BENCHMARK_RESOLUTION, PROCESSED_SCENE_DATA_DIR, INPUT_RESOLUTION
from orb.third_party.nerfactor.data_gen.util import gen_data


BOUND_FACTOR = .75


def buggy_main(input_dir: str, output_dir: str):
    poses_path = os.path.join(input_dir, 'poses_bounds.npy')
    poses_arr = np.load(poses_path)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    assert poses.shape == (3, 5, 70), poses.shape
    bds = poses_arr[:, -2:].transpose([1, 0])

    # Load and resize images
    filenames = list(sorted(os.listdir(os.path.join(input_dir, 'images'))))
    assert len(filenames) == 70, len(filenames)
    imgs = []
    downsize_factor = Image.open(os.path.join(input_dir, 'images', filenames[0])).size[0] / BENCHMARK_RESOLUTION
    img_paths = []
    for filename in tqdm(filenames, desc="Loading images"):
        img_path = os.path.join(input_dir, 'images', filename)
        img = load_rgb_png(img_path, downsize_factor=downsize_factor)
        mask = load_mask_png(os.path.join(input_dir, 'masks', filename), downsize_factor=downsize_factor)
        img = np.concatenate([img, mask[:, :, None]], axis=2)
        imgs.append(img)  # 512, 512, 4
        img_paths.append(img_path)
    imgs = np.stack(imgs, axis=-1)  # 512, 512, 4, 70
    assert imgs.shape == (512, 512, 4, 70), imgs.shape

    # Sanity check
    n_poses = poses.shape[-1]
    n_imgs = imgs.shape[-1]
    assert n_poses == n_imgs, (
        "Mismatch between numbers of images ({n_imgs}) and "
        "poses ({n_poses})").format(n_imgs=n_imgs, n_poses=n_poses)
    # Update poses according to downsampling
    poses[:2, 4, :] = np.array(
        imgs.shape[:2]).reshape([2, 1]) # override image size
    poses[2, 4, :] = poses[2, 4, :] * 1. / downsize_factor # scale focal length

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # Nx3x5
    imgs = np.moveaxis(imgs, -1, 0) # NxHxWx4
    bds = np.moveaxis(bds, -1, 0).astype(np.float32) # Nx2

    # Rescale according to a default bd factor
    scale = 1. / (bds.min() * BOUND_FACTOR)
    poses[:, :3, 3] *= scale # scale translation
    bds *= scale
    #
    # # FIXME DEBUG
    # with open(os.path.join(input_dir, '../blender_format_LDR/transforms_train.json')) as f:
    #     transforms_train = json.load(f)
    # with open(os.path.join(input_dir, '../blender_format_LDR/transforms_test.json')) as f:
    #     transforms_test = json.load(f)

    with open(os.path.join(input_dir, 'test_id.txt'), 'r') as f:
        test_ids: List[str] = f.read().splitlines()
    test_indices = [filenames.index(test_id) for test_id in test_ids]
    gen_data(poses, imgs, img_paths, test_indices, output_dir)


def gen_data_from_blender_format(imgs: np.ndarray, img_paths: List[str], ind_train: List[int], ind_vali: List[int], outroot: str):
    with open(os.path.join(os.path.dirname(img_paths[0]), '../../blender_format_LDR/transforms_train.json')) as f:
        blender_transforms_train = json.load(f)
    with open(os.path.join(os.path.dirname(img_paths[0]), '../../blender_format_LDR/transforms_test.json')) as f:
        blender_transforms_test = json.load(f)

    view_folder = '{mode}_{i:03d}'
    # Only the original NeRF and JaxNeRF implementations need these
    train_json = os.path.join(outroot, 'transforms_train.json')
    vali_json = os.path.join(outroot, 'transforms_val.json')
    test_json = os.path.join(outroot, 'transforms_test.json')
    # Training-validation split
    # n_imgs = imgs.shape[0]
    # ind_train = np.array([x for x in np.arange(n_imgs) if x not in ind_vali])
    cam_angle_x = blender_transforms_train['camera_angle_x']

    # Training frames
    train_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_train):
        view_folder_ = view_folder.format(mode='train', i=vi)
        os.makedirs(os.path.join(outroot, view_folder_), exist_ok=True)
        # Write image
        img = imgs[i, :, :, :]

        Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(os.path.join(outroot, view_folder_, 'rgba.png'))

        frame_meta_blender = blender_transforms_train['frames'][vi]
        if os.path.basename(frame_meta_blender['file_path'] + '.png') != os.path.basename(img_paths[i]):
            print(frame_meta_blender['file_path'] + '.png', img_paths[i])
            import ipdb; ipdb.set_trace()
        c2w = np.array(frame_meta_blender['transform_matrix'])

        frame_meta = {
            'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
            'transform_matrix': c2w.tolist()}
        train_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        with open(os.path.join(outroot, view_folder_, 'metadata.json'), 'w') as f:
            json.dump(frame_meta, f, indent=4)

    # Validation views
    vali_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    for vi, i in enumerate(ind_vali):
        view_folder_ = view_folder.format(mode='val', i=vi)
        os.makedirs(os.path.join(outroot, view_folder_), exist_ok=True)
        # Write image
        img = imgs[i, :, :, :]
        Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(os.path.join(outroot, view_folder_, 'rgba.png'))

        frame_meta_blender = blender_transforms_test['frames'][vi]
        if os.path.basename(frame_meta_blender['file_path'] + '.png') != os.path.basename(img_paths[i]):
            print(frame_meta_blender['file_path'] + '.png', img_paths[i])
            import ipdb; ipdb.set_trace()
        c2w = np.array(frame_meta_blender['transform_matrix'])

        frame_meta = {
            'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
            'transform_matrix': c2w.tolist()}
        vali_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': cam_angle_x,
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        with open(os.path.join(outroot, view_folder_, 'metadata.json'), 'w') as f:
            json.dump(frame_meta, f, indent=4)

    # Write training and validation JSONs
    with open(train_json, 'w') as f:
        json.dump(train_meta, f, indent=4)
    with open(vali_json, 'w') as f:
        json.dump(vali_meta, f, indent=4)

    # # Test views
    # test_meta = {'camera_angle_x': cam_angle_x, 'frames': []}
    # for i in range(test_poses.shape[0]):
    #     view_folder_ = view_folder.format(mode='test', i=i)
    #     # Record metadata
    #     pose = test_poses[i, :, :]
    #     c2w = np.vstack((pose[:3, :4], np.array([0, 0, 0, 1]).reshape(1, 4)))
    #     frame_meta = {
    #         'file_path': '', 'rotation': 0, 'transform_matrix': c2w.tolist()}
    #     test_meta['frames'].append(frame_meta)
    #     # Write the nearest input to this test view folder
    #     dist = np.linalg.norm(pose[:, 3] - poses[:, :, 3], axis=1)
    #     nn_i = np.argmin(dist)
    #     nn_img = imgs[nn_i, :, :, :]
    #     xm.io.img.write_float(
    #         nn_img, join(outroot, view_folder_, 'nn.png'), clip=True)
    #     # Write this frame's metadata to the view folder
    #     frame_meta = {
    #         'cam_angle_x': cam_angle_x,
    #         'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
    #         'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
    #         'imw': img.shape[1], 'scene': '', 'spp': 0, 'original_path': ''}
    #     xm.io.json.write(
    #         frame_meta, join(outroot, view_folder_, 'metadata.json'))
    #
    # # Write JSON
    # xm.io.json.write(test_meta, test_json)


def main(input_dir: str, output_dir: str):
    filenames = list(sorted(os.listdir(os.path.join(input_dir, 'images'))))
    input_resolution = Image.open(os.path.join(input_dir, 'images', filenames[0])).size[0]
    assert input_resolution == INPUT_RESOLUTION, (input_resolution, INPUT_RESOLUTION)
    downsize_factor = input_resolution / BENCHMARK_RESOLUTION
    imgs = []
    img_paths = []
    for filename in tqdm(filenames, desc="Loading images"):
        img_path = os.path.join(input_dir, 'images', filename)
        img = load_rgb_png(img_path, downsize_factor=downsize_factor)
        mask = load_mask_png(os.path.join(input_dir, 'masks', filename), downsize_factor=downsize_factor)
        img = np.concatenate([img, mask[:, :, None]], axis=2)
        assert img.shape == (BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 4), img.shape
        imgs.append(img)
        img_paths.append(img_path)
    imgs = np.stack(imgs, axis=0)  # 70, 512, 512, 4
    with open(os.path.join(input_dir, 'test_id.txt'), 'r') as f:
        test_ids: List[str] = f.read().splitlines()
    with open(os.path.join(input_dir, 'train_id.txt'), 'r') as f:
        train_ids: List[str] = f.read().splitlines()
    assert set(test_ids).intersection(set(train_ids)) == set(), (set(test_ids).intersection(set(train_ids)))
    test_indices = [filenames.index(test_id) for test_id in test_ids]
    train_indices = [filenames.index(train_id) for train_id in train_ids]

    gen_data_from_blender_format(imgs, img_paths, train_indices, test_indices, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.input_dir, args.output_dir)
