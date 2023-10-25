import os
from functools import partial
import json
from orb.utils.preprocess import load_rgb_png, load_mask_png
from orb.third_party.idr.code.preprocess.preprocess_cameras import get_Ps, get_normalization_function
import torch
import numpy as np

from orb.third_party.idr.code.utils import rend_util
from orb.utils.colmap.read_write_model import read_model, qvec2rotmat
from orb.utils.colmap.volsdf_normalize_cameras import _normalize_cameras
from orb.utils.colmap.load_colmap import load_pinhole_camera
from orb.third_party.idr.code.datasets.scene_dataset import SceneDataset
from orb.constant import PROCESSED_SCENE_DATA_DIR
import datetime
from pathlib import Path


def get_all_mask_points(mask_paths, downsize_factor=None):
    mask_points_all = []
    mask_ims = []
    for path in mask_paths:
        cur_mask = load_mask_png(path, downsize_factor)
        mask_points = np.where(cur_mask)
        xs = mask_points[1]
        ys = mask_points[0]
        mask_points_all.append(np.stack((xs,ys,np.ones_like(xs))).astype(np.float32))
        mask_ims.append(cur_mask)
    return mask_points_all, np.array(mask_ims)


def preprocess_cameras(base_dir, downsize_factor=None):
    # For runs before 05/25, normalize with only training cameras
    with open(os.path.join(base_dir, 'train_id.txt')) as f:
        train_ids = f.read().splitlines()
    with open(os.path.join(base_dir, 'test_id.txt')) as f:
        test_ids = f.read().splitlines()

    sparse_dir = os.path.join(base_dir, 'sparse/0')
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, '.bin')
    K = np.eye(3)
    assert colmap_cameras[1].params[2] == colmap_cameras[1].width / 2, colmap_cameras
    assert colmap_cameras[1].params[3] == colmap_cameras[1].height / 2, colmap_cameras
    K[0, 0] = colmap_cameras[1].params[0]
    K[1, 1] = colmap_cameras[1].params[1]
    K[0, 2] = colmap_cameras[1].params[2]
    K[1, 2] = colmap_cameras[1].params[3]
    print(K)

    if downsize_factor is not None:
        K[:2, :] /= downsize_factor
    print(K)

    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    cameras_npz_format = {}
    image_paths = []
    mask_paths = []
    counter = 0
    for cur_image in colmap_images.values():
        img_name = cur_image.name
        if img_name in train_ids:
            pass
        elif img_name in test_ids:
            continue
        else:
            import ipdb; ipdb.set_trace()
            raise RuntimeError
        image_paths.append(os.path.join(image_dir, img_name))
        mask_paths.append(os.path.join(mask_dir, img_name))

        M = np.zeros((3, 4))
        M[:, 3] = cur_image.tvec
        M[:3, :3] = qvec2rotmat(cur_image.qvec)

        P = np.eye(4)
        P[:3, :] = K @ M
        cameras_npz_format['world_mat_%d' % counter] = P
        counter += 1

    # volsdf normalize cameras
    cameras = _normalize_cameras(cameras_npz_format, len(cameras_npz_format))

    # idr normalize cameras
    number_of_normalization_points = 100
    mask_points_all, masks_all = get_all_mask_points(mask_paths, downsize_factor=downsize_factor)
    number_of_cameras = len(masks_all)
    Ps = get_Ps(cameras, number_of_cameras)

    normalization, all_Xs = get_normalization_function(Ps, mask_points_all, number_of_normalization_points,
                                                       number_of_cameras, masks_all)

    cameras_new = {}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i], np.array([[0, 0, 0, 1.0]])), axis=0).astype(
            np.float32)
    assert len(image_paths) == len(train_ids), (len(image_paths), len(train_ids), base_dir)
    return cameras_new, image_paths, mask_paths


def preprocess_cameras_core(base_dir, downsize_factor=None):
    sparse_dir = os.path.join(base_dir, 'sparse/0')
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, '.bin')
    if len(colmap_cameras) != 1:
        raise RuntimeError((base_dir, colmap_cameras))
    cam, = colmap_cameras.values()
    K = load_pinhole_camera(cam, downsize_factor)

    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    cameras_npz_format = {}
    image_paths = []
    mask_paths = []
    counter = 0
    for cur_image in colmap_images.values():
        img_name = cur_image.name
        image_paths.append(os.path.join(image_dir, img_name))
        mask_paths.append(os.path.join(mask_dir, img_name))

        M = np.zeros((3, 4))
        M[:, 3] = cur_image.tvec
        M[:3, :3] = qvec2rotmat(cur_image.qvec)

        P = np.eye(4)
        P[:3, :] = K @ M
        cameras_npz_format['world_mat_%d' % counter] = P
        counter += 1

    # volsdf normalize cameras
    cameras = _normalize_cameras(cameras_npz_format, len(cameras_npz_format))

    # idr normalize cameras
    # number_of_normalization_points = 100
    number_of_normalization_points = 1000
    mask_points_all, masks_all = get_all_mask_points(mask_paths, downsize_factor=downsize_factor)
    # number_of_normalization_points = len(mask_points_all[0][0, :])
    print('number of normalization points', number_of_normalization_points)
    number_of_cameras = len(masks_all)
    Ps = get_Ps(cameras, number_of_cameras)

    normalization, all_Xs = get_normalization_function(Ps, mask_points_all, number_of_normalization_points,
                                                       number_of_cameras, masks_all)

    print('IDR normalization', normalization)

    cameras_new = {}
    for i in range(number_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = np.concatenate((Ps[i], np.array([[0, 0, 0, 1.0]])), axis=0).astype(
            np.float32)
    if os.environ.get('IDR_NO_CACHE') != '1':
        cache_dir = os.path.join(PROCESSED_SCENE_DATA_DIR, Path(base_dir).parent.parent.stem, 'idr_format', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now()))
        print('cache saved to', cache_path)
        np.savez(cache_path, **cameras_new, image_paths=image_paths, mask_paths=mask_paths)
    else:
        print('not saving cache')
    return cameras_new, image_paths, mask_paths


def preprocess_cameras_split(base_dir, downsize_factor, split):
    cameras, image_paths, mask_paths = preprocess_cameras_core(base_dir, downsize_factor)

    with open(os.path.join(base_dir, 'train_id.txt')) as f:
        train_ids = f.read().splitlines()
    with open(os.path.join(base_dir, 'test_id.txt')) as f:
        test_ids = f.read().splitlines()
    include_ids = {'train': train_ids, 'test': test_ids}[split]
    exclude_ids = {'train': test_ids, 'test': train_ids}[split]

    include_indices = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        if image_name in include_ids:
            include_indices.append(True)
        elif image_name in exclude_ids:
            include_indices.append(False)
        else:
            raise RuntimeError(image_path)

    cameras_new = dict()
    image_paths_new = []
    mask_paths_new = []
    counter = 0
    for ind in range(len(image_paths)):
        if include_indices[ind]:
            cameras_new['scale_mat_%d' % counter] = cameras['scale_mat_%d' % ind]
            cameras_new['world_mat_%d' % counter] = cameras['world_mat_%d' % ind]
            image_paths_new.append(image_paths[ind])
            mask_paths_new.append(mask_paths[ind])
            counter += 1
    assert len(image_paths_new) == len(include_ids), (len(image_paths_new), len(include_ids), base_dir)
    assert len(mask_paths_new) == len(include_ids), (len(mask_paths_new), len(include_ids), base_dir)
    assert len(cameras_new) == len(include_ids) * 2, (len(cameras_new), len(include_ids), base_dir)
    return cameras_new, image_paths_new, mask_paths_new


class Dataset(SceneDataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 scan_id='',
                 cam_file=None,
                 split='train',
                 ):

        self.instance_dir = os.path.join(data_dir, scan_id, 'final_output/llff_format_LDR')

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        input_image_shape = load_rgb_png(os.path.join(self.instance_dir, 'images', '0000.png')).shape
        assert input_image_shape == (2048, 2048, 3), input_image_shape
        downsize_factor = input_image_shape[0] / img_res[0]
        self.split = split
        camera_dict, image_paths, mask_paths = preprocess_cameras_split(self.instance_dir, downsize_factor=downsize_factor, split=self.split)

        self.n_images = len(image_paths)
        print(f"Loaded {self.n_images} training images.")

        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = load_rgb_png(path, downsize_factor=downsize_factor) * 2 - 1
            rgb = rgb.transpose(2, 0, 1).reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        self.object_masks = []
        for path in mask_paths:
            object_mask = load_mask_png(path, downsize_factor=downsize_factor)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

    def get_pose_init(self):
        init_pose = torch.stack(self.pose_all).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

    def get_scale_mat(self):
        camera_dict, _, _ = preprocess_cameras_split(self.instance_dir, downsize_factor=4, split=self.split)
        return camera_dict['scale_mat_0']
