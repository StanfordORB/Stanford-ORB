from orb.third_party.physg.code.datasets.scene_dataset import SceneDataset
from orb.utils.colmap.read_write_model import read_model, read_cameras_binary, read_images_binary
from orb.constant import BENCHMARK_RESOLUTION
from orb.utils.preprocess import load_mask_png, load_rgb_exr
import numpy as np
from orb.utils.colmap.nerfplusplus_normalize_cam_dict import _normalize_cam_dict
import torch
from pathlib import Path
import os
from pyquaternion import Quaternion
from orb.utils.convert_cameras import llff_to_colmap
from orb.constant import INPUT_RESOLUTION
import imageio


# https://github.com/Kai-46/nerfplusplus/blob/ebf2f3e75fd6c5dfc8c9d0b533800daaf17bd95f/colmap_runner/extract_sfm.py#LL48C9-L48C9
def parse_camera_dict(colmap_cameras, colmap_images):
    assert len(colmap_cameras) == 1, colmap_cameras
    cam, = colmap_cameras.values()
    # cam = colmap_cameras[1]
    assert (cam.model == 'PINHOLE')
    img_size = [cam.width, cam.height]
    # https://github.com/colmap/colmap/blob/1555ff03e9fce85a2a1596095fee0f161524d844/src/base/camera_models.h#L243
    params = list(cam.params)
    fx, fy, cx, cy = params
    assert cx == cam.width / 2 and cy == cam.height / 2, cam
    K = np.eye(4)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    camera_dict = {}
    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name

        qvec = list(image.qvec)
        tvec = list(image.tvec)

        # w, h, fx, fy, cx, cy, qvec, tvec
        # camera_dict[img_name] = img_size + params + qvec + tvec
        camera_dict[img_name] = {}
        camera_dict[img_name]['img_size'] = img_size

        camera_dict[img_name]['K'] = list(K.flatten())

        rot = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)
        camera_dict[img_name]['W2C'] = list(W2C.flatten())

    return camera_dict


def read_cam_dict(cam_dict):
    for x in sorted(cam_dict.keys()):
        K = np.array(cam_dict[x]['K']).reshape((4, 4))
        W2C = np.array(cam_dict[x]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)

        cam_dict[x]['K'] = K
        cam_dict[x]['W2C'] = W2C
        cam_dict[x]['C2W'] = C2W
    return cam_dict


def preprocess_cameras(base_dir):
    with open(os.path.join(base_dir, 'train_id.txt')) as f:
        train_ids = f.read().splitlines()
    with open(os.path.join(base_dir, 'test_id.txt')) as f:
        test_ids = f.read().splitlines()

    sparse_dir = os.path.join(base_dir, 'sparse/0')
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, '.bin')
    camera_dict = parse_camera_dict(colmap_cameras, colmap_images)
    camera_dict, _ = _normalize_cam_dict(camera_dict)

    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    image_paths = []
    mask_paths = []
    for cur_image in colmap_images.values():
        img_name = cur_image.name
        if img_name in train_ids:
            pass
        elif img_name in test_ids:
            continue
        else:
            import ipdb; ipdb.set_trace()
            raise RuntimeError

        image_paths.append(os.path.join(image_dir, img_name.replace('.png', '.exr')))
        mask_paths.append(os.path.join(mask_dir, img_name))
    assert len(image_paths) == len(train_ids), (len(image_paths), len(train_ids), base_dir)
    return camera_dict, image_paths, mask_paths


def preprocess_cameras_core_backup(base_dir, novel):
    w2c_mats, _, hwf = llff_to_colmap(base_dir, filename='poses_bounds_novel.npy' if novel else 'poses_bounds.npy')
    h, w, f = hwf
    print('[DEBUG] loaded hwf from pose bounds', hwf)

    img_size = [w, h]
    fx, fy, cx, cy = f, f, w / 2, h / 2
    K = np.eye(4)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    if novel:
        with open(os.path.join(base_dir, 'novel_id.txt'), 'r') as f:
            novel_ids = f.read().splitlines()
        img_names = [tuple(novel_id.split('/')) for novel_id in novel_ids]
    else:
        image_dir = os.path.join(base_dir, 'images')
        img_names = [f.replace('.exr', '.png') for f in sorted(os.listdir(image_dir)) if f.endswith('exr')]  # assume HDR
    assert len(img_names) == len(w2c_mats), (len(img_names), len(w2c_mats), base_dir)

    camera_dict = {}
    for i in range(len(w2c_mats)):
        img_name = img_names[i]
        camera_dict[img_name] = {
            'img_size': img_size,
            'K': list(K.flatten()),
            'W2C': list(w2c_mats[i].flatten()),
        }

    return camera_dict




def preprocess_cameras_core(base_dir, return_before_normalize=False):
    sparse_dir = os.path.join(base_dir, 'sparse/0')
    if os.path.exists(sparse_dir):
        colmap_cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        colmap_images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        # if len(os.listdir(os.path.join(base_dir, 'sparse'))) > 1:  # FIXME: change it back once data is ready!!!
        if not base_dir.startswith('/viscam/projects/imageint/yzzhang/data/capture_scene_data_v0/data/'):
            assert len(os.listdir(os.path.join(base_dir, 'sparse'))) == 3, sparse_dir  # FIXME comment back
            scene1, scene2 = sorted([scene for scene in os.listdir(os.path.join(base_dir, 'sparse')) if scene != '0'])
            colmap_cameras_novel1 = read_cameras_binary(os.path.join(base_dir, 'sparse', scene1, "cameras.bin"))
            colmap_images_novel1 = read_images_binary(os.path.join(base_dir, 'sparse', scene1, "images.bin"))
            colmap_cameras_novel2 = read_cameras_binary(os.path.join(base_dir, 'sparse', scene2, "cameras.bin"))
            colmap_images_novel2 = read_images_binary(os.path.join(base_dir, 'sparse', scene2, "images.bin"))
        else:
            colmap_cameras_novel1 = colmap_cameras
            colmap_images_novel1 = {}
            colmap_cameras_novel2 = colmap_cameras
            colmap_images_novel2 = {}

        camera_dict = parse_camera_dict(colmap_cameras, colmap_images)
        camera_dict_joint = {
            **camera_dict,
            **{(scene1, k): v for k, v in parse_camera_dict(colmap_cameras_novel1, colmap_images_novel1).items()},
            **{(scene2, k): v for k, v in parse_camera_dict(colmap_cameras_novel2, colmap_images_novel2).items()},
        }

        if True:
            camera_dict_joint_backup = {
                **preprocess_cameras_core_backup(base_dir, novel=False),
                **preprocess_cameras_core_backup(base_dir, novel=True),
            }
            if set(camera_dict_joint.keys()) != set(camera_dict_joint_backup.keys()):
                import ipdb; ipdb.set_trace()
            for k, v in camera_dict_joint.items():
                for k_, v_ in v.items():
                    if not np.allclose(v_, camera_dict_joint_backup[k][k_], atol=1e-3):
                        if k_ == 'K':
                            print(f'[ERROR] mismatch: {k}, {k_}')
                            print(v_, camera_dict_joint_backup[k][k_])
                        else:
                            import ipdb; ipdb.set_trace()
            print('[DEBUG] passed llff-colmap conversion check')

    else:
        camera_dict_joint = {
            **preprocess_cameras_core_backup(base_dir, novel=False),
            **preprocess_cameras_core_backup(base_dir, novel=True),
        }

    if return_before_normalize:
        return camera_dict_joint
    camera_dict_joint, _ = _normalize_cam_dict(camera_dict_joint)
    camera_dict = dict()
    camera_dict_novel = dict()
    for k, v in camera_dict_joint.items():
        if isinstance(k, str):
            camera_dict[k] = v
        else:
            scene, k = k
            if scene not in camera_dict_novel:
                camera_dict_novel[scene] = dict()
            camera_dict_novel[scene][k] = v
    # assert len(camera_dict_novel) == 2, camera_dict_novel  # FIXME comment back
    return camera_dict, camera_dict_novel


def preprocess_cameras_split(base_dir, split='train'):
    if split in ['train', 'test']:
        return preprocess_cameras_train_test(base_dir, split)
    if split == Path(base_dir).parent.parent.stem:
        return preprocess_cameras_train_test(base_dir, 'test')
    return preprocess_cameras_novel(base_dir, split)


def preprocess_cameras_train_test(base_dir, split):
    camera_dict, _ = preprocess_cameras_core(base_dir)

    with open(os.path.join(base_dir, 'train_id.txt')) as f:
        train_ids = f.read().splitlines()
    with open(os.path.join(base_dir, 'test_id.txt')) as f:
        test_ids = f.read().splitlines()

    include_ids = {'train': train_ids, 'test': test_ids}[split]
    exclude_ids = {'train': test_ids, 'test': train_ids}[split]

    image_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    image_paths = []
    mask_paths = []
    new_camera_dict = dict()
    for img_name in sorted(camera_dict.keys()):
        if img_name in include_ids:
            pass
        elif img_name in exclude_ids:
            continue
        else:
            import ipdb; ipdb.set_trace()
            raise RuntimeError(img_name, base_dir)

        new_camera_dict[img_name] = camera_dict[img_name]
        image_paths.append(os.path.join(image_dir, img_name.replace('.png', '.exr')))
        mask_paths.append(os.path.join(mask_dir, img_name))
    assert len(image_paths) == len(include_ids) == len(new_camera_dict), (len(image_paths), len(include_ids), len(new_camera_dict), base_dir)
    return new_camera_dict, image_paths, mask_paths


def preprocess_cameras_novel(base_dir, split):
    _, camera_dict_novel = preprocess_cameras_core(base_dir)
    camera_dict_novel = camera_dict_novel[split]
    image_dir = os.path.join(base_dir, '../../../', split, 'final_output/llff_format_HDR/images')
    mask_dir = os.path.join(base_dir, '../../../', split, 'final_output/llff_format_HDR/masks')
    image_paths = []
    mask_paths = []
    for img_name in sorted(camera_dict_novel.keys()):
        image_paths.append(os.path.join(image_dir, img_name.replace('.png', '.exr')))
        mask_paths.append(os.path.join(mask_dir, img_name))
    return camera_dict_novel, image_paths, mask_paths


class Dataset(SceneDataset):

    def __init__(self, gamma, exposure, instance_dir, train_cameras, split='train'):

        self.instance_dir = os.path.join(instance_dir, 'final_output/llff_format_HDR')
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.gamma = gamma
        self.exposure = exposure
        print(f'Applying inverse gamma = {gamma}, exposure = {exposure}')
        self.train_cameras = train_cameras

        cam_dict, image_paths, mask_paths = preprocess_cameras_split(self.instance_dir, split)
        cam_dict = read_cam_dict(cam_dict)
        print('Found # images, # masks, # cameras: ', len(image_paths), len(mask_paths), len(cam_dict))
        self.n_cameras = len(image_paths)
        self.image_paths = image_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        self.intrinsics_all = []
        self.pose_all = []
        input_image_shape = imageio.imread(image_paths[0]).shape
        assert input_image_shape == (INPUT_RESOLUTION, INPUT_RESOLUTION, 3), input_image_shape
        downsize_factor = input_image_shape[0] / BENCHMARK_RESOLUTION
        for x in sorted(cam_dict.keys()):
            intrinsics = cam_dict[x]['K'].astype(np.float32)
            if downsize_factor is not None:
                assert intrinsics.shape == (4, 4), intrinsics.shape
                intrinsics[:2, :] /= downsize_factor
            pose = cam_dict[x]['C2W'].astype(np.float32)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        assert (len(image_paths) == self.n_cameras)
        self.has_groundtruth = True
        self.rgb_images = []
        print(f'Applying gamma correction: {self.gamma}')
        for path in image_paths:
            rgb = load_rgb_exr(path, downsize_factor=downsize_factor)
            rgb = 2 ** self.exposure * rgb
            rgb = np.power(rgb, self.gamma)
            rgb = rgb.transpose(2, 0, 1)  # HWC -> CHW

            H, W = rgb.shape[1:3]
            self.img_res = [H, W]
            self.total_pixels = self.img_res[0] * self.img_res[1]

            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        assert (len(mask_paths) == self.n_cameras)
        self.object_masks = []
        for path in mask_paths:
            object_mask = load_mask_png(path, downsize_factor=downsize_factor)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())
