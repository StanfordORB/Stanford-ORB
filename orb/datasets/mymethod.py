from orb.constant import BENCHMARK_RESOLUTION, PROJ_ROOT
import glob
from functools import partial
from orb.utils.preprocess import load_mask_png, load_rgb_exr, load_rgb_png
from orb.utils.colmap.read_write_model import read_model, read_cameras_binary, read_images_binary
import numpy as np
import torch
import json
import os
from pathlib import Path
from pyquaternion import Quaternion


LLFF_HDR_SCENE_DATA_DIR = os.path.join(PROJ_ROOT, 'data/stanfordorb/llff_colmap_HDR')
LLFF_LDR_SCENE_DATA_DIR = os.path.join(PROJ_ROOT, 'data/stanfordorb/llff_colmap_LDR')
BLENDER_HDR_SCENE_DATA_DIR = os.path.join(PROJ_ROOT, 'data/stanfordorb/blender_HDR')
BLENDER_LDR_SCENE_DATA_DIR = os.path.join(PROJ_ROOT, 'data/stanfordorb/blender_LDR')


""" LLFF utils """
# https://github.com/Kai-46/nerfplusplus/blob/ebf2f3e75fd6c5dfc8c9d0b533800daaf17bd95f/colmap_runner/extract_sfm.py#LL48C9-L48C9
def parse_camera_dict(colmap_cameras, colmap_images):
    assert len(colmap_cameras) == 1, colmap_cameras
    cam, = colmap_cameras.values()
    assert cam.model == 'PINHOLE', cam.model
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


def preprocess_cameras_core(base_dir, normalize_fn=None, return_before_normalize=False):
    sparse_dir = os.path.join(base_dir, 'sparse/0')
    colmap_cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    colmap_images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
    assert len(os.listdir(os.path.join(base_dir, 'sparse'))) == 3, sparse_dir
    scene1, scene2 = sorted([scene for scene in os.listdir(os.path.join(base_dir, 'sparse')) if scene != '0'])
    colmap_cameras_novel1 = read_cameras_binary(os.path.join(base_dir, 'sparse', scene1, "cameras.bin"))
    colmap_images_novel1 = read_images_binary(os.path.join(base_dir, 'sparse', scene1, "images.bin"))
    colmap_cameras_novel2 = read_cameras_binary(os.path.join(base_dir, 'sparse', scene2, "cameras.bin"))
    colmap_images_novel2 = read_images_binary(os.path.join(base_dir, 'sparse', scene2, "images.bin"))

    camera_dict = parse_camera_dict(colmap_cameras, colmap_images)
    camera_dict_joint = {
        **camera_dict,
        **{(scene1, k): v for k, v in parse_camera_dict(colmap_cameras_novel1, colmap_images_novel1).items()},
        **{(scene2, k): v for k, v in parse_camera_dict(colmap_cameras_novel2, colmap_images_novel2).items()},
    }
    if return_before_normalize:
        return camera_dict_joint
    if normalize_fn is not None:
        # the normalization considers **all** splits (i.e. train, test, and 2 novel splits) for the scene
        camera_dict_joint = normalize_fn(camera_dict_joint)
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
    assert len(camera_dict_novel) == 2, camera_dict_novel
    return camera_dict, camera_dict_novel


def preprocess_cameras_split(base_dir, split='train', ext='.exr'):
    if split in ['train', 'test']:
        return preprocess_cameras_train_test(base_dir, split, ext)
    assert split != Path(base_dir).parent.parent.stem, (base_dir, split)
    return preprocess_cameras_novel(base_dir, split, ext)


def preprocess_cameras_train_test(base_dir, split, image_ext):
    assert image_ext in ['.exr', '.png'], image_ext
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
        image_paths.append(os.path.join(image_dir, img_name.replace('.png', image_ext)))
        mask_paths.append(os.path.join(mask_dir, img_name))
    assert len(image_paths) == len(include_ids) == len(new_camera_dict), (len(image_paths), len(include_ids), len(new_camera_dict), base_dir)
    return new_camera_dict, image_paths, mask_paths


def preprocess_cameras_novel(base_dir, split, image_ext):
    assert image_ext in ['.exr', '.png'], image_ext
    _, camera_dict_novel = preprocess_cameras_core(base_dir)
    camera_dict_novel = camera_dict_novel[split]
    image_dir = os.path.join(base_dir, '../', split, 'images')
    mask_dir = os.path.join(base_dir, '../', split, 'masks')
    image_paths = []
    mask_paths = []
    for img_name in sorted(camera_dict_novel.keys()):
        image_paths.append(os.path.join(image_dir, img_name.replace('.png', image_ext)))
        mask_paths.append(os.path.join(mask_dir, img_name))
    return camera_dict_novel, image_paths, mask_paths


class LLFFDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, split: str, hdr: bool = True):
        # split is one of ['train', 'test', 'novel_scene_name']
        self.data_root = data_root
        assert os.path.exists(self.data_root), f"Data directory is empty: {self.data_root}"

        cam_dict, image_paths, mask_paths = preprocess_cameras_split(self.data_root, split, ext='.exr' if hdr else '.png')
        cam_dict = read_cam_dict(cam_dict)
        print('Found # images, # masks, # cameras: ', len(image_paths), len(mask_paths), len(cam_dict))
        self.data_size = len(image_paths)
        assert len(image_paths) == len(mask_paths) == len(cam_dict) == self.data_size

        self.data = {
            'image_path': image_paths,
            'mask_path': mask_paths,
            'intrinsics': [],
            'pose': [],
            'image': [],
            'mask': []
        }
        load_rgb = load_rgb_exr if hdr else load_rgb_png
        input_image_shape = load_rgb(image_paths[0]).shape
        assert input_image_shape == (2048, 2048, 3), input_image_shape
        downsize_factor = input_image_shape[0] / BENCHMARK_RESOLUTION
        for x in sorted(cam_dict.keys()):
            intrinsics = cam_dict[x]['K'].astype(np.float32)
            if downsize_factor is not None:
                assert intrinsics.shape == (4, 4), intrinsics.shape
                intrinsics[:2, :] /= downsize_factor
            pose = cam_dict[x]['C2W'].astype(np.float32)
            self.data['intrinsics'].append(intrinsics)
            self.data['pose'].append(pose)

        for path in image_paths:
            image = load_rgb(path, downsize_factor=downsize_factor)
            self.data['image'].append(image)
        for path in mask_paths:
            mask = load_mask_png(path, downsize_factor=downsize_factor)
            self.data['mask'].append(mask)

        """ 
        self.data contains the following fields:
        image_paths: list[str] 
        mask_paths: list[str]
        intrinsics: Float[torch.Tensor, "N 4 4"], camera intrinsics
        poses: Float[torch.Tensor, "N 4 4"], camera-to-world matrices
        images: Float[torch.Tensor, "N 3 H=512 W=512"], RGB images, pixel range [0, 1]
        masks: Float[torch.Tensor, "N H=512 W=512"], binary masks with values 0 or 1
        """
        self.data['intrinsics'] = torch.tensor(np.stack(self.data['intrinsics']), dtype=torch.float32)
        self.data['pose'] = torch.tensor(np.stack(self.data['pose']), dtype=torch.float32)
        self.data['image'] = torch.tensor(np.stack(self.data['image']), dtype=torch.float32).permute(0, 3, 1, 2)
        self.data['mask'] = torch.tensor(np.stack(self.data['mask']), dtype=torch.float32)[:, None, :, :]

    def __getitem__(self, index):
        return {
            'intrinsics': self.data['intrinsics'][index],
            'pose': self.data['pose'][index],
            'image': self.data['image'][index],
            'mask': self.data['mask'][index],
        }

    def __len__(self):
        return self.data_size


LLFFHDRDataset = partial(LLFFDataset, hdr=True)
LLFFLDRDataset = partial(LLFFDataset, hdr=False)


class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, split: str, hdr: bool = True):
        # split is one of ['train', 'test', 'novel_scene_name']
        self.data_root = data_root
        assert os.path.exists(self.data_root), f"Data directory is empty: {self.data_root}"

        with open(os.path.join(self.data_root, 'transforms_{}.json'.format(split if split in ['train', 'test'] else 'novel')), 'r') as f:
            metadata = json.load(f)
        image_ext = '.exr' if hdr else '.png'
        load_rgb = load_rgb_exr if hdr else load_rgb_png
        factor = load_rgb(next(iter(glob.glob(os.path.join(self.data_root, "train", '*' + image_ext))))).shape[0] // BENCHMARK_RESOLUTION
        #
        camera_angle_x = None
        if 'camera_angle_x' not in metadata:
            for frame in metadata['frames']:
                if frame['scene_name'] == split:
                    camera_angle_x = float(frame['camera_angle_x'])
                    break
        else:
            camera_angle_x = float(metadata["camera_angle_x"])
        focal = .5 * BENCHMARK_RESOLUTION / np.tan(.5 * camera_angle_x)
        intrinsics = np.array([
            [focal, 0, 0.5 * BENCHMARK_RESOLUTION, 0],
            [0, focal, 0.5 * BENCHMARK_RESOLUTION, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.data_size = len(metadata['frames'])
        self.data = {
            'image_path': [],
            'mask_path': [],
            'intrinsics': [],
            'pose': [],
            'image': [],
            'mask': []
        }
        frame_data_root = self.data_root if split in ['train', 'test'] else os.path.join(self.data_root, '..', split)
        for frame in metadata['frames']:
            if split not in ['train', 'test'] and frame['scene_name'] != split:
                continue
            image_path = os.path.join(frame_data_root, frame["file_path"] + image_ext)
            image = load_rgb(image_path, downsize_factor=factor)
            mask_path = os.path.join(frame_data_root, os.path.dirname(frame['file_path']) + '_mask', os.path.basename(frame['file_path']) + '.png')
            mask = load_mask_png(mask_path, downsize_factor=factor)
            pose = np.array(frame['transform_matrix'])

            self.data['image_path'].append(image_path)
            self.data['mask_path'].append(mask_path)
            self.data['intrinsics'].append(intrinsics)
            self.data['pose'].append(pose)
            self.data['image'].append(image)
            self.data['mask'].append(mask)


        """ 
        self.data contains the following fields:
        image_paths: list[str] 
        mask_paths: list[str]
        focal: float, focal length
        intrinsics: Float[torch.tensor, "N 4 4"], camera intrinsics
        poses: Float[torch.Tensor, "N 4 4"], camera-to-world matrices
        images: Float[torch.Tensor, "N 3 H=512 W=512"], RGB images, pixel range [0, 1]
        masks: Float[torch.Tensor, "N 1 H=512 W=512"], binary masks with values 0 or 1
        """

        self.data['intrinsics'] = torch.tensor(np.stack(self.data['intrinsics']), dtype=torch.float32)
        self.data['pose'] = torch.tensor(np.stack(self.data['pose']), dtype=torch.float32)
        self.data['image'] = torch.tensor(np.stack(self.data['image']), dtype=torch.float32).permute(0, 3, 1, 2)
        self.data['mask'] = torch.tensor(np.stack(self.data['mask']), dtype=torch.float32)[:, None, :, :]

    def __getitem__(self, index):
        return {
            'intrinsics': self.data['intrinsics'][index],
            'pose': self.data['pose'][index],
            'image': self.data['image'][index],
            'mask': self.data['mask'][index],
        }


BlenderHDRDataset = partial(BlenderDataset, hdr=True)
BlenderLDRDataset = partial(BlenderDataset, hdr=False)


if __name__ == "__main__":
    dataset = LLFFHDRDataset(data_root=os.path.join(LLFF_HDR_SCENE_DATA_DIR, "baking_scene001"), split='train')
    for k, v in dataset[0].items():
        print(k, v.shape)

    dataset = LLFFHDRDataset(data_root=os.path.join(LLFF_HDR_SCENE_DATA_DIR, "baking_scene001"), split='baking_scene002')
    for k, v in dataset[0].items():
        print(k, v.shape)

    dataset = LLFFLDRDataset(data_root=os.path.join(LLFF_LDR_SCENE_DATA_DIR, "baking_scene001"), split='test')
    for k, v in dataset[0].items():
        print(k, v.shape)

    dataset = BlenderHDRDataset(data_root=os.path.join(BLENDER_HDR_SCENE_DATA_DIR, 'baking_scene001'), split='train')
    for k, v in dataset[0].items():
        print(k, v.shape)

    dataset = BlenderHDRDataset(data_root=os.path.join(BLENDER_HDR_SCENE_DATA_DIR, 'baking_scene001'), split='baking_scene002')
    for k, v in dataset[0].items():
        print(k, v.shape)

    dataset = BlenderLDRDataset(data_root=os.path.join(BLENDER_LDR_SCENE_DATA_DIR, 'baking_scene001'), split='test')
    for k, v in dataset[0].items():
        print(k, v.shape)
