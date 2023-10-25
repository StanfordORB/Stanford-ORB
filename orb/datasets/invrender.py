from orb.third_party.invrender.code.datasets.syn_dataset import SynDataset
import os
import torch
import numpy as np

from orb.third_party.invrender.code.utils import rend_util
from orb.constant import BENCHMARK_RESOLUTION
from orb.utils.preprocess import load_hdr_rgba
import json
import imageio


class Dataset(SynDataset):
    def __init__(self,
                 instance_dir,
                 frame_skip,
                 split='train'
                 ):
        scene = os.path.basename(instance_dir)
        self.instance_dir = os.path.join(instance_dir, 'final_output/blender_format_HDR')
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.split = split

        json_path = os.path.join(self.instance_dir, 'transforms_{}.json'.format(split))
        print('Read cam from {}'.format(json_path))
        with open(json_path, 'r') as fp:
            meta = json.load(fp)

        image_paths = []
        poses = []
        env_map_paths = []
        for frame in meta['frames']:
            poses.append(np.array(frame['transform_matrix']))
            if 'scene_name' in frame:
                image_paths.append(os.path.join(self.instance_dir.replace(scene, frame['scene_name']), frame['file_path'] + '.exr'))
            else:
                image_paths.append(os.path.join(self.instance_dir, frame['file_path'] + '.exr'))
            if split == 'test':
                env_map_paths.append(os.path.join(self.instance_dir, 'env_map', os.path.basename(frame['file_path'] + '.exr')))

        img_h = img_w = BENCHMARK_RESOLUTION
        input_image_shape = load_hdr_rgba(image_paths[0]).shape
        assert input_image_shape == (2048, 2048, 4), input_image_shape
        downsize_factor = input_image_shape[0] / img_h

        if split != 'novel':
            camera_angle_x = float(meta['camera_angle_x'])
            focal = .5 * img_w / np.tan(.5 * camera_angle_x)
        poses = np.array(poses)
        # print("focal {}, img_w {}, img_h {}".format(focal, img_w, img_h))
        scale = 2.0
        print("Scale {}".format(scale))
        poses[..., 3] /= scale

        # skip for training
        image_paths = image_paths[::frame_skip]
        poses = poses[::frame_skip, ...]
        print('Training image: {}'.format(len(image_paths)))
        self.n_cameras = len(image_paths)
        self.image_paths = image_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None

        self.intrinsics_all = []
        self.pose_all = []

        if split == 'novel':
            def get_intrinsics(camera_angle_x):
                focal = .5 * img_w / np.tan(.5 * camera_angle_x)
                intrinsics = [[focal, 0, img_w / 2], [0, focal, img_h / 2], [0, 0, 1]]
                intrinsics = np.array(intrinsics).astype(np.float32)
                return intrinsics
        else:
            intrinsics = [[focal, 0, img_w / 2], [0, focal, img_h / 2], [0, 0, 1]]
            intrinsics = np.array(intrinsics).astype(np.float32)
        for i in range(self.n_cameras):
            if split == 'novel':
                self.intrinsics_all.append(torch.from_numpy(get_intrinsics(meta['frames'][i]['camera_angle_x'])).float())
            else:
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(poses[i]).float())

        self.rgb_images = []
        self.object_masks = []

        self.img_res = [img_h, img_w]
        self.total_pixels = self.img_res[0] * self.img_res[1]

        # read training images
        for path in image_paths:
            rgba = load_hdr_rgba(path, downsize_factor)
            rgb = rgba[:, :, :3].reshape(-1, 3)
            self.rgb_images.append(torch.from_numpy(rgb).float())

            object_mask = rgba[:, :, 3] > .5
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())
            self.has_groundtruth = True

        # read relight image only for test
        if self.split == 'test':
            self.envmap6_images = []
            self.envmap12_images = []
            envmap6_image_paths = env_map_paths[::frame_skip]
            envmap12_image_paths = env_map_paths[::frame_skip]
            for path in envmap6_image_paths:
                rgb = rend_util.load_rgb(path).reshape(-1, 3)
                self.envmap6_images.append(torch.from_numpy(rgb).float())
            for path in envmap12_image_paths:
                rgb = rend_util.load_rgb(path).reshape(-1, 3)
                self.envmap12_images.append(torch.from_numpy(rgb).float())
