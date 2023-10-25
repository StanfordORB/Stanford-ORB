from orb.third_party.nvdiffrecmc.dataset.dataset_nerf import DatasetNERF
from orb.utils.preprocess import load_hdr_rgba
from orb.third_party.nvdiffrecmc.render import util
from pathlib import Path
import os
import torch
import numpy as np


def _load_img(path, downsize_factor):
    rgba = load_hdr_rgba(path + '.exr', downsize_factor)
    rgba = torch.tensor(rgba, dtype=torch.float32)
    return rgba


class DatasetCapture(DatasetNERF):
    def _parse_frame(self, cfg, idx):
        # Config projection matrix (static, so could be precomputed)
        if 'camera_angle_x' not in cfg:  # hack
            fovx = cfg['frames'][idx]['camera_angle_x']
        else:
            fovx = cfg['camera_angle_x']
        fovy   = util.fovx_to_fovy(fovx, self.aspect)
        proj   = util.perspective(fovy, self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Load image data and modelview matrix
        assert self.aspect == 1 and self.FLAGS.train_res[0] == self.FLAGS.train_res[1]
        if 'scene_name' in cfg['frames'][idx]:
            base_dir = self.base_dir.replace(Path(self.base_dir).parent.parent.name, cfg['frames'][idx]['scene_name'])
        else:
            base_dir = self.base_dir
        img    = _load_img(os.path.join(base_dir, cfg['frames'][idx]['file_path']), downsize_factor=self.resolution[0] / self.FLAGS.train_res[0])

        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv     = mv @ util.rotate_x(-np.pi / 2)

        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension
