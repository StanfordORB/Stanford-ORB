import imageio
imageio.plugins.freeimage.download()
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import imageio
from tqdm import tqdm
import cv2
import os
import glob
from orb.third_party.invrender.code.model.sg_render import compute_envmap
from pathlib import Path
from orb.constant import PROCESSED_SCENE_DATA_DIR, DEFAULT_SCENE_DATA_DIR, VERSION
from orb.utils.env_map import env_map_to_cam_to_world_by_convention
import pyexr


TINY_NUMBER = 1e-8
NUM_SG = 128
ENV_WIDTH = 512
ENV_HEIGHT = 256
if VERSION == 'extension':
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, required=True)
    parser.add_argument('-o', '--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.join(SCENE_DATA_DIR, args.scene, 'final_output/llff_format_HDR/env_map/')
    output_dir = os.path.join(PROCESSED_SCENE_DATA_DIR, args.scene, 'invrender_format/env_map/')
    with open(os.path.join(SCENE_DATA_DIR, args.scene, 'final_output/blender_format_HDR/transforms_test.json')) as f:
        test_frames = json.load(f)['frames']
    for frame in test_frames:
        test_name = os.path.basename(frame['file_path']) + ".exr"
        input_path = os.path.join(input_dir, test_name)
        output_subdir = os.path.join(output_dir, Path(input_path).stem)
        os.makedirs(output_subdir, exist_ok=args.overwrite)

        envmap = pyexr.open(input_path).get()[..., :3]
        envmap = cv2.resize(envmap, (ENV_WIDTH, ENV_HEIGHT), interpolation=cv2.INTER_AREA)
        assert envmap.shape == (ENV_HEIGHT, ENV_WIDTH, 3), envmap.shape
        pyexr.write(os.path.join(output_subdir, 'envmap_cam.exr'), envmap)

        c2w = np.array(frame['transform_matrix'])
        c2w[..., 3] /= 2  # hard-coded, should be consistent with the dataloader
        envmap = env_map_to_cam_to_world_by_convention(envmap, c2w, convention='invrender')
        pyexr.write(os.path.join(output_subdir, 'envmap_world_invrender.exr'), envmap)

        fit_sg(envmap, output_subdir)


def fit_sg(gt_envmap: np.ndarray, out_dir: str):
    gt_envmap = torch.from_numpy(gt_envmap).cuda()
    H, W = gt_envmap.shape[:2]

    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    assert (os.path.isdir(out_dir))

    numLgtSGs = NUM_SG
    lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
    lgtSGs.data[..., 3:4] *= 100.
    lgtSGs.requires_grad = True

    optimizer = torch.optim.Adam([lgtSGs,], lr=1e-2)

    # reload sg parameters if exists
    # pretrained_file = os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs))
    # if os.path.isfile(pretrained_file):
    #     assert overwrite, pretrained_file
        # print('Loading: ', pretrained_file)
        # lgtSGs.data.copy_(torch.from_numpy(np.load(pretrained_file)).cuda())

    # N_iter = 100000
    N_iter = 10000  # FIXME
    for step in tqdm(range(N_iter)):
        optimizer.zero_grad()
        env_map = compute_envmap(lgtSGs, H, W)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        loss.backward()
        optimizer.step()

        # if step % 50 == 0:
        #     print('step: {}, loss: {}'.format(step, loss.item()))

        if step % 100 == 0:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
            im = np.clip(np.power(im, 1./2.2), 0., 1.)
            im = np.uint8(im * 255.)
            imageio.imwrite(os.path.join(out_dir, 'log_im_{}.png'.format(numLgtSGs)), im)

            # np.save(os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())


if __name__ == '__main__':
    main()
