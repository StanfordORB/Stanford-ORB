import imageio
import pyexr
from tqdm import tqdm
import json
imageio.plugins.freeimage.download()
import torch
import torch.nn as nn
import numpy as np
import imageio
import cv2
import argparse
import os
from pathlib import Path
from orb.constant import PROCESSED_SCENE_DATA_DIR, DEFAULT_SCENE_DATA_DIR
from orb.utils.preprocess import load_rgb_exr
from orb.utils.env_map import env_map_to_cam_to_world_by_convention

TINY_NUMBER = 1e-8
if DEFAULT_SCENE_DATA_DIR:
    SCENE_DATA_DIR = DEFAULT_SCENE_DATA_DIR
else:
    SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


def parse_raw_sg(sg):
    SGLobes = sg[..., :3] / (torch.norm(sg[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    SGLambdas = torch.abs(sg[..., 3:4])
    SGMus = torch.abs(sg[..., -3:])
    return SGLobes, SGLambdas, SGMus


def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    # print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    # print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    # print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


def fit_sg(gt_envmap: np.ndarray, out_dir: str, overwrite: bool):
    gt_envmap = torch.tensor(gt_envmap, dtype=torch.float32, device='cuda')
    H, W = gt_envmap.shape[:2]
    assert H == 256 and W == 512, (H, W)

    print(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    assert (os.path.isdir(out_dir))

    numLgtSGs = 128
    lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
    lgtSGs.data[..., 3:4] *= 100.
    lgtSGs.requires_grad = True

    optimizer = torch.optim.Adam([lgtSGs,], lr=1e-2)

    # N_iter = 100000
    N_iter = 10000  # FIXME

    # ignore saved files
    pretrained_file = os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs))
    if os.path.isfile(pretrained_file):
        assert overwrite, pretrained_file
        # print('Loading: ', pretrained_file)
        # lgtSGs.data.copy_(torch.from_numpy(np.load(pretrained_file)).cuda())

    for step in tqdm(range(N_iter)):
        optimizer.zero_grad()
        env_map = SG2Envmap(lgtSGs, H, W)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            # print('step: {}, loss: {}'.format(step, loss.item()))
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
            im = np.power(im, 1./2.2)
            im = np.clip(im, 0., 1.)
            # im = (im - im.min()) / (im.max() - im.min() + TINY_NUMBER)
            im = np.uint8(im * 255.)
            imageio.imwrite(os.path.join(out_dir, 'log_im_{}.png'.format(numLgtSGs)), im)

            # np.save(os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())
    np.save(os.path.join(out_dir, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scene', type=str, required=True)
    parser.add_argument('-o', '--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.join(SCENE_DATA_DIR, args.scene, 'final_output/llff_format_HDR/env_map/')
    output_dir = os.path.join(PROCESSED_SCENE_DATA_DIR, args.scene, 'physg_format/env_map/')

    with open(os.path.join(SCENE_DATA_DIR, args.scene, 'final_output/blender_format_HDR/transforms_test.json')) as f:
        test_frames = json.load(f)['frames']

    for frame in test_frames:
        test_name = os.path.basename(frame['file_path']) + ".exr"
        input_path = os.path.join(input_dir, test_name)
        output_subdir = os.path.join(output_dir, Path(input_path).stem)
        os.makedirs(output_subdir, exist_ok=args.overwrite)

        env_map = load_rgb_exr(input_path, downsize_factor=None)
        env_map = cv2.resize(env_map, (512, 256), interpolation=cv2.INTER_AREA)
        pyexr.write(os.path.join(output_subdir, 'envmap_cam.exr'), env_map)
        c2w = np.array(frame['transform_matrix'])
        env_map = env_map_to_cam_to_world_by_convention(env_map, c2w, convention='physg')
        pyexr.write(os.path.join(output_subdir, 'envmap_world_physg.exr'), env_map)
        fit_sg(env_map, os.path.join(output_dir, Path(input_path).stem), args.overwrite)


if __name__ == '__main__':
    main()
