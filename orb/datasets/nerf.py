import os
import torch
import numpy as np
import imageio
import glob
import json
from orb.utils.preprocess import load_mask_png, load_rgb_png
from orb.constant import BENCHMARK_RESOLUTION


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_capture_data(basedir, testskip):
    basedir = os.path.join(basedir, 'final_output/blender_format_LDR')

    splits = ['train', 'val', 'test']
    splits = ['train', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
    metas['val'] = metas['test']
    splits = ['train', 'val', 'test']

    factor = load_rgb_png(next(iter(glob.glob(os.path.join(basedir, "train", "*.png"))))).shape[0] // BENCHMARK_RESOLUTION

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            rgb = load_rgb_png(os.path.join(basedir, frame["file_path"] + ".png"), downsize_factor=factor)
            assert rgb.shape == (512, 512, 3)
            alpha = load_mask_png(os.path.join(basedir, os.path.dirname(frame['file_path']) + '_mask', os.path.basename(frame['file_path']) + '.png'), downsize_factor=factor).astype(np.float32)
            imgs.append(np.concatenate([rgb, alpha[..., None]], -1))
            poses.append(np.array(frame['transform_matrix']))
        # imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        imgs = np.array(imgs).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal], i_split
