import numpy as np
import os, imageio
from imageint.third_party.nerfpytorch.load_llff import load_llff_data
from imageint.utils.preprocess import load_mask_png, load_rgb_png


def load_capture_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    images, poses, bds, render_poses, i_test = load_llff_data(
        basedir,
        factor=factor,
        recenter=recenter,
        bd_factor=bd_factor,
        spherify=spherify,
        path_zflat=path_zflat,
    )
    imgdir = os.path.join(basedir, 'images')
    maskdir = os.path.join(basedir, 'masks')
    filenames = [f for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    sh = images[0].shape
    factor = load_rgb_png(os.path.join(imgdir, filenames[0])).shape[0] // sh[0]
    new_images = []
    for i in range(len(images)):
        rgb = load_rgb_png(os.path.join(imgdir, filenames[i]), downsize_factor=factor)
        alpha = load_mask_png(os.path.join(maskdir, filenames[i]), downsize_factor=factor)
        new_images.append(np.concatenate([rgb, alpha[..., None]], -1))
    new_images = np.stack(new_images, 0)
    assert new_images.shape == (*images.shape[:3], 4), f'{new_images.shape} != {(*images.shape[:3], 4)}'
    return new_images, poses, bds, render_poses, i_test
