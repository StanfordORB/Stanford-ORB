import cv2
import os
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)
try:
    import pyexr
except:
    logger.error('pyexr not found')
    pyexr = None
import imageio


def cv2_downsize(f: np.ndarray, downsize_factor: Optional[int]):
    if downsize_factor is not None:
        f = cv2.resize(f, (0, 0), fx=1 / downsize_factor, fy=1 / downsize_factor, interpolation=cv2.INTER_AREA)
    return f


def load_rgb_exr(path: str, downsize_factor: Optional[int] = None) -> np.ndarray:
    # NO correction
    if pyexr is not None:
        f = pyexr.open(path).get()
    else:
        f = imageio.imread(path)
    assert f.dtype == np.float32, f.dtype
    assert len(f.shape) == 3 and f.shape[2] == 3, f.shape
    f = cv2_downsize(f, downsize_factor)
    return f


def load_rgb_png(path: str, downsize_factor: Optional[int] = None) -> np.ndarray:
    f = imageio.imread(path)
    assert f.dtype == np.uint8, f.dtype
    assert len(f.shape) == 3 and f.shape[2] == 3, f.shape
    f = f.astype(np.float32) / 255
    f = cv2_downsize(f, downsize_factor)
    return f


def load_mask_png(path: str, downsize_factor: Optional[int] = None) -> np.ndarray:
    f = imageio.imread(path)
    assert f.dtype == np.uint8, f.dtype
    f = f / 255
    f = cv2_downsize(f, downsize_factor)
    f = f > .5
    assert len(f.shape) == 2, f.shape
    return f


def rgb_to_srgb(f: np.ndarray):
    # f is loaded from .exr
    # output is NOT clipped to [0, 1]
    assert len(f.shape) == 3, f.shape
    assert f.shape[2] == 3, f.shape
    f = np.where(f > 0.0031308, np.power(np.maximum(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * f)
    return f


def srgb_to_rgb(f: np.ndarray):
    # f is LDR
    assert len(f.shape) == 3, f.shape
    assert f.shape[2] == 3, f.shape
    f = np.where(f <= 0.04045, f / 12.92, np.power((np.maximum(f, 0.04045) + 0.055) / 1.055, 2.4))
    return f


def load_hdr_rgba(path, downsize_factor: Optional[int] = None) -> np.ndarray:
    rgb = load_rgb_exr(path, downsize_factor)
    mask = imageio.imread(os.path.join(os.path.dirname(path) + '_mask', os.path.basename(path).replace('.exr', '.png')))
    assert mask.shape == (2048, 2048), mask.shape
    assert mask.dtype == np.uint8, mask.dtype
    mask = (mask / 255).astype(np.float32)
    mask = cv2_downsize(mask, downsize_factor)
    rgba = np.concatenate([rgb, mask[:, :, None]], axis=2)
    return rgba
