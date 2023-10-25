import numpy as np
from typing import Optional
from .read_write_model import Camera


def load_pinhole_camera(camera: Camera, downsize_factor: Optional[int] = None):
    assert camera.model == 'PINHOLE'
    K = np.eye(3)
    assert camera.params[2] == camera.width / 2, camera
    assert camera.params[3] == camera.height / 2, camera
    K[0, 0] = camera.params[0]
    K[1, 1] = camera.params[1]
    K[0, 2] = camera.params[2]
    K[1, 2] = camera.params[3]

    if downsize_factor is not None:
        print('applying downsize factor', downsize_factor)
        K[:2, :] /= downsize_factor
    print(K)

    return K
