import numpy as np
import mcubes
import trimesh
from typing import Callable


def clean_mesh(mesh: trimesh.Trimesh):
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float32)
    mesh_clean = components[areas.argmax()]
    return mesh_clean


def extract_mesh_from_nerf(fn: Callable[[np.ndarray], np.ndarray],
                           bounds: tuple = (1.2, 1.2, 1.2)) -> trimesh.Trimesh:
    N = 256
    t0 = np.linspace(-bounds[0], bounds[0], N + 1)  # gap = BOUND * 2 / N, mapped to arange(0, N + 2) with gap = 1 and N + 1 points
    t1 = np.linspace(-bounds[1], bounds[1], N + 1)
    t2 = np.linspace(-bounds[2], bounds[2], N + 1)
    query_pts = np.stack(np.meshgrid(t0, t1, t2), -1).astype(np.float32)
    sh = query_pts.shape
    flat = query_pts.reshape([-1, 3])

    chunk = 1024 * 64
    raw = np.concatenate([fn(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)], 0)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    sigma = np.maximum(raw[..., -1], 0.)

    threshold = 50.
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    output_mesh = trimesh.Trimesh(vertices, triangles)
    scale_mat = np.diag([1 / N * bounds[0] * 2, 1 / N * bounds[1] * 2, 1 / N * bounds[2] * 2, 1.0])
    output_mesh.apply_transform(scale_mat)
    trans_mat = trimesh.transformations.translation_matrix(-np.asarray(bounds))
    output_mesh.apply_transform(trans_mat)

    a = np.pi / 2
    s, c = np.sin(a), np.cos(a)
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, c, s, 0],
                        [0, -s, c, 0],
                        [0, 0, 0, 1]])
    output_mesh.apply_transform(rot_mat)
    rot_mat = np.array([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]])
    output_mesh.apply_transform(rot_mat)
    a = np.pi
    s, c = np.sin(a), np.cos(a)
    rot_mat = np.array([[c, 0, s, 0],
                        [0, 1, 0, 0],
                        [-s, 0, c, 0],
                        [0, 0, 0, 1]])
    output_mesh.apply_transform(rot_mat)

    a = -np.pi / 2
    s, c = np.sin(a), np.cos(a)
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, c, s, 0],
                        [0, -s, c, 0],
                        [0, 0, 0, 1]])
    output_mesh.apply_transform(rot_mat)
    return output_mesh
