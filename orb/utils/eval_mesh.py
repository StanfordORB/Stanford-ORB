import numpy as np
import trimesh
from scipy.spatial import cKDTree as KDTree
import logging

logger = logging.getLogger(__name__)

target_volume = 900  # roughly corresponds to a witdth of 10cm
num_points_align = 1000
max_iterations = 100
cost_threshold = 0
num_points_chamfer = 30000


def sample_surface_point(mesh, num_points, even=False):
    if even:
        sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
        while len(sample_points) < num_points:
            more_sample_points, indexes = trimesh.sample.sample_surface_even(mesh, count=num_points)
            sample_points = np.concatenate([sample_points, more_sample_points], axis=0)
    else:
        sample_points, indexes = trimesh.sample.sample_surface(mesh, count=num_points)
    return sample_points[:num_points]


def load_mesh(fpath: str) -> trimesh.Trimesh:
    if fpath.endswith('.npz'):
        mesh_npz = np.load(fpath)
        verts = mesh_npz['verts']
        faces = mesh_npz['faces']
        faces = np.concatenate((faces, faces[:, list(reversed(range(faces.shape[-1])))]), axis=0)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    else:
        mesh = trimesh.load_mesh(fpath)
    return mesh


# https://github.com/facebookresearch/DeepSDF/blob/main/deep_sdf/metrics/chamfer.py
def compute_trimesh_chamfer(gt_points, gen_mesh, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """
    if gen_mesh is None:
        gen_points_sampled = np.zeros((num_mesh_samples, 3))
    else:
        gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    # only need numpy array of points
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer


def compute_shape_score(output_mesh, target_mesh):
    if output_mesh is None:
        logger.error('output mesh not found')
        return {}
    try:
        mesh_result = load_mesh(output_mesh)
    except ValueError:
        import traceback; traceback.print_exc()
        mesh_result = None
    mesh_scan = load_mesh(target_mesh)
    gt_to_gen_chamfer, gen_to_gt_chamfer = compute_trimesh_chamfer(mesh_scan, mesh_result, num_points_chamfer)
    bidir_chamfer = (gt_to_gen_chamfer + gen_to_gt_chamfer) / 2.
    return {'bidir_chamfer': bidir_chamfer}


def calculate_scale(mesh, target_volume, method='volume'):
    if method == 'bounding_box':
        width, height, length = mesh.extents
        bounding_box_volume = (width * height * length)
        scale = (target_volume / bounding_box_volume)**(1/3)
    elif method == 'volume':
        voxel_length = mesh.extents.min() /100
        voxel = mesh.voxelized(voxel_length).fill()
        voxel_volume = voxel.volume
        scale = (target_volume / voxel_volume)**(1/3)
    return scale
