import numpy as np
import imageio
import os
# from tu.configs import get_attrdict
from tu2.ppp import get_attrdict
from orb.datasets.nvdiffrecmc import DatasetCapture
from orb.constant import BENCHMARK_RESOLUTION
from PIL import Image
import torch
import torch.nn as nn
from PIL import ImageFile

import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)
from pytorch3d.renderer.blending import (
    BlendParams,
)
from pytorch3d.ops import interpolate_face_attributes

ImageFile.LOAD_TRUNCATED_IMAGES = True
RADIUS = 3.0


class NormalShader(nn.Module):
    def __init__(
        self,
        device = "cpu",
        cameras = None,
        lights = None,
        materials = None,
        blend_params = None,
    ) -> None:
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        return self
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        # cameras = kwargs.get("cameras", self.cameras)
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            # fragments.pix_to_face, ones, faces_normals
            fragments.pix_to_face, fragments.bary_coords, faces_normals
        )
        # blend_params = kwargs.get("blend_params", self.blend_params)
        # znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        # zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        # images = softmax_rgb_blend(
        #     pixel_normals, fragments, blend_params, znear=znear, zfar=zfar
        # )
        return pixel_normals


def render_geometry(data_config, mesh_dir, out_dir):
    FLAGS = get_attrdict({
        "spp": 1,
        "train_res": [BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION],
        "cam_near_far": [
            0.1,
            1000.0
        ],
        "pre_load": True,
        'background': 'white',
    })
    dataset_validate = DatasetCapture(data_config, FLAGS)
    # fov = camera_dict['cam_focal'][it]  # * 180 / np.pi
    fov = dataset_validate.cfg['camera_angle_x']
    focal_ratio = 1 / np.tan(fov / 2)
    # focal_ratio = focal_ratio / (FLAGS.resize / (FLAGS.resize - 2 * FLAGS.pad))
    fov = 2 * np.arctan(1 / focal_ratio)
    fov = fov * 180 / np.pi
    device = "cuda"
    os.makedirs(out_dir, exist_ok=True)
    base_mesh = os.path.join(mesh_dir, 'mesh.obj')
    verts, faces, _ = load_obj(base_mesh)
    # faces_idx = faces.verts_idx.to(torch.int64)
    mesh = load_objs_as_meshes([base_mesh], device=device)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)
    for it, target in enumerate(dataloader_validate):
        c2w = torch.linalg.inv(target['mv'].cuda().squeeze(0))
        original_c2w = c2w.clone().cpu().detach().numpy()
        c2w[:, 1:2] *= -1
        c2w[:, 2:3] *= -1
        w2c = torch.linalg.inv(c2w)
        R = w2c[None, :3, :3].to(device)
        T = w2c[None, :3, 3].to(device)
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = T.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1
        cameras = FoVPerspectiveCameras(device=device, R=R_pytorch3d, T=T_pytorch3d, fov=fov)
        raster_settings = RasterizationSettings(
            image_size=BENCHMARK_RESOLUTION, # FLAGS.resize,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        # Create a rasterizer using the settings
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=NormalShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )
        normal_map = renderer(mesh.extend(len(cameras)))
        normal_map = normal_map.squeeze().cpu().numpy()  # HxWx3
        normal_map = normal_map / (np.linalg.norm(normal_map, axis=-1, keepdims=True) + 1e-9)
        normal_map_flat = normal_map.reshape(-1, 3).T
        normal_map_flat = original_c2w[:3, :3].T @ normal_map_flat
        normal_map = normal_map_flat.T.reshape(BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 3)
        # Rasterize the mesh to get the fragments
        fragments = rasterizer(mesh)
        # Extract the depth from the fragments and invert it for visualization
        depth_map = fragments.zbuf[0, ..., :1].squeeze().cpu().numpy()
        # depth_map = depth_map * (depth_map != -1)  # set background to 0
        img_name = f"{it:06d}.png"
        np.save(os.path.join(out_dir, 'depth_' + img_name.replace(".png", ".npy")), depth_map)
        np.save(os.path.join(out_dir, 'normal_' + img_name.replace(".png", ".npy")), normal_map)
        imageio.imwrite(os.path.join(out_dir, 'depth_' + img_name.replace(".png", ".exr")), depth_map)
        imageio.imwrite(os.path.join(out_dir, 'normal_' + img_name.replace(".png", ".exr")), normal_map)

        Image.fromarray(((normal_map + 1) / 2 * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out_dir, 'normal_' + img_name))
        depth_map_vis = depth_map.copy()
        depth_map_vis -= depth_map_vis[depth_map_vis != -1].min()
        depth_map_vis /= depth_map_vis[depth_map != -1].max()
        depth_map_vis = depth_map_vis.clip(0, 1)
        Image.fromarray(((1 - depth_map_vis) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(out_dir, 'depth_' + img_name))
