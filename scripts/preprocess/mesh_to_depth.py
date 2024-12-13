# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import ImageFile
import shutil
import pickle
import numpy as np
import pytorch3d
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    HardPhongShader,
    SoftPhongShader,
)
import imageio
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
        cameras = kwargs.get("cameras", self.cameras)
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        # blend_params = kwargs.get("blend_params", self.blend_params)
        # znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        # zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        # images = softmax_rgb_blend(
        #     pixel_normals, fragments, blend_params, znear=znear, zfar=zfar
        # )
        return pixel_normals
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--pose_prior', type=str, default=None)
    parser.add_argument('--resize', type=int, default=1024)
    parser.add_argument('--pad', type=int, default=50, help='how much padding is add to the image')
    FLAGS = parser.parse_args()
    device = "cuda"
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    verts, faces, _ = load_obj(FLAGS.base_mesh)
    faces_idx = faces.verts_idx.to(torch.int64)
    mesh = load_objs_as_meshes([FLAGS.base_mesh], device=device)
    # look_at_view_transform(2.7, 0, 180)
    # mesh = Meshes(
    #     verts=[verts],
    #     faces=[faces_idx],
    # )
    camera_dict = pickle.load(open(FLAGS.pose_prior, "rb"))
    normal_output_dir = os.path.join(FLAGS.out_dir, "normal_maps")
    z_output_dir = os.path.join(FLAGS.out_dir, "z_maps")
    os.makedirs(normal_output_dir, exist_ok=True)
    os.makedirs(z_output_dir, exist_ok=True)
    all_img_list = [x.strip() for x in open(os.path.join(FLAGS.out_dir, "../../outputs_for_nvdiffrec/view_imgs.txt"), "r").readlines()]
    test_img_list = ["%04d.png"%int(x.strip()) for x in open(os.path.join(FLAGS.out_dir, "../../test_id.txt"), "r").readlines()]
    for it, c2w in tqdm.tqdm(enumerate(camera_dict['cam_c2w'])):
        img_name = all_img_list[it]
        if img_name not in test_img_list:
            continue
        original_c2w = c2w.clone().cpu().detach().numpy()
        c2w[:,1:2] *= -1
        c2w[:,2:3] *= -1
        w2c = torch.linalg.inv(c2w)
        R = w2c[None,:3,:3].to(device)
        T = w2c[None,:3,3].to(device)
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = T.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1
        fov = camera_dict['cam_focal'][it] # * 180 / np.pi
        focal_ratio = 1 / np.tan(fov / 2)
        focal_ratio = focal_ratio / (FLAGS.resize / (FLAGS.resize-2*FLAGS.pad))
        fov = 2 * np.arctan(1 / focal_ratio)
        fov = fov * 180 / np.pi
        cameras = FoVPerspectiveCameras(device=device, R=R_pytorch3d, T=T_pytorch3d, fov=fov)
        raster_settings = RasterizationSettings(
            image_size=FLAGS.resize,
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
        normal_map = normal_map.squeeze().cpu().numpy() # HxWx3
        normal_map = normal_map / (np.linalg.norm(normal_map, axis=-1, keepdims=True)+1e-9)
        normal_map_flat = normal_map.reshape(-1, 3).T
        normal_map_flat = original_c2w[:3,:3].T @ normal_map_flat
        normal_map = normal_map_flat.T.reshape(FLAGS.resize, FLAGS.resize, 3)
        normal_map_vis = ((normal_map+1)/2 * 255).clip(0, 255).astype(np.uint8)
        # Rasterize the mesh to get the fragments
        fragments = rasterizer(mesh)
        # Extract the depth from the fragments and invert it for visualization
        depth_map = fragments.zbuf[0, ..., :1].squeeze().cpu().numpy()
        depth_map = depth_map * (depth_map != -1)  # set background to 0
        depth_map_vis = depth_map - depth_map.min()
        depth_map_vis = depth_map_vis / depth_map_vis.max()
        depth_map_vis = ((1.0 - depth_map_vis)*255).clip(0, 255).astype(np.uint8)  # invert depth map
        # # Display the depth map
        # # plt.figure(figsize=(10, 10))
        # # plt.imshow(depth_map, cmap='gray')
        # # plt.show()
        # depth_map = (depth_map * 255).clip(0, 255).astype(np.uint8)
        # imageio.imsave("./temp.png", depth_map)
        # # imageio.imsave("./temp_img.png", (images[0, ..., :3].cpu().numpy()* 255).clip(0, 255).astype(np.uint8))
        # imageio.imsave("./temp_img.png", (normal_map[..., :3].cpu().numpy()* 255).clip(0, 255).astype(np.uint8))
        np.save(os.path.join(z_output_dir, img_name.replace(".png", ".npy")), depth_map)
        np.save(os.path.join(normal_output_dir, img_name.replace(".png", ".npy")), normal_map)
        imageio.imsave(os.path.join(z_output_dir, img_name), depth_map_vis)
        imageio.imsave(os.path.join(normal_output_dir, img_name), normal_map_vis)