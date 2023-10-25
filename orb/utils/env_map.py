import numpy as np
import cv2


def env_map_to_physg(env_map: np.ndarray):
    # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
    H, W = env_map.shape[:2]
    env_map = np.roll(env_map, W // 2, axis=1)
    return env_map


def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w, convention):
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(-0.5*np.pi, 1.5*np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                           axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
    if convention == 'ours':
        return envmap_remapped
    if convention == 'physg':
        # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
        envmap_remapped_physg = np.roll(envmap_remapped, W//2, axis=1)
        return envmap_remapped_physg
    if convention == 'nerd':
        # change convention from ours (Left +Z-X, Up +Y) to nerd (Left +Z+X, Up +Y)
        envmap_remapped_nerd = envmap_remapped[:,::-1,:]
        return envmap_remapped_nerd

    assert convention == 'invrender', convention
    # change convention from ours (Left +Z-X, Up +Y) to invrender (Left -X+Y, Up +Z)
    theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([np.cos(theta) * np.sin(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(phi)], axis=-1)    # [H, W, 3]
    # viewdirs = np.stack([-viewdirs[...,0], viewdirs[...,2], viewdirs[...,1]], axis=-1)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped_Inv = cv2.remap(envmap_remapped, coord_x, coord_y, cv2.INTER_LINEAR)
    return envmap_remapped_Inv
