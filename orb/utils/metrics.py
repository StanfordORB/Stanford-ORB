import cv2
import numpy as np
from .preprocess import rgb_to_srgb as _tonemap_srgb


def mse_to_psnr(mse):
    """Compute PSNR given an MSE (we assume the maximum pixel value is 1)."""
    return -10. / np.log(10.) * np.log(mse)


def calc_PSNR(img_pred, img_gt, mask_gt,
              max_value, use_gt_median, tonemapping, scale_invariant,
              divide_mask=True):
    # make sure img_pred, img_gt are linear
    '''
        calculate the PSNR between the predicted image and ground truth image.
        a scale is optimized to get best possible PSNR.
        images are clip by max_value_ratio.
        params:
        img_pred: numpy.ndarray of shape [H, W, 3]. predicted HDR image.
        img_gt: numpy.ndarray of shape [H, W, 3]. ground truth HDR image.
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
        max_value: Float. the maximum value of the ground truth image clipped to.
            This is designed to prevent the result being affected by too bright pixels.
        tonemapping: Bool. Whether the images are tone-mapped before comparion.
        divide_mask: Bool. Whether the mse is divided by the foreground area.
    '''
    if mask_gt.ndim == 3:
        mask_gt = mask_gt[..., 0]
    if mask_gt.dtype == np.float32:
        mask_gt = (mask_gt * 255).clip(0, 255).astype(np.uint8)
    else:
        import ipdb; ipdb.set_trace()
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask_gt = cv2.erode(mask_gt, kernel)
    mask_gt = (mask_gt > 127).astype(np.float32)

    img_pred = img_pred * mask_gt[..., None]
    img_gt = img_gt * mask_gt[..., None]
    img_gt[img_gt < 0] = 0
    if use_gt_median:  # image in linear space are usually too dark, need to re-normalize
        if img_gt.clip(0, 1).mean() > 1e-8:
            gt_median = _tonemap_srgb(img_gt.clip(0, 1)).mean() / img_gt.clip(0, 1).mean()
            img_pred = img_pred * gt_median
            img_gt = img_gt * gt_median

    if scale_invariant:
        img_pred_pixels = img_pred[np.where(mask_gt > 0.5)]
        img_gt_pixels = img_gt[np.where(mask_gt > 0.5)]
        for c in range(3):
            if (img_pred_pixels[:, c] ** 2).sum() <= 1e-6:
                img_pred_pixels[:, c] = np.ones_like(img_pred_pixels[:, c])
                # import ipdb; ipdb.set_trace()
        scale = (img_gt_pixels * img_pred_pixels).sum(axis=0) / (img_pred_pixels ** 2).sum(axis=0)
        assert scale.shape == (3,), scale.shape
        if (scale < 0).any():
            import ipdb; ipdb.set_trace()
        if (img_pred < 0).any():
            import ipdb; ipdb.set_trace()
        if (img_gt < 0).any():
            import ipdb; ipdb.set_trace()
        img_pred = scale * img_pred
        # if not tonemapping:
    #     imageio.imsave("./rescaled.exr", img_pred)
    #     imageio.imsave("./rescaled_gt.exr", img_gt)

    # clip the prediction and the gt img by the maximum_value
    img_pred = np.clip(img_pred, 0, max_value)
    img_gt = np.clip(img_gt, 0, max_value)

    if tonemapping:
        img_pred = _tonemap_srgb(img_pred)
        img_gt = _tonemap_srgb(img_gt)
        # imageio.imsave("./rescaled.png", (img_pred*255).clip(0,255).astype(np.uint8))
        # imageio.imsave("./rescaled_gt.png", (img_gt*255).clip(0,255).astype(np.uint8))

    if not divide_mask:
        mse = ((img_pred - img_gt) ** 2).mean()
        lb = ((np.ones_like(img_gt) * .5 * mask_gt[:, :, None] - img_gt) ** 2).mean()
    else:
        mse = ((img_pred - img_gt) ** 2).sum() / mask_gt.sum()
        lb = ((np.ones_like(img_gt) * .5 * mask_gt[:, :, None] - img_gt) ** 2).sum() / mask_gt.sum()
    out = mse_to_psnr(mse)
    lb = mse_to_psnr(lb)
    out = max(out, lb)
    return out, img_pred, img_gt


def erode_mask(mask, target_size):
    if mask.ndim == 3:
        mask = mask[...,0]
    if mask.dtype == np.float32:
        mask = (mask*255).clip(0, 255).astype(np.uint8)
    # shrink mask by a small margin to prevent inaccurate mask boundary.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    if target_size is not None:
        mask = cv2.resize(mask, (target_size, target_size))
    return (mask > 127).astype(np.float32)


def calc_normal_distance(normal_pred, normal_gt, mask_gt):
    '''
        params:
        normal_pred: numpy.ndarray of shape [H, W, 3].
        normal_gt: numpy.ndarray of shape [H, W, 3].
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
    '''
    assert normal_pred.shape == normal_gt.shape, (normal_pred.shape, normal_gt.shape)

    eps = 1e-6
    normal_pred = normal_pred / (np.linalg.norm(normal_pred, axis=-1, keepdims=True) + eps)
    normal_gt = normal_gt / (np.linalg.norm(normal_gt, axis=-1, keepdims=True) + eps)

    mask_gt = erode_mask(mask_gt, normal_pred.shape[0])
    cos_dist = (1-(normal_pred * normal_gt).sum(axis=-1)) * mask_gt
    return float(cos_dist.sum() / mask_gt.sum())


def calc_depth_distance(depth_pred, depth_gt, mask_gt):
    '''
        params:
        depth_pred: numpy.ndarray of shape [H, W].
        depth_gt: numpy.ndarray of shape [H, W].
        mask_gt: numpy.ndarray of shape [H, W]. ground truth foreground mask.
    '''
    assert depth_pred.shape == depth_gt.shape
    mask_gt = erode_mask(mask_gt, depth_pred.shape[0])
    depth_gt_masked = depth_gt[np.where(mask_gt>0.5)]
    depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
    if (depth_pred_masked ** 2).sum() <= 1e-6:
        depth_pred = np.ones_like(depth_gt) # np.maximum(depth_pred, .99 * np.min(depth_gt_masked) * np.ones_like(depth_pred))
        depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
    scale = (depth_gt_masked * depth_pred_masked).sum() / (depth_pred_masked**2).sum()
    depth_pred = scale * depth_pred
    return float((((depth_pred - depth_gt)**2) * mask_gt).mean())


def calc_depth_distance_per_scene(depth_pred_list, depth_gt_list, mask_gt_list):
    # depth_pred, depth_gt, mask_gt: list[Float[np.ndarray, "H W"]
    assert len(depth_pred_list) == len(depth_gt_list) == len(mask_gt_list), (
        len(depth_pred_list), len(depth_gt_list), len(mask_gt_list)
    )
    scale_nom_sum = 0
    scale_denom_sum = 0
    for depth_pred, depth_gt, mask_gt in zip(depth_pred_list, depth_gt_list, mask_gt_list):
        assert depth_pred.shape == depth_gt.shape
        mask_gt = erode_mask(mask_gt, depth_pred.shape[0])
        depth_gt_masked = depth_gt[np.where(mask_gt>0.5)]
        depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
        if (depth_pred_masked ** 2).sum() <= 1e-6:
            depth_pred = np.ones_like(depth_gt) # np.maximum(depth_pred, .99 * np.min(depth_gt_masked) * np.ones_like(depth_pred))
            depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
        scale_nom = (depth_gt_masked * depth_pred_masked).sum()
        scale_denom = (depth_pred_masked**2).sum()
        scale_nom_sum += scale_nom
        scale_denom_sum += scale_denom

    scale = scale_nom_sum / scale_denom_sum
    out = []
    for depth_pred, depth_gt, mask_gt in zip(depth_pred_list, depth_gt_list, mask_gt_list):
        mask_gt = erode_mask(mask_gt, depth_pred.shape[0])
        depth_pred_masked = depth_pred[np.where(mask_gt>0.5)]
        if (depth_pred_masked ** 2).sum() <= 1e-6:
            depth_pred = np.ones_like(depth_gt) # np.maximum(depth_pred, .99 * np.min(depth_gt_masked) * np.ones_like(depth_pred))
        depth_pred = scale * depth_pred
        out.append(float((((depth_pred - depth_gt) ** 2) * mask_gt).mean()))
    return np.mean(out)
