from PIL import Image
import json
import argparse
from typing import List
import os
import numpy as np
from tqdm import tqdm
from orb.utils.preprocess import load_rgb_png, load_mask_png
from orb.constant import BENCHMARK_RESOLUTION, PROCESSED_SCENE_DATA_DIR, INPUT_RESOLUTION


def main(input_dir: str, outroot: str):
    with open(os.path.join(input_dir, 'novel_id.txt'), 'r') as f:
        novel_ids: List[str] = f.read().splitlines()
    imgs = []
    img_paths = []
    for novel_id in novel_ids:
        img_path = os.path.join(input_dir, f'../../../{novel_id.split("/")[0]}/final_output/llff_format_LDR/images/{novel_id.split("/")[1]}')
        img = load_rgb_png(img_path, downsize_factor=INPUT_RESOLUTION / BENCHMARK_RESOLUTION)
        mask_path = os.path.join(input_dir, f'../../../{novel_id.split("/")[0]}/final_output/llff_format_LDR/masks/{novel_id.split("/")[1]}')
        mask = load_mask_png(mask_path, downsize_factor=INPUT_RESOLUTION / BENCHMARK_RESOLUTION)
        img = np.concatenate([img, mask[:, :, None]], axis=2)
        assert img.shape == (BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION, 4), img.shape
        imgs.append(img)
        img_paths.append(img_path)
    imgs = np.stack(imgs, axis=0)  # 70, 512, 512, 4
    ind_novel = list(range(len(novel_ids)))

    with open(os.path.join(input_dir, '../blender_format_LDR/transforms_novel.json')) as f:
        blender_transforms_novel = json.load(f)

    view_folder = '{mode}_{i:03d}'
    # novel_json = os.path.join(outroot, 'transforms_novel.json')
    novel_meta = {'camera_angle_x': None, 'frames': []}
    for vi, i in enumerate(ind_novel):
        view_folder_ = view_folder.format(mode='novel', i=vi)
        os.makedirs(os.path.join(outroot, view_folder_), exist_ok=True)
        # Write image
        img = imgs[i, :, :, :]
        Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8)).save(os.path.join(outroot, view_folder_, 'rgba.png'))

        frame_meta_blender = blender_transforms_novel['frames'][vi]
        if os.path.basename(frame_meta_blender['file_path'] + '.png') != os.path.basename(img_paths[i]):
            print(frame_meta_blender['file_path'] + '.png', img_paths[i])
            import ipdb; ipdb.set_trace()
        c2w = np.array(frame_meta_blender['transform_matrix'])

        frame_meta = {
            'file_path': './%s/rgba' % view_folder_, 'rotation': 0,
            'transform_matrix': c2w.tolist()}
        novel_meta['frames'].append(frame_meta)
        # Write this frame's metadata to the view folder
        frame_meta = {
            'cam_angle_x': frame_meta_blender['camera_angle_x'],
            'cam_transform_mat': ','.join(str(x) for x in c2w.ravel()),
            'envmap': '', 'envmap_inten': 0, 'imh': img.shape[0],
            'imw': img.shape[1], 'scene': '', 'spp': 0,
            'original_path': img_paths[i]}
        with open(os.path.join(outroot, view_folder_, 'metadata.json'), 'w') as f:
            json.dump(frame_meta, f, indent=4)
    # with open(novel_json, 'w') as f:
    #     json.dump(novel_meta, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.input_dir, args.output_dir)
