import json
import datetime
import copy
import os
import glob
from orb.constant import PROJ_ROOT
from orb.utils.paper import load_scores
import logging


logger = logging.getLogger(__name__)
LEADERBOARD_DIR = os.path.join(PROJ_ROOT, 'logs/leaderboard')


# MAIN = True
MAIN = False

HEADER = [
    'Depth$\\downarrow$',
    'Normal$\\downarrow$',
    'Shape$\\downarrow$',
    'PSNR-H$\\uparrow$', 'PSNR-L$\\uparrow$', 'SSIM$\\uparrow$', 'LPIPS$\\downarrow$',
    'PSNR-H$\\uparrow$', 'PSNR-L$\\uparrow$', 'SSIM$\\uparrow$', 'LPIPS$\\downarrow$',
]
if not MAIN:
    HEADER.extend(['PSNR-L$\\uparrow$', 'SSIM$\\uparrow$', 'LPIPS$\\downarrow$'])

SCORES_STATS_DUMMY = {
    'shape': {'bidir_chamfer': None},
    'geometry': {'depth_mse': None, 'normal_angle': None},
    'light': {'psnr_hdr': None, 'psnr_ldr': None, 'ssim': None, 'lpips': None},
    'view': {'psnr_hdr': None, 'psnr_ldr': None, 'ssim': None, 'lpips': None},
}

METHOD_TO_STRING = {
    'idr': 'IDR~\\cite{yariv2020multiview}',
    'nerf': 'NeRF~\\cite{mildenhall2020nerf}',
    'nvdiffrec': 'NVDiffRec~\\cite{Munkberg2022nvdiffrec}',
    'nerd': 'NeRD~\\cite{boss2021nerd}',
    'nerfactor': 'NeRFactor~\\cite{zhang2021nerfactor}',
    'neuralpil': 'Neural-PIL~\\cite{boss2021neural}',
    'physg': 'PhySG~\\cite{physg2021}',
    'nvdiffrecmc': 'NVDiffRecMC~\\cite{hasselgren2022nvdiffrecmc}',
    'invrender': 'InvRender~\\cite{zhang2022invrender}',
    'singleimage': 'SI-SVBRDF~\\cite{li2020inverse}',
    'sirfs': 'SIRFS~\\cite{barron2014shape}',
    'nvdiffrec_pseudo_gt': 'NVDiffRec~\\cite{Munkberg2022nvdiffrec}\\textdagger',
    'nvdiffrecmc_pseudo_gt': 'NVDiffRecMC~\\cite{hasselgren2022nvdiffrecmc}\\textdagger',
}

COLS = [
    'nvdiffrecmc_pseudo_gt',
    'nvdiffrec_pseudo_gt',
    '\\midrule',
    'idr', 'nerf',
    '\\midrule',
    'neuralpil',
    'physg',
    'nvdiffrec',
    'nerd',
    'nerfactor',
    'invrender',
    'nvdiffrecmc',
    '\\midrule',
    'singleimage',
    'sirfs',
]


def load_latest_leaderboard():
    # with open(os.path.join(LEADERBOARD_DIR, 'latest.json')) as f:
    #     data = json.load(f)
    data = {'scores': {}}
    for method in METHOD_TO_STRING.keys():
        data['scores'][method] = load_scores(method)

    # # compute failures
    # for method in METHOD_TO_STRING.keys():
    #     fail = 0
    #     count = 0
    #     if method in ['singleimage', 'sirfs', 'nvdiffrec_pseudo_gt', 'nvdiffrecmc_pseudo_gt']:
    #         continue
    #     if method != 'nerf':
    #         continue
    #     for scene in sorted(data['scores'][method]['scores']['geometry_all'].keys()):
    #         count += 1
    #         if data['scores'][method]['scores']['geometry_all'][scene]['normal_angle'] > .9:
    #         # if data['scores'][method]['scores']['view_all'][scene]['psnr_hdr'] < 20:
    #             fail += 1
    #             print(method, scene, data['scores'][method]['scores']['geometry_all'][scene]['normal_angle'])
    #             # print(method, scene, data['scores'][method]['scores']['view_all'][scene]['psnr_hdr'])
    #             print_vcv_url(os.path.dirname(data['scores'][method]['info'][scene]['view'][0]['output_image']))
    #     print(method, fail, count, fail / count)
    # exit()

    rows = ["& " + " & ".join(HEADER) + '\\\\' + '\\midrule']
    for method in COLS:
        if method == '\\midrule':
            rows.append('\\midrule')
            continue
        if method not in data['scores'].keys():
            scores_stats = {}
        else:
            scores_stats = data['scores'][method]['scores_stats']

        if 'pseudo_gt' in method:
            scores_stats['shape'].pop('bidir_chamfer')
        for benchmark_name in scores_stats.keys():
            for metric_name in scores_stats[benchmark_name].keys():
                if metric_name == 'scene_count':
                    continue
                mean, std = scores_stats[benchmark_name][metric_name]
                if metric_name == 'bidir_chamfer':
                    # hack
                    mean = mean * 1000 * 2
                    std = std * 1000 * 2
                elif metric_name == 'depth_mse':
                    mean = mean * 1000
                    std = std * 1000
                if metric_name in ['ssim', 'lpips']:
                    mean = f"{mean:.3f}"
                    std = f"{std:.3f}"
                else:
                    mean = f"{mean:.2f}"
                    std = f"{std:.2f}"
                if not MAIN:
                    mean = f'${mean}$\\xpm' + '{' + f'{std}' + '}'
                else:
                    mean = f'${mean}$'
                scores_stats[benchmark_name][metric_name] = mean
            scores_stats[benchmark_name].pop('scene_count')

        if 'geometry' not in scores_stats.keys():
            # hack
            rows.append(METHOD_TO_STRING[method] + '\\\\')
            continue
        if 'material' not in scores_stats.keys():
            scores_stats['material'] = {}

        cols = []
        if len(scores_stats['geometry']) == 0:
            cols.append('\\multicolumn{3}{c}{N/A}')
        else:
            if 'depth_mse' not in scores_stats['geometry']:
                cols.append('N/A')
            else:
                cols.append(scores_stats['geometry']['depth_mse'])
            cols.append(scores_stats['geometry']['normal_angle'])
            if 'bidir_chamfer' not in scores_stats['shape']:
                cols.append('N/A')
            else:
                cols.append(scores_stats['shape']['bidir_chamfer'])
        if len(scores_stats['light']) == 0:
            cols.append('\\multicolumn{4}{c}{N/A}')
        else:
            cols.extend([
                scores_stats['light']['psnr_hdr'],
                scores_stats['light']['psnr_ldr'],
                scores_stats['light']['ssim'],
                scores_stats['light']['lpips'],
            ])
        if len(scores_stats['view']) == 0:
            cols.append('\\multicolumn{4}{c}{N/A}')
        else:
            cols.extend([
                scores_stats['view']['psnr_hdr'],
                scores_stats['view']['psnr_ldr'],
                scores_stats['view']['ssim'],
                scores_stats['view']['lpips'],
            ])

        if not MAIN:
            if len(scores_stats['material']) == 0:
                cols.append('\\multicolumn{3}{c}{N/A}')
            else:
                cols.extend([
                    scores_stats['material']['psnr_ldr'],
                    scores_stats['material']['ssim'],
                    scores_stats['material']['lpips'],
                ])

        rows.append(METHOD_TO_STRING[method] + ' & ' + ' & '.join(cols) + '\\\\')
    print('======')
    print('Final:')
    print('======')
    print('\n'.join(rows))


def main():
    load_latest_leaderboard()


if __name__ == '__main__':
    main()
