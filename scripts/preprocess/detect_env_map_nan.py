import os
from typing import List, Dict
import json
from pathlib import Path
import numpy as np
import glob
from orb.constant import ALL_SCENES, PROJ_ROOT
from orb.utils.preprocess import load_rgb_exr


SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'
ERROR_METADATA_PATH = os.path.join(PROJ_ROOT, 'error_metadata.json')
# http://vcv.stanford.edu/viscam/projects/imageint/yzzhang/imageint/error_metadata.json


def main():
    error_files: List[Dict[str, str]] = []
    for scene in ALL_SCENES:
        path_pattern = os.path.join(SCENE_DATA_DIR, scene, 'final_output/llff_format_HDR/env_map/*.exr')
        for path in glob.glob(path_pattern):
            arr = load_rgb_exr(path)
            if np.isnan(arr).any():
                error_files.append({'scene': scene, 'path': path, 'filename': Path(path).stem})
    with open(ERROR_METADATA_PATH, 'w') as f:
        json.dump({'env_map_nan': error_files}, f, indent=4)
    print(ERROR_METADATA_PATH)


if __name__ == "__main__":
    main()
