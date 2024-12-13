import os
from imageint.constant import DEFAULT_SCENE_DATA_DIR
from typing import List


def get_novel_scenes(scene: str, data_root: str = DEFAULT_SCENE_DATA_DIR) -> List[str]:
    with open(os.path.join(data_root, scene, 'final_output/llff_format_HDR/novel_id.txt'), 'r') as f:
        novel_ids = f.read().splitlines()
    return list(sorted(set(tuple(novel_id.split('/'))[0] for novel_id in novel_ids)))
