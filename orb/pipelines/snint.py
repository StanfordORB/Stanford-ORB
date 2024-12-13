import os
import numpy as np
from orb.pipelines.base import BasePipeline
from typing import Dict, Any, List
import logging
logger = logging.getLogger(__name__)


SNINT_ROOT = "/viscam/projects/imageint/szwu/results/snint_results_512"
SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data'


class Pipeline(BasePipeline):
    def test_new_light(self, scene: str, overwrite=True):
        return []
