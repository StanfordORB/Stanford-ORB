import os
import datetime
from typing import Dict
import json
from orb.constant import PROJ_ROOT


LEADERBOARD_DIR = os.path.join(PROJ_ROOT, 'logs/leaderboard')


def load_scores(method: str) -> Dict:
    print(
        method, 'updated: ',
        datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(LEADERBOARD_DIR, 'baselines', f'{method}.json'))).strftime('%Y-%m-%d %H:%M:%S')
          )
    with open(os.path.join(LEADERBOARD_DIR, 'baselines', f'{method}.json')) as f:
        data = json.load(f)
    return data
