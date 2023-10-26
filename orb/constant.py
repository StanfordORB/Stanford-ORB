import os

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_RESOLUTION = 512

PROCESSED_SCENE_DATA_DIR = os.path.join(PROJ_ROOT, 'processed_data')

SCENES_FULL = [
    'baking_scene001', 'baking_scene002', 'baking_scene003',
    'ball_scene002', 'ball_scene003', 'ball_scene004',
    'blocks_scene002', 'blocks_scene005', 'blocks_scene006',
    'cactus_scene001', 'cactus_scene005', 'cactus_scene007',
    'car_scene002', 'car_scene004', 'car_scene006',
    'chips_scene002', 'chips_scene003', 'chips_scene004',
    'cup_scene003', 'cup_scene006', 'cup_scene007',
    'curry_scene001', 'curry_scene005', 'curry_scene007',
    'gnome_scene003', 'gnome_scene005', 'gnome_scene007',
    'grogu_scene001', 'grogu_scene002', 'grogu_scene003',
    'pepsi_scene002', 'pepsi_scene003', 'pepsi_scene004',
    'pitcher_scene001', 'pitcher_scene005', 'pitcher_scene007',
    'salt_scene004', 'salt_scene005', 'salt_scene007',
    'teapot_scene001', 'teapot_scene002', 'teapot_scene006'
]

SCENES_LIGHT = [
    "teapot_scene001",
    "grogu_scene002",
    "gnome_scene003",
    "car_scene004",
    "pitcher_scene005",
    "blocks_scene006",
    "cactus_scene007",
]


def get_scenes_from_id(id: str):
    assert id in ['light', 'full', 'example'], id
    return {
        'light': SCENES_LIGHT,
        'full': SCENES_FULL,
        'example': SCENES_LIGHT[0:1],
    }[id]
