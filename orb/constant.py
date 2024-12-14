import os

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_RESOLUTION = 512

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


### Below is irrelevant if you are evaluating your own methods
### It's for baseline methods reported in the paper

IDR_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/idr')
INVRENDER_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/invrender')
NERD_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/nerd')
NERF_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/nerfpytorch')
NERFACTOR_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/nerfactor')
NEURALPIL_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/neuralpil')
NVDIFFREC_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/nvdiffrec')
NVDIFFRECMC_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/nvdiffrecmc')
PHYSG_ROOT = os.path.join(PROJ_ROOT, 'orb/third_party/physg')

DEBUG_SAVE_DIR = os.path.join(PROJ_ROOT, 'debug')
os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

# SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data/data'
# SCENE_METADATA_DIR = '/viscam/projects/imageint/yzzhang/data/capture_scene_data/metadata'

VERSION = os.getenv('VERSION', 'extension')
assert VERSION in ['submission', 'rebuttal', 'revision', 'release', 'extension'], VERSION

if VERSION == 'extension':
    PROCESSED_SCENE_DATA_DIR = '/svl/u/yzzhang/projects/aria/processed_data'
    INPUT_RESOLUTION = 2000
    DEFAULT_SCENE_DATA_DIR = '/svl/data/Aria_DTC/ORB_eval_format_spherified'
else:
    PROCESSED_SCENE_DATA_DIR = '/viscam/projects/imageint/yzzhang/data/processed_data'
    INPUT_RESOLUTION = 2048
    DEFAULT_SCENE_DATA_DIR = '/viscam/projects/imageint/capture_scene_data/data/'
assert BENCHMARK_RESOLUTION == 512, "BENCHMARK_RESOLUTION must be 512 as hard-coded in some configs"
DOWNSIZE_FACTOR = INPUT_RESOLUTION / BENCHMARK_RESOLUTION

print(f'[INFO] INPUT_RESOLUTION: {INPUT_RESOLUTION}, BENCHMARK_RESOLUTION: {BENCHMARK_RESOLUTION}')

ALL_SCENES = [
    "scene001_obj003_baking",
    "scene001_obj008_grogu",
    "scene001_obj016_pitcher",
    "scene001_obj018_teapot",
    "scene002_obj003_baking",
    "scene002_obj008_grogu",
    "scene002_obj010_pepsi",
    "scene002_obj012_cart",
    "scene002_obj017_ball",
    "scene002_obj018_teapot",
    "scene002_obj019_blocks",
    "scene002_obj020_chips",
    "scene003_obj003_baking",
    "scene003_obj007_gnome",
    "scene003_obj008_grogu",
    "scene003_obj010_pepsi",
    "scene003_obj013_cup",
    "scene003_obj017_ball",
    "scene003_obj020_chips",
    "scene004_obj001_salt",
    "scene004_obj010_pepsi",
    "scene004_obj012_cart",
    "scene004_obj017_ball",
    "scene004_obj020_chips",
    "scene005_obj001_salt",
    "scene005_obj007_gnome",
    "scene005_obj016_pitcher",
    "scene005_obj019_blocks",
    "scene006_obj012_cart",
    "scene006_obj013_cup",
    "scene006_obj018_teapot",
    "scene006_obj019_blocks",
    "scene007_obj001_salt",
    "scene007_obj007_gnome",
    "scene007_obj013_cup",
    "scene007_obj016_pitcher"
]

SUBMISSION_SCENES = [
    "scene001_obj003_baking",
    "scene001_obj018_teapot",
    "scene002_obj008_grogu",
    "scene002_obj017_ball",
    "scene002_obj019_blocks",
    "scene002_obj020_chips",
    "scene003_obj003_baking",
    "scene003_obj007_gnome",
    "scene003_obj008_grogu",
    "scene003_obj010_pepsi",
    "scene003_obj020_chips",
    "scene004_obj010_pepsi",
    "scene004_obj012_cart",
    "scene004_obj017_ball",
    "scene005_obj001_salt",
    "scene005_obj016_pitcher",
    "scene005_obj019_blocks",
    "scene006_obj012_cart",
    "scene006_obj013_cup",
    "scene006_obj018_teapot",
    "scene007_obj001_salt",
    "scene007_obj007_gnome",
    "scene007_obj013_cup",
    "scene007_obj016_pitcher"
]

REBUTTAL_SCENES = [
    "scene001_obj021_cactus",
    "scene005_obj021_cactus",
    "scene007_obj021_cactus",
    "scene001_obj022_curry",
    "scene005_obj022_curry",
    "scene007_obj022_curry"
]

SUBMISSION_ADD_SCENES = [
    "scene001_obj008_grogu",
    "scene001_obj016_pitcher",
    "scene002_obj003_baking",
    "scene002_obj010_pepsi",
    "scene002_obj012_cart",
    "scene002_obj018_teapot",
    "scene003_obj013_cup",
    "scene003_obj017_ball",
    "scene004_obj001_salt",
    "scene004_obj020_chips",
    "scene005_obj007_gnome",
    "scene006_obj019_blocks",
]
assert tuple(ALL_SCENES) == tuple(sorted(SUBMISSION_SCENES + SUBMISSION_ADD_SCENES))

REVISION_SCENES = SUBMISSION_SCENES + REBUTTAL_SCENES
RELEASE_SCENES = ALL_SCENES + REBUTTAL_SCENES

EXTENSION_SCENES = [
    # 'scene000_000_Airplane_B097C7SHJH_WhiteBlue',
    'scene000_000_Birdhouse',
]





if __name__ == "__main__":
    print('submission scenes', len(SUBMISSION_SCENES))  # 24
    print('submission add scenes', len(SUBMISSION_ADD_SCENES)) # 12
    print('rebuttal scenes', len(REBUTTAL_SCENES))  # 6
    print('release scenes', len(RELEASE_SCENES))
