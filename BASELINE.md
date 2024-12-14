### Benchmarks

Below contains the information for preprocessing, training, and testing baseline methods from the paper. 

#### Installation

Training each baseline require one (and only one) of the two environments:

Environment `dmodel`, created following [NVDiffRec](https://github.com/NVlabs/nvdiffrec?tab=readme-ov-file#installation):
```bash
conda env create -f envs/dmodel.yml
```
Environment `neuralpil`, created following [Neural-PIL](https://github.com/cgtuebingen/Neural-PIL?tab=readme-ov-file#setup):
```bash
conda env create -f envs/neuralpil.yml
```

Testing NVDiffRec and NVDiffRecMC requires the following environment:

Environment `dmodel3`, created following [NVDiffRec](https://github.com/zju3dv/d-model?tab=readme-ov-file#installation), with `pytorch3d` installed:

```bash
conda env create -f envs/dmodel3.yml
```

In each environment, also install a util package:
```bash
pip install git+https://github.com/zzyunzhi/tu2
```

#### Training Scripts


|No.|Method|Format| Script                                    |
|---|---|---|-------------------------------------------|
|1|[IDR](https://github.com/lioryariv/idr)|llff_format_LDR| [Script](./scripts/extension/run_idr.sh)         |
|2|[PhySG](https://github.com/Kai-46/PhySG)|llff_format_HDR| [Script](./scripts/extension/run_physg_slurm.sh)       |
|3|[InvRender](https://github.com/zju3dv/InvRender)|blender_format_HDR| [Script](./scripts/extension/run_invrender.sh)   |
|4|[NeRD](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition/)|blender_format_LDR| [Script](./scripts/extension/run_nerd_slurm.sh)        |
|5|[Neural-PIL](https://github.com/cgtuebingen/Neural-PIL)|blender_format_LDR| [Script](./scripts/extension/run_neuralpil_slurm.sh)   |
|6|[NeRF](https://github.com/yenchenlin/nerf-pytorch)|blender_format_LDR| [Script](./scripts/extension/run_nerf_slurm.sh)        |
|7|[NeRFactor](https://github.com/google/nerfactor)|blender_format_LDR| [Script](./scripts/extension/run_nerfactor_slurm.sh)   |
|8|[NVDiffRec](https://github.com/NVlabs/nvdiffrec)|blender_format_HDR| [Script](./scripts/extension/run_nvdiffrec_slurm.sh)   |
|9|[NVDiffRecMC](https://github.com/NVlabs/nvdiffrecmc)|blender_format_HDR| [Script](./scripts/extension/run_nvdiffrecmc_slurm.sh) |

Adapt the scripts listed above. You need modify `DATA_ROOT` to be your data path, `CODE_ROOT` to be the path to this codebase, `SCENE` to be the name of the folder name of the scene you want to train, and `EXP_ID` to be your custom experiment ID which will be used to identify experiments during evaluation. The data format required is listed [here](./scripts/extension/README.md#data-format).


#### Testing Scripts

##### Data Preparation
Assume data are stored in `"my/data/path"`. It will be the same as `DATA_ROOT` used in the training scripts above. 
Data paths are configured in [constant.py](./imageint/constant.py). 
Within this file, do the following:
1. Change `EXTENSION_SCENES` to the scenes you want to test.
2. Change `DEFAULT_SCENE_DATA_DIR` under the if clause `if VERSION == "extension"` to `"my/data/path`. 
3. Change `PROCESSED_SCENE_DATA_DIR` (output of preprocessing) to your desired path. 

Adapt the following script for preprocessing: [preprocess_slurm.sh](./scripts/extension/preprocess_slurm.sh).

##### Testing
Adapt the following script for testing: [test.sh](./scripts/extension/test.sh).
Results will be saved under `imageint/logs/leaderboard/baselines/`. 

Each testing script used in `test.sh` corresponds to a baseline method. It evokes a pipeline file, e.g., [mymethod.py](./imageint/pipelines/mymethod.py). Running it does the following:
1. Test view synthesis with `test_new_view`. 
2. Test relighting with `test_new_light`.
3. Test depth and normal with `test_geometry`.
4. Test material with `test_material`.
5. Test mesh output with `test_shape`. 

In each of the pipeline class methods 1) computes and saves the baseline outputs, e.g., predicted RGB images under test views, as local files, and 2) returns the local file paths. Step 1) will be skipped if you run the test script again, unless you turn on the `OVERWRITE*` flags from [this](./imageint/utils/test.py) file. 

#### Custom Method
To evaluate your method, follow the steps below:
1. Adapt [this](./imageint/pipelines/mymethod.py) class which handles the test-time pipeline to your method. Examples of the implementation can be found under [here](./imageint/pipelines/).
2. Run ```python scripts/test/my_method.py```
3. Outputs will be saved to `imageint/logs/leaderboard/baselines/my_method.json`, where `score_stats` contains the mean and standard deviation for all metrics averaged across scenes. If you are running multiple methods, results for all methods will be aggregated to `imageint/logs/leaderboard/baselines/latest.json`. 
