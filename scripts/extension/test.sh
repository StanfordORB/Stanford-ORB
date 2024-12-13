#!/bin/bash

source ~/.bashrc

CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
cd ${CODE_ROOT} || exit

actenv dmodel2
conda activate /svl/u/yzzhang/envs/dmodel2

# python scripts/test/idr.py
# python scripts/test/invrender.py
# python scripts/test/nerf.py
# python scripts/test/nvdiffrec.py
# python scripts/test/nvdiffrecmc.py
python scripts/test/physg.py
exit

# Compute shape
actenv pytorch3d  # [INST] a conda env with pytorch3d installed
NO_SCORE_VIEW=1 NO_SCORE_LIGHT=1 NO_SCORE_MATERIAL=1 NO_SCORE_SHAPE=1 python scripts/test/nvdiffrec.py
NO_SCORE_VIEW=1 NO_SCORE_LIGHT=1 NO_SCORE_MATERIAL=1 NO_SCORE_SHAPE=1 python scripts/test/nvdiffrecmc.py
exit

# Compute depth and normal
source /viscam/u/zhengfei/anaconda3/etc/profile.d/conda.sh
conda activate dmodel  # [INST] this corresponds to conda env `dmodel3` from README
NO_SCORE_SHAPE=1 python scripts/test/nvdiffrec.py
NO_SCORE_SHAPE=1 python scripts/test/nvdiffrecmc.py


actenv2 neuralpil2
conda activate /svl/u/yzzhang/envs/neuralpil2
# python scripts/test/nerd.py
# python scripts/test/neuralpil.py
# exit

actenv dmodel2
conda activate /svl/u/yzzhang/envs/dmodel2
# python scripts/test/nerd.py
# python scripts/test/neuralpil.py