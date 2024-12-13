
#!/bin/bash

source ~/.bashrc
conda activate /svl/u/yzzhang/envs/dmodel2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT=${CODE_ROOT}/imageint/third_party/nvdiffrecc

cd ${THIRD_PARTY_CODE_ROOT} || exit

SCENE=scene099_056_Vase_B0BV44B4R4_BlueBirdsYellowBirds
EXP_ID=1031

python train.py \
--config "${CODE_ROOT}/configs/nvdiffrecmc.json" \
-rm "${DATA_ROOT}/${SCENE}/final_output/blender_format_HDR" \
-o ${SCENE}/${EXP_ID}
