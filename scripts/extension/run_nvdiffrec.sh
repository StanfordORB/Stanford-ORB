#!/bin/bash

source ~/.bashrc
actenv dmodel2
conda activate /svl/u/yzzhang/envs/dmodel2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT=${CODE_ROOT}/imageint/third_party/nvdiffrec

cd ${THIRD_PARTY_CODE_ROOT} || exit

SCENE=scene005_003_BirdHouseRedRoofYellowWindows
EXP_ID=1105

python train.py \
--config "${CODE_ROOT}/configs/nvdiffrec.json" \
-rm "${DATA_ROOT}/${SCENE}/final_output/blender_format_HDR" \
-o ${SCENE}/${EXP_ID}
