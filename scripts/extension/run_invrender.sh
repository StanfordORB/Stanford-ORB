#!/bin/bash

source ~/.bashrc
conda activate /svl/u/yzzhang/envs/dmodel2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/invrender/code"

SCENE=scene000_000_Airplane_B097C7SHJH_WhiteBlue

cd ${THIRD_PARTY_CODE_ROOT} || exit

EXP_ID=1109

# Sequentially run three training stages
stages=("IDR" "Illum" "Material")
for STAGE in "${stages[@]}"; do
    python training/exp_runner.py \
--conf "${CODE_ROOT}/configs/invrender.conf" \
--data_split_dir "${DATA_ROOT}/${SCENE}" \
--expname ${SCENE}/${EXP_ID} \
--trainstage ${STAGE}
done