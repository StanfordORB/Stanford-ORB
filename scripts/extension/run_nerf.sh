#!/bin/bash

source ~/.bashrc
conda activate /svl/u/yzzhang/envs/dmodel2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/nerfpytorch"

SCENE=scene000_000_Airplane_B097C7SHJH_WhiteBlue

cd ${THIRD_PARTY_CODE_ROOT} || exit

EXP_ID=1109

python run_nerf.py \
--config "${CODE_ROOT}/configs/nerf.txt" \
--expname "${SCENE}/${EXP_ID}" \
--datadir "${DATA_ROOT}/${SCENE}"
