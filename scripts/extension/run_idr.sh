#!/bin/bash

source ~/.bashrc
conda activate /svl/u/yzzhang/envs/dmodel2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/idr/code"

SCENE=scene000_000_Airplane_B097C7SHJH_WhiteBlue

cd ${THIRD_PARTY_CODE_ROOT} || exit

EXP_NAME=1108

python training/exp_runner.py \
--conf ${CODE_ROOT}/configs/idr.conf \
--expname _${EXP_NAME} \
--scan_id ${SCENE} \
--data_dir ${DATA_ROOT}
