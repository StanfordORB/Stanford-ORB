#!/bin/bash

source ~/.bashrc
conda activate /svl/u/yzzhang/envs/dmodel2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/physg/code"

cd ${THIRD_PARTY_CODE_ROOT} || exit

EXP_ID=1110_debug
SCENE=scene000_000_Birdhouse

python ${THIRD_PARTY_CODE_ROOT}/training/exp_runner.py \
--conf ${CODE_ROOT}/configs/physg.conf \
--data_split_dir ${DATA_ROOT}/${SCENE} \
--expname ${SCENE}/${EXP_ID} \
--nepoch 2000 --max_niter 200001 \
--gamma 1.0  --exposure 0.5
