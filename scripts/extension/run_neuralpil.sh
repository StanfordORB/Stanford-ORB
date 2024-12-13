#!/bin/bash

source ~/.bashrc
actenv neuralpil2
conda activate /svl/u/yzzhang/envs/neuralpil2

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/neuralpil"

cd ${THIRD_PARTY_CODE_ROOT} || exit

SCENE=scene000_000_Airplane_B097C7SHJH_WhiteBlue
EXP_ID=1031

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python train_neural_pil.py \
--datadir "${DATA_ROOT}/${SCENE}" \
--basedir ./logs/ --expname ${SCENE}/${EXP_ID} --gpu 0 \
--config ${CODE_ROOT}/configs/neuralpil.txt
