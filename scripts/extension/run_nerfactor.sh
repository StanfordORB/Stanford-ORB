#!/bin/bash

source ~/.bashrc
actenv neuralpil2
conda activate /svl/u/yzzhang/envs/neuralpil2  # doesn't work on viscam7

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
PROCESSED_DATA_ROOT=/svl/u/yzzhang/projects/aria/processed_data
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/nerfactor"

SCENE=scene000_000_Birdhouse

cd ${CODE_ROOT} || exit

SCENE_DATA_DIR="${DATA_ROOT}/${SCENE}/final_output/llff_format_LDR"
PROCESSED_SCENE_DATA_DIR="${PROCESSED_DATA_ROOT}/${SCENE}/nerfactor_format"

# [INST] Run the following 2 commands for preprocessing

# python ${CODE_ROOT}/scripts/preprocess/colmap_to_nerfactor.py \
# -i $SCENE_DATA_DIR -o $PROCESSED_SCENE_DATA_DIR

# python ${CODE_ROOT}/scripts/preprocess/colmap_to_nerfactor_novel.py \
# -i $SCENE_DATA_DIR -o $PROCESSED_SCENE_DATA_DIR
# exit

# [INST] Ignore the following

cd ${THIRD_PARTY_CODE_ROOT} || exit

EXP_ID=1110

# PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_GPU_THREAD_MODE=gpu_private REPO_DIR=${THIRD_PARTY_CODE_ROOT} \
# python nerfactor/trainvali.py \
# --config="${CODE_ROOT}/configs/nerfactor_nerf.ini" \
# --config_override="data_root=${PROCESSED_SCENE_DATA_DIR},outroot=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerf/${EXP_ID}"
# exit


PREV_EXP_ID=${EXP_ID}
EXP_ID=1110
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_GPU_THREAD_MODE=gpu_private REPO_DIR=${THIRD_PARTY_CODE_ROOT} \
TF_FORCE_GPU_ALLOW_GROWTH=true \
python nerfactor/geometry_from_nerf.py \
--data_root="${PROCESSED_DATA_ROOT}" \
--trained_nerf="${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerf/${PREV_EXP_ID}/lr1e-4" \
--out_root="${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${EXP_ID}" \
--imh=512 --scene_bbox='' --occu_thres=0.5 \
--mlp_chunk=1000000
exit

PREV_EXP_ID=${EXP_ID}
EXP_ID=1110
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_GPU_THREAD_MODE=gpu_private REPO_DIR=${THIRD_PARTY_CODE_ROOT} \
python nerfactor/trainvali.py \
--config="${CODE_ROOT}/configs/nerfactor_shape.ini" \
--config_override="data_root=${PROCESSED_DATA_ROOT},data_nerf_root=${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${PREV_EXP_ID},outroot=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_shape/${EXP_ID}"

PREV_EXP_ID=${EXP_ID}
EXP_ID=1110
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_GPU_THREAD_MODE=gpu_private REPO_DIR=${THIRD_PARTY_CODE_ROOT} \
python nerfactor/trainvali.py \
--config="${CODE_ROOT}/configs/nerfactor_nerfactor.ini" \
--config_override="data_root=${PROCESSED_DATA_ROOT},data_nerf_root=${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${PREV_EXP_ID},shape_model_ckpt=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_shape/${PREV_EXP_ID}/lr1e-2/checkpoints/ckpt-2,outroot=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerfactor/${EXP_ID},vali_period=999"

PREV_EXP_ID=${EXP_ID}
EXP_ID=1110
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_GPU_THREAD_MODE=gpu_private REPO_DIR=${THIRD_PARTY_CODE_ROOT} \
TF_FORCE_GPU_ALLOW_GROWTH=true \
python nerfactor/geometry_from_nerf.py \
--data_root="${PROCESSED_DATA_DIR}" \
--trained_nerf="${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerf/${PREV_EXP_ID}/lr1e-4" \
--out_root="${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${EXP_ID}" \
--imh=512 \
--scene_bbox='' \
--occu_thres=0.5 \
--mlp_chunk=1000000