#!/bin/bash

source ~/.bashrc
actenv py311

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
PROCESSED_DATA_ROOT=/svl/u/yzzhang/projects/aria/processed_data
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/nerfactor"

SCENE=scene000_000_Birdhouse

SCENE_DATA_DIR="${DATA_ROOT}/${SCENE}/final_output/llff_format_LDR"
PROCESSED_SCENE_DATA_DIR="${PROCESSED_DATA_ROOT}/${SCENE}/nerfactor_format"

# [INST] Follow run_nerfactor.sh for preprocessing

EXP_ID=1110

# [INST] Run each of the following 4 commands sequentially

# EXTRA_ENV_VARS=""
# COMMAND="python nerfactor/trainvali.py \
# --config=${CODE_ROOT}/configs/nerfactor_nerf.ini \
# --config_override='data_root=${PROCESSED_SCENE_DATA_DIR},outroot=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerf/${EXP_ID}'"

# EXTRA_ENV_VARS="TF_FORCE_GPU_ALLOW_GROWTH=true"
# COMMAND="python nerfactor/geometry_from_nerf.py \
# --data_root=${PROCESSED_SCENE_DATA_DIR} \
# --trained_nerf=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerf/${EXP_ID}/lr1e-4 \
# --out_root=${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${EXP_ID} \
# --imh=512 --scene_bbox='' --occu_thres=0.5 \
# --mlp_chunk=1000000"

# # [INST] DEBUGGING ONLY!!! This stage is too slow
# EXTRA_ENV_VARS="TF_FORCE_GPU_ALLOW_GROWTH=true"
# COMMAND="python nerfactor/geometry_from_nerf.py \
# --data_root=${PROCESSED_SCENE_DATA_DIR} \
# --trained_nerf=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerf/${EXP_ID}/lr1e-4 \
# --out_root=${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${EXP_ID} \
# --imh=512 --scene_bbox='' --occu_thres=0.5 \
# --mlp_chunk=1000000 \
# --light_h 1"

EXTRA_ENV_VARS=""
COMMAND="python nerfactor/trainvali.py \
--config='${CODE_ROOT}/configs/nerfactor_shape.ini' \
--config_override='data_root=${PROCESSED_SCENE_DATA_DIR},data_nerf_root=${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${EXP_ID},outroot=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_shape/${EXP_ID}'"

# EXTRA_ENV_VARS=""
# COMMAND="python nerfactor/trainvali.py \
# --config='${CODE_ROOT}/configs/nerfactor_nerfactor.ini' \
# --config_override='data_root=${PROCESSED_SCENE_DATA_DIR},data_nerf_root=${THIRD_PARTY_CODE_ROOT}/output/surf/${SCENE}/${EXP_ID},shape_model_ckpt=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_shape/${EXP_ID}/lr1e-2/checkpoints/ckpt-2,outroot=${THIRD_PARTY_CODE_ROOT}/output/train/${SCENE}_nerfactor/${EXP_ID},vali_period=999'"

python -m tu.sbatch.sbatch_sweep --time 72:00:00 \
--proj_dir "${THIRD_PARTY_CODE_ROOT}" --conda_env neuralpil2 \
--env_vars "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TF_GPU_THREAD_MODE=gpu_private REPO_DIR=${THIRD_PARTY_CODE_ROOT} $EXTRA_ENV_VARS" \
--console_output_dir /svl/u/yzzhang/projects/aria/sbatch_console_outputs/ \
--job nerfactor_${SCENE}_${EXP_ID} --command "$COMMAND" \
--partition viscam --cpus_per_task 8 --mem 120G \
exclude=viscam1,viscam7,svl[1-6]
