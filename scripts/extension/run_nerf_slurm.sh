#!/bin/bash

source ~/.bashrc
actenv py311

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/nerfpytorch"

SCENE=scene000_000_Birdhouse
EXP_ID=1110

COMMAND="python run_nerf.py \
--datadir ${DATA_ROOT}/${SCENE} \
--config ${CODE_ROOT}/configs/nerf.txt \
--expname ${SCENE}/${EXP_ID}"

python -m tu.sbatch.sbatch_sweep --time 72:00:00 \
--proj_dir ${THIRD_PARTY_CODE_ROOT} --conda_env dmodel2 \
--console_output_dir /svl/u/yzzhang/projects/aria/sbatch_console_outputs/ \
--job nerf_${SCENE}_${EXP_ID} --command "$COMMAND" \
--partition svl --gpu_type titanrtx --cpus_per_task 8 --mem 20G
