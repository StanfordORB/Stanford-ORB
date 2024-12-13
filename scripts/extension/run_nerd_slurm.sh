#!/bin/bash

source ~/.bashrc
actenv py311

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/nerd"

SCENE=scene000_000_Birdhouse
EXP_ID=1110

COMMAND="python train_nerd.py \
--datadir ${DATA_ROOT}/${SCENE} \
--basedir ./logs/ --expname ${SCENE}/${EXP_ID} --gpu 0 \
--config ${CODE_ROOT}/configs/nerd.txt"

python -m tu.sbatch.sbatch_sweep --time 72:00:00 \
--proj_dir ${THIRD_PARTY_CODE_ROOT} --conda_env neuralpil2 \
--env_vars PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
--console_output_dir /svl/u/yzzhang/projects/aria/sbatch_console_outputs/ \
--job nerd_${SCENE}_${EXP_ID} --command "$COMMAND" \
--partition svl --gpu_type titanrtx --cpus_per_task 8 --mem 20G
