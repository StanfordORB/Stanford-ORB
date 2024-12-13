#!/bin/bash

source ~/.bashrc
actenv py311

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint
THIRD_PARTY_CODE_ROOT="${CODE_ROOT}/imageint/third_party/physg/code"

SCENE=scene000_000_Birdhouse
EXP_ID=1110

COMMAND="python training/exp_runner.py \
--conf ${CODE_ROOT}/configs/physg.conf \
--data_split_dir ${DATA_ROOT}/$SCENE \
--expname $SCENE/${EXP_ID} \
--nepoch 2000 --max_niter 200001 \
--gamma 1 --exposure 0.5"

python -m tu.sbatch.sbatch_sweep --time 72:00:00 \
--proj_dir ${THIRD_PARTY_CODE_ROOT} --conda_env dmodel2 \
--console_output_dir /svl/u/yzzhang/projects/aria/sbatch_console_outputs/ \
--job physg_${SCENE}_${EXP_ID} --command "$COMMAND" \
--partition viscam --cpus_per_task 8 --mem 20G \
exclude=viscam1,viscam7,svl[1-6]
