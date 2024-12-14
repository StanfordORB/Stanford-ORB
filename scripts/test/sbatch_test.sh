#!/bin/bash

GPU_INFO="--partition viscam --cpus_per_task 8 --mem 20G"
BASH_OUT_PATH="/viscam/projects/imageint/yzzhang/imageint/scripts/exp/test_sbatch_test.txt"
for METHOD in \
"nerd" \
"neuralpil" \
"pseudo_gt_nvdiffrec" \
"pseudo_gt_nvdiffrecmc" \
"nvdiffrec" \
"nvdiffrecmc" \
"idr" \
"physg" \
"invrender" \
; do
ENV_VARS="NO_SCORE_VIEW=1 NO_SCORE_LIGHT=1 NO_SCORE_GEOMETRY=1 NO_SCORE_MATERIAL=0"
#ENV_VARS=""
#for METHOD in "nerd" "neuralpil"; do
#for METHOD in "nerf"; do
  COMMAND="python scripts/test/$METHOD.py"
  echo "$COMMAND" | tee -a $BASH_OUT_PATH
  python -m tu.sbatch.sbatch_sweep --time 168:00:00 \
--proj_dir /viscam/projects/imageint/yzzhang/imageint --conda_env dmodel \
--env_vars "$ENV_VARS" \
--job test_${METHOD} --command "$COMMAND" $GPU_INFO exclude=viscam1,viscam7,svl[1-6] |& tee -a $BASH_OUT_PATH
done
