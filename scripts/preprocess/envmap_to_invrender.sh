#!/bin/bash

GPU_INFO="--partition viscam --cpus_per_task 8 --mem 20G"
BASH_OUT_PATH="/viscam/projects/imageint/yzzhang/imageint/scripts/exp/preprocess_envmap_to_invrender.txt"
for SCENE in \
"scene001_obj003_baking" \
"scene001_obj008_grogu" \
"scene001_obj016_pitcher" \
"scene001_obj018_teapot" \
"scene002_obj003_baking" \
"scene002_obj008_grogu" \
"scene002_obj010_pepsi" \
"scene002_obj012_cart" \
"scene002_obj017_ball" \
"scene002_obj018_teapot" \
"scene002_obj019_blocks" \
"scene002_obj020_chips" \
"scene003_obj003_baking" \
"scene003_obj007_gnome" \
"scene003_obj008_grogu" \
"scene003_obj010_pepsi" \
"scene003_obj013_cup" \
"scene003_obj017_ball" \
"scene003_obj020_chips" \
"scene004_obj001_salt" \
"scene004_obj010_pepsi" \
"scene004_obj012_cart" \
"scene004_obj017_ball" \
"scene004_obj020_chips" \
"scene005_obj001_salt" \
"scene005_obj007_gnome" \
"scene005_obj016_pitcher" \
"scene005_obj019_blocks" \
"scene006_obj012_cart" \
"scene006_obj013_cup" \
"scene006_obj018_teapot" \
"scene006_obj019_blocks" \
"scene007_obj001_salt" \
"scene007_obj007_gnome" \
"scene007_obj013_cup" \
"scene007_obj016_pitcher" \
; do
  COMMAND="python scripts/preprocess/envmap_to_invrender.py -s $SCENE -o"
  echo "$COMMAND" | tee -a $BASH_OUT_PATH
  python -m tu.sbatch.sbatch_sweep --time 168:00:00 \
--proj_dir /viscam/projects/imageint/yzzhang/imageint --conda_env dmodel \
--job invrender_${SCENE} --command "$COMMAND" $GPU_INFO exclude=viscam1,viscam7,svl[1-6] |& tee -a $BASH_OUT_PATH
done
