#!/bin/bash

source ~/.bashrc
actenv neuralpil  # doesn't work on viscam7

for SCENE in \
"scene001_obj003_baking" \
"scene001_obj018_teapot" \
"scene002_obj008_grogu" \
"scene002_obj017_ball" \
"scene002_obj019_blocks" \
"scene002_obj020_chips" \
"scene003_obj003_baking" \
"scene003_obj007_gnome" \
"scene003_obj008_grogu" \
"scene003_obj010_pepsi" \
"scene003_obj020_chips" \
"scene004_obj010_pepsi" \
"scene004_obj012_cart" \
"scene004_obj017_ball" \
"scene005_obj001_salt" \
"scene005_obj016_pitcher" \
"scene005_obj019_blocks" \
"scene006_obj012_cart" \
"scene006_obj013_cup" \
"scene006_obj018_teapot" \
"scene007_obj001_salt" \
"scene007_obj007_gnome" \
"scene007_obj013_cup" \
"scene007_obj016_pitcher" \
; do
  SCENE_DATA_DIR="/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529/data/$SCENE/final_output/llff_format_LDR"
  PROCESSED_SCENE_DATA_DIR="/viscam/projects/imageint/yzzhang/data/processed_data/$SCENE/nerfactor_format"
#  python /viscam/projects/imageint/yzzhang/imageint/scripts/preprocess/colmap_to_nerfactor.py \
#-i $SCENE_DATA_DIR -o $PROCESSED_SCENE_DATA_DIR
  python /viscam/projects/imageint/yzzhang/imageint/scripts/preprocess/colmap_to_nerfactor_novel.py \
-i $SCENE_DATA_DIR -o $PROCESSED_SCENE_DATA_DIR
done
