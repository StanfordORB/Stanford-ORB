#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

source ~/.bashrc
cd /viscam/projects/imageint/yzzhang/imageint || exit
#actenv neuralpil
###python scripts/test/nerd.py
#python scripts/test/neuralpil.py
#exit
##
actenv dmodel
#python scripts/test/singleimage.py
#python scripts/test/sirfs.py
#python scripts/test/nerf.py
##
#NO_SCORE_VIEW=1 NO_SCORE_LIGHT=1 python scripts/test/nerd.py
#NO_SCORE_VIEW=1 python scripts/test/neuralpil.py
#
#IMAEGINT_PSEUDO_GT=1 python scripts/test/nvdiffrec.py
#python scripts/test/nvdiffrec.py
#IMAEGINT_PSEUDO_GT=1 python scripts/test/nvdiffrecmc.py
#python scripts/test/nvdiffrecmc.py
#
#python scripts/test/idr.py
#python scripts/test/physg.py
#python scripts/test/invrender.py

#source /viscam/u/zhengfei/anaconda3/etc/profile.d/conda.sh
#conda activate dmodel
#cd /viscam/projects/imageint/yzzhang/imageint || exit
##OVERWRITE_GEOMETRY=1 python scripts/test/nvdiffrec.py
#OVERWRITE_GEOMETRY=1 python scripts/test/nvdiffrecmc.py

export NO_SCORE_VIEW=1
export NO_SCORE_LIGHT=1
export NO_SCORE_GEOMETRY=1
#python scripts/test/invrender.py
#python scripts/test/nerd.py
python scripts/test/neuralpil.py
