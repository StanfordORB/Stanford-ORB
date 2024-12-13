#!/bin/bash

source ~/.bashrc
actenv py311

DATA_ROOT=/svl/data/Aria_DTC/ORB_eval_format_spherified
CODE_ROOT=/svl/u/yzzhang/projects/aria/imageint

# SCENE=scene000_000_Airplane_B097C7SHJH_WhiteBlue
# SCENE=scene001_000_Airplane_B097C7SHJH_WhiteBlue
SCENE=scene000_000_Birdhouse
SCENE=scene001_000_Birdhouse

# COMMAND="python scripts/preprocess/envmap_to_invrender.py -s $SCENE"
# COMMAND="python scripts/preprocess/envmap_to_invrender.py -s $SCENE -o"
COMMAND="python scripts/preprocess/envmap_to_physg.py -s $SCENE"
# COMMAND="python scripts/preprocess/envmap_to_physg.py -s $SCENE -o"
# COMMAND="python scripts/preprocess/envmap_to_nerd.py -s $SCENE"
# COMMAND="python scripts/preprocess/envmap_to_nerd.py -s $SCENE -o"

python -m tu.sbatch.sbatch_sweep --time 72:00:00 \
--proj_dir ${CODE_ROOT} --conda_env neuralpil2 \
--console_output_dir /svl/u/yzzhang/projects/aria/sbatch_console_outputs/ \
--job preprocess_${SCENE} --command "$COMMAND" \
--env_vars PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
--partition viscam --cpus_per_task 4 --mem 12G \
exclude=viscam1,viscam7,svl[1-6]
