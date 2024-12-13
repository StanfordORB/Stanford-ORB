#!/bin/bash
src_base="/viscam/projects/imageint/capture_scene_data"
#dest_base="/viscam/projects/imageint/yzzhang/data/capture_scene_data"
dest_base="/viscam/projects/imageint/yzzhang/data/capture_scene_data_0529"

# loop over all directories matching the pattern in the source base
find "$src_base" -type d -name 'final_output' | while read src_dir; do
    # extract the relative path
    rel_path=${src_dir#$src_base/}
    # create the corresponding destination directory, if it does not exist
    mkdir -p "$dest_base/$rel_path"
    # copy the directory using rsync
    rsync -avh --progress "$src_dir/" "$dest_base/$rel_path/"
done
