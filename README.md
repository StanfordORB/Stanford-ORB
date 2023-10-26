# Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark

This is the official repository of the <b>Stanford-ORB</b> dataset to test 3D object-centric inverse rendering models. 

The dataset consists of:

- __2,795__ HDR images of __14__ objects captured in __7__ in-the-wild scenes;
- __418__ HDR ground truth environment maps aligned with image captures;
- Textured scanned mesh of __14__ objects captured from studio;
- Comphehensive benchmarks for evaluating the inverse rendering methods;
- Reported results of concurrent state-of-the-art models;
- A full set of scripts & guideline to run the benchmarks.

This repository contains instructions for dataset downloads and evaluation tools.



> __Real-World 3D Object Inverse Rendering Benchmark__  
> [Zhengfei Kuang](https://zhengfeikuang.com), [Yunzhi Zhang](https://https://cs.stanford.edu/~yzzhang/), [Hong-Xing Yu](https://kovenyu.com/), [Samir Agarwala](https://samiragarwala.github.io/), [Shangzhe Wu](https://elliottwu.com/), [Jiajun Wu](https://jiajunwu.com/)
> _NeurIPS 2023 Datasets and Benchmarks Track, December 2023_  
> __[Project page](https://stanfordorb.github.io)&nbsp;/ [Paper](https://arxiv.org/abs/2310.16044)__

## Dataset Structure

Below is the overall structure of our dataset. The dataset is provided under three commonly used representations with LDR and HDR captures: 
1. The blender representation from the NeRF blender dataset (`blender_HDR/LDR`);
2. The LLFF representation (`llff_colmap_HDR/LDR`);
3. The COLMAP representation(`llff_colmap_HDR/LDR`).

Download links are provided on the [project page](https://stanfordorb.github.io).

```
data	
----- blender_HDR/LDR		
    ----- <obj_name>_scene00x               (name of the object and scene)
        ----- test/%04d.{exr/png}           (Test images in HDR/LDR)	
        ----- test_mask/%04d.png            (Test mask images in LDR)	
        ----- train/%04d.{exr/png}          (Train images in HDR/LDR)	
        ----- train_mask/%04d.png           (Train mask images in LDR)	
        ----- transforms_test.json          (Test metadata)	
        ----- transforms_train.json         (Train metadata)	
        ----- transforms_novel.json         (metadata of novel scene test images. Note that each frame has a unique camera angle)				
----- llff_colmap_HDR/LDR
    ----- <obj_name>_scene00x               (name of the object and scene)
        ----- images/%04d.{exr/png}         (All images in HDR/LDR)
        ----- masks/%04d.png                (All image masks in LDR)
        ----- sparse/0/*.bin                (Colmap Files)	
        ----- sparse/<nv_sn_name>/*.bin     (Colmap Files of novel scenes)	
        ----- poses_bounds.npy              (LLFF's camera files)	
        ----- train_id.txt                  (name of training images)	
        ----- test_id.txt                   (name of test images)	
        ----- novel_id.txt                  (name of test images from novel scenes)	
        ----- poses_bounds_novel.npy        (camera poses of novel scene test images indexed by order of names in novel_id.txt)					
----- ground_truth
    ----- <obj_name>_scene00x               (name of the object and scene)	
        ----- env_map/%04d.exr              (Ground truth HDRI environment maps (share the same name with the corresponding test image))
        ----- surface_normal/%04d.npy       (Ground truth surface normal maps)
        ----- z_depth/%04d.npy              (Ground truth Z-depth maps)
        ----- pseudo_gt_albedo/%04d.npy     (Pseudo GT albedo maps)
        ----- mesh_blender/mesh.obj         (Scanned Mesh aligned with the blender format data)
```

While most part of our data are identical to the original representations, some nuances still exist.

First, we provide object masks for all images, stored in the folder of ``*_mask`` for blender representation and ``masks`` for LLFF/Colmap representation. Feel free to use/ignore them during the training.

Second, to support the task of relighting, in each data folder (e.g. `baking_scene001`), we also provide the camera poses of test images from other scenes (e.g. `baking_scene002` and `baking_scene003`). In the blender representation, the data is stored in ``transforms_novel.json``; In the LLFF representation it's stored in ``novel_id.txt`` and ``poses_bounds_novel.npy``; In the COLMAP representation it's stored in ``sparse/<novel_scene_name>/*.bin``. Note that all the novel camera poses are transformed to align with the original poses, so no further adaptation is required for the users. In other words, you can directly use them as additional test poses as shown in the example dataloader [here](https://github.com/StanfordORB/Stanford-ORB/blob/9a559af9de855a0f37f96dd2670c9a5f970e22c0/orb/datasets/mymethod.py#L323). 


## Quick Start

With the dataset downloaded, you can train/test you model within only a few steps:

### 1. Training

We provide the example dataloaders for all structures [here](./orb/datasets/mymethod.py). Select one of the data dataloaders that best fit your method, and integrate it to your code. 

For accurate evaluation, your model must be trained by each capture separately, which results in 42 sets of learned weights in total.

### 2.Inferring

One the model is trained with a certain capture (e.g. `baking_scene001`), 
evaluation is done with the test data from this capture (denoted as <i>test dataset</i> below) 
and other captures of the same object, such as `baking_scene002` and `baking_scene003` (denoted as <i>novel datasets</i> below).

Here we explain the input and output of each benchmark tests:

- Novel View Synthesis
  - Input 1: Poses from the test dataset;
  - Output 1: Rendered LDR/HDR images of the test views.
- Relighting: 
  - Input 1: Poses from novel datasets;
  - Input 2: Ground truth environment maps of the test views (located in ground_truth/<capture_name>/env_map);
  - Output 1: Rendered LDR/HDR images of the test views.
- Geometry Estimation: 
  - Input 1: Poses from test dataset;
  - Output 1: Rendered Z-depth maps of the test views.
  - Output 2: Rendered camera-space normal maps of the test views.
  - Output 3: Reconstructed 3D mesh.

Pack up the path to all predictions and the corresponding ground truths in a json file,
in the same structure as [this example](./examples/test/mymethod.json).

### 3.Evaluation

Simply run this one-line command:
```bash
python scripts/test.py --input-path <path_to_input_json_file> --output-path <path_to_output_json_file> --scenes full
```

Ta-da! All evaluation results (summed-up scores and per-capture scores) will be wrote to the output json file.

### 4.Result Visualization
To further generate visulization figures as in our paper, first install several required packages:
```bash
pip install numpy matplotlib glob2 tqdm seaborn==0.12.2 pandas==1.4.4 scipy==1.9.1 
```
Then, move the output json file to `visulize/methods/<your_method_name>.json`.
Add your model's name in `visulize/visualize.ipynb` and run it. 
It will automaticly load your scores and generate the figure.

Feel free to play with the visualize script for better layout.

## Acknowledgement

```bibtex
@misc{kuang2023stanfordorb,
      title={Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark}, 
      author={Zhengfei Kuang and Yunzhi Zhang and Hong-Xing Yu and Samir Agarwala and Shangzhe Wu and Jiajun Wu},
      year={2023},
      eprint={2310.16044},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
