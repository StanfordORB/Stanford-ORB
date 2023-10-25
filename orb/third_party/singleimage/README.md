# Learning to Reconstruct Shape and Spatially-Varying Reflectance from a Single Image 
Zhengqin Li, Zexiang Xu, Ravi Ramamoorthi, Kalyan Sunkavalli and Manmohan Chandraker. SIGGRAPH ASIA, 2018

## Real Examples
This is the official code release of paper [Learning to Reconstruct Shape and Spatially-Varying Reflectance from a Single Image ](https://drive.google.com/file/d/17K3RrWQ48gQynOhZHq1g5sQgjLjoMiPk/view). To test all the real images included in our paper, supplementary material and video, run
```
python testReal.py --cuda 
```
The input images are included in folder `real`. Their names are listed in file `imList.txt`, which will be loaded when running `testReal.py`. Results are by default saved in folder `output`. Definitions of results are as follows. `(n)` is the name of input and `(m)` is the level of cascade. Since we use three levels of cascade, the value of `(m)` can be 0, 1 or 2.
* `(n)albedo_(m).png`: Diffuse albedo prediction without gama correction. 
* `(n)normal_(m).png`: Normal prediction.
* `(n)rough_(m).png`: Roughness prediction.
* `(n)depth_(m).png/hdf5`: `.hdf5` is real depth. `.png` is tranformed depth in range of 0 to 1.  
* `(n)shCoef_(m).hdf5`: Spherical harmonics coefficents prediction for environment lighting. 
* `(n)renderedBounce1_(m).png`: First bounce image rendered using predicted shape and SVBRDF. 
* `(n)renderedBounce2_(m).png`: Second bounce image predicted by global illumination network. 
* `(n)renderedBounce3_(m).png`: Third bounce image predicted by global illumination network.
* `(n)renderedEnv_(m).png`: Image rendered using predicted environment lighting, shape and SVBRDF. 
* `(n)input.png`: Input image. 
* `(n)mask.png`: Segmentation mask. 

## Test
To test the pretrained mode, first download the full dataset from the [link](http://cseweb.ucsd.edu/~viscomp/projects/SIGA18ShapeSVBRDF/Data.zip), which is around 128GB. Unzip the dataset under the same directory where you save the code. And then run 
```
python test.py --cuda 
```
The testing errors will be saved in the folder `test_render2_refine1_cascade2` and painted on the screen. We have corrected a bug when rendering the second bounce image. Therefore the testing errors are slightly different from the number in the paper, which are summarized as follows. The error of second bounce is smaller (`Bounce2_(m)`). 

|      |Albedo_0|Albedo_1|Albedo_2|Normal_0|Normal_1|Normal_2|
|------|-------|-------|-------|-------|-------|-------|
|New|5.649x10<sup>-2</sup>|5.116x10<sup>-2</sup>|4.843x10<sup>-2</sup>|4.513x10<sup>-2</sup>|3.898x10<sup>-2</sup>|3.815x10<sup>-2</sup>|
|Origin|5.670x10<sup>-2</sup>|5.132x10<sup>-2</sup>|4.868x10<sup>-2</sup>|4.580x10<sup>-2</sup>|3.907x10<sup>-2</sup>|3.822x10<sup>-2</sup>|
|      |**Roughness_0**|**Roughness_1**|**Roughness_2**|**Depth_0**|**Depth_1**|**Depth_2**|
|New|2.061x10<sup>-1</sup>|2.0072x10<sup>-1</sup>|1.938x10<sup>-1</sup>|1.865x10<sup>-2</sup>|1.620x10<sup>-2</sup>|1.501x10<sup>-2</sup>| 
|Origin|2.064x10<sup>-1</sup>|2.011x10<sup>-1</sup>|1.943x10<sup>-1</sup>|1.871x10<sup>-2</sup>|1.624x10<sup>-2</sup>|1.505x10<sup>-2</sup>|
|      |**Bounce1_0**|**Bounce1_1**|**Bounce1_2**|**Bounce2_0**|**Bounce2_1**|**Bounce2_2**|
|New|3.309x10<sup>-3</sup>|2.042x10<sup>-3</sup>|1.648x10<sup>-3</sup>|2.65x10<sup>-4</sup>|2.22x10<sup>-4</sup>|2.19x10<sup>-4</sup>|
|Origin|3.291x10<sup>-3</sup>|2.046x10<sup>-3</sup>|1.637x10<sup>-3</sup>|2.76x10<sup>-4</sup>|2.47x10<sup>-4</sup>|2.45x10<sup>-4</sup>|
|      |**Bounce3_0**|**Bounce3_1**|**Bounce3_2**| | | |
|New|6.5x10<sup>-5</sup>|5.9x10<sup>-5</sup>|5.8x10<sup>-5</sup>||||
|Origin|6.4x10<sup>-5</sup>|5.9x10<sup>-5</sup>|5.8x10<sup>-5</sup>||||

In the following, we also report the L2 errors of predicted environment lighting (`Envmap_(m)`) and the image rendered with predicted environment lighting and the predicted shape and SVBRDF  (`Bounce1Env_(m)`), which we did not include in the paper.

|Envmap_0|Envmap_1|Envmap_2|Bounce1Env_0|Bounce1Env_1|Bounce1Env_2|
|-------|-------|-------|-------|-------|-------|
|6.04x10<sup>-4</sup>|5.39x10<sup>-4</sup>|4.75x10<sup>-4</sup>|7.915x10<sup>-3</sup>|7.502<sup>-3</sup>|6.995x10<sup>-3</sup>|

## Train
Again before you train the network, you need to first download the data from [link](http://cseweb.ucsd.edu/~viscomp/projects/SIGA18ShapeSVBRDF/Data.zip), which is around 128GB and unzip the dataset under the same directory where you save the code. In our paper, we train three levels of cascade network separately. Once we finish training one level of cascade, we will output the intermediate predictions and use that to train the next level of cascade. To begin with, we first train the network for global illumination prediction. 
```
python trainGlobalIllumination.py --cuda
```
The trained model will be saved in `check_globalillumination`. To test the results, run 
```
python testGlobalIllumination.py --cuda 
```
The error will be saved in `test_globalillumination`. Then we train the first level of cascade structure by running
```
python trainInitEnv.py --cuda
```
The trained model will be saved in `check_initEnvGlob_cascade0`. To test the results, run 
```
python testInitEnv.py --cuda
```
Then output the intermediate predictions by running 
```
python outputInitEnv.py --cuda --dataRoot ../Data/train 
python outputInitEnv.py --cuda --dataRoot ../Data/test
```
You can train the second level of cascade network in a similar way by running
```
python trainCascadeEnv_step.py --cuda --cascadeLevel 1 --nepoch 8
python testCascadeEnv_step.py --cuda --cascadeLevel 1 --epochId 7
python outputCascadeEnv_step.py --cuda --cascadeLevel 1 --epochId 7 --dataRoot ../Data/train
python outputCascadeEnv_step.py --cuda --cascadeLevel 1 --epochId 7 --dataRoot ../Data/test
```
and the third level by running 
```
python trainCascadeEnv_step.py --cuda --cascadeLevel 2 --nepoch 6
python testCascadeEnv_step.py --cuda --cascadeLevel 2 --epochId 5
```
After that, you should be able to test all three levels of cascade network together by running 
```
python test.py --cuda
```
which should give you the same performance as you tested each level of cascade separately. 

## Other codes
We also include codes to get some other results in the paper. To count the energy distribution of the first, second and third bounce (Fig 3 in the paper), run 
```
python multiBounceDistribution.py 
```
To verify that rendering data with global illumination can improve shape and SVBRDF reconstruction performance, run 
```
python trainInit.py --cuda --inputMode 1
python testInit.py --cuda --inputMode 1
```
and 
```
python trainInit.py --cuda --inputMode 0
python testInit.py --cuda --inputMode 0
```
The first two lines of code will train and test images rendered with global illumination. The last two lines of code will train the network using images without global illumination and test the network using images with global illumination.  The conclusion is it's worth considering global illumination to achieve good shape and SVBRDF estimation. 
