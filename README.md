Gym-UnrealCV: Realistic virtual worlds for visual reinforcement learning [Modified for Scan-RL]
===

This is a fork of [gym-unrealcv](https://github.com/zfw1226/gym-unrealcv) modified for [Scan-RL](https://github.com/darylperalta/ScanRL) implementation. You may look at [gym-unrealcv](https://github.com/zfw1226/gym-unrealcv) for the original implementation and documentation.

# Introduction
**This project integrates Unreal Engine with OpenAI Gym for visual reinforcement learning based on [UnrealCV](http://unrealcv.org/).**
In this project, you can run your RL algorithms in various realistic UE4 environments easily without any knowledge of Unreal Engine and UnrealCV.
The framework of this project is shown as below:

![framework](./doc/framework.JPG)

- ```UnrealCV``` is the basic bridge between ```Unreal Engine``` and ```OpenAI Gym```.
- ```OpenAI Gym``` is a toolkit for developing RL algorithm, compatible with most of numerical computation library, such as Tensorflow or Theano.


![search1](./doc/search1.gif)

<!-- ![search2](./doc/search2.gif) -->

Snapshots of RL based visual navigation for object searching and obstacle avoidance.

# Installation
## Dependencies
- UnrealCV
- Gym
- CV2
- Matplotlib
- Numpy
- Docker(Optional)
- Nvidia-Docker(Optional)

We recommend you to use [anaconda](https://www.continuum.io/downloads) to install and manage your python environment.
```CV2``` is used for images processing, like extracting object mask and bounding box.```Matplotlib``` is used for visualization.
## Install Gym-UnrealCV for Scan-RL

It is easy to install gym-unrealcv, just run
```buildoutcfg
git clone https://github.com/darylperalta/gym-unrealcv.git
cd gym-unrealcv
pip install -e .
```
While installing gym-unrealcv, dependencies including [OpenAI Gym](https://github.com/openai/gym), unrealcv, numpy and matplotlib are installed.
`Opencv` is should be installed additionally.
If you use ```anaconda```,you can run
```buildoutcfg
conda update conda
conda install --channel menpo opencv
```
## Prepare Unreal Environment for Scan-RL
You need to prepare an unreal environment to run the Scan-RL Experiments.
Manually download the environments for the different Scan-RL experiments to the [UnrealEnv](gym_unrealcv/envs/UnrealEnv) directory using the following links:
- [Single House Policy Experiment](https://drive.google.com/drive/folders/13o0pDj4KXhSVl0r-lLmsu5TMEi7hSiM2?usp=sharing)
- Multiple Houses Policy Experiments
    - [Batch 1](https://drive.google.com/drive/folders/159dxiqj__jXNplmkLUGo0KMPEpSFLrnB?usp=sharing)
    - [Batch 2](https://drive.google.com/drive/folders/1Vas8e65JxXJdFhZ6-Uu6EyQxpAh659K3?usp=sharing)
    - [Batch 3](https://drive.google.com/drive/folders/1ui-Vvuf_CthetWm7_FAeMULqplE0l1yp?usp=sharing)
    - [Batch 4](https://drive.google.com/drive/folders/1-zpie4kiIfht4wNyEXBIMUwBP5x8s9oW?usp=sharing)
    - [Batch 5](https://drive.google.com/drive/folders/1Vkxu2cxpKk5_0W39gw8B0iuSOf7n6Xtl?usp=sharing)
    - [Batch 6](https://drive.google.com/drive/folders/1LU7Br-kdaeoSYE9nD2giDAuOIMyVgZXV?usp=sharing)
    - [Batch 7](https://drive.google.com/drive/folders/1ptJctlhgkcR2kFgw7f4ZVVkjbbkDyrrA?usp=sharing)
    - [Batch 8](https://drive.google.com/drive/folders/1TTm0eFsGC7X4zWO4fMUes0KKjfFVzFWC?usp=sharing)
    - [Batch 9](https://drive.google.com/drive/folders/1nAtSqy48JHVZBCXFjL9loy85HTpcpjOF?usp=sharing)
    - [Batch 10](https://drive.google.com/drive/folders/18Bk1V0DmG2aiUdh8D9Ma0IsC-JhSmqA1?usp=sharing)
    - [Batch 11](https://drive.google.com/drive/folders/1Wz7INDIJvNiOI7mQDddtN3UX241v55rr?usp=sharing)
    - [Batch 12](https://drive.google.com/drive/folders/1_r0eR6jNqyWa7p4v_h50IT_MEQHPC6Rp?usp=sharing)
- [Stanford Bunny Experiment](https://drive.google.com/drive/folders/1vLEaJDuJmt3Rx7zRoPveD1zQHSkCs6dC?usp=sharing)

## Prepare Ground truth points
To replicate our experiments, you also need to download the groundtruth point clouds we used for each experiment in the following links:
- [Single House Policy Experiment Ground Truth Point Cloud](https://drive.google.com/file/d/19p8tdLxdFnoJBe5kAg7VwEpgeUApHMMK/view?usp=sharing)

- [Multiple Houses Single Policy Experiment and Stanford Bunny Ground Truth Point Clouds](https://drive.google.com/drive/folders/1xJsqBZpJfqAFiq9hQoDlcbTguBcXfPav?usp=sharing).
    - You'll find in the link folders *house_BATX_full* where X is the Houses3K batch number.
    - You'll also find a folder named *bunny* for the Non-House Target Model Experiment using Stanford Bunny.


You can save the ground truth point cloud anywhere. You'll just need to specify the path in the setting file (See *Prepare Environment setting files*).


## Prepare Environment setting files

### Single House Policy Experiment
You can change some environment settings using this [setting file](gym_unrealcv/envs/setting/depth_fusionB_keras.json). Specify the location of the ground truth point cloud in variable ```pointcloud_path```. To change the azimuth resolution, you can modify variable ```discrete_actions```. If using 2 distance levels set ```start_pose_rel``` to [0.0, 45.0, 125.0] else if 3 distance levels set ```start_pose_rel``` to [0.0, 45.0, 150.0]. ```min_elevation, max_elevation, min_distance and max_distance``` can be used to vary the range of eleavtion and distance the agent can go to as discussed in the paper.

### Multiple Houses Single Policy Experiment
You can change some environment settings using this [setting file](gym_unrealcv/envs/setting/depth_fusionB_keras_multHouse_rand_setA.json). Variables to be set are shown below:
```
"batch": "6",
"pcl_path1": "/hdd/AIRSCAN/datasets//houses3k_gt_ply/house_BAT1_full/groundtruth/",
"pcl_path2": "/hdd/AIRSCAN/datasets/houses3k_gt_ply/house_BAT2_full/groundtruth/",
"pcl_path3": "/hdd/AIRSCAN/datasets/houses3k_gt_ply/house_BAT3_full/groundtruth/",
"pcl_path4": "/hdd/AIRSCAN/datasets/house_BAT4_full/groundtruth/",
"pcl_path5": "/hdd/AIRSCAN/datasets/house_BAT5_full/groundtruth/",
"pcl_path6": "/hdd/AIRSCAN/datasets/houses3k_gt_ply/house_BAT6_full/groundtruth/",
"pcl_path7": "/hdd/AIRSCAN/datasets/house_BAT7_full/groundtruth/",
"pcl_path8": "/hdd/AIRSCAN/datasets/houses3k_gt_ply/house_BAT8_full/groundtruth/",
"pcl_path9": "/hdd/AIRSCAN/datasets/house_BAT9_full/groundtruth_resized/",
"pcl_path10": "/hdd/AIRSCAN/datasets/house_BAT10_full/groundtruth_resized/",
"pcl_path11": "/hdd/AIRSCAN/datasets/house_BAT11_full/groundtruth/",
"pcl_path12": "/hdd/AIRSCAN/datasets/house_BAT12_full/groundtruth/",
"pcl_path_bunny": "/home/daryl/datasets/bunny/groundtruth/",
"nn_distance_path": "envs/utils/tf_nndistance_so.so",
"test": 1,
"testSet": 0,
"test_all": 0,
"testSetA": 0,
"testSetB": 0,
"testSetC": 0,
"testSetD": 0,
"testSetE": 0,
"new_split": 1,
"test_baseline": 1,
"save_pcd": 0,
"disp_houses": 1
"disp_coverage": 0
```
- ```batch``` is used to indicate the batch from Houses3K. Use 0-12 for Houses3k and 'bunny' if using Stanford Bunny.
- ```pcl_path1, ...``` are the path for ground truth point clouds for the different batches.
- ```nn_distance_path``` is used for the tf operator for the chamfer distance.
- ```test, testSet, test_all, testSetA, ..., testSetE``` are used to control if you're using train set or a specific test set. If training using the train set set all of them to 0.
- ```test``` is assigned to 1 when testing ScanRL or the circular baseline in the train set.
- ```testSet``` is assigned with 1 to do the test on test set. Make sure to clear ```test``` when using the test set.
- ```test_all``` is used to use multiple subsets for each batch of Houses3K.
- ``` testSetA, ..., testSetE``` allow to choose which subset to use.
- ```new_split``` should be 1 if using geometry split and 0 for random split.
- ```test_baseline``` should be  1 if running the circular baselines.
- ```disp_coverage``` can be set to 1 to display surface coverage per step.

### Stanford Bunny Experiment
For stanford bunny experiment this is the [setting file](gym_unrealcv/envs/setting/bunny.json). Specify the location of the ground truth point cloud in variable ```pcl_path_bunny```. Make sure `batch` is set to "bunny".


## Usage for Circular Baselines
- Single House Policy Experiment
```
cd example/circular_agent
python circular_agent_close_depth.py
```
Use argument ```--circ_type``` to choose which circular baseline type to use.
- Multiple Houses Single Policy Experiment
```
cd example/circular_agent
python circular_agent_baseline.py
```

## Notes on chamfer distance computation

Computation of Chamfer Distance was based on [Pointnet Autoencoder](https://github.com/charlesq34/pointnet-autoencoder). The default [tf_operators](gym_unrealcv/envs/utils/tf_nndistance_so.so) in this repo were compiled for CUDA 8. You can recompile for other CUDA versions using the code in [Pointnet Autoencoder](https://github.com/charlesq34/pointnet-autoencoder).
