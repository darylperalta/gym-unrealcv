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
- [Single House Policy Experiment](https://drive.google.com/drive/folders/1O7cFs-JP4uCYjV2jmmhEUudgAh_DUUHx?usp=sharing)
- Multiple Houses Policy Experiments
    - [Batch 1](https://drive.google.com/drive/folders/13_JwdAZ9hZwNkVzInoGh8WznCSO4fa8u?usp=sharing)
    - [Batch 2](https://drive.google.com/drive/folders/1nqAOMUUZyWVn1kd97_Qlqiul_Cqo_FwK?usp=sharing)
    - [Batch 3](https://drive.google.com/drive/folders/1z_V98VJUZWqVFSR63zOyQm0--Nxsc3me?usp=sharing)
    - [Batch 4](https://drive.google.com/drive/folders/17vLE-YmxxjiHXFvlziDR507TSBPQ_fJv?usp=sharing)
    - [Batch 5](https://drive.google.com/drive/folders/1rRYzwPMh_i4MdPvmsAofCNXczyiln9fL?usp=sharing)
    - [Batch 6](https://drive.google.com/drive/folders/1KpWnLoyq0-1snRPDWTS36M9VBR3JjhV5?usp=sharing)
    - [Batch 7](https://drive.google.com/drive/folders/1yF90Lga7EyL2zUczupsIA_Rs5dTPV1au?usp=sharing)
    - [Batch 8](https://drive.google.com/drive/folders/1_ubfD_GtZ9BrchdrSZ5nQDxT4VbLr93x?usp=sharing)
    - [Batch 9](https://drive.google.com/drive/folders/1BM08AQUEc_gzhmLO29HaOYL6pgLkAsXu?usp=sharing)
    - [Batch 10](https://drive.google.com/drive/folders/1KvAYEPV7h1Bl7wibcRMooUOuUjyBsbq1?usp=sharing)
    - [Batch 11](https://drive.google.com/drive/folders/1t0lCz_RQ1E29d0lKxa-CFA2SR0_7eSoJ?usp=sharing)
    - [Batch 12](https://drive.google.com/drive/folders/1nqAOMUUZyWVn1kd97_Qlqiul_Cqo_FwK?usp=sharing)
- [Stanford Bunny Experiment](https://drive.google.com/drive/folders/1me7pgsLLTZ6a_Gx8gLLLcvUIvLPDOMOy?usp=sharing)

## Prepare Ground truth points
To replicate our experiments, you also need to download the groundtruth point clouds we used for each experiment in the following links:
- [Single House Policy Experiment Ground truth Point Cloud](https://drive.google.com/file/d/19p8tdLxdFnoJBe5kAg7VwEpgeUApHMMK/view?usp=sharing)

You can save the ground truth point cloud anywhere. You'll just need to specify the pat in the setting file (See *Prepare Environment setting files*).


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
- ```disp_coverage``` can be set to 1 to diusplay surface coverage per step.

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

Computation of Chamfer Distance was based on this [Pointnet Autoencoder](https://github.com/charlesq34/pointnet-autoencoder). The default [tf_operators](gym_unrealcv/envs/utils/tf_nndistance_so.so) in this repo were compiled for CUDA 8. You can recompile for other CUDA versions using the code in [Pointnet Autoencoder](https://github.com/charlesq34/pointnet-autoencoder).

<!--
## Prepare Unreal Environment for Random Agent Example
You need prepare an unreal environment to run the demo envirnment.
You can do it by running the script [RealisticRendering.sh](RealisticRendering.sh)
```buildoutcfg
sh RealisticRendering.sh
```
or manually download the `RealisticRendering` env from this [link](https://s3-us-west-1.amazonaws.com/unreal-rl/RealisticRendering_RL_3.10.zip),
then unzip and move it to the [UnrealEnv](gym_unrealcv/envs/UnrealEnv) directory.

**Note that you can download more environments from [UnrealCV Model Zoo](http://docs.unrealcv.org/en/master/reference/model_zoo.html).**

There are two ways to launch the unreal environment in gym-unrealcv, called ```docker-based``` and ```docker-free```.
The ```docker-based``` way depends on [docker](https://docs.docker.com/engine/installation/linux/ubuntu/#install-from-a-package) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
The ```docker-free``` way launches the env binary directly.
The ```docker-based``` way promise more stable unrealcv connection and support to run the env parallelly.
On the contrast, the ```docker-free``` way only support running an unreal environment in the same time.
So the ```docker-based``` way is highly recommended to get better experience.
You can learn to install and use the ```docker-based``` way in this [page](doc/run_docker.md).

**Note that the default config runs in the ``Docker-free`` way.**

# Usage for Random Agent
## Run a random agent

Once ```gym-unrealcv``` is installed successfully, you will see that your agent is walking randomly in first-person view to find a door, after you run:
```
cd example/random
python random_agent.py -e 'Search-RrDoorDiscrete-v0'
```
It will take a few minutes for the image to pull if you runs environment based on docker at the first time.
After that, if all goes well，a pre-defined gym environment ```Search-RrDoorDiscrete-v0``` will be launched.
And then you will see that your agent is moving around the realistic room randomly.

We list the pre-defined environments in this [page](doc/EnvLists.md), for object searching and active object tracking.
# Tutorials from the original repo
We provide a set of tutorials to help you get started with Gym-UnrealCV.
### 1. Modify the pre-defined environment
You can follow the [modify_env_tutorial](doc/config_env.md) to modify the configuration of the pre-defined environment.

### 2. Add a new unreal environment
You can follow the [add_new_env_tutorial](doc/addEnv.md) to add new unreal environment for your RL task.

### 3.   Training a reinforcement learning agent
Besides, we also provide examples, such as [DQN](doc/dqn.md) and [DDPG](doc/ddpg.md), to demonstrate how to train agent in gym-unrealcv.

## Cite the original gym-unrealcv
If you use Gym-UnrealCV in your academic research, we would be grateful if you could cite it as follow:
```buildoutcfg
@misc{gymunrealcv2017,
    author = {Fangwei Zhong, Weichao Qiu, Tingyun Yan, Alan Yuille, Yizhou Wang},
    title = {Gym-UnrealCV: Realistic virtual worlds for visual reinforcement learning},
    howpublished={Web Page},
    url = {https://github.com/unrealcv/gym-unrealcv},
    year = {2017}
}
```
 -->
