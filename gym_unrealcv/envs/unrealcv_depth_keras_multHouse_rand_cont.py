import gym # openai gym
# from unrealcv_cmd import  UnrealCv # a lib for using unrealcv client command
from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
from gym import spaces
import numpy as np
import math
import os
from gym_unrealcv.envs.utils import env_unreal
from gym_unrealcv.envs.navigation.interaction import Navigation
import random
from math import sin, cos, radians

from gym_unrealcv.envs.utils.utils_depthFusion import write_pose, write_depth, depth_fusion, depth_conversion, poseRelToAbs, poseOrigin, depth_fusion_mult
import time
# import pcl
import open3d as o3d
o3d.set_verbosity_level(o3d.VerbosityLevel.Error)
# import run_docker # a lib for run env in a docker container
# import gym_unrealcv.envs.utils.run_docker

import tensorflow as tf
from tensorflow.python.framework import ops
import keras.backend as K




class depthFusion_keras_multHouse_rand_cont(gym.Env):
    # init the Unreal Gym Environment
    def __init__(self,
                setting_file = 'depth_fusion.json',
                reset_type = 'waypoint',       # testpoint, waypoint,
                augment_env = None,   #texture, target, light
                test = True,               # if True will use the test_xy as start point
                action_type = 'discrete',  # 'discrete', 'continuous'
                observation_type = 'rgbd', # 'color', 'depth', 'rgbd'
                reward_type = 'bbox', # distance, bbox, bbox_distance,
                docker = False,
                # resolution = (84,84)
                # resolution = (640,480),
                resolution = (640,480),
                log_dir='log/'
    ):
     self.test = False
     # self.test = True
     self.testSet = False
     self.test_all= False
     self.testSetA = False
     self.testSetB = False
     self.testSetC = False
     self.testSetD = False
     self.testSetE = False
     self.batch12 = True
     self.batch11 = False
     self.batch10 = False
     self.batch9 = False
     self.batch8 = False
     self.batch7 = False
     self.batch5 = False
     self.batch4 = False
     self.batch3 = False
     self.batch2 = False
     self.batch1 = False
     self.bunny = False
     self.test_baseline = False
     self.unsolved_ctr = 0
     self.unsolved_list = []

     self.save_pcd = False
     self.disp_houses = False
     setting = self.load_env_setting(setting_file)
     self.cam_id = 0
     # self.reset_type = 'random'
     self.reset_type = 'test'
     self.log_dir = log_dir
     # gt_pcl = pcl.load('house-000024-gt.ply')

     # run virtual enrionment in docker container
     # self.docker = run_docker.RunDocker()
     # env_ip, env_dir = self.docker.start(ENV_NAME=ENV_NAME)

     # start unreal env
     docker = False
     # self.unreal = env_unreal.RunUnreal(ENV_BIN="house_withfloor/MyProject2/Binaries/Linux/MyProject2")
     self.unreal = env_unreal.RunUnreal(ENV_BIN=self.env_bin)
     env_ip,env_port = self.unreal.start(docker,resolution)

     # connect unrealcv client to server
     # self.unrealcv = UnrealCv(self.cam_id, ip=env_ip, env=env_dir)


     # connect UnrealCV
     self.unrealcv = Navigation(cam_id=self.cam_id,
                              port= env_port,
                              ip=env_ip,
                              targets=self.target_list,
                              env=self.unreal.path2env,
                              resolution=resolution)
     self.unrealcv.pitch = self.pitch

      # define action
     self.action_type = action_type
     assert self.action_type == 'discrete' or self.action_type == 'continuous'
     if self.action_type == 'discrete':
         self.action_space = spaces.Discrete(len(self.discrete_actions))
     elif self.action_type == 'continuous':
         self.action_space = spaces.Box(low = np.array(self.continous_actions['low']),high = np.array(self.continous_actions['high']))

     self.observation_type = observation_type
     assert self.observation_type == 'color' or self.observation_type == 'depth' or self.observation_type == 'rgbd' or self.observation_type == 'gray'
     self.observation_shape = self.unrealcv.define_observation(self.cam_id,self.observation_type)

     self.startpose = self.unrealcv.get_pose(self.cam_id)

     #try hardcode start pose
     # self.startpose = [750.0, 295.0, 212.3,356.5,190.273, 0.0]
     # self.startpose = [0.0, 707.1068, 707.1067,0.0,270.0, -45.0] # [0,45,1000]
     # self.startpose = [0.0,99.6,8.72,0.0,270.0,-5.0] #[for depth fusion] [0,5,100]
     # self.startpose = [0.0,70.7,70.7,0.0,270.0,-45.0]
     azimuth, elevation, distance = self.start_pose_rel
     print('start pose rel', azimuth,elevation,distance)
     self.startpose = poseRelToAbs(azimuth, elevation, distance)
     print('start_pose: ', self.startpose)
     ''' create base frame '''
     poseOrigin(self.log_dir+'frame-{:06}.pose.txt'.format(1000))
     # ACTION: (Azimuth, Elevation, Distance)

     self.count_steps = 0
     self.count_house_frames = 0
     # self.max_steps = 35
     self.target_pos = ( -60,   0,   50)
     # self.gt_pclpose_prev = np.array(self.start_pose_rel)
     # self.action_space = gym.spaces.Discrete(len(self.ACTION_LIST))
     # state = self.unrealcv.read_image(self.cam_id, 'lit')
     # self.observation_space = gym.spaces.Box(low=0, high=255, shape=state.shape)

     self.nn_distance_module =tf.load_op_library('/home/daryl/gym-unrealcv/gym_unrealcv/envs/utils/tf_nndistance_so.so')
     self.total_distance = 0

     objects = self.unrealcv.get_objects()
     # print('objects', objects)
     # self.houses = [(obj) for obj in objects if obj.startswith('BAT6_')]
     if self.batch7:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT7_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT7_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT7_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT7_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT7_SETE')]
     elif self.batch1:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT1_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT1_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT1_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT1_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT1_SETE')]
     elif self.batch2:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT2_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT2_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT2_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT2_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT2_SETE')]
     elif self.batch3:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT3_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT3_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT3_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT3_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT3_SETE')]
     elif self.batch4:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT4_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT4_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT4_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT4_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT4_SETE')]
     elif self.batch5:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT5_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT5_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT5_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT5_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT5_SETE')]
     elif self.batch8:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT8_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT8_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT8_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT8_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT8_SETE')]
     elif self.batch9:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT9_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT9_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT9_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT9_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT9_SETE')]
     elif self.batch10:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT10_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT10_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT10_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT10_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT10_SETE')]
     elif self.batch11:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT11_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT11_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT11_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT11_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT11_SETE')]
     elif self.batch12:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT12_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT12_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT12_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT12_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT12_SETE')]
     elif self.bunny:
         self.houses = [(obj) for obj in objects if obj.startswith('textured')]
     else:
         self.housesA = [(obj) for obj in objects if obj.startswith('BAT6_SETA')]
         self.housesB = [(obj) for obj in objects if obj.startswith('BAT6_SETB')]
         self.housesC = [(obj) for obj in objects if obj.startswith('BAT6_SETC')]
         self.housesD = [(obj) for obj in objects if obj.startswith('BAT6_SETD')]
         self.housesE = [(obj) for obj in objects if obj.startswith('BAT6_SETE')]
     if not self.bunny:
         # print('A', self.housesA)
         # self.housesA.sort()
         # print('B', self.housesB)
         # self.housesB.sort()
         # print('B sorted', self.housesB)
         # self.housesC.sort()
         # print('C', self.housesC)
         # self.housesD.sort()
         # print('D', self.housesD)
         # self.housesE.sort()
         # print('E', self.housesE)

         # hide houses
         for house in self.housesA:
             self.unrealcv.hide_obj(house)
         for house in self.housesB:
             self.unrealcv.hide_obj(house)
         for house in self.housesC:
             self.unrealcv.hide_obj(house)
         for house in self.housesD:
             self.unrealcv.hide_obj(house)
         for house in self.housesE:
             self.unrealcv.hide_obj(house)

     if (not self.testSet) and (not self.bunny):


         if self.batch1:
             remove_numA = [1,16,19,35,36]
             remove_numB = [6,7,18,29,44]
             remove_numC = [5,8,33,38,48]
             remove_numD = [1,15,17,28,30]
             remove_numE = [8,9,27,31,48]
         elif self.batch2:
             remove_numA = [6,10,25,35,38]
             remove_numB = [9,11,20,28,43]
             remove_numC = [12,14,21,35,40]
             remove_numD = [3,10,11,37,42]
             remove_numE = [3,5,19,31,40]
         elif self.batch3:
             remove_numA = [7,20,22,29,50]
             remove_numB = [2,24,36,43,49]
             remove_numC = [2,4,10,29,35]
             remove_numD = [4,22,24,30,36]
             remove_numE = [13,14,21,33,34]
         elif self.batch4:
             remove_numA = [1,14,18,32,42]
             remove_numB = [2,3,32,11,13]
             remove_numC = [5,9,22,34,46]
             remove_numD = [5,7,10,13,22]
             remove_numE = [7,22,24,32,34]
         elif self.batch5:
             remove_numA = [3,20,32,46,48]
             remove_numB = [8,40,42,46,48]
             remove_numC = [7,19,24,34,39]
             remove_numD = [1,11,34,38,48]
             remove_numE = [13,34,39,40,43]
         elif self.batch7:
             remove_numA = [8,18,19,49]
             remove_numB = [6,7,17,39]
             remove_numC = [3,26,38,39]
             remove_numD = [10,21,27,49]
             remove_numE = [3,10,23,48]
         elif self.batch8:
             remove_numA = [4, 11, 15, 22, 48]
             remove_numB = [3, 20, 25, 40, 42]
             remove_numC = [8, 14, 16, 31, 50]
             remove_numD = [1, 12, 14, 38, 41]
             remove_numE = [2, 22, 28, 30, 43]
         elif self.batch9:
             remove_numA = [6, 14, 19, 35, 38]
             remove_numB = [4, 28, 38, 39, 43]
             remove_numC = [6, 10, 21, 44, 47]
             remove_numD = [5, 7, 19, 25, 44]
             remove_numE = [3, 8, 27, 36, 43]
         elif self.batch10:
             remove_numA = [43, 39, 27, 46, 22]
             remove_numB = [2, 46, 39, 22, 47]
             remove_numC = [9, 47, 33, 27, 10]
             remove_numD = [41, 47, 26, 33, 45]
             remove_numE = [36, 20, 48, 46, 37]
         elif self.batch11:
             remove_numA = [6, 7, 16, 30, 31]
             remove_numB = [2, 14, 18, 20, 28]
             remove_numC = [1, 9, 22, 24, 49]
             remove_numD = [8, 16, 33, 40, 41]
             remove_numE = [2, 13, 21, 25, 34]
         elif self.batch12:
             remove_numA = [2, 5, 27, 34, 46]
             remove_numB = [15, 27, 33, 39, 50]
             remove_numC = [10, 17, 19, 36, 44]
             remove_numD = [5, 21, 30, 32, 47]
             remove_numE = [9, 12, 32, 33, 46]
         else:
             remove_numA = [1,4,10,35,48,50] # remove House 50
             remove_numB = [3,4,21,45,49,50] # remove House 50
             remove_numC = [3,11,13,36,43,50] # remove House 50
             remove_numD = [6,21,22,26,30,50] # remove House 50
             remove_numE = [4,10,11,46,48,50] # remove House 50
         remove_numA.sort()
         remove_numB.sort()
         remove_numC.sort()
         remove_numD.sort()
         remove_numE.sort()
         # Remove for SetA
         i = 0

         while(1):
             num = int((self.housesA[i].split('HOUSE')[1]).split('_')[0])
             if num in remove_numA:
                 print('removed', self.housesA.pop(i))
                 remove_numA.remove(num)
                 if len(remove_numA) == 0:
                     break
             else:
                 i += 1
         # Remove for B
         i = 0

         while(1):
             num = int((self.housesB[i].split('HOUSE')[1]).split('_')[0])
             if num in remove_numB:
                 print('removed', self.housesB.pop(i))
                 remove_numB.remove(num)
                 if len(remove_numB) == 0:
                     break
             else:
                 i += 1
         # Remove for SetC
         i = 0

         while(1):
             num = int((self.housesC[i].split('HOUSE')[1]).split('_')[0])
             if num in remove_numC:
                 print('removed', self.housesC.pop(i))
                 remove_numC.remove(num)
                 if len(remove_numC) == 0:
                     break
             else:
                 i += 1
         # Remove for SetD
         i = 0

         while(1):
             num = int((self.housesD[i].split('HOUSE')[1]).split('_')[0])
             if num in remove_numD:
                 print('removed', self.housesD.pop(i))
                 remove_numD.remove(num)
                 if len(remove_numD) == 0:
                     break
             else:
                 i += 1
         # Remove for SetE
         i = 0

         while(1):
             num = int((self.housesE[i].split('HOUSE')[1]).split('_')[0])
             if num in remove_numE:
                 print('removed', self.housesE.pop(i))
                 remove_numE.remove(num)
                 if len(remove_numE) == 0:
                     break
             else:
                 i += 1

     if not self.bunny:
         self.houses = self.housesA + self.housesB + self.housesC + self.housesD + self.housesE
     # display houses
     if self.disp_houses == True:
         print('display houses')
         for i in range(len(self.houses)):
             print('id', i)
             print('filename', self.houses[i])

     self.num_house = len(self.houses)
     # print('houses new', self.houses)
     self.house_ids_ordered = list(range(len(self.houses)))
     # print('house_ids_ordered',self.house_ids_ordered)
     self.house_ids_shuffle = random.shuffle(self.house_ids_ordered)
     # print(self.house_ids_shuffle)

     # self.house
     print('num houses', self.num_house)

     for house in self.houses:
         self.unrealcv.hide_obj(house)

     self.ids = list(range(self.num_house))
     # print('ids', self.ids)
     self.shuffle_ids = self.ids.copy()
     random.shuffle(self.shuffle_ids)
     # print('shuffled ids', self.shuffle_ids)
     # self.house_id = 406, 7,
     if self.test == True:
         self.house_id = -1
         # self.house_id = 144
     elif self.testSet == True:
         self.house_id = -1
         self.testA_id = 0
         self.testB_id = 0
         self.testC_id = 0
         self.testD_id = 0
         self.testE_id = 0

     else:
         self.house_id = random.randint(0, self.num_house-1) # randomize houses

     # self.unrealcv.show_obj(self.houses[self.house_id])

     # gt_dir = '/hdd/AIRSCAN/datasets/house38_44_2/groundtruth/'
     # gt_dir = '/hdd/AIRSCAN/datasets/house_10/groundtruth/'
     # gt_dir = '/hdd/AIRSCAN/datasets/house_setA_comp/groundtruth/'
     if self.batch12 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT12_full/groundtruth/'
     elif self.batch11 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT11_full/groundtruth/'
     elif self.batch10 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT10_full/groundtruth_resized/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT10_full/groundtruth/'
     elif self.batch9 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT9_full/groundtruth_resized/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT10_full/groundtruth/'
     elif self.batch8 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT8_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT8_full/groundtruth/'
     elif self.batch7 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT7_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT7_full/groundtruth/'
     elif self.batch5 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT5_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT5_full/groundtruth/'
     elif self.batch4 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT4_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT4_full/groundtruth/'
     elif self.batch3 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT3_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT3_full/groundtruth/'
     elif self.batch2 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT2_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT2_full/groundtruth/'
     elif self.batch1 == True:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT1_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT1_full/groundtruth/'
     elif self.bunny == True:
         gt_dir = '/home/daryl/datasets/bunny/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/bunny/groundtruth/'
     else:
         gt_dir = '/hdd/AIRSCAN/datasets/house_BAT6_full/groundtruth/'
         # gt_dir = '/home/justine/airscan_gym/gym-unrealcv/house_BAT6_full/groundtruth/'

     self.gt_pcl = []
     for i in range(len(self.houses)):
         # gt_fn = gt_dir + self.houses[i] + '_sampled_10k.ply'
         if self.bunny:
             gt_fn = gt_dir + self.houses[i].split('_2')[0] + '_rotate.ply'
         elif not (self.batch1 or self.batch2 or self.batch3 or self.batch4 or self.batch5 or self.batch8 or self.batch9 or self.batch10 or self.batch11 or self.batch12):
             gt_fn = gt_dir + self.houses[i].split('_WTR')[0] + '_WTR.ply'
         else:
             gt_fn = gt_dir + self.houses[i].split('_')[0]+'_'+self.houses[i].split('_')[1]+'_'+self.houses[i].split('_')[2]+ '.ply'
         # print('gt_fn before: ', gt_fn)
         if not self.bunny:
             if ('SETB' in gt_fn):
                 gt_fn = gt_fn.replace('SETB', 'SETA')
             elif ('SETC' in gt_fn):
                 gt_fn = gt_fn.replace('SETC', 'SETA')
             elif ('SETD' in gt_fn):
                 gt_fn = gt_fn.replace('SETD', 'SETA')
             elif ('SETE' in gt_fn):
                 gt_fn = gt_fn.replace('SETE', 'SETA')
         # print('gt_fn after: ', gt_fn)

         # print('gt', gt_fn)
         # gt_pcl = pcl.load(gt_fn)
         # gt_pcl = pcl.load('/home/daryl/datasets/BAT6_SETA_HOUSE8_WTR_sampled_10k.ply')
         # gt_pcl = np.asarray(gt_pcl)
         # print('gt pcl')
         # print('i', i)
         # print('gt_pcl', gt_fn)
         gt_pcl = o3d.read_point_cloud(gt_fn)
         gt_pcl = np.asarray(gt_pcl.points, dtype=np.float32)
         # print('o3d pcl', gt_pcl)


         self.gt_pcl.append(np.expand_dims(gt_pcl,axis=0))

    def _step(self, action = 0):
        # (velocity, angle) = self.ACTION_LIST[action]
        # print('action', action)
        self.count_steps += 1
        self.count_house_frames +=1
        # azimuth, elevation, distance  = self.discrete_actions[action]
        # change_pose = np.array((azimuth, elevation, distance))

        azimuth, elevation, distance  = action
        # print('az, elev, dist', azimuth, elevation, distance)
        change_pose = np.array((azimuth, elevation, distance))

        pose_prev = np.array(self.pose_prev)
        # print('pose prev', pose_prev)
        # print('action', change_pose)

        # MIN_elevation = 20
        MIN_elevation = 10
        # MAX_elevation = 70
        MAX_elevation = 80
        MIN_distance = 100
        MAX_distance = 150
        # if self.batch9 == True:
        #     MIN_distance = 250
        #     MAX_distance = 300


        # MAX_distance = 175
        # MAX_distance = 125

        pose_new = pose_prev + change_pose
        # pose_new = pose_prev + np.array([30,0,0]) # to test ICM
        if pose_new[2] > MAX_distance:
            pose_new[2] = MAX_distance
        elif pose_new[2] < MIN_distance:
            pose_new[2] = MIN_distance
        if (pose_new[1] >= MAX_elevation):
            pose_new[1] = MAX_elevation
        elif (pose_new[1] <= MIN_elevation):
            pose_new[1] = MIN_elevation
        # else:
            # pose_new[1] = 45.0
        if (pose_new[0] < 0):
            pose_new[0] = 360 + pose_new[0]
        elif (pose_new[0]>=359.99):
            pose_new[0] = pose_new[0] - 360

        # print('action ', action)
        # print('pose new', pose_new)
        # print(azimuth, elevation, distance )
        # collision, move_dist = self.unrealcv.move_rel2(self.cam_id, azimuth, elevation, distance)
        collision, move_dist = self.unrealcv.move_rel2(self.cam_id, pose_new[0], pose_new[1], pose_new[2])
        # print('collision', collision)
        # print('distance:   ', move_dist)

        self.pose_prev = pose_new
        # state = self.unrealcv.read_image(self.cam_id, 'lit')
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        # print('state shape', state.shape)
        depth_pt = self.unrealcv.read_depth(self.cam_id,mode='depthFusion')
        pose = self.unrealcv.get_pose(self.cam_id,'soft')
        depth = depth_conversion(depth_pt, 320)
        # pose_filename = self.log_dir+'frame-{:06}.pose.txt'.format(self.count_steps)
        # depth_filename = self.log_dir+'frame-{:06}.depth.npy'.format(self.count_steps)
        pose_filename = self.log_dir+'frame-{:06}.pose-{:06}.txt'.format(self.count_house_frames, self.house_id)
        depth_filename = self.log_dir+'frame-{:06}.depth-{:06}.npy'.format(self.count_house_frames, self.house_id)
        write_pose(pose, pose_filename)
        np.save(depth_filename, depth)
        reward, done = self.reward(collision,move_dist)


        # limit max step per episode
        if self.count_steps > self.max_steps:
            done = True
            # print('Reach Max Steps')

        return state, reward, done, {}

    # reset the environment
    def _reset(self, start_pose_rel = None):

       x,y,z,_, yaw, _ = self.startpose
       # self.house_id = 0
       #
       # for house in self.houses:
       #     self.unrealcv.hide_obj(house)
       #
       # self.unrealcv.show_obj(self.houses[self.house_id])

       if self.reset_type == 'random':
           distance = 1000
           azimuth = 0
           elevation = 45

           p=90
           distance = distance + random.randint(-250,250)
           azimuth = random.randint(0,359)
           elevation = random.randint(35,55)

           yaw_exp = 270 - azimuth
           pitch = -1*elevation

           y = distance*sin(radians(p-elevation))*cos(radians(azimuth))
           x = distance*sin(radians(p-elevation))*sin(radians(azimuth))

           z = distance*cos(radians(p-elevation))

           self.unrealcv.set_pose(self.cam_id,[x,y,z,0,yaw_exp,pitch]) # pose = [x, y, z, roll, yaw, pitch]

       else:
           for house in self.houses:
               self.unrealcv.hide_obj(house)
           if self.test == True:
               self.house_id += 1

               if self.house_id >= len(self.houses):
                   print('No. of unsolved houses:', self.unsolved_ctr)
                   print('Unsolved Houses')
                   for unsolved_id in self.unsolved_list:
                       print(self.houses[unsolved_id])

               print('Testing House: ', self.houses[self.house_id])

               print(self.house_id)
               print('house id', self.house_id)
               self.unrealcv.show_obj(self.houses[self.house_id])

           elif self.testSet == True:
               # self.house_id += 1
               #8 'dont use'
               # self.house_id = 41
               # self.house_id = 70
               # Add extracting house id's from a list of house numbers

               if self.testSetA == True:
                   # testA = [3,4,21,45,49]
                   if self.batch7 == True:
                       testA = [8,18,19,49]
                       testA = [8]
                   elif self.batch8 == True:
                       # testA =  [4, 11, 15, 22, 48]
                       testA =  [4]
                   elif self.batch9 == True:
                       testA =  [6, 14, 19, 35, 38]
                       testA =  [14]
                   elif self.batch10 == True:
                       testA =  [43, 39, 27, 46, 22]
                       testA =  [ 46]
                   elif self.batch11 == True:
                       testA =  [6, 7, 30, 30, 31]
                       testA =  [30]
                   elif self.batch12 == True:
                       testA =  [2, 5, 27, 34, 46]
                   elif self.batch1 == True:
                       testA = [1,16,19,35,36]
                       testA = [35]
                   elif self.batch2 == True:
                       testA = [6,10,25,35,38]
                       # testA = [6]
                   elif self.batch3 == True:
                       testA = [7,20,22,29,50]
                       # testA = [7]
                   elif self.batch4 == True:
                       testA = [1,14,18,32,42]
                       # testA = [18,32,42]
                   elif self.batch5 == True:
                       testA = [3,20,32,46,48]
                   else:
                       testA = [1,4,10,35,48]
                       # testA = [48]
                   testA.sort()
                   print('test set: ', testA)
                   if self.testA_id >= len(testA):
                       print('No. of unsolved houses:', self.unsolved_ctr)
                       print('Unsolved Houses')
                       print('id ', self.testA_id)
                       print('len ', len(testA))
                       for unsolved_id in self.unsolved_list:
                           print(self.housesA[unsolved_id])

                   i = 0
                   while(1):
                       num = int((self.housesA[i].split('HOUSE')[1]).split('_')[0])
                       if num == testA[self.testA_id]:
                           self.house_id = i
                           self.testA_id += 1
                           break
                       else:
                          i += 1
                   if self.test_all and (self.testA_id == len(testA)):
                       self.testSetA = False

                   print('Testing House: ', self.housesA[self.house_id])
                   self.unrealcv.show_obj(self.housesA[self.house_id])

               elif self.testSetB == True:
                   if self.batch7 == True:
                       testB = [6,7,17,39]
                   elif self.batch8 == True:
                       testB = [3, 20, 25, 40, 42]
                   elif self.batch9 == True:
                       testB = [4, 28, 38, 39, 43]
                   elif self.batch10 == True:
                       testB = [2, 46, 39, 22, 47]
                       testB = [ 46, 39, 22, 47]
                   elif self.batch11 == True:
                       testB = [2, 14, 18, 20, 28]
                       testB = [2]
                   elif self.batch12 == True:
                       testB = [15, 27, 33, 39, 50]
                   elif self.batch1 == True:
                       testB = [6,7,18,29,44]
                       # testB = [29]

                   elif self.batch2 == True:
                       testB = [9,11,20,28,43]
                       # testB = [11]
                   elif self.batch3 == True:
                       testB = [2,24,36,43,49]
                   elif self.batch4 == True:
                       testB = [2,3,32,11,13]
                       # testB = [32,11,13]
                   elif self.batch5 == True:
                       testB = [8,40,42,46,48]
                       testB = [40]

                   else:
                       testB = [3,4,21,45,49]
                   testB.sort()
                   if self.testB_id >= len(testB):
                       print('No. of unsolved houses:', self.unsolved_ctr)
                       print('Unsolved Houses')
                       for unsolved_id in self.unsolved_list:
                           print(self.housesB[unsolved_id])

                   i = 0
                   while(1):
                       num = int((self.housesB[i].split('HOUSE')[1]).split('_')[0])
                       if num == testB[self.testB_id]:
                           self.house_id = i
                           self.testB_id += 1
                           break
                       else:
                          i += 1

                   if self.test_all and (self.testB_id == len(testB)):
                       self.testSetB = False

                   print('house id', self.house_id)
                   print('Testing House: ', self.housesB[self.house_id])
                   print('Using: ', self.houses[self.house_id])
                   self.unrealcv.show_obj(self.housesB[self.house_id])

               elif self.testSetC == True:
                   if self.batch7 ==True:
                       testC = [3,26,38,39]
                   elif self.batch8 == True:
                       testC =  [8, 14, 16, 31, 50]
                   elif self.batch9 == True:
                       testC =  [6, 10, 21, 44, 47]
                   elif self.batch10 == True:
                       testC =  [9, 47, 33, 27, 10]
                   elif self.batch11 == True:
                       testC =  [1, 9, 22, 24, 49]
                   elif self.batch12 == True:
                       testC =  [10, 17, 19, 36, 44]
                   elif self.batch1 == True:
                       testC = [5,8,33,38,48]
                       # testC = [33]
                   elif self.batch2 == True:
                       testC = [12,14,21,35,40]
                   elif self.batch3 == True:
                       testC = [2,4,10,29,35]
                   elif self.batch4 == True:
                       testC = [5,9,22,34,46]
                       # testC = [22]
                   elif self.batch5 == True:
                       testC = [7,19,24,34,39]
                   else:
                       testC = [3,11,13,36,43]
                   testC.sort()
                   if self.testC_id >= len(testC):
                       print('No. of unsolved houses:', self.unsolved_ctr)
                       print('Unsolved Houses')
                       for unsolved_id in self.unsolved_list:
                           print(self.housesC[unsolved_id])

                   i = 0
                   while(1):
                       num = int((self.housesC[i].split('HOUSE')[1]).split('_')[0])
                       if num == testC[self.testC_id]:
                           self.house_id = i
                           self.testC_id += 1
                           break
                       else:
                          i += 1
                   if self.test_all and (self.testC_id == len(testC)):
                       self.testSetC = False
                   print('Testing House: ', self.housesC[self.house_id])
                   self.unrealcv.show_obj(self.housesC[self.house_id])

               elif self.testSetD == True:
                   if self.batch7 == True:
                       testD = [10,21,27,49]
                   elif self.batch8 == True:
                       testD = [1, 12, 14, 38, 41]
                   elif self.batch9 == True:
                       testD = [5, 7, 19, 25, 44]
                   elif self.batch10 == True:
                       testD = [41, 47, 26, 33, 45]
                       testD = [26]
                   elif self.batch11 == True:
                       testD = [8, 16, 33, 40, 41]
                   elif self.batch12 == True:
                       testD = [5, 21, 30, 32, 47]
                   elif self.batch1 == True:
                       testD = [1,15,17,28,30]
                       # testD = [28]
                   elif self.batch2 == True:
                       testD = [3,10,11,37,42]
                   elif self.batch3 == True:
                       testD = [4,22,24,30,36]
                   elif self.batch4 == True:
                       testD = [5,7,10,13,22]
                   elif self.batch5 == True:
                       testD = [1,11,34,38,48]
                   else:
                       testD = [6,21,22,26,30]
                   testD.sort()
                   if self.testD_id >= len(testD):
                       print('No. of unsolved houses:', self.unsolved_ctr)
                       print('Unsolved Houses')
                       for unsolved_id in self.unsolved_list:
                           print(self.housesD[unsolved_id])
                   i = 0
                   while(1):
                       num = int((self.housesD[i].split('HOUSE')[1]).split('_')[0])
                       if num == testD[self.testD_id]:
                           self.house_id = i
                           self.testD_id += 1
                           break
                       else:
                          i += 1
                   if self.test_all and (self.testD_id == len(testD)):
                       self.testSetD = False
                   print('Testing House: ', self.housesD[self.house_id])
                   self.unrealcv.show_obj(self.housesD[self.house_id])

               elif self.testSetE == True:
                   if self.batch7 == True:
                       testE = [3,10,23,48]
                   elif self.batch8 == True:
                       testE = [2, 22, 28, 30, 43]
                   elif self.batch9 == True:
                       testE = [3, 8, 27, 36, 43]
                   elif self.batch10 == True:
                       testE = [36, 20, 48, 46, 37]
                   elif self.batch11 == True:
                       testE = [2, 13, 21, 25, 34]
                   elif self.batch12 == True:
                       testE = [9, 12, 32, 33, 46]
                   elif self.batch1 == True:
                       testE = [8,9,27,31,48]
                   elif self.batch2 == True:
                       testE = [3,5,19,31,40]
                   elif self.batch3 == True:
                       testE = [13,14,21,33,34]
                   elif self.batch4 == True:
                       testE = [7,22,24,32,34]
                       # testE = [24]
                   elif self.batch5 == True:
                       testE = [13,34,39,40,43]
                   else:
                       testE = [4,10,11,46,48]
                       # testE = [10,11,46,48]
                       # testE = [11]

                   testE.sort()
                   i = 0
                   if self.test_all and (self.testE_id == len(testE)):
                       print('Test all')
                       print('No. of unsolved houses:', self.unsolved_ctr)
                   elif self.testE_id >= len(testE):
                       print('No. of unsolved houses:', self.unsolved_ctr)
                       print('Unsolved Houses')
                       for unsolved_id in self.unsolved_list:
                           print(self.housesE[unsolved_id])

                   while(1):
                       num = int((self.housesE[i].split('HOUSE')[1]).split('_')[0])
                       if num == testE[self.testE_id]:
                           # print('removed', self.housesB.pop(i))
                           self.house_id = i
                           self.testE_id += 1
                           break
                       else:
                          i += 1

                   print('Testing House: ', self.housesE[self.house_id])
                   self.unrealcv.show_obj(self.housesE[self.house_id])

           else:
               # self.house_id = random.randint(0, self.num_house-1) # randomize houses
               # self.house_id = 39 #House 46
               self.house_id = self.shuffle_ids.pop()
               if len(self.shuffle_ids) == 0:
                   self.shuffle_ids = self.ids.copy()
                   random.shuffle(self.shuffle_ids)

               self.unrealcv.show_obj(self.houses[self.house_id])

           self.unrealcv.set_pose(self.cam_id,self.startpose) # pose = [x, y, z, roll, yaw, pitch]

       state = self.unrealcv.get_observation(self.cam_id, self.observation_type)

       self.count_steps = 0
       self.count_house_frames = 0

       depth_pt = self.unrealcv.read_depth(self.cam_id,mode='depthFusion')
       pose = self.unrealcv.get_pose(self.cam_id,'soft')
       depth = depth_conversion(depth_pt, 320)
       # depth_filename = self.log_dir+'frame-{:06}.depth-{:06}.npy'.format(self.count_steps)
       # pose_filename = self.log_dir+'frame-{:06}.pose-{:06}.txt'.format(self.count_steps)
       pose_filename = self.log_dir+'frame-{:06}.pose-{:06}.txt'.format(self.count_house_frames, self.house_id)
       depth_filename = self.log_dir+'frame-{:06}.depth-{:06}.npy'.format(self.count_house_frames, self.house_id)
       write_pose(pose, pose_filename)
       np.save(depth_filename, depth)

       out_pcl_np = depth_fusion_mult(self.log_dir, first_frame_idx =0, base_frame_idx=1000, num_frames = self.count_house_frames + 1, save_pcd = self.save_pcd, max_depth = 1.0, house_id=self.house_id)
       # out_fn = 'log/house-' + '{:06}'.format(self.count_steps+1) + '.ply'
       # out_pcl = pcl.load(out_fn)
       # out_pcl_np = np.asarray(out_pcl)
       out_pcl_np = np.expand_dims(out_pcl_np,axis=0)
       self.cd_old = self.compute_chamfer(out_pcl_np)
       # print('cd old ', self.cd_old)
       self.pose_prev = np.array(self.start_pose_rel)

       return  state

    # close docker while closing openai gym
    # def _close(self):
       # self.docker.close()

    # calcuate reward according to your task
    def reward(self,collision, move_dist):

       done = False

       depth_start = time.time()

       out_pcl_np = depth_fusion_mult(self.log_dir, first_frame_idx =0, base_frame_idx=1000, num_frames = self.count_house_frames + 1, save_pcd = self.save_pcd, max_depth = 1.0, house_id=self.house_id)
       # print('out_pcl_np', out_pcl_np.shape)
       if out_pcl_np.shape[0] != 0:
           out_pcl_np = np.expand_dims(out_pcl_np,axis=0)
           cd = self.compute_chamfer(out_pcl_np)
       else:
           cd = 0.0
       cd_delta = cd - self.cd_old
       depth_end = time.time()

       # print('coverage: ', cd)
       # print("Depth Fusion time: ", depth_end - depth_start)
       if self.test or self.testSet:
           print('coverage: ', cd)
       # print('house frames: ', self.count_house_frames)
       # if cd > 94.0:
       # if cd > 87.0:
       if cd > 96.0:
           done = True
           reward = 100

       else:
           # reward = cd_delta*0.2
           reward = cd_delta
           # reward = cd_delta*0.4
           # reward += -2 # added to push minimization of steps
           reward += -0.02 * move_dist
           # print('move_dist', move_dist)
           # print('reward', reward)


       self.cd_old = cd
       self.total_distance += move_dist
       if ((self.test == True) or (self.testSet)) and (self.count_house_frames == 50):
           print('UNSOLVED')
           self.unsolved_ctr += 1
           self.unsolved_list.append(self.house_id)
           done = True
       elif (self.test_baseline) and (self.count_house_frames == 27):
           print('UNSOLVED')
           self.unsolved_ctr += 1
           self.unsolved_list.append(self.house_id)
           done = True
       # elif (self.count_house_frames == 50):
       #     done = True

       # print('total distance: ', self.total_distance)
       return reward, done

    # calcuate the 2D distance between the target and camera
    def cauculate_distance(self,target,current):
       error = abs(np.array(target) - np.array(current))[:2]# only x and y
       distance = math.sqrt(sum(error * error))
       return distance

    def load_env_setting(self,filename):
       f = open(self.get_settingpath(filename))
       type = os.path.splitext(filename)[1]
       if type == '.json':
            import json
            setting = json.load(f)
       elif type == '.yaml':
            import yaml
            setting = yaml.load(f)
       else:
            print('unknown type')

       self.cam_id = setting['cam_id']
       self.target_list = setting['targets']
       self.max_steps = setting['maxsteps']
       self.trigger_th = setting['trigger_th']
       self.height = setting['height']
       self.pitch = setting['pitch']
       self.start_pose_rel = setting['start_pose_rel']
       self.discrete_actions = setting['discrete_actions']
       self.continous_actions = setting['continous_actions']
       self.env_bin = setting['env_bin']
       self.env_name = setting['env_name']
       print('env name: ', self.env_name)
       print('env id: ', setting['env_bin'])
       return setting

    def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)

    def compute_chamfer(self, output):
       # with tf.Session('') as sess:
       # sess = K.get_session()
       # self.sess.run(tf.global_variables_initializer())
       # loss_out = self.sess.run(loss,feed_dict={inp_placeholder: output})
       with tf.device('/gpu:0'):
           sess = K.get_session()
           with sess.as_default():
           # with tf.Session('') as sess:

               # inp_placeholder = tf.placeholder(tf.float32)
               # reta,retb,retc,retd=self.nn_distance(inp_placeholder,self.gt_pcl)
               # with tf.name_scope('chamfer'):
               # reta,retb,retc,retd=self.nn_distance(output,self.gt_pcl)
               # print('house id: ', self.house_id)
               # print('gt name: ', self.houses[self.house_id].split('_WTR')[0] + '_WTR.ply')

               # house_17 = o3d.read_point_cloud('/home/daryl/Deep-RL-Keras/BAT2_SETA_HOUSE6.ply')
               # print(type(house_17))
               # house_17_np = np.asarray(house_17.points, dtype = np.float32)
               # house_17_np = np.expand_dims(house_17_np,axis=0)
               # output2 = house_17_np

               _,_,retc,_=self.nn_distance(output,self.gt_pcl[self.house_id])
               # _,_,retc,_=self.nn_distance(output,output2)

               # loss=tf.reduce_sum(reta)+tf.reduce_sum(retc)

               # loss=tf.reduce_sum(retc)
               dist_thresh = tf.greater(0.0008, retc)
               dist_mean = tf.reduce_mean(tf.cast(dist_thresh, tf.float32))

               # loss_out = tf.Tensor.eval(loss)
               coverage = tf.Tensor.eval(dist_mean)
               # coverage2 = tf.Tensor.eval(dist_mean2)
               # print('coverage2 ', coverage2)
               # loss_out = self.sess.run(loss,feed_dict={inp_placeholder: output})
               # print('coverage ', coverage)
               return coverage*100

    def nn_distance(self,xyz1,xyz2):
       '''
     Computes the distance of nearest neighbors for a pair of point clouds
     input: xyz1: (batch_size,#points_1,3)  the first point cloud
     input: xyz2: (batch_size,#points_2,3)  the second point cloud
     output: dist1: (batch_size,#point_1)   distance from first to second
     output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
     output: dist2: (batch_size,#point_2)   distance from second to first
     output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
       '''
       return self.nn_distance_module.nn_distance(xyz1,xyz2)
