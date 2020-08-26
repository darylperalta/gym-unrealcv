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
from gym_unrealcv.envs.utils.utils_depthFusion import write_pose, write_depth, depth_fusion, depth_conversion, poseRelToAbs, poseOrigin
import time
import open3d as o3d

o3d.set_verbosity_level(o3d.VerbosityLevel.Error)

import tensorflow as tf
from tensorflow.python.framework import ops
import keras.backend as K

class depthFusion_keras(gym.Env):
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
                resolution = (640,480),
                log_dir='log/'
    ):
     setting = self.load_env_setting(setting_file)
     self.cam_id = 0
     # self.reset_type = 'random'
     self.reset_type = 'test'
     self.log_dir = log_dir
     # print('gt_fn', self.gt_fn)
     gt_pcl = o3d.read_point_cloud(self.gt_fn)
     self.gt_pcl = np.asarray(gt_pcl.points, dtype=np.float32)
     self.gt_pcl = np.expand_dims(self.gt_pcl,axis=0)

     # start unreal env
     docker = False
     # self.unreal = env_unreal.RunUnreal(ENV_BIN="house_withfloor/MyProject2/Binaries/Linux/MyProject2")
     self.unreal = env_unreal.RunUnreal(ENV_BIN=self.env_bin)
     env_ip,env_port = self.unreal.start(docker,resolution)

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

     azimuth, elevation, distance = self.start_pose_rel
     # print('start pose rel', azimuth,elevation,distance)
     self.startpose = poseRelToAbs(azimuth, elevation, distance)
     # print('start_pose: ', self.startpose)
     ''' create base frame '''
     poseOrigin(self.log_dir+'frame-{:06}.pose.txt'.format(1000))
     # ACTION: (Azimuth, Elevation, Distance)

     self.count_steps = 0
     # self.max_steps = 35
     self.target_pos = ( -60,   0,   50)
     self.pose_prev = np.array(self.start_pose_rel)
     self.nn_distance_module =tf.load_op_library(self.nn_distance_path)

     self.total_distance = 0

    def _step(self, action = 0):
        self.count_steps += 1
        azimuth, elevation, distance  = self.discrete_actions[action]
        change_pose = np.array((azimuth, elevation, distance))
        pose_prev = np.array(self.pose_prev)

        MIN_elevation = 20
        MAX_elevation = 70
        MIN_distance = 100
        MAX_distance = 150
        # MAX_distance = 125

        pose_new = pose_prev + change_pose

        if pose_new[2] > MAX_distance:
            pose_new[2] = MAX_distance
        elif pose_new[2] < MIN_distance:
            pose_new[2] = MIN_distance
        if (pose_new[1] >= MAX_elevation):
            pose_new[1] = MAX_elevation
        elif (pose_new[1] <= MIN_elevation):
            pose_new[1] = MIN_elevation
        else:
            pose_new[1] = 45.0
        if (pose_new[0] < 0):
            pose_new[0] = 360 + pose_new[0]
        elif (pose_new[0]>=359):
            pose_new[0] = pose_new[0] - 360


        collision, move_dist = self.unrealcv.move_rel2(self.cam_id, pose_new[0], pose_new[1], pose_new[2])

        self.pose_prev =pose_new
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        depth_pt = self.unrealcv.read_depth(self.cam_id,mode='depthFusion')
        pose = self.unrealcv.get_pose(self.cam_id,'soft')
        depth = depth_conversion(depth_pt, 320)
        pose_filename = self.log_dir+'frame-{:06}.pose.txt'.format(self.count_steps)
        depth_filename = self.log_dir+'frame-{:06}.depth.npy'.format(self.count_steps)
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
           self.unrealcv.set_pose(self.cam_id,self.startpose) # pose = [x, y, z, roll, yaw, pitch]

       state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
       self.count_steps = 0
       depth_pt = self.unrealcv.read_depth(self.cam_id,mode='depthFusion')
       pose = self.unrealcv.get_pose(self.cam_id,'soft')
       depth = depth_conversion(depth_pt, 320)
       pose_filename = self.log_dir+'frame-{:06}.pose.txt'.format(self.count_steps)
       depth_filename = self.log_dir+'frame-{:06}.depth.npy'.format(self.count_steps)
       write_pose(pose, pose_filename)
       np.save(depth_filename, depth)

       out_pcl_np = depth_fusion(self.log_dir, first_frame_idx =0, base_frame_idx=1000, num_frames = self.count_steps + 1, save_pcd = False, max_depth = 1.0)
       out_pcl_np = np.expand_dims(out_pcl_np,axis=0)
       self.cd_old = self.compute_chamfer(out_pcl_np)
       # print('cd old ', self.cd_old)
       self.pose_prev = np.array(self.start_pose_rel)

       return  state

    def reward(self,collision, move_dist):

       done = False
       # depth_start = time.time()
       out_pcl_np = depth_fusion(self.log_dir, first_frame_idx =0, base_frame_idx=1000, num_frames = self.count_steps + 1, save_pcd = False, max_depth = 1.0)
       # print('out_pcl_np', out_pcl_np.shape)
       if out_pcl_np.shape[0] != 0:
           out_pcl_np = np.expand_dims(out_pcl_np,axis=0)
           cd = self.compute_chamfer(out_pcl_np)
       else:
           cd = 0.0
       cd_delta = cd - self.cd_old

       # depth_end = time.time()
       target_coverage = 96.0
       # target_coverage = 97.0
       if cd > target_coverage:
           done = True
           reward = 100
            # print('covered', self.count_steps)
       else:
           reward = cd_delta
           reward += -2 + -0.02 * move_dist # added to push minimization of steps

       self.cd_old = cd
       self.total_distance += move_dist

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
       self.gt_fn = setting['pointcloud_path']
       self.nn_distance_path = self.get_nn_distance(setting['nn_distance_path'])
       print('env name: ', self.env_name)
       print('env id: ', setting['env_bin'])
       return setting

    def get_settingpath(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, 'envs/setting', filename)

    def get_nn_distance(self, filename):
       import gym_unrealcv
       gympath = os.path.dirname(gym_unrealcv.__file__)
       return os.path.join(gympath, filename)

    def compute_chamfer(self, output):
       with tf.device('/gpu:0'):
           sess = K.get_session()
           with sess.as_default():
               _,_,retc,_=self.nn_distance(output,self.gt_pcl)
               dist_thresh = tf.greater(0.0008, retc)
               dist_mean = tf.reduce_mean(tf.cast(dist_thresh, tf.float32))
               coverage = tf.Tensor.eval(dist_mean)

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
