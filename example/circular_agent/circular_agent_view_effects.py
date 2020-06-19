import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K
# from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e","--env_id", nargs='?', default='DepthFusionViewEffects-v0', help='Select the environment to run')
    args = parser.parse_args()
    print('envid: ', args.env_id)
    env = gym.make(args.env_id)

    env.rendering = True
    env.rendering = False

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.rendering = False
    env.seed(0)
    agent = RandomAgent(env.action_space)
    print('circular agent')

    reward = 0
    done = False

    # init_pose = np.array([0.0, 45.0,distance])
    ob = env.reset()
    print('after reset first')
    i = 0
    filename = 'ob%d'%i
    filename = filename + '.png'
    # ob, reward, done, _ = env.step(action)
    azimuth_type = -1 # -1 - 45 deg resolution, 0 - 22.5, +1 - 15 deg res
    # distance_levels_num = 1 # 2 - 2 levels, # 3 - 3 levels

    if azimuth_type == -1:
        left_num = 7
    elif azimuth_type == 0:
        left_num = 15
    else:
        left_num = 23
    # pri

    steps = 0
    for i in range(left_num):
        action = 0
        ob, reward, done, _ = env.step(action)
        steps += 1
        print('steps', steps)
        print('done:', done)
        if done == True:
            ob = env.reset()

    ob = env.reset()
