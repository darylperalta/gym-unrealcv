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
    parser.add_argument("-e","--env_id", nargs='?', default='DepthFusionBGrayMultHouseRand-v0', help='Select the environment to run')
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

    episode_count = 100
    reward = 0
    done = False

    num_views = 16
    num_views = 34
    num_first_elev = 8

    distance = 150


    # actions = [4,4,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,2,2,0,0,0,0,0,0,0,0]
    actions = [4,4,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,2,2,0,0,0,0,0,0,0,0]

    while(True):
        ob = env.reset()
        steps = 0

        for a in actions:
            ob, reward, done, _ = env.step(a)
            steps += 1
            # print('steps', steps)
            if done == True:
                print('steps', steps)
                break

    env.close()
