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
    parser.add_argument("-e","--env_id", nargs='?', default='DepthFusionBGray-v0', help='Select the environment to run')
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
    # init_pose = np.array([0.0, 45.0,distance])
    ob = env.reset()
    print('after reset first')
    i = 0
    filename = 'ob%d'%i
    filename = filename + '.png'
    # ob, reward, done, _ = env.step(action)
    baseline_type = 1 # 0 - usual baseline, # 1 - bottom to top, # 2 top to bottom
    azimuth_type = 0 # 0 - 45 deg resolution, 1 - 22.5 deg res
    distance_levels_num = 3 # 2 - 2 levels, # 3 - 3 levels

    print('Baseline type:', baseline_type)
    print('Distance levels num:', distance_levels_num)

    if distance_levels_num == 2:
        if baseline_type == 0:
            '''for 45 azimuth resolution '''
            steps = 0
            print('steps', steps)
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done:', done)
                print('get close')
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            action = 3
            ob, reward, done, _ = env.step(action)
            steps += 1
            print('steps', steps)
            print('done:', done)
            for i in range(8):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                if done ==True:
                    ob = env.reset()
            action = 3
            ob, reward, done, _ = env.step(action)
            steps += 1
            print('steps', steps)
            for i in range(8):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
        elif baseline_type == 1:

            '''for 45 azimuth resolution '''
            steps = 0
            print('steps', steps)

            for i in range(1):
                action = 3
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go down')
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 2
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go up')
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            print('steps', steps)
            print('done:', done)
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 2
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go up')
                if done ==True:
                    ob = env.reset()

            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                if done ==True:
                    ob = env.reset()
        elif baseline_type == 2:

            '''for 45 azimuth resolution '''
            steps = 0
            print('steps', steps)
            for i in range(1):
                action = 2
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go up')
                print('done', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done', done)
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 3
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go down')
                print('done', done)
                if done ==True:
                    ob = env.reset()
            print('steps', steps)
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 3
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go down')
                print('done', done)
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done', done)
                if done ==True:
                    ob = env.reset()
    if distance_levels_num == 3:
        if baseline_type == 0:
            '''for 45 azimuth resolution '''
            steps = 0
            print('steps', steps)
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done:', done)
                print('get close')
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            action = 3
            ob, reward, done, _ = env.step(action)
            steps += 1
            print('steps', steps)
            print('done:', done)
            for i in range(8):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                if done ==True:
                    ob = env.reset()
            action = 3
            ob, reward, done, _ = env.step(action)
            steps += 1
            print('steps', steps)
            for i in range(8):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
        elif baseline_type == 1:

            '''for 45 azimuth resolution '''
            steps = 0
            print('steps', steps)

            for i in range(1):
                action = 3
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go down')
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 2
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go up')
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            print('steps', steps)
            print('done:', done)
            for i in range(8):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 2
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go up')
                if done ==True:
                    ob = env.reset()

            for i in range(8):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                if done ==True:
                    ob = env.reset()
        elif baseline_type == 2:

    		# [45, 0, 0],
    		# [-45, 0, 0],
    		# [0, 25, 0],
    		# [0, -25, 0],
    		# [0, 0, -25],
    		# [0, 0, 25]
            '''for 45 azimuth resolution '''
            steps = 0
            print('steps', steps)
            for i in range(1):
                action = 2
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go up')
                print('done', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done', done)
                if done ==True:
                    ob = env.reset()

            for i in range(1):
                action = 4
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('get close')
                print('done:', done)
                if done ==True:
                    ob = env.reset()

            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 3
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go down')
                print('done', done)
                if done ==True:
                    ob = env.reset()
            print('steps', steps)
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(1):
                action = 3
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('go down')
                print('done', done)
                if done ==True:
                    ob = env.reset()
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('done', done)
                if done ==True:
                    ob = env.reset()

    # Close the env and write monitor result info to disk
    env.close()
