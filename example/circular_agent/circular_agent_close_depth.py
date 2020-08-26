"""
Code for Single House Policy Experiment - Circular Path Baselines
"""

import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Circular Baseline for Single Houuse Policy Experiment")
    parser.add_argument("-e","--env_id", nargs='?', default='DepthFusionBGray-v0', help='Select the environment to run')
    parser.add_argument("--circ_type", dest='circ_type', type=int, default=2, help='circular baseline type (1,2,3). 1 for Circ 1 (bottom to top), 2 for Circ 2 (Mid, Bot, Top), 3 for Circ 3 (Top to bottom). Default is 2.')
    parser.add_argument("--azimuth_type", dest='azimuth_type', type=int, default=0, help='Azimuth resolution. Use 0 for 45 degree resolution and 1 for 22.5 degree resolution.')
    parser.add_argument("--distance_levels", dest='distance_levels', type=int, default=3, help='Number of distance levels (2 or 3). Default is 3.')

    args = parser.parse_args()

    print('envid: ', args.env_id)
    env = gym.make(args.env_id)
    env.seed(0)
    print('circular agent')

    print('args circ', args.circ_type)
    reward = 0
    done = False

    # init_pose = np.array([0.0, 45.0,distance])
    ob = env.reset()
    print('after reset first')
    i = 0
    filename = 'ob%d'%i
    filename = filename + '.png'
    # ob, reward, done, _ = env.step(action)

    if args.circ_type == 3:
        baseline_type = 1 # 1 - bottom to top,
    elif args.circ_type == 1:
        baseline_type = 2 # 2 top to bottom
    elif args.circ_type == 2:
        baseline_type = 0 # 0 - usual baseline
    else:
        print("Invalid Circular Type. Choose among (circ1,circ2,circ3).")
        print('Using default circ2')
        baseline_type = 0 # 0 - usual baseline

    if args.azimuth_type == 0:
        azimuth_type = 0 # 1 - bottom to top,
    elif args.azimuth_type == 1:
        azimuth_type = 0 # 2 top to bottom
    else:
        print("Invalid Azimuth Type. Choose 0 or 1.")
        print('Using default 0')
        azimuth_type = 0  # 0 - usual baseline

    if args.distance_levels == 3:
        distance_levels_num = 3
    elif args.distance_levels == 2:
        distance_levels_num = 2
    else:
        print("Invalid distance_levels. Choose 2 or 3.")
        print('Using default 3')
        distance_levels_num = 3

    # baseline_type = 0 # 0 - usual baseline, # 1 - bottom to top, # 2 top to bottom
    # azimuth_type = 0 # 0 - 45 deg resolution, 1 - 22.5 deg res
    # distance_levels_num = 3 # 2 - 2 levels, # 3 - 3 levels

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
            for i in range(7):
                action = 0
                ob, reward, done, _ = env.step(action)
                steps += 1
                print('steps', steps)
                print('reward: ', reward)
                print('done:', done)
                if done ==True:
                    ob = env.reset()
            for i in range(2):
                action = 2
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
            # action = 3
            # ob, reward, done, _ = env.step(action)
            # steps += 1
            # print('steps', steps)
            # for i in range(8):
            #     action = 0
            #     ob, reward, done, _ = env.step(action)
            #     steps += 1
            #     print('steps', steps)
            #     print('reward: ', reward)
            #     print('done:', done)
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
