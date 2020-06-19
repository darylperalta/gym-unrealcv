import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import numpy as np
from example.utils.odm import run_opensfm
# IMAGE_PATH = '/home/daryl/OpenDroneMap_v0_3_1/images_icm'
IMAGE_PATH = '/home/daryl/OpenSfM-0.2.0/data/unreal_test'
DEPTHMAPS_PATH = '/home/daryl/OpenSfM-0.2.0/data/unreal_depth'
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e","--env_id", nargs='?', default='Search-RrMultiPlantsDiscreteTest-v0', help='Select the environment to run')
    args = parser.parse_args()
    print('envid: ', args.env_id)
    env = gym.make(args.env_id)

    env.rendering = True

    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # You provide the directory to write to (can be an existing
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    poses_file = '/hdd/AIRSCAN/daryl/experiments/48views.txt'
    poses = np.loadtxt(poses_file)


    num_views = poses.shape[0]
    num_views = 36
    ob = env.reset()
    # distance = 1000
    init_pose = np.array([0.0, 45.0, 1000.0])
    # action = init_pose
    print('num views: ', num_views)
    for i in range(num_views):

        action = poses[i]

        # action[0] = action[0]+10
        # print(action)
        # print(type(action))
        # ob, reward, done, _ = env.step(action)
        ob, reward, done, _ = env.step(action)
        # print('ob shape: ', ob.shape)
        print('reward: ', reward)
        print('done:', done)
        filename = '/ob%d'%i
        filename = IMAGE_PATH+'/images'+filename + '.png'

        cv2.imwrite(filename,ob)
        # action = agent.act(ob,reward,done)
        #action = [57.0, 30.0, 1484.0]
        print('Reconstructing images...')
        print(IMAGE_PATH)
        ret_val = run_opensfm(IMAGE_PATH, DEPTHMAPS_PATH,i)
        print('returned: ', ret_val)
        # if done:
              # break
    # print('Reconstructing images...')
    # print(IMAGE_PATH)
    # ret_val = run_opensfm(IMAGE_PATH)
    # print('returned: ', ret_val)

    # for i in range(episode_count):
    #     ob = env.reset()
    #     while True:
    #         action = agent.act(ob, reward, done)
    #         ob, reward, done, _ = env.step(action)
    #         if done:
    #             break

    # Close the env and write monitor result info to disk
    env.close()
