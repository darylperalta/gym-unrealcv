import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e","--env_id", nargs='?', default='curiosityChangePos-v0', help='Select the environment to run')
    args = parser.parse_args()
    print('envid: ', args.env_id)
    env = gym.make(args.env_id)

    env.rendering = True

    AE_dataset_path = '/home/daryl/datasets/unreal_images'
    out_path = '/home/daryl/datasets/unreal_images'

    delta_elev = 6
    delta_azimuth = 6
    delta_distance = 20

    MIN_elevation = 25
    MAX_elevation = 65
    MIN_distance = 600
    MAX_distance = 2000
    # MAX_steps = 10000
    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    num_views = 36
    ob = env.reset()
    distance = 1000
    # init_pose = np.array([0.0, 45.0, 1000.0])
    init_pose = np.array([0.0, MIN_elevation, MIN_distance])
    action = init_pose
    ob, reward, done, _ = env.step(action)
    for d in range(MIN_distance, MAX_distance, delta_distance):

        for e in range(MIN_elevation, MAX_elevation, delta_elev):

            for a in range(0,359,delta_azimuth):
                filename = 'ob_a%d_e%d_d%d'%(a,e,d)
                filename = out_path+'/'+filename + '.png'

                cv2.imwrite(filename,ob)

                # action[0] = action[0]+10
                action[0] = a
                action[1] = e
                action[2] = d

                ob, reward, done, _ = env.step(action)

                print('d, e, a: ', d, e, a)


        # action = agent.act(ob,reward,done)
        #action = [57.0, 30.0, 1484.0]

        # if done:
              # break



    # for i in range(episode_count):
    #     ob = env.reset()
    #     while True:
    #         action = agent.act(ob, reward, done)
    #         ob, reward, done, _ = env.step(action)
    #         if done:
    #             break

    # Close the env and write monitor result info to disk
    env.close()
