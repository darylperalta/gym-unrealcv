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

    num_views = 18
    ob = env.reset()
    distance = 100
    init_pose = np.array([0.0, 45.0, 100.0])
    action = init_pose
    for i in range(num_views):
        filename = 'ob%d'%i
        filename = filename + '.png'
        # ob, reward, done, _ = env.step(action)

        cv2.imwrite(filename,ob)

        action[0] = action[0]+20
        # print(action)
        # print(type(action))
        # ob, reward, done, _ = env.step(action)
        ob, reward, done, _ = env.step(action)
        # print('ob shape: ', ob.shape)
        print('reward: ', reward)
        print('done:', done)

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
