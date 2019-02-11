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

    num_views = 16
    num_views = 24
    num_first_elev = 8
    # num_views = 26
    # num_first_elev = 13

    # num_views = 4

    distance = 100
    init_pose = np.array([0.0, 45.0, 100.0])
    ob = env.reset()
    i = 0
    filename = 'ob%d'%i
    filename = filename + '.png'
    # ob, reward, done, _ = env.step(action)

    cv2.imwrite(filename,ob)
    # init_pose = np.array([0.0, 45.0, 50.0])
    action = init_pose.copy()
    for i in range(1,num_views):



        action[0] = action[0]+45
        # action[0] = action[0]+90


        # print(action)
        # print(type(action))
        # ob, reward, done, _ = env.step(action)
        if i == (num_first_elev):
            action[1] = action[1]-25
        if i == (num_first_elev*2):
            action[1] = action[1]+50
        ob, reward, done, _ = env.step(action)
        filename = 'ob%d'%i
        filename = filename + '.png'


        # ob, reward, done, _ = env.step(action)

        cv2.imwrite(filename,ob)
        # print('ob shape: ', ob.shape)
        print('reward: ', reward)
        print('done:', done)
    # i = i+1
    # print('i:: ', i)
    # action = init_pose
    # action[1] = 89.9
    # print('action', action)
    # ob, reward, done, _ = env.step(action)
    # filename = 'ob%d'%i
    # filename = filename + '.png'
    # cv2.imwrite(filename,ob)
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
