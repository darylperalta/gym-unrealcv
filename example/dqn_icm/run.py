import json
import os
import time
from distutils.dir_util import copy_tree
import gym_unrealcv
import dqn
import gym
from constants import *
# from constants_pred import *
from example.utils import preprocessing, io_util
from gym import wrappers
import numpy as np

if __name__ == '__main__':
    # print('params: ', params_json)
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    # env.observation_space = env.observation_shape
    print('observation_space:', env.observation_space.shape)
    env.rendering = SHOW
    # must be discrete action
    assert env.action_type == 'discrete'

    ACTION_SIZE = env.action_space.n
    print('action size:         ', ACTION_SIZE)
    #ACTION_LIST = env.discrete_actions
    INPUT_CHANNELS = env.observation_space.shape[2]
    OBS_HIGH = env.observation_space.high
    OBS_LOW = env.observation_space.low
    OBS_RANGE = OBS_HIGH - OBS_LOW

    print('ACTIONS: ')
    discrete_actions = env.discrete_actions
    # print()

    process_img = preprocessing.preprocessor(observation_space=env.observation_space, length=INPUT_CHANNELS,
                                             size=(INPUT_HEIGHT,INPUT_WIDTH))

    #init log file
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(PARAM_DIR):
        os.makedirs(PARAM_DIR)


    #load init param
    if not CONTINUE:
        print('Not CONTINUE')
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0
        Agent = dqn.DeepQ_icm(ACTION_SIZE, MEMORY_SIZE, GAMMA, LEARNING_RATE,
                          INPUT_HEIGHT, INPUT_WIDTH,INPUT_CHANNELS,USE_TARGET_NETWORK,
                          enc_shape=ENC_SHAPE, icm_lr =LEARNINGRATE_ICM, pretrained=PRETRAINED, enc_path = ENC_PATH, vae =VAE)
        env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,force=True)

        io_util.create_csv_header(TRA_DIR)

    else:
        print('CONTINUE')
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            # print("ENTEREDDDD")
            # print("ENTEREDDDD")
            # print("ENTEREDDDD")
            # print("ENTEREDDDD")
            # print("ENTEREDDDD")
            print("ENTEREDDDD")
            print('params_json: ', params_json)
            d = json.load(outfile)
            explorationRate = d.get('explorationRate')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')
            Agent = dqn.DeepQ_icm(
                ACTION_SIZE,
                MEMORY_SIZE,
                GAMMA,
                LEARNING_RATE,
                INPUT_HEIGHT,
                INPUT_WIDTH,
                INPUT_CHANNELS,
                USE_TARGET_NETWORK
            )
            print('weights path: ', weights_path)

            Agent.loadWeights(weights_path)
            io_util.clear_monitor_files(MONITOR_DIR + 'tmp')
            copy_tree(monitor_path, MONITOR_DIR + 'tmp')
            env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,resume=True)

        io_util.create_csv_header(TRA_DIR)

    #main loop
    try:
        start_time = time.time()
        for epoch in range(current_epoch, MAX_EPOCHS, 1):
            obs = env.reset()

            #observation = io_util.preprocess_img((obs-OBS_LOW)/OBS_RANGE)
            observation = process_img.process_gray(obs, reset=True) # converts to gray

            if COLOR == True:
                observation = process_img.process_color(obs,reset=True)
            else:
                observation = process_img.process_gray(obs,reset=True)

            cumulated_reward = 0
            if (epoch % TEST_INTERVAL_EPOCHS != 0 or stepCounter < LEARN_START_STEP) and TRAIN is True :  # explore
                print('EXPLORE')
                EXPLORE = True
            else:
                EXPLORE = False
                print ("Evaluate Model")

            pose_prev = [0,45,1000]
            obs_new = obs
            for t in range(1000):

                start_req = time.time()

                if EXPLORE is True: #explore
                    action = Agent.feedforward(observation, explorationRate)

                    # action_env = action * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
                    # action_env = np.array(env.discrete_actions[action])
                    print('action type ', type(action))
                    action_env = np.array(discrete_actions[action])
                    # print('action env', action_env)
                    # print(action_env.type)
                    # obs_new, reward, done, info = env.step(action_env)

                    pose_new = pose_prev + action_env
                    # pose_new = pose_prev + np.array([30,0,0]) # to test ICM
                    if pose_new[2] > MAX_distance:
                        pose_new[2] = MAX_distance
                    elif pose_new[2] < MIN_distance:
                        pose_new[2] = MIN_distance
                    if (pose_new[1] > MAX_elevation):
                        pose_new[1] = MAX_elevation
                    elif (pose_new[1] < MIN_elevation):
                        pose_new[1] = MIN_elevation

                    print('action', action)
                    print('action env', action_env)
                    print('pose_new', pose_new)
                    obs_new, reward, done, info = env.step(pose_new)
                    #newObservation = io_util.preprocess_img((obs_new-OBS_LOW)/OBS_RANGE)
                    newObservation = process_img.process_gray(obs_new)

                    if COLOR == True:
                        newObservation = process_img.process_color(obs_new)
                    else:
                        newObservation = process_img.process_gray(obs_new)

                    stepCounter += 1

                    reward_i, l_i = Agent.get_intrinsic_reward(observation, action, newObservation)
                    reward = reward_i
                    Agent.addMemory(observation, action, reward, newObservation, done)

                    pose_prev = pose_new
                    observation = newObservation
                    if stepCounter == LEARN_START_STEP:
                        print("Starting learning")

                    if Agent.getMemorySize() >= LEARN_START_STEP:
                        Agent.learnOnMiniBatch(BATCH_SIZE,icm_only=False)

                        if explorationRate > FINAL_EPSILON and stepCounter > LEARN_START_STEP:
                            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / MAX_EXPLORE_STEPS
                    elif Agent.getMemorySize() >= LEARN_START_STEP_ICM:
                        Agent.learnOnMiniBatch(BATCH_SIZE,icm_only=True)
                        #elif stepCounter%(MAX_EXPLORE_STEPS * 1.5) == 0 :
                            #explorationRate = 0.99
                            #print 'Reset Exploration Rate'

                #test
                else:
                    action = Agent.feedforward(observation,0)
                    obs_new, reward, done, info = env.step(action)
                    newObservation = process_img.process_gray(obs_new)
                    #newObservation = io_util.preprocess_img((obs_new-OBS_LOW)/OBS_RANGE)
                    observation = newObservation

                if MAP:
                    io_util.live_plot(info)

                #io_util.save_trajectory(info,TRA_DIR,epoch)

                cumulated_reward += reward
                if done:
                    m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                    h, m = divmod(m, 60)

                    print ("EP " + str(epoch) +" Csteps= " + str(stepCounter) + " - {} steps".format(t + 1) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s) )
                        # SAVE SIMULATION DATA
                    if epoch % SAVE_INTERVAL_EPOCHS == 0 and TRAIN is True:
                        # save model weights and monitoring data
                        print('Save model')
                        Agent.saveModel(MODEL_DIR + '/dqn_ep' + str(epoch) + '.h5')

                        #backup monitor file
                        copy_tree(MONITOR_DIR+ 'tmp', MONITOR_DIR + str(epoch))

                        parameter_keys = ['explorationRate', 'current_epoch','stepCounter', 'FINAL_EPSILON','loadsim_seconds']
                        parameter_values = [explorationRate, epoch, stepCounter,FINAL_EPSILON, int(time.time() - start_time + loadsim_seconds)]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(PARAM_DIR + '/dqn_ep' + str(epoch) + '.json','w') as outfile:
                            json.dump(parameter_dictionary, outfile)



                    break



    except KeyboardInterrupt:
        print("Shutting down")
        env.close()
