import gym
import gym_unrealcv
from distutils.dir_util import copy_tree
import os
import json
from constants import *
# from constants_pred import *
from ddpg import DDPG, DDPG_icm
from gym import wrappers
import time
from example.utils import preprocessing, io_util
import numpy as np
import cv2
from essentialMat import getMatrices, eulerAnglesToRotationMatrix
from example.utils.odm import run_opensfm


if __name__ == '__main__':
    print('Environment: ', ENV_NAME)
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    # env.observation_space = env.observation_shape
    print('observation_space:', env.observation_space.shape) # (240, 320, 3)
    env.rendering = SHOW
    assert env.action_type == 'continuous'
    ACTION_SIZE = env.action_space.shape[0]
    print('action size: ', ACTION_SIZE) # 3
    ACTION_HIGH = env.action_space.high
    print('Action HIGH: ', ACTION_HIGH) # [100 45 1]
    ACTION_LOW = env.action_space.low
    print('Action LOW: ', ACTION_LOW) # [0 -45 1]
    INPUT_CHANNELS = env.observation_space.shape[2]
    OBS_HIGH = env.observation_space.high
    OBS_LOW = env.observation_space.low
    OBS_RANGE = OBS_HIGH - OBS_LOW
    # print('OBS High: ', OBS_HIGH)
    # print('OBS Low: ', OBS_LOW)
    # print('OBS Range: ', OBS_RANGE)

    process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))

    #init log file
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(PARAM_DIR):
        os.makedirs(PARAM_DIR)

    # channel = 1
    # Agent = DDPG_icm(ACTION_SIZE, MEMORY_SIZE, GAMMA,
    #              LEARNINGRATE_CRITIC, LEARNINGRATE_ACTOR, TARGET_UPDATE_RATE,
    #              INPUT_SIZE, INPUT_SIZE, 3, icm_lr=LEARNINGRATE_ICM)
    Agent = DDPG_icm(ACTION_SIZE, MEMORY_SIZE, GAMMA,
                 LEARNINGRATE_CRITIC, LEARNINGRATE_ACTOR, TARGET_UPDATE_RATE,
                 INPUT_HEIGHT, INPUT_WIDTH, 1, enc_shape=(288,),icm_lr=LEARNINGRATE_ICM)
    # input size = 84

    #load init param
    if not CONTINUE:
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0
        env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,force=True)

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            explorationRate = d.get('explorationRate')
            current_epoch = d.get('current_epoch')
            stepCounter = d.get('stepCounter')
            loadsim_seconds = d.get('loadsim_seconds')
            Agent.loadWeights(critic_weights_path, actor_weights_path,state_encoder_weights_path,inverse_model_weights_path,forward_model_weights_path)
            io_util.clear_monitor_files(MONITOR_DIR + 'tmp')
            copy_tree(monitor_path, MONITOR_DIR + 'tmp')
            env = wrappers.Monitor(env, MONITOR_DIR + 'tmp', write_upon_reset=True,resume=True)

    if not os.path.exists(TRA_DIR):
        io_util.create_csv_header(TRA_DIR)

    try:
        start_time = time.time()
        metric_reward = 0
        num_success = 0
        for epoch in range(current_epoch + 1, MAX_EPOCHS + 1, 1):
            obs = env.reset()
            obs_new = obs
            #observation = io_util.preprocess_img(obs)
            observation = process_img.process_gray(obs,reset=True)
            # print('observation shape: ', observation.shape)
            # cv2.imshow('Image', observation)
            # cv2.waitKey(0)
            # `print('shape of obs:       ', observation.shape)
            cumulated_reward = 0
            #if ((epoch) % TEST_INTERVAL_EPOCHS != 0 or stepCounter < LEARN_START_STEP) and TRAIN is True :  # explore
            if TRAIN is True:
                EXPLORE = True
            else:
                EXPLORE = False

            #else:
            #    EXPLORE = False
            #    print ("Evaluate Model")
            pose_prev = [0,45,1000]
            obs_prev = obs
            for t in range(MAX_STEPS_PER_EPOCH):

                start_req = time.time()

                if EXPLORE is True: #explore

                    action_pred = Agent.actor.model.predict(observation)
                    # print('action pred', action_pred)
                    action = Agent.Action_Noise(action_pred, explorationRate)
                    # print('action',action)
                    # print(action.shape)
                    #print action

                    action_env = action * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
                    # print('action env', action_env)
                    # obs_new, reward, done, info = env.step(action_env)

                    pose_new = pose_prev + action_env
                    if pose_new[2] > MAX_distance:
                        pose_new[2] = MAX_distance
                    elif pose_new[2] < MIN_distance:
                        pose_new[2] = MIN_distance
                    if (pose_new[1] > MAX_elevation):
                        pose_new[1] = MAX_elevation
                    elif (pose_new[1] < MIN_elevation):
                        pose_new[1] = MIN_elevation

                    # print('pose_new: ', pose_new)
                    obs_new, reward, done, info = env.step(pose_new)
                    _,_,R,_ = getMatrices(obs_prev,obs_new)
                    R2= eulerAnglesToRotationMatrix(action_env)
                    # print('R diff: ', R2-R)
                    rdiff = R2-R
                    r_rot = - np.mean(np.square(rdiff))
                    print('rdiff: ', r_rot)
                    # print('R: ', R)

                    # print('action_env: ',action_env)
                    # print('action shape: ', action_env.shape)

                    newObservation = process_img.process_gray(obs_new)
                    #newObservation = io_util.preprocess_img(obs_new)
                    stepCounter += 1

                    # additions for ICM
                    # print("obs shape: ",observation.shape)
                    action_batch = np.zeros((1,)+action_env.shape)
                    action_batch[0] = action
                    reward_i = Agent.get_intrinsic_reward(observation, action_batch, newObservation)
                    print('reward_dist, reward_i: ', reward, reward_i)
                    # print('reward:  ', reward)
                    # reward_total = 0.2*reward_i + 0.8*reward
                    # reward_total = 0.8*reward_i + 0.2*reward
                    # reward_total = 0.6*reward_i + 0.2*reward + 0.2*r_rot
                    reward_total = 0.5*reward_i + 0.5*r_rot
                    # reward_total = reward_i
                    # Agent.addMemory(observation, action, reward, newObservation, done)
                    Agent.addMemory(observation, action, reward_total, newObservation, done)
                    observation = newObservation

                    pose_prev = pose_new

                    if stepCounter == LEARN_START_STEP:
                        print("Starting learning")


                    if Agent.getMemorySize() >= LEARN_START_STEP:
                        Agent.learnOnMiniBatch(BATCH_SIZE)
                        if explorationRate > FINAL_EPSILON and stepCounter > LEARN_START_STEP:
                            explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / MAX_EXPLORE_STEPS
                        #elif stepCounter % (MAX_EXPLORE_STEPS * 1.5) == 0:
                        #    explorationRate = 0.99
                        #    print 'Reset Exploration Rate'
                #test
                else:
                    # action_batch = Agent.actor.model.predict(observation)
                    # action_batch_norm = action_batch
                    # for i in range(len(action_batch[0])):
                    #     # noise = np.random.normal(0.5,0.5)
                    #     # action_batch[0][i] = (1 - explorationRate) * action_pred[0][i] + explorationRate * noise
                    #     action_batch_norm[0][i] = max(0,action_batch_norm[0][i])
                    #     action_batch_norm[0][i] = min(1,action_batch_norm[0][i])
                    # action = action_batch_norm[0]
                    filename = '/ob%d'%t
                    # filename = LOG_NAME_SAVE+filename + '.png'
                    IMAGE_PATH2 = '_EP%d' % t
                    IMAGE_PATH2 = IMAGE_PATH + IMAGE_PATH2
                    os.mkdir(IMAGE_PATH2)
                    # filename = IMAGE_PATH + '/images' + filename + '.png'
                    IMAGE_PATH2_images = IMAGE_PATH2+'/images'
                    os.mkdir(IMAGE_PATH2_images)

                    IMAGE_PATH2_OLD = '_EP%d' % (t-1)
                    IMAGE_PATH2_OLD = IMAGE_PATH + IMAGE_PATH2_OLD
                    IMAGE_PATH2_OLD_images = IMAGE_PATH2_OLD + '/images'


                    copy_cmd = 'cp -r '+IMAGE_PATH2_OLD+ ' ' +IMAGE_PATH2

                    os.system(copy_cmd)
                    filename = IMAGE_PATH2 + '/images' + filename + '.png'
                    cv2.imwrite(filename,obs_new)

                    print('Reconstructing images...')
                    print(IMAGE_PATH)
                    # ret_val = run_opensfm(IMAGE_PATH, DEPTHMAPS_PATH,t)
                    # DEPTHMAPS_PATH2 = '_EP%d' % t
                    # DEPTHMAPS_PATH2 = DEPTHMAPS_PATH + DEPTHMAPS_PATH2
                    # os.mkdir(DEPTHMAPS_PATH2)
                    # ret_val = run_opensfm(IMAGE_PATH2, DEPTHMAPS_PATH,t)
                    # print('returned: ', ret_val)
                    action_pred = Agent.actor.model.predict(observation)
                    action = Agent.Action_Noise(action_pred, 0)



                    action_env = action * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
                    print('action env: ', action_env)

                    pose_new = pose_prev + action_env
                    if pose_new[2] > 2000:
                        pose_new[2] = 2000
                    if (pose_new[1] > 65):
                        pose_new[1] = 65
                    elif (pose_new[1]<25):
                        pose_new[1] = 25

                    # obs_new, reward, done, info = env.step(action_env_prev+action_env)
                    obs_new, reward, done, info = env.step(pose_new)
                    newObservation = process_img.process_gray(obs_new)
                    #newObservation = io_util.preprocess_img(obs_new)
                    action_batch = np.zeros((1,)+action_env.shape)
                    action_batch[0] = action
                    reward_i = Agent.get_intrinsic_reward(observation, action_batch, newObservation)
                    print('reward_i: ', reward_i)
                    # print('diff: ', diff)

                    observation = newObservation
                    # reward_total = reward
                    reward_total = reward_i
                    pose_prev = pose_new

                #print 'step time:' + str(time.time() - start_req)
                if MAP:
                    io_util.live_plot(info)
                #io_util.save_trajectory(info, TRA_DIR, epoch)

                # cumulated_reward += reward
                cumulated_reward += reward_total
                if done:
                    m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                    h, m = divmod(m, 60)
                    metric_reward += cumulated_reward
                    if reward>= 0:
                        num_success += 1
                        print('success')
                    print('num success: ', num_success)
                    print ("EP " + str(epoch) +" Csteps= " + str(stepCounter) + " - {} steps".format(t + 1) + " - CReward: " + str(
                        round(cumulated_reward, 2)) + "  Eps=" + str(round(explorationRate, 2)) + "  Time: %d:%02d:%02d" % (h, m, s) )
                        # SAVE SIMULATION DATA
                    print('Metric reward: ', metric_reward)
                    if (epoch) % SAVE_INTERVAL_EPOCHS == 0 and TRAIN is True:
                        # save model weights and monitoring data
                        print('Save model')
                        Agent.saveModel( MODEL_DIR + '/ep' +str(epoch))

                        copy_tree(MONITOR_DIR + 'tmp', MONITOR_DIR + str(epoch))
                        # save simulation parameters.
                        parameter_keys = ['explorationRate', 'current_epoch','stepCounter', 'FINAL_EPSILON','loadsim_seconds']
                        parameter_values = [explorationRate, epoch, stepCounter,FINAL_EPSILON, int(time.time() - start_time + loadsim_seconds)]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(PARAM_DIR + '/' + str(epoch) + '.json','w') as outfile:
                            json.dump(parameter_dictionary, outfile)

                    break

    except KeyboardInterrupt:
        print("Shutting down")
        env.close()
