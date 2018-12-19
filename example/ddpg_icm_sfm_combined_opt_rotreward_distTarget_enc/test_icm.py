import gym
import gym_unrealcv
from distutils.dir_util import copy_tree
import os
import json
# from constants import *
from constants_pred import *
from ddpg import DDPG, DDPG_icm
from gym import wrappers
import time
from example.utils import preprocessing, io_util
from example.utils.odm import run_opensfm
import numpy as np
import cv2
from essentialMat import getMatrices, eulerAnglesToRotationMatrix
from autoencodercolored_512_notVAE import build_decoder



def test_icm2():
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
    print(forward_model_weights_path)

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
    Agent = DDPG_icm(ACTION_SIZE, MEMORY_SIZE, GAMMA,
                 LEARNINGRATE_CRITIC, LEARNINGRATE_ACTOR, TARGET_UPDATE_RATE,
                 INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS, enc_shape=ENC_SHAPE, icm_lr=LEARNINGRATE_ICM, pretrained=PRETRAINED, enc_path = ENC_PATH, vae=VAE )

    # input size = 84

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
    print('explore rate: ', explorationRate)
    if not os.path.exists(TRA_DIR):
        io_util.create_csv_header(TRA_DIR)
    #Build decoder
    decoder_path = '/home/daryl/gym-unrealcv/example/ddpg_icm_sfm_combined_opt_rotreward_distTarget_enc/state_encoder/decoder-512.hdf5'
    # decoder = build_decoder()
    decoder = build_decoder(VAE, ENC_SHAPE, channels=INPUT_CHANNELS)
    decoder.load_weights(decoder_path)

    for epoch in range(current_epoch + 1, MAX_EPOCHS + 1, 1):
        obs = env.reset()
        obs_new = obs

        # cv2.imshow('Initial observation', obs)
        # cv2.waitKey(0)
        #observation = io_util.preprocess_img(obs)
        # observation = process_img.process_gray(obs,reset=True)
        if COLOR == True:
            observation = process_img.process_color(obs,reset=True)
            # obs_show = process_img.process_color_ae(obs,reset=True)
        else:
            observation = process_img.process_gray(obs,reset=True)
        print('shape obs init ', observation[0].shape)
        # cv2.imshow('Initial observation gray', observation[0])
        # cv2.imshow('Initial observation color', obs_show)
        cv2.imshow('initial_obs color', obs_new)
        # cv2.imwrite('initial_obs.png',observation[0])
        # cv2.waitKey(0)

        init_state = Agent.icm.state_encoder.predict(observation)


        decoded_image = decoder.predict(init_state)

        cv2.imshow('out_input',decoded_image[0])
        # cv2.imwrite('out_input.png', decoded_image[0])
        cv2.waitKey(0)


        if TRAIN is True:
            EXPLORE = True
        else:
            EXPLORE = False

        pose_prev = [0,45,1000]
        save_path = '/hdd/AIRSCAN/icm_viewprediction/'
        for t in range(MAX_STEPS_PER_EPOCH):
        # action_env = np.array([45,10,100])
        # action_env = np.array([179.98845,-64.9999995,-499.99982])
            # action_env = np.array([73.385,-44.981,-199.9826])
            # action = (action_env - ACTION_LOW)/(ACTION_HIGH - ACTION_LOW)

            action_pred = Agent.actor.model.predict(observation)
            # print('action pred', action_pred)
            action = Agent.Action_Noise(action_pred, 0.0)
            action_env = action * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW

            print('action_env', action_env)
            print('action', action)
            print('action shape', action.shape)



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

            # print('pose_new: ', pose_new)
            obs_new, reward, done, info = env.step(pose_new)

            # cv2.imshow('new observation', obs_new)
            # cv2.waitKey(0)
            # observation_new = process_img.process_gray(obs_new,reset=True)
            if COLOR == True:
                observation_new = process_img.process_color(obs_new,reset=True)
            else:
                observation_new = process_img.process_gray(obs_new,reset=True)
            cv2.imshow('new observation', observation_new[0])
            fn_gt = 'gt_new%d'%t
            fn_gt = save_path+fn_gt+'.png'
            cv2.imwrite(fn_gt, (observation_new[0]*255).astype(np.uint8))
            print('new obs: ', fn_gt)

            # cv2.imwrite('new_obs.png', observation_new[0])
            # cv2.waitKey(0)

            action_batch = np.zeros((1,)+action_env.shape)
            action_batch[0] = action

            # new_state = Agent.icm.forward_model.predict([init_state,action])
            new_state = Agent.icm.forward_model.predict([init_state,action_batch])
            new_state_gt = Agent.icm.state_encoder.predict(observation_new)
            decoded_image_new_gt = decoder.predict(new_state_gt)
            cv2.imshow('decoded new observation', decoded_image_new_gt[0])
            fn_gt_decoded = 'gt_new_decoded%d'%t
            fn_gt_decoded = save_path+fn_gt_decoded+'.png'
            cv2.imwrite(fn_gt_decoded, (decoded_image_new_gt[0]*255).astype(np.uint8))

            # cv2.imwrite(decoded_image_new_gt)

            print('new obs: ', fn_gt_decoded)


            decoded_image_new = decoder.predict(new_state)
            cv2.imshow('out_input_icm',decoded_image_new[0])

            fn_out_icm = 'out_icm_decoded%d'%t
            fn_out_icm = save_path+fn_out_icm+'.png'
            cv2.imwrite(fn_out_icm, (decoded_image_new[0]*255).astype(np.uint8))
            # cv2.imwrite('decoded_image_new.png', decoded_image_new[0])
            cv2.waitKey(0)

            observation = observation_new
            pose_prev = pose_new
            init_state =  Agent.icm.state_encoder.predict(observation_new)


def test_icm():
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
    print(forward_model_weights_path)
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
    Agent = DDPG_icm(ACTION_SIZE, MEMORY_SIZE, GAMMA,
                 LEARNINGRATE_CRITIC, LEARNINGRATE_ACTOR, TARGET_UPDATE_RATE,
                 INPUT_HEIGHT, INPUT_WIDTH, 1, icm_lr=LEARNINGRATE_ICM)

    # input size = 84

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

    obs = env.reset()
    obs_new = obs

    # cv2.imshow('Initial observation', obs)
    # cv2.waitKey(0)
    #observation = io_util.preprocess_img(obs)
    # observation = process_img.process_gray(obs,reset=True)
    if COLOR == True:
        observation = process_img.process_color(obs,reset=True)
    else:
        observation = process_img.process_gray(obs,reset=True)
    print('shape obs init ', observation[0].shape)
    cv2.imshow('Initial observation gray', observation[0])
    # cv2.imwrite('initial_obs.png',observation[0])
    cv2.waitKey(0)

    init_state = Agent.icm.state_encoder.predict(observation)



    #Build decoder
    decoder = build_decoder()
    # decoder = build_decoder(VAE, ENC_SHAPE, channels=INPUT_CHANNELS)
    decoder.load_weights('/hdd/AIRSCAN/icm_models/decoder_checkpointsmodel-50_cont.hdf5')
    decoded_image = decoder.predict(init_state)

    cv2.imshow('out_input',decoded_image[0])
    # cv2.imwrite('out_input.png', decoded_image[0])
    cv2.waitKey(0)


    if TRAIN is True:
        EXPLORE = True
    else:
        EXPLORE = False

    pose_prev = [0,45,1000]
    # action_env = np.array([45,10,100])
    # action_env = np.array([179.98845,-64.9999995,-499.99982])
    action_env = np.array([73.385,-44.981,-199.9826])
    action = (action_env - ACTION_LOW)/(ACTION_HIGH - ACTION_LOW)
    print('action_env', action_env)
    print('action', action)
    print('action shape', action.shape)



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

    # print('pose_new: ', pose_new)
    obs_new, reward, done, info = env.step(pose_new)

    # cv2.imshow('new observation', obs_new)
    # cv2.waitKey(0)
    # observation_new = process_img.process_gray(obs_new,reset=True)
    if COLOR == True:
        observation = process_img.process_color(obs,reset=True)
    else:
        observation = process_img.process_gray(obs,reset=True)
    cv2.imshow('new observation', observation_new[0])
    # cv2.imwrite('new_obs.png', observation_new[0])
    cv2.waitKey(0)

    action_batch = np.zeros((1,)+action_env.shape)
    action_batch[0] = action

    # new_state = Agent.icm.forward_model.predict([init_state,action])
    new_state = Agent.icm.forward_model.predict([init_state,action_batch])

    decoded_image_new = decoder.predict(new_state)
    cv2.imshow('out_input_icm',decoded_image_new[0])
    # cv2.imwrite('decoded_image_new.png', decoded_image_new[0])
    cv2.waitKey(0)



if __name__ == '__main__':
    test_icm2()
