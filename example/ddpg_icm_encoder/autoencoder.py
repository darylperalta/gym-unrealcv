import numpy as np
import math
# from keras.initializations import normal, identity
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Activation,Conv2DTranspose, Lambda, Reshape
from keras.layers import Conv2D, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import ELU
from keras.layers import Activation
import keras.backend as K
import tensorflow as tf
from icm_model import icm_class
from constants import *
from glob import glob
import random
import os
from example.utils import preprocessing
# from constants import *
import gym
import gym_unrealcv
import cv2
from keras.callbacks import ModelCheckpoint
# channels = 1
# img_rows = 240
# img_cols = 320
# img_shape = (channels, img_rows, img_cols)
# action_size = 3
# enc_shape = (288,)

def build_encoder(channels = 1,img_rows = 240,img_cols = 320, action_size = 3,enc_shape = (288,)):

    with tf.device(TF_DEVICE):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)


    img_shape = (channels, img_rows, img_cols)
    if K.image_dim_ordering() == 'tf':
        img_shape = (img_rows, img_cols, channels)

    icm = icm_class(sess, img_shape, action_size, enc_shape,  0.001)
    print('state encoder')
    icm.state_encoder.summary()
    encoder = icm.state_encoder

    return encoder

def build_decoder(enc_shape= (288,)):

    S = Input(shape= enc_shape)
    x = Dense(6400)(S)
    x = Reshape((5,10,128))(x)
    x = Conv2DTranspose(64, (3,3),padding='same', strides = (2,2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2DTranspose(64, (3,3),padding='same', strides = (3,2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2DTranspose(64, (3,3),padding='same', strides = (2,2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2DTranspose(32, (3,3),padding='same', strides = (2,2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2DTranspose(1, (3,3),padding='same', strides = (2,2))(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs = S, outputs = x, name = 'decoder')
    model.summary()
    return model


def create_test(dataset_path = '/home/daryl/datasets/unreal_images',test_path='/home/daryl/datasets/unreal_images_test'):
    imageSet = glob(dataset_path+'/*.png')
    # imageSet.sort()
    print(imageSet[0:10])
    random.shuffle(imageSet)
    print(imageSet[0:10])
    print('imageSet: ', type(imageSet))
    num_images = len(imageSet)
    print('num_images: ', num_images)
    split_ratio = 0.9
    # print(num_images * 0.1)
    num_test =int( num_images * (1-split_ratio))
    print('num test', num_test)
    imageSet_test = imageSet[:num_test]
    for i in imageSet_test:
        i_split = i.split('/')

        source_file = i
        dest_file = test_path+'/'+i_split[-1]
        print('source ', source_file)
        print('dest ', dest_file)
        os.rename(source_file,dest_file)

# def dataloader_ae(env,dataset_path = '/home/daryl/datasets/unreal_images',test_path='/home/daryl/datasets/unreal_images_test', batch_size =32,img_rows = 240,img_cols = 320):
#     print('blahblah')
#     imageSet = glob(dataset_path+'/*.png')
#     # imageSet.sort()
#     print(imageSet[0:10])
#     random.shuffle(imageSet)
#     # print(imageSet[0:10])
#     # print('imageSet: ', type(imageSet))
#     num_images = len(imageSet)
#     print('num_images: ', num_images)
#     INPUT_HEIGHT = img_rows
#     INPUT_WIDTH = img_cols
#     # obs_shape = np.array([240, 320, 3])
#     # print(obs_shape.shape)
#
#     process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))
#
#     # """Generator to be used with model.fit_generator()"""
#     while True:
#     #     files = glob.glob(os.path.join(path, '*.npz'))
#         random.shuffle(imageSet)
#         # print(imageSet[-10:])
#         new_imageSet = [imageSet[n:n+batch_size] for n in range(0, len(imageSet), batch_size)]
#         # print(len(new_imageSet))
#         image_batch = np.zeros((batch_size,INPUT_HEIGHT,INPUT_WIDTH,1))
#         while new_imageSet:
#             image_batch_fn = new_imageSet.pop()
#             # print('new batch')
#             if len(image_batch_fn) == batch_size:
#                 for i in range(len(image_batch_fn)):
#                     image = cv2.imread(image_batch_fn[i])
#                     cv2.imshow('image', image)
#                     cv2.waitKey(0)
#                     observation = process_img.process_gray(image,reset=True)
#                     # print('observation shape', observation.shape)
#                     image_batch[i] = observation[0]
#                 # yield image_batch, image_batch

def dataloader(env,batch_size =32,dataset_path = '/home/daryl/datasets/unreal_images',test_path='/home/daryl/datasets/unreal_images_test', img_rows = 240,img_cols = 320):
    # print('dataloader')
    imageSet = glob(dataset_path+'/*.png')
    # imageSet.sort()
    # print(imageSet[0:10])
    random.shuffle(imageSet)
    # print(imageSet[0:10])
    # print('imageSet: ', type(imageSet))
    num_images = len(imageSet)
    # print('num_images: ', num_images)
    INPUT_HEIGHT = img_rows
    INPUT_WIDTH = img_cols
    # obs_shape = np.array([240, 320, 3])
    # print(obs_shape.shape)

    process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))

    # """Generator to be used with model.fit_generator()"""

    while True:
    #     files = glob.glob(os.path.join(path, '*.npz'))
        random.shuffle(imageSet)
        # print(imageSet[-10:])
        new_imageSet = [imageSet[n:n+batch_size] for n in range(0, len(imageSet), batch_size)]
        # print(len(new_imageSet))
        image_batch = np.zeros((batch_size,INPUT_HEIGHT,INPUT_WIDTH,1))
        while new_imageSet:
            image_batch_fn = new_imageSet.pop()
            # print('new batch')
            if len(image_batch_fn) == batch_size:
                for i in range(len(image_batch_fn)):
                    image = cv2.imread(image_batch_fn[i])
                    # cv2.imshow('image', image)
                    # cv2.waitKey(0)
                    observation = process_img.process_gray(image,reset=True)
                    # print('observation shape', observation.shape)
                    image_batch[i] = observation[0]
                yield image_batch, image_batch


def test_load(env,batch_size =32,dataset_path = '/home/daryl/datasets/unreal_images',test_path='/home/daryl/datasets/unreal_images_test', img_rows = 240,img_cols = 320, test=False):
    imageSet = glob(test_path+'/*.png')
    random.shuffle(imageSet)
    test_image_fn = imageSet[0:batch_size]

    INPUT_HEIGHT = img_rows
    INPUT_WIDTH = img_cols


    process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))
    image_batch = np.zeros((batch_size,INPUT_HEIGHT,INPUT_WIDTH,1))
    for i in range(len(test_image_fn)):
        image = cv2.imread(test_image_fn[i])
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        observation = process_img.process_gray(image,reset=True)
        # print('observation shape', observation.shape)
        image_batch[i] = observation[0]
    return image_batch

def main():

    channels = 1
    img_rows = 240
    img_cols = 320
    img_shape = (channels, img_rows, img_cols)
    action_size = 3
    enc_shape = (288,)

    encoder = build_encoder()
    decoder = build_decoder()


    img_shape = (channels, img_rows, img_cols)
    if K.image_dim_ordering() == 'tf':
        img_shape = (img_rows, img_cols, channels)

    image_in = Input(shape = img_shape)
    z = encoder(image_in)
    image_out = decoder(z)
    autoencoder = Model(name = 'autoencoder',inputs=image_in,outputs=image_out)
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    # print('skip?')
    # dataloader(env)
    # dataloader_ae(env)
    # print('end')

    checkpt_path = '/hdd/AIRSCAN/icm_models/autoencoder_checkpoints'

    epochs = 50
    num_train =26460
    batch_size = 32
    steps = num_train//batch_size
    train = True
    cont = True
    init_epoch = 4
    old_model = '/hdd/AIRSCAN/icm_models/autoencoder_checkpointsmodel-04.hdf5'

    if train == True:
        if cont==True:
            print('loading model: ', old_model)
            checkpointer = ModelCheckpoint(filepath=checkpt_path + 'model-{epoch:02d}_cont.hdf5', verbose=1)
            autoencoder = load_model(old_model)
        else:
            checkpointer = ModelCheckpoint(filepath=checkpt_path + 'model-{epoch:02d}.hdf5', verbose=1)
        autoencoder.summary()
        optimizer = RMSprop(0.0003)
        autoencoder.compile(optimizer=optimizer, loss = 'mse')

        autoencoder.fit_generator(dataloader(env,batch_size), epochs=epochs, initial_epoch = init_epoch,steps_per_epoch=steps, shuffle=True, callbacks=[checkpointer])
    else:
        autoencoder = load_model(old_model)
        print('test')
        test_image_batch = test_load(env,batch_size,test = True)
        out_image = autoencoder.predict(test_image_batch)
        print('out_image shape: ', out_image.shape)
        input_image_sample = test_image_batch[0]
        out_image_sample = out_image[0]
        print(type(out_image_sample))
        # out_image_sample = out_image_sample * 255.0
        # out_image_sample =out_image_sample.astype(np.int)
        print(np.min(out_image_sample))
        print(np.max(out_image_sample))
        print(out_image_sample)
        cv2.imshow('input',input_image_sample)
        cv2.imshow('out',out_image_sample)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
