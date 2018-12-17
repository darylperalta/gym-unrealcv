import numpy as np
import math
# from keras.initializations import normal, identity
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Activation,Conv2DTranspose, Lambda, Reshape
from keras.layers import Conv2D, Concatenate,  BatchNormalization, LeakyReLU
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
# from example.utils import preprocessing
# from constants import *
import gym
import gym_unrealcv
import cv2
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.losses import mse
import keras
# from eval_callback import EvalCallBack
# channels = 1
# img_rows = 240
# img_cols = 320
# img_shape = (channels, img_rows, img_cols)
# action_size = 3
# enc_shape = (288,)

class preprocessor():
    def __init__(self,channels = 1, length = 3, size = (84,84) ):
        self.length = channels
        self.previous = np.zeros((1, length,size[0],size[1]))
        self.size = size
        self.image_channels = channels
        # self.image_high = observation_space.high
        self.image_high = 255
        # self.image_low = observation_space.low
        self.image_low = 0
        self.image_range = self.image_high - self.image_low
        # print('size: ', self.size)

    def resize(self,image):
        # print('resize to ', self.size)
        # cv_image = cv2.resize(image, self.size)
        cv_image = cv2.resize(image, (self.size[1], self.size[0]))
        return cv_image

    def color2gray(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def reshape(self,image):
        reshape_image = image.reshape(1, self.image_channels, self.size[0], self.size[1])
        if K.image_dim_ordering() == 'tf':
            reshape_image= reshape_image.transpose(0, 2, 3, 1)
        return reshape_image

    def normalize(self,image):
        normalized_image = (image - self.image_low) / (self.image_range)
        return  normalized_image

    def process_color(self,image, reset=False):
        resize_image = self.resize(image)
        color_image = resize_image/255.0

        return color_image

    def process_gray(self,image, reset= False):
        resize_image = self.resize(image)
        # print('resize image shape: ', resize_image.shape)
        gray_image = self.color2gray(resize_image)
        gray_image = gray_image/255.0
        # print('gray image shape: ', gray_image.shape)
        # cv2.imshow('gray image ', gray_image)
        # cv2.waitKey(0)
        # if reset:
        #     #print 'reset'
        #     for i in range(self.length):
        #         self.previous[0][i] = gray_image
        # else:
        #     #print 'update'
        #     self.previous = np.insert(self.previous, 0, gray_image, axis=1)
        #     self.previous = np.delete(self.previous, -1, axis=1)
        self.previous = np.insert(self.previous, 0, gray_image, axis=1)
        self.previous = np.delete(self.previous, -1, axis=1)


        #print self.previous.shape
        if K.image_dim_ordering() == 'tf':
            processed = self.previous.transpose(0, 2, 3, 1)
        else:
            processed = self.previous
        return processed

        # print img_processed.shape
        return img_processed


class EvalCallBack(keras.callbacks.Callback):
    """ This class hacks the callback mechanism through saving trainig status of the network """
    def __init__(self, foldpath= '/home/daryl/gym-unrealcv/example/ddpg_icm_sfm_combined_opt_rotreward_distTarget_enc/out_dir_ae',batch_size=8,img_rows = 240,img_cols = 320, interval=1):
        self.foldpath = foldpath
        # self.stage = stage
        self.interval = interval # how many epochs before running evaluation
        # self.env = env
        self.batch_size = batch_size
        self.img_rows = img_rows
        self.img_cols = img_cols
    def run_eval(self, epoch):

        test_image_batch = test_load(self.batch_size,test = True,img_rows = self.img_rows,img_cols = self.img_cols)
        out_image = self.model.predict(test_image_batch)
        for i in range(out_image.shape[0]):
            # print('out_image shape: ', out_image.shape)
            input_image_sample = test_image_batch[i]
            out_image_sample = out_image[i]

            cv2.imwrite(self.foldpath+'/autoencoder_in_ep'+str(epoch)+'_'+str(i)+'.png',(input_image_sample*255).astype(np.uint8))
            cv2.imwrite(self.foldpath+'/autoencoder_out'+str(epoch)+'_'+str(i)+'.png',(out_image_sample*255).astype(np.uint8))



    def on_epoch_end(self, epoch, logs=None):

        if (epoch+1) % self.interval == 0:
            self.run_eval(epoch)


def build_encoder(channels = 1,img_rows = 120,img_cols = 160, action_size = 3,enc_shape = (288,)):

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

def build_decoder(vae = False, enc_shape= (288,), channels = 1):

    if vae == True:
        latent_inputs = Input(shape=enc_shape, name='z_sampling')
        x = Dense(5*10*256)(latent_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((5, 10, 256))(x)


        # x = Conv2DTranspose(64, (3,3), strides=(2,2), padding ='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        #
        # x = Conv2DTranspose(64, (3,3), strides=(3,2), padding ='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        #
        # x = Conv2DTranspose(32,(3,3),strides=(2,2),padding ='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        #
        # x = Conv2DTranspose(32,(3,3),strides=(2,2),padding ='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        #
        # x = Conv2DTranspose(1,(3,3),strides=(2,2),padding ='same')(x)

        x = Conv2DTranspose(256, (5,5), strides=(2,2), padding ='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(.2)(x)

        x = Conv2DTranspose(256, (5,5), strides=(3,2), padding ='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(.2)(x)

        x = Conv2DTranspose(128,(5,5),strides=(2,2),padding ='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(.2)(x)

        x = Conv2DTranspose(64,(5,5),strides=(2,2),padding ='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(.2)(x)

        # x = Conv2DTranspose(64,(5,5),strides=(2,2),padding ='same')(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(.2)(x)


        x = Conv2DTranspose(channels,(5,5),strides=(1,1),padding ='same')(x)
        outputs = Activation('sigmoid')(x)

        # instantiate decoder model
        model = Model(latent_inputs, outputs, name='decoder')
        model.summary()
        plot_model(model, to_file='vaegan_decoder.png', show_shapes=True)

    else:

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
    imageSet = glob(test_path+'/*.png')
    # imageSet.sort()
    print(imageSet[0:10])
    random.shuffle(imageSet)
    print(imageSet[0:10])
    print('imageSet: ', type(imageSet))
    num_images = len(imageSet)
    print('num_images: ', num_images)
    # split_ratio = 0.9
    # # print(num_images * 0.1)
    # num_test =int( num_images * (1-split_ratio))
    # print('num test', num_test)
    # imageSet_test = imageSet[:num_test]
    # for i in imageSet_test:
    #     i_split = i.split('/')
    #
    #     source_file = i
    #     dest_file = test_path+'/'+i_split[-1]
    #     print('source ', source_file)
    #     print('dest ', dest_file)
    #     os.rename(source_file,dest_file)

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

def dataloader(batch_size =32,vae = False, dataset_path = '/home/justine/datasets/unreal_images',test_path='/home/justine/datasets/unreal_images_test', img_rows = 240,img_cols = 320,channels=3,augment=True):
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

    process_img = preprocessor(channels=channels, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))
    # process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))

    # """Generator to be used with model.fit_generator()"""

    while True:
    #     files = glob.glob(os.path.join(path, '*.npz'))
        random.shuffle(imageSet)
        # print(imageSet[-10:])
        new_imageSet = [imageSet[n:n+batch_size] for n in range(0, len(imageSet), batch_size)]
        # print(len(new_imageSet))
        image_batch = np.zeros((batch_size,INPUT_HEIGHT,INPUT_WIDTH,channels))
        while new_imageSet:
            image_batch_fn = new_imageSet.pop()
            # print('new batch')
            if len(image_batch_fn) == batch_size:
                for i in range(len(image_batch_fn)):
                    image = cv2.imread(image_batch_fn[i])
                    # cv2.imshow('image', image)
                    # cv2.waitKey(0)
                    # observation = process_img.process_gray(image,reset=True)
                    observation = process_img.process_color(image,reset=True)
                    if augment == True:
                        chance = np.random.uniform(low=0, high=1)
                        if chance >= 0.5:
                            # print("AUGMENT")
                            random_gamma = np.random.uniform(low=0.8, high=1.2)
                            random_brightness = np.random.uniform(low = 0.8, high=1.5)
                            observation = observation**random_gamma
                            observation = observation*random_brightness
                            np.clip(observation, a_min=0, a_max=1)

                    # print('observation s img_rows = 240,img_cols = 320hape', observation.shape)
                    # image_batch[i] = observation[0]
                    image_batch[i] = observation
                if vae ==True:
                    yield image_batch, None
                else:
                    yield image_batch, image_batch


def test_load(batch_size =32,dataset_path = '/home/justine/datasets/unreal_images',test_path='/home/justine/datasets/unreal_images_test', img_rows = 240,img_cols = 320, channels= 3, test=False):
    imageSet = glob(test_path+'/*.png')
    random.shuffle(imageSet)
    test_image_fn = imageSet[0:batch_size]

    INPUT_HEIGHT = img_rows
    INPUT_WIDTH = img_cols


    # process_img = preprocessing.preprocessor(observation_space=env.observation_space, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))
    process_img = preprocessor(channels=channels, length = 1, size = (INPUT_HEIGHT,INPUT_WIDTH))
    image_batch = np.zeros((batch_size,INPUT_HEIGHT,INPUT_WIDTH,channels))
    for i in range(len(test_image_fn)):
        image = cv2.imread(test_image_fn[i])
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # observation = process_img.process_gray(image,reset=True)
        observation = process_img.process_color(image,reset=True)
        # print('observation shape', observation.shape)
        # image_batch[i] = observation[0]
        image_batch[i] = observation
    return image_batch

def main():
    vae = True
    # channels = 1
    channels = 3
    img_rows = 120
    img_cols = 160
    img_shape = (channels, img_rows, img_cols)
    action_size = 3
    enc_shape = (256,)
    metrics = [mse]
    encoder = build_encoder(enc_shape = enc_shape, channels = channels)
    decoder = build_decoder(vae,enc_shape,channels=channels)


    img_shape = (channels, img_rows, img_cols)
    if K.image_dim_ordering() == 'tf':
        img_shape = (img_rows, img_cols, channels)

    image_in = Input(shape = img_shape)


    if vae == True:
        z_mean, z_log_var, z= encoder(image_in)
        image_out = decoder(z)
    else:
        z = encoder(image_in)
        image_out = decoder(z)
    autoencoder = Model(name = 'autoencoder',inputs=image_in,outputs=image_out)
    # env = gym.make(ENV_NAME)
    # env = env.unwrapped
    # print('skip?')
    # dataloader(env)
    # dataloader_ae(env)
    # print('end')

    # checkpt_path = '/hdd/AIRSCAN/icm_models/autoencoder_checkpoints'
    checkpt_path = '/home/justine/airscan_gym/checkpoints/' # VAE4 - changed to leaky relu changed kernel_sizess

    epochs = 120
    # num_train =26460
    num_train = 46688
    batch_size = 16
    steps = num_train//batch_size
    # steps = 1
    train = True
    cont = False
    init_epoch = 0
    # old_model = '/hdd/AIRSCAN/icm_models/autoencoder_checkpointsmodel-50_cont.hdf5'
    # old_model = '/hdd/AIRSCAN/icm_models/vae2_checkpointsmodel-13.hdf5'
    # old_model = '/hdd/AIRSCAN/icm_models/vae3_checkpointsmodel-10.hdf5'
    # old_model = '/hdd/AIRSCAN/icm_models/vae4_checkpointsmodel-10_cont.hdf5'
    old_model = '/home/justine/airscan_gym/checkpoints/model-99.hdf5'
    # eval_check = EvalCallBack(env)
    eval_check = EvalCallBack(foldpath= '/home/justine/airscan_gym/checkpoints',img_rows = img_rows,img_cols = img_cols)
    if train == True:
        if cont==True:
            print('loading model: ', old_model)
            checkpointer = ModelCheckpoint(filepath=checkpt_path + 'model-{epoch:02d}_cont.hdf5', verbose=1)

        else:
            checkpointer = ModelCheckpoint(filepath=checkpt_path + 'model-{epoch:02d}.hdf5', verbose=1)
        autoencoder.summary()
        # optimizer = RMSprop(0.0003)
        if vae == True:
            print('compiling VAE encoder')
            # optimizer = RMSprop(0.000001)
            optimizer = RMSprop(0.0003)
            # optimizer = RMSprop(0.00003)
            image_in_flat = K.flatten(image_in)
            image_out_flat = K.flatten(image_out)
            # reconstruction_loss = mse(image_in,image_out)
            reconstruction_loss = mse(image_in_flat, image_out_flat)
            # reconstructin_loss =K.mean(K.square(image_in - image_out), axis=-1)

            print('shapes')
            print(K.shape(image_in_flat))
            print(K.int_shape(reconstruction_loss))

            # reconstruction_loss = K.mean(K.square(image_in - image_out), axis=-1)
            # reconstruction_loss *= channels*img_rows*img_cols
            reconstruction_loss *= img_shape[0]*img_shape[1]*img_shape[2]
            kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            # print(K.shape(z_mean))
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # print(K.int_shape(kl_loss))
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            # vae_loss = K.mean(K.mean(K.square(image_in - image_out), axis=-1)+ kl_loss, axis = -1)
            # vae_loss = K.mean(image_in  + kl_loss, axis = -1)

            autoencoder.add_loss(vae_loss)
            # autoencoder.add_loss(kl_loss)
            autoencoder.compile(optimizer = optimizer, metrics = metrics)
            if cont == True:
                autoencoder.load_weights(old_model)
            autoencoder.fit_generator(dataloader(batch_size,img_rows = img_rows,img_cols = img_cols,vae = vae), epochs=epochs, initial_epoch = init_epoch,steps_per_epoch=steps, shuffle=True, callbacks=[checkpointer, eval_check])
            # autoencoder.fit_generator(dataloader(env,batch_size,vae = vae), epochs=epochs, initial_epoch = init_epoch,steps_per_epoch=2, shuffle=True, callbacks=[checkpointer, eval_check])
            # autoencoder.compile(optimizer=optimizer, loss = 'mse')
        else:
            optimizer = RMSprop(0.000001)
            # reconstruction_loss = mse(image_in,image_out)
            # reconstruction_loss *= channels*img_rows*img_cols
            # kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
            # kl_loss = K.sum(kl_loss, axis=-1)
            # kl_loss *= -0.5
            # vae_loss = K.mean(reconstruction_loss + kl_loss)
            # autoencoder.add_loss(vae_loss)
            # autoencoder.compile(optimizer=optimizer)
            autoencoder.compile(optimizer=optimizer, loss = 'mse')
            autoencoder.fit_generator(dataloader(batch_size,img_rows = img_rows,img_cols = img_cols), epochs=epochs, initial_epoch = init_epoch,steps_per_epoch=steps, shuffle=True, callbacks=[checkpointer])
        # autoencoder.fit_generator(dataloader(env,batch_size), epochs=epochs, initial_epoch = init_epoch,steps_per_epoch=steps, shuffle=True, callbacks=[checkpointer])
    else:
        # autoencoder = load_model(old_model)
        autoencoder.load_weights(old_model)
        encoder_path = '/hdd/AIRSCAN/icm_models/vae4_encoder_checkpointsmodel-7.hdf5'
        decoder_path = '/hdd/AIRSCAN/icm_models/vae4_decoder_checkpointsmodel-7.hdf5'
        encoder.save_weights(encoder_path)
        decoder.save_weights(decoder_path)


        encoder2 = build_encoder(enc_shape = enc_shape)




        image_in = Input(shape = img_shape)
        # z = encoder2(image_in)
        # image_out = decoder2(z)

        if vae == True:
            decoder2 = build_decoder(vae,enc_shape)
            [z_mean, z_log_var, z]= encoder2(image_in)
            image_out = decoder2(z)
        else:
            decoder2 = build_decoder(enc_shape)
            z = encoder2(image_in)
            image_out = decoder2(z)

        # encoder2.load_weights('/hdd/AIRSCAN/icm_models/encoder_checkpointsmodel-50_cont.hdf5')
        # decoder2.load_weights('/hdd/AIRSCAN/icm_models/decoder_checkpointsmodel-50_cont.hdf5')
        encoder2.load_weights(encoder_path)
        decoder2.load_weights(decoder_path)
        # encoder2.load_weights(old_model)
        autoencoder2 = Model(name = 'autoencoder2',inputs=image_in,outputs=image_out)

        print('test')
        test_image_batch = test_load(env,batch_size,test = True)
        # out_image = autoencoder.predict(test_image_batch)
        # out_image = autoencoder2.predict(test_image_batch)

        [z_mean_out, z_log_var_out, z_out] = encoder2.predict(test_image_batch)
        print('max', np.max(z_out))
        print('min', np.min(z_out))
        out_image1 = decoder2.predict(z_out)
        z_out[:,0] += 1.0
        z_out[:,1] += 1.0
        z_out[:,2] += 1.0
        z_out[:,3] += 1.0
        z_out[:,4] += 1.0
        z_out[:,5] += 1.0
        z_out[:,6] += 1.0
        z_out[:,7] += 1.0
        z_out[:,8] += 1.0
        z_out[:,9] += 1.0
        z_out[:,10] += 2.0
        z_out[:,11] += 2.0
        z_out[:,12] += 2.0
        z_out[:,13] += 2.0
        z_out[:,14] += 2.0
        z_out[:,15] += 2.0
        z_out[:,16] += 2.0
        z_out[:,17] += 2.0
        z_out[:,18] += 2.0
        z_out[:,19] += 2.0
        out_image2 = decoder2.predict(z_out)
        z_out[:,0] += 2.0
        z_out[:,1] += 2.0
        z_out[:,2] += 2.0
        z_out[:,3] += 2.0
        z_out[:,4] += 2.0
        z_out[:,5] += 2.0
        z_out[:,6] += 2.0
        z_out[:,7] += 2.0
        z_out[:,8] += 2.0
        z_out[:,9] += 2.0
        z_out[:,10] += 2.0
        z_out[:,11] += 2.0
        z_out[:,12] += 2.0
        z_out[:,13] += 2.0
        z_out[:,14] += 2.0
        z_out[:,15] += 2.0
        z_out[:,16] += 2.0
        z_out[:,17] += 2.0
        z_out[:,18] += 2.0
        z_out[:,19] += 2.0
        out_image3 = decoder2.predict(z_out)

        # for i in range(out_image.shape[0]):
        for i in range(out_image1.shape[0]):
            # print('out_image shape: ', out_image.shape)
            input_image_sample = test_image_batch[i]
            # out_image_sample = out_image[i]
            out_image_sample1 = out_image1[i]
            out_image_sample2 = out_image2[i]
            out_image_sample3 = out_image3[i]
            # print(type(out_image_sample))
            # out_image_sample = out_image_sample * 255.0
            # out_image_sample =out_image_sample.astype(np.int)
            # print(np.min(out_image_sample))
            # print(np.max(out_image_sample))
            # print(out_image_sample)

            cv2.imshow('input',input_image_sample)
            # cv2.imshow('out',out_image_sample)
            cv2.imshow('out1',out_image_sample1)
            cv2.imshow('out2',out_image_sample2)
            cv2.imshow('out3',out_image_sample3)
            cv2.waitKey(0)


            # print(old_model+str(i)+'.png')
            # cv2.imwrite('autoencoder_in'+str(i)+'.png',(input_image_sample*255).astype(np.uint8))
            # cv2.imwrite('autoencoder_out'+str(i)+'.png',(out_image_sample*255).astype(np.uint8))

if __name__ == '__main__':
    main()
    # create_test()
