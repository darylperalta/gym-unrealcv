import numpy as np
import math
# from keras.initializations import normal, identity
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Activation,Convolution2D, MaxPooling2D, Lambda
from keras.layers import Conv2D, Concatenate, BatchNormalization, LeakyReLU,LSTM, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import ELU
import keras.backend as K
import tensorflow as tf
from keras.utils import plot_model

def icm_loss(y_true,y_pred):
    return y_pred
#
# def state_encoder(state_shape, enc_shape):
#     # state -> phi
#     S = Input(shape= state_shape)
#
#     '''
#     x = Convolution2D(32, 3, 3, activation='elu')(S)
#     x = Convolution2D(32, 3, 3, activation='elu')(x)
#     x = Convolution2D(32, 3, 3, activation='elu')(x)
#     x = Convolution2D(32, 3, 3, activation='elu')(x)
#     x = Flatten()(x)
#     '''
#     x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(S)
#     x = ELU(alpha=1.0)(x)
#     x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
#     x = ELU(alpha=1.0)(x)
#     x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
#     x = ELU(alpha=1.0)(x)
#     x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
#     x = ELU(alpha=1.0)(x)
#     x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
#     x = ELU(alpha=1.0)(x)
#
#     phi = Flatten()(x)
#     model = Model(input = S, output = phi)
#
#     return model
#
# def forward_model(action_size, enc_shape, output_dim = 288):
#     # phi_t, a_t -> s_t
#
#     phi_t = Input(shape = enc_shape)
#     a_t = Input(shape = action_size)
#
#     x = Concatenate()([phi_t, a_t])
#     x = Dense(256, activation='relu')(x)
#     out = Dense(output_dim, activation='linear')(x)
#
#     model = Model([phi_t,a_t],out)
#     return model
#
# def inverse_model(enc_shape, action_size = (6,)):
#     # phi_t, phi_t+1 -> a_t_hat
#     phi_t = Input(shape = enc_shape)
#     phi_t_1 = Input(shape = enc_shape)
#
#     x = Concatenate()([phi_t, phi_t_1])
#     x = Dense(256, activation='relu')(x)
#     out = Dense(action_size, activation='linear')(x)
#
#     model = Model([phi_t, phi_t_1], out)
#     return model

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class icm_class(object):
    def __init__(self, state_shape, action_size, enc_shape, LEARNING_RATE, pretrained=False, enc_path = '/hdd/AIRSCAN/icm_models/encoder_checkpointsmodel-50_cont.hdf5', vae =False):
        # self.sess = sess
        # self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.enc_shape = enc_shape
        self.action_shape = (action_size,)
        self.state_shape = state_shape
        self.pretrained = pretrained
        self.enc_path = enc_path
        self.vae = vae
        # K.set_session(sess)

        self.state_encoder = self.create_state_encoder(state_shape,enc_shape)
        self.forward_model = self.create_forward_model(action_size,enc_shape, enc_shape[0])
        # self.inverse_model = self.create_inverse_model()
        # self.train_model_inv = self.create_train_model_inv()
        # self.train_model_forward = self.create_train_model_forward()
        self.train_model, self.reward_model = self.create_train_combined()

    def get_intrinsic_reward(self,state_t, a_t, state_t_1):

        [r_i, l_i] = self.reward_model.predict([state_t, a_t, state_t_1])
        #
        # phi_t_1 = self.state_encoder.predict(state_t_1)
        # phi_t_1_hat = self.train_model_forward.predict([state_t, a_t])
        #
        #
        # diff = phi_t_1-phi_t_1_hat
        # r_i = 0.5*np.sum(np.square(phi_t_1-phi_t_1_hat))
        # return r_i
        return r_i[0], l_i

    def train(self, state_t, a_t, state_t_1, batch_size):
        # input: obs, action, obs_new
        # phi_t_1 = self.state_encoder.predict(state_t_1)
        # self.train_model_inv.train_on_batch([state_t,state_t_1], a_t)
        # self.train_model_forward.train_on_batch([state_t, a_t],phi_t_1)
        self.train_model.train_on_batch([state_t, a_t, state_t_1],np.zeros((batch_size,)))
        # print('trained')

    def create_inverse_model(self,use_lstm = True):
        # inv model: state_t, state_t_1 -> a_hat
        phi_t = Input(shape = self.enc_shape)
        phi_t_1 = Input(shape = self.enc_shape)

        x = Concatenate()([phi_t, phi_t_1])
        x = Dense(256, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)        #added after seeing view prediction bad
        x = Dense(256, activation='relu')(x)
        if use_lstm:
            x = Reshape((1,-1))(x)
            x = LSTM(256, activation='relu')(x)
        else:
            x = Dense(256, activation='relu')(x)
        # x = Dense(256, activation='relu')(x)
        out = Dense(self.action_size, activation = 'linear')(x)

        model = Model([phi_t,phi_t_1],out,name='inverse_model')
        print('inverse_model: ')
        model.summary()

        return model



    def create_forward_model(self, action_size, enc_shape, output_dim = 288, use_lstm = False):
        # phi_t, a_t -> phi_t+1

        action_shape = (action_size,)
        print('action shape: ', action_shape)
        phi_t = Input(shape = enc_shape)
        a_t = Input(shape = action_shape)

        x = Concatenate()([phi_t, a_t])
        # x = Dense(256, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        # x = Dense(128, activation='relu')(x)        #added after seeing view prediction bad
        # x = Dense(256, activation='relu')(x)

        # if use_lstm:
        #     x = Reshape((1,-1))(x)
        #     x = LSTM(256, activation='relu')(x)
        # else:
        #     x = Dense(256, activation='relu')(x)
        # x = Dense(256, activation='relu')(x)
        # out = Dense(output_dim, activation='linear')(x)
        # out = Dense(output_dim, activation='sigmoid')(x)
        out = Dense(output_dim, activation='relu')(x)

        model = Model([phi_t,a_t],out,name='forward_model')
        print('forward model: ')
        model.summary()

        return model

    def create_state_encoder(self, state_shape, enc_shape):
        # state -> phi
        s_t = Input(shape= state_shape)

        if self.vae == True:
            # x = Conv2D(32,(3,3), strides =(2,2),padding='same', name= 'enc_conv1')(s_t)
            # x = BatchNormalization(name= 'enc_bn1')(x)
            # #x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU1')(x)
            #
            #
            # x = Conv2D(32,(3,3), strides =(2,2),padding='same', name= 'enc_conv2')(x)
            # x = BatchNormalization(name= 'enc_bn2')(x)
            # #x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU2')(x)
            #
            #
            # x = Conv2D(64,(3,3), strides =(2,2),padding='same', name= 'enc_conv3')(x)
            # x = BatchNormalization(name= 'enc_bn3')(x)
            # #x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU3')(x)
            #
            # x = Conv2D(64,(3,3), strides =(3,2),padding='same', name= 'enc_conv4')(x)
            # x = BatchNormalization(name= 'enc_bn4')(x)
            # #x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU4')(x)
            #
            # x = Conv2D(128,(3,3), strides =(2,2),padding='same', name= 'enc_conv5')(x)
            # x = BatchNormalization(name= 'enc_bn5')(x)
            # #x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU5')(x)

            x = Conv2D(64,(5,5), strides =(2,2),padding='same', name= 'enc_conv1')(s_t)
            x = BatchNormalization(name= 'enc_bn1')(x)
            #x = Activation('relu')(x)
            x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU1')(x)


            x = Conv2D(64,(5,5), strides =(2,2),padding='same', name= 'enc_conv2')(x)
            x = BatchNormalization(name= 'enc_bn2')(x)
            #x = Activation('relu')(x)
            x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU2')(x)


            x = Conv2D(128,(5,5), strides =(2,2),padding='same', name= 'enc_conv3')(x)
            x = BatchNormalization(name= 'enc_bn3')(x)
            #x = Activation('relu')(x)
            x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU3')(x)

            x = Conv2D(256,(5,5), strides =(3,2),padding='same', name= 'enc_conv4')(x)
            x = BatchNormalization(name= 'enc_bn4')(x)
            #x = Activation('relu')(x)
            x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU4')(x)

            if state_shape[0]==240:
                x = Conv2D(256,(5,5), strides =(2,2),padding='same', name= 'enc_conv5')(x)
                x = BatchNormalization(name= 'enc_bn5')(x)
            #x = Activation('relu')(x)
                x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU5')(x)

            x = Flatten()(x)

            x_mean = Dense(enc_shape[0], name='x_mean')(x)
            x_mean = BatchNormalization()(x_mean)
            z_mean = LeakyReLU(alpha = 0.2, name = 'z_mean')(x_mean)


            x_log_var = Dense(enc_shape[0], name='x_log_var')(x)
            x_log_var = BatchNormalization()(x_log_var)
            z_log_var = LeakyReLU(alpha = 0.2, name='z_log_var')(x_log_var)

            # use reparameterization trick to push the sampling out as input
            # note that "output_shape" isn't necessary with the TensorFlow backend
            z = Lambda(sampling, output_shape=(enc_shape[0],), name='z')([z_mean, z_log_var])
            #encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
            model = Model(input = s_t, output = [z_mean, z_log_var, z],name='state_encoder')



        else:

            # x = Conv2D(32, (3,3), padding='same', strides=(2, 2))(s_t)
            # x = ELU(alpha=1.0)(x)
            # x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
            # x = ELU(alpha=1.0)(x)
            # x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
            # x = ELU(alpha=1.0)(x)
            # x = Conv2D(64, (3, 3), padding='same', strides=(3, 2))(x)
            # x = ELU(alpha=1.0)(x)
            # x = Conv2D(128, (3, 3), padding='same', strides=(2, 2))(x)
            # x = ELU(alpha=1.0)(x)

            x = Conv2D(32,(8,8), strides =(4,4),padding='valid', name= 'enc_conv1')(s_t)
            # x = BatchNormalization(name= 'enc_bn1')(x)
            x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU1')(x)


            x = Conv2D(64,(4,4), strides =(2,2),padding='valid', name= 'enc_conv2')(x)
            # x = BatchNormalization(name= 'enc_bn2')(x)
            x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2, name = 'enc_LReLU2')(x)


            x = Conv2D(64,(3,3), strides =(1,1),padding='valid', name= 'enc_conv3')(x)
            # x = BatchNormalization(name= 'enc_bn3')(x)
            x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU3')(x)

            # x = Conv2D(256,(5,5), strides =(3,2),padding='same', name= 'enc_conv4')(x)
            # x = BatchNormalization(name= 'enc_bn4')(x)
            #x = Activation('relu')(x)
            # x = LeakyReLU(alpha = 0.2,name = 'enc_LReLU4')(x)

            x = Flatten()(x)
            # phi = Dense(enc_shape[0], name='phi', activation ='relu')(x)
            phi = Dense(enc_shape[0], name='phi', activation ='relu')(x)
            model = Model(input = s_t, output = phi,name='state_encoder')

        plot_model(model, to_file='encoder.png', show_shapes=True)
        print('state_encoder: ')
        model.summary()
        return model



    # def create_train_combined(self,lmd=1.0, beta=0.01):
    def create_train_combined(self,lmd=1.0, beta=0.8, freeze = True):

        s_t = Input(shape = self.state_shape)
        a_t = Input(shape = self.action_shape)
        s_t_1 = Input(shape = self.state_shape)

        if self.vae == True:
            phi_t = self.state_encoder(s_t)[2]
            phi_t_1 = self.state_encoder(s_t_1)[2]

        else:
            phi_t = self.state_encoder(s_t)
            phi_t_1 = self.state_encoder(s_t_1)

        if self.pretrained == True:
            print('pretrained encoder')
            print('path: ', self.enc_path)
            if freeze == True:
                self.state_encoder.trainable = False
            self.state_encoder.load_weights(self.enc_path)
            # self.state_encoder.trainable = False
        if freeze == True:
            self.state_encoder.trainable = False


        phi_t_1_hat = self.forward_model([phi_t,a_t])
        # a_hat = self.inverse_model([phi_t, phi_t_1])

        # r_in = 0.5 * K.sum(K.square(phi_t_1 - phi_t_1_hat))
        # r_in = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1])), name = 'reward_in')([phi_t_1,phi_t_1_hat])
        r_in = Lambda(lambda x: 0.5 * K.mean(K.square(x[0] - x[1])), name = 'reward_in')([phi_t_1,phi_t_1_hat])
        # l_i = 0.5 * -K.sum(a_t * K.log(a_hat + K.epsilon()))
        # l_i = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon())))([a_t,a_hat])
        # l_i = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1])),name ='inverse_model_loss')([a_t,a_hat])
        # l_i = Lambda(lambda x: 0.5 * K.mean(K.square(x[0] - x[1])),name ='inverse_model_loss')([a_t,a_hat])

        # loss0 =  beta * r_in + (1.0 - beta) * l_i
        # loss0 = Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1], name = 'l_0')([r_in,l_i])
        # loss0 = Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1], name = 'l_0')([r_in,l_i])
        loss0 = r_in
        # print('loss: ', type(loss0))
        print('r_in: ', type(r_in))
        icm_combined = Model([s_t,a_t,s_t_1], loss0)
        optimizer = RMSprop(lr=self.LEARNING_RATE)
        optimizer = Adam(lr=self.LEARNING_RATE)
        print('combined model:')
        # print()
        icm_combined.compile(loss=icm_loss,
                              optimizer=optimizer)
        icm_combined.summary()
        plot_model(icm_combined, to_file ="icm_combined.png")
        # reward_model = Model([s_t,a_t,s_t_1],[r_in,l_i])
        reward_model = Model([s_t,a_t,s_t_1],[r_in,r_in])

        return icm_combined, reward_model

    def create_train_model_forward(self):
        #s_t, a_t -> phi_t+1
        state_encoder.trainable = False

        s_t = Input(shape = self.state_shape)
        a_t = Input(shape = self.action_shape)
        phi_t = self.state_encoder(s_t)
        phi_t_1 = self.forward_model([phi_t,a_t])
        model_train_forward = Model([s_t,a_t],phi_t_1)

        optimizer = RMSprop(lr=self.LEARNING_RATE)
        model_train_forward.compile(loss='mse',
                              optimizer=optimizer)

        # self.inverse_model.predict(s_t)

        # model_train_inverse

        return model_train_forward

    def create_train_model_inv(self, train_encoder = False):
        if train_encoder == True:
            state_encoder.trainable = True
        else:
            state_encoder.trainable = False

        s_t = Input(shape = self.state_shape)
        s_t_1 = Input(shape = self.state_shape)

        phi_t = self.state_encoder(s_t)
        phi_t_1 = self.state_encoder(s_t_1)

        a_hat = self.inverse_model([phi_t, phi_t_1])



        model_train_inv = Model([s_t,s_t_1], a_hat)

        optimizer = RMSprop(lr=self.LEARNING_RATE)

        model_train_inv.compile(loss='mse',
                              optimizer=optimizer)

        return model_train_inv


def test():

    state_shape = (82,82,1)
    enc_shape = (288,)
    # model = state_encoder(state_shape, enc_shape)
    # model.summary()
    #
    # action_size = (3,)
    # model_forward = forward_model(action_size, enc_shape)
    # model_forward.summary()
    # num_actions = 3
    # model_inv = inverse_model(enc_shape, num_actions)
    # model_inv.summary()

def test_class():
    TF_DEVICE = '/gpu:0'
    state_shape = (82,82,1)
    enc_shape = (288,)
    num_actions = 3



    with tf.device(TF_DEVICE):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    lr = 0.001

    forward = forward_dynamics(sess,state_shape, num_actions, enc_shape,  lr)
    forward.train_model.summary()

def test_train():
    TF_DEVICE = '/gpu:0'
    state_shape = (82,82,3)
    enc_shape = (288,)
    num_actions = 3



    with tf.device(TF_DEVICE):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    lr = 0.001

    icm_module = icm(sess,state_shape, num_actions, enc_shape,  lr)
    a = np.random.rand(2,num_actions)
    state = np.random.rand(2,state_shape[0], state_shape[1], state_shape[2])
    state_2 = np.random.rand(2,state_shape[0], state_shape[1], state_shape[2])

    phi_t = icm_module.state_encoder.predict(state)
    phi_t_1 = icm_module.state_encoder.predict(state_2)

    a_hat = icm_module.inverse_model.predict([phi_t, phi_t_1])
    print('a_hat: ', a_hat)

    icm_module.train(state,a,state_2)

    # forward.train_model.summary()
    # print('a ',a.shape)
    # print('state ', state.shape)
    #
    #
    #
    # phi_t_1_hat = forward.train_model.predict([state,a])
    # phi_t_1 = forward.state_encoder.predict([state])
    # forward.train_model.train_on_batch([state,a], phi_t_1)
    #
    # a_try = np.random.rand(num_actions)
    # reward = forward.get_intrinsic_reward(state,a,state_2)
    # print('rewarddd:  ', reward)
    # print('phi')
    # print(phi.shape)




if __name__ == '__main__':
    test_train()
