import numpy as np
import math
# from keras.initializations import normal, identity
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
#from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Activation,Convolution2D, MaxPooling2D
from keras.layers import Conv2D, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras.layers.advanced_activations import ELU
import keras.backend as K
import tensorflow as tf

def state_encoder(state_shape, enc_shape):
    # state -> phi
    S = Input(shape= state_shape)

    '''
    x = Convolution2D(32, 3, 3, activation='elu')(S)
    x = Convolution2D(32, 3, 3, activation='elu')(x)
    x = Convolution2D(32, 3, 3, activation='elu')(x)
    x = Convolution2D(32, 3, 3, activation='elu')(x)
    x = Flatten()(x)
    '''
    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(S)
    x = ELU(alpha=1.0)(x)
    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = ELU(alpha=1.0)(x)
    x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = ELU(alpha=1.0)(x)

    phi = Flatten()(x)
    model = Model(input = S, output = phi)

    return model

def forward_model(action_size, enc_shape, output_dim = 288):
    # phi_t, a_t -> s_t

    phi_t = Input(shape = enc_shape)
    a_t = Input(shape = action_size)

    x = Concatenate()([phi_t, a_t])
    x = Dense(256, activation='relu')(x)
    out = Dense(output_dim, activation='linear')(x)

    model = Model([phi_t,a_t],out)
    return model

def inverse_model(enc_shape, action_size = (6,)):
    # phi_t, phi_t+1 -> a_t_hat
    phi_t = Input(shape = enc_shape)
    phi_t_1 = Input(shape = enc_shape)

    x = Concatenate()([phi_t, phi_t_1])
    x = Dense(256, activation='relu')(x)
    out = Dense(action_size, activation='linear')(x)

    model = Model([phi_t, phi_t_1], out)
    return model


class forward_dynamics(object):
    def __init__(self, sess, state_shape, action_size, enc_shape, LEARNING_RATE):
        self.sess = sess
        # self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.enc_shape = enc_shape
        self.action_shape = (action_size,)
        self.state_shape = state_shape
        # K.set_session(sess)

        self.state_encoder = self.create_state_encoder(state_shape,enc_shape)
        self.forward_model = self.create_forward_model(action_size,enc_shape, enc_shape[0])
        self.train_model = self.create_train_model()


    def get_intrinsic_reward(self,state_t, a_t, state_t_1):

        phi_t_1 = self.state_encoder.predict(state_t_1)
        phi_t_1_hat = self.train_model.predict([state_t, a_t])

        r_i = 0.5*np.sum(np.square(phi_t_1-phi_t_1_hat))

        return r_i

    def train(self, state_t, a_t, state_t_1):
        # input: obs, action, obs_new
        phi_t_1 = self.state_encoder.predict(state_t_1)
        self.train_model.train_on_batch([state_t, a_t],phi_t_1)


    def create_forward_model(self, action_size, enc_shape, output_dim = 288):
        # phi_t, a_t -> phi_t+1

        action_shape = (action_size,)
        print('action shape: ', action_shape)
        phi_t = Input(shape = enc_shape)
        a_t = Input(shape = action_shape)

        x = Concatenate()([phi_t, a_t])
        x = Dense(256, activation='relu')(x)
        out = Dense(output_dim, activation='linear')(x)

        model = Model([phi_t,a_t],out)

        return model

    def create_state_encoder(self, state_shape, enc_shape):
        # state -> phi
        s_t = Input(shape= state_shape)

        '''
        x = Convolution2D(32, 3, 3, activation='elu')(S)
        x = Convolution2D(32, 3, 3, activation='elu')(x)
        x = Convolution2D(32, 3, 3, activation='elu')(x)
        x = Convolution2D(32, 3, 3, activation='elu')(x)
        x = Flatten()(x)
        '''
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(s_t)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)

        phi = Flatten()(x)
        model = Model(input = s_t, output = phi)

        return model

    def create_train_model(self):
        #s_t, a_t -> phi_t+1
        s_t = Input(shape = self.state_shape)
        a_t = Input(shape = self.action_shape)
        phi_t = self.state_encoder(s_t)
        phi_t_1 = self.forward_model([phi_t,a_t])
        model = Model([s_t,a_t],phi_t_1)

        optimizer = RMSprop(lr=self.LEARNING_RATE)
        model.compile(loss='mse',
                              optimizer=optimizer,
                              metrics=['accuracy'])

        return model


class icm_class(object):
    def __init__(self, sess, state_shape, action_size, enc_shape, LEARNING_RATE):
        self.sess = sess
        # self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        self.enc_shape = enc_shape
        self.action_shape = (action_size,)
        self.state_shape = state_shape
        # K.set_session(sess)

        self.state_encoder = self.create_state_encoder(state_shape,enc_shape)
        self.forward_model = self.create_forward_model(action_size,enc_shape, enc_shape[0])
        self.inverse_model = self.create_inverse_model()
        self.train_model_inv = self.create_train_model_inv()
        self.train_model_forward = self.create_train_model_forward()


    def get_intrinsic_reward(self,state_t, a_t, state_t_1):

        phi_t_1 = self.state_encoder.predict(state_t_1)
        phi_t_1_hat = self.train_model_forward.predict([state_t, a_t])

        r_i = 0.5*np.sum(np.square(phi_t_1-phi_t_1_hat))

        return r_i

    def train(self, state_t, a_t, state_t_1):
        # input: obs, action, obs_new
        phi_t_1 = self.state_encoder.predict(state_t_1)
        self.train_model_inv.train_on_batch([state_t,state_t_1], a_t)
        self.train_model_forward.train_on_batch([state_t, a_t],phi_t_1)
        print('trained')

    def create_inverse_model(self):
        # inv model: state_t, state_t_1 -> a_hat
        phi_t = Input(shape = self.enc_shape)
        phi_t_1 = Input(shape = self.enc_shape)

        x = Concatenate()([phi_t, phi_t_1])
        x = Dense(256, activation='relu')(x)
        out = Dense(self.action_size, activation = 'linear')(x)

        model = Model([phi_t,phi_t_1],out)

        return model



    def create_forward_model(self, action_size, enc_shape, output_dim = 288):
        # phi_t, a_t -> phi_t+1

        action_shape = (action_size,)
        print('action shape: ', action_shape)
        phi_t = Input(shape = enc_shape)
        a_t = Input(shape = action_shape)

        x = Concatenate()([phi_t, a_t])
        x = Dense(256, activation='relu')(x)
        out = Dense(output_dim, activation='linear')(x)

        model = Model([phi_t,a_t],out)

        return model

    def create_state_encoder(self, state_shape, enc_shape):
        # state -> phi
        s_t = Input(shape= state_shape)

        '''
        x = Convolution2D(32, 3, 3, activation='elu')(S)
        x = Convolution2D(32, 3, 3, activation='elu')(x)
        x = Convolution2D(32, 3, 3, activation='elu')(x)
        x = Convolution2D(32, 3, 3, activation='elu')(x)
        x = Flatten()(x)
        '''
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(s_t)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)
        x = Conv2D(32, (3, 3), padding='same', strides=(2, 2))(x)
        x = ELU(alpha=1.0)(x)

        phi = Flatten()(x)
        model = Model(input = s_t, output = phi)

        return model

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

    def create_train_model_inv(self):
        state_encoder.trainable = True

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
