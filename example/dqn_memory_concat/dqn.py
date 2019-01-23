import time
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Flatten, ZeroPadding2D, LSTM, Reshape
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from example.utils import memory
import keras.backend as K
# from constants import *
from icm_model import icm_class
from keras.utils import plot_model

class DeepQ_icm:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, outputs, memorySize, discountFactor, learningRate, img_rows, img_cols, img_channels ,useTargetNetwork, enc_shape=(256,), icm_lr =0.001, pretrained=True, enc_path = '/hdd/AIRSCAN/icm_models/vae4_encoder_checkpointsmodel-7.hdf5', vae =True):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0
        self.enc_shape =enc_shape
        self.icm_lr = icm_lr
        self.img_shape = (self.img_channels, self.img_rows, self.img_cols)
        self.pretrained = pretrained
        self.enc_path = enc_path
        self.vae = vae

        if K.image_dim_ordering() == 'tf':
            self.img_shape = (self.img_rows, self.img_cols, self.img_channels)

        if K.backend() == 'tensorflow':
            # with KTF.tf.device(TF_DEVICE):
                # config = tf.ConfigProto()
                # config.gpu_options.allow_growth = True
                # KTF.set_session(tf.Session(config=config))
            self.initNetworks()
        else :
            self.initNetworks()

    def initNetworks(self):

        self.model = self.createModel()
        if self.useTargetNetwork:
            self.targetModel = self.createModel()
        # self.icm = icm_class(self.img_shape, self.output_size, self.enc_shape,  self.icm_lr, pretrained = self.pretrained, enc_path=self.enc_path, vae=self.vae)

    def get_intrinsic_reward(self,observation, action, obs_new):
        return self.icm.get_intrinsic_reward(observation, action, obs_new)

    def createModel(self):
        input_shape = (self.img_channels, self.img_rows, self.img_cols)
        if K.image_dim_ordering() == 'tf':
            input_shape = ( self.img_rows, self.img_cols, self.img_channels)

        model = Sequential()
        # model.add(Convolution2D(32, 3, 3,border_mode='same', input_shape = input_shape))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        #
        # model.add(Convolution2D(64, 3, 3, border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(Convolution2D(64, 3, 3, border_mode='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        #
        #
        # model.add(Dense(256))
        # # model.add(Reshape((1,-1)))
        # # model.add(LSTM(256, activation='relu'))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(256))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.25))
        # model.add(Dense(self.output_size,activation='linear'))
        # model.compile(Adam(lr=self.learningRate), 'MSE')

        model.add(Convolution2D(32, (8, 8), strides=(4, 4),input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(Dense(nb_actions))
        # model.add(Activation('linear'))
        model.add(Dense(self.output_size,activation='linear'))
        model.compile(Adam(lr=self.learningRate), 'MSE')

        model.summary()
        plot_model(model, to_file='DQN.png', show_shapes=True)
        # print('state_encoder: ')

        return model


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)
        print('update target network')

    # predict Q values for all the actions
    def getQValues(self, state):
        if self.useTargetNetwork:
            predicted = self.targetModel.predict(state)
        else:
            predicted = self.model.predict(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)


    def getMemorySize(self):
        return self.memory.getCurrentSize()


    def learnOnMiniBatch(self, miniBatchSize, icm_only=False):

        self.count_steps += 1

        state_batch,action_batch,reward_batch,newState_batch,isFinal_batch\
        = self.memory.getMiniBatch(miniBatchSize)

        qValues_batch = self.model.predict(np.array(state_batch),batch_size=miniBatchSize)

        isFinal_batch = np.array(isFinal_batch) + 0

        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if self.useTargetNetwork:
            qValuesNewState_batch = self.targetModel.predict_on_batch(np.array(newState_batch))
        else :
            qValuesNewState_batch = self.model.predict_on_batch(np.array(newState_batch))

        Y_sample_batch = reward_batch + (1 - isFinal_batch) * self.discountFactor * np.max(qValuesNewState_batch, axis=1)

        X_batch = np.array(state_batch)
        Y_batch = np.array(qValues_batch)

        action_onehat = np.zeros(qValues_batch.shape)
        for i,action in enumerate(action_batch):
            Y_batch[i][action] = Y_sample_batch[i]
            action_onehat[i][action] = 1

        self.model.fit(X_batch, Y_batch, validation_split=0.0, batch_size = miniBatchSize, nb_epoch=1, verbose = 0)

        # if icm_only == False:
        #     self.model.fit(X_batch, Y_batch, validation_split=0.0, batch_size = miniBatchSize, nb_epoch=1, verbose = 0)
        #     self.icm.train(np.array(state_batch), np.array(action_onehat), np.array(newState_batch), miniBatchSize)
        # else:
        #     self.icm.train(np.array(state_batch), np.array(action_onehat), np.array(newState_batch), miniBatchSize)

        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()


    def saveModel(self, path,state_encoder_weights_path,inverse_model_weights_path,forward_model_weights_path):
        if self.useTargetNetwork:
            self.targetModel.save(path)
        else:
            self.model.save(path)
        self.icm.state_encoder.save_weights(state_encoder_weights_path)
        # self.icm.inverse_model.save_weights(inverse_model_weights_path)
        self.icm.forward_model.save_weights(forward_model_weights_path)

    def loadWeights(self, path,state_encoder_weights_path,inverse_model_weights_path,forward_model_weights_path):
        self.model.load_weights(path)
        if self.useTargetNetwork:
            self.targetModel.load_weights(path)
        self.icm.state_encoder.load_weights(state_encoder_weights_path)
        # self.icm.inverse_model.load_weights(inverse_model_weights_path)
        self.icm.forward_model.load_weights(forward_model_weights_path)



    def feedforward(self,observation,explorationRate):
        qValues = self.getQValues(observation)
        action = self.selectAction(qValues, explorationRate)
        return action, qValues


class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, outputs, memorySize, discountFactor, learningRate, img_rows, img_cols, img_channels ,useTargetNetwork):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.useTargetNetwork = useTargetNetwork
        self.count_steps = 0
        if K.backend() == 'tensorflow':
            # with KTF.tf.device(TF_DEVICE):
                # config = tf.ConfigProto()
                # config.gpu_options.allow_growth = True
                # KTF.set_session(tf.Session(config=config))
            self.initNetworks()
        else :
            self.initNetworks()

    def initNetworks(self):

        self.model = self.createModel()
        if self.useTargetNetwork:
            self.targetModel = self.createModel()

    def createModel(self):
        input_shape = (self.img_channels, self.img_rows, self.img_cols)
        if K.image_dim_ordering() == 'tf':
            input_shape = ( self.img_rows, self.img_cols, self.img_channels)

        model = Sequential()
        model.add(Convolution2D(32, 3, 3,border_mode='same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(LSTM(256))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(self.output_size,activation='linear'))
        model.compile(Adam(lr=self.learningRate), 'MSE')
        model.summary()


        return model


    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)
        print('update target network')

    # predict Q values for all the actions
    def getQValues(self, state):
        if self.useTargetNetwork:
            predicted = self.targetModel.predict(state)
        else:
            predicted = self.model.predict(state)
        return predicted[0]

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
        else :
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)


    def getMemorySize(self):
        return self.memory.getCurrentSize()


    def learnOnMiniBatch(self, miniBatchSize,):

        self.count_steps += 1

        state_batch,action_batch,reward_batch,newState_batch,isFinal_batch\
        = self.memory.getMiniBatch(miniBatchSize)

        qValues_batch = self.model.predict(np.array(state_batch),batch_size=miniBatchSize)

        isFinal_batch = np.array(isFinal_batch) + 0

        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if self.useTargetNetwork:
            qValuesNewState_batch = self.targetModel.predict_on_batch(np.array(newState_batch))
        else :
            qValuesNewState_batch = self.model.predict_on_batch(np.array(newState_batch))

        Y_sample_batch = reward_batch + (1 - isFinal_batch) * self.discountFactor * np.max(qValuesNewState_batch, axis=1)

        X_batch = np.array(state_batch)
        Y_batch = np.array(qValues_batch)

        for i,action in enumerate(action_batch):
            Y_batch[i][action] = Y_sample_batch[i]

        self.model.fit(X_batch, Y_batch, validation_split=0.0, batch_size = miniBatchSize, nb_epoch=1, verbose = 0)

        if self.useTargetNetwork and self.count_steps % 1000 == 0:
            self.updateTargetNetwork()


    def saveModel(self, path):
        if self.useTargetNetwork:
            self.targetModel.save(path)
        else:
            self.model.save(path)

    def loadWeights(self, path):
        self.model.load_weights(path)
        if self.useTargetNetwork:
            self.targetModel.load_weights(path)


    def feedforward(self,observation,explorationRate):
        qValues = self.getQValues(observation)
        action = self.selectAction(qValues, explorationRate)
        return action
