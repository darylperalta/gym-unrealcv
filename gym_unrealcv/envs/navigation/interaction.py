from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
import numpy as np
import time
from gym import spaces
import cv2

class Navigation(UnrealCv):
    def __init__(self, env, cam_id = 0, port = 9000,
                 ip = '127.0.0.1' , targets = None, resolution=(160,120)):

        super(Navigation, self).__init__(env=env, port = port,ip = ip , cam_id=cam_id,resolution=resolution)

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)

        self.img_color = np.zeros(1)
        self.img_depth = np.zeros(1)
        self.img_counter = 0
        from gym_unrealcv.envs.navigation.plot_img_flag import plot_img_flag
        self.plot = plot_img_flag

    def get_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            self.img_color = state = self.read_image(cam_id, 'lit')
        elif observation_type == 'depth':
            self.img_depth = state = self.read_depth(cam_id)
        elif observation_type == 'rgbd':
            self.img_color = self.read_image(cam_id, 'lit')
            self.img_depth = self.read_depth(cam_id)
            state = np.append(self.img_color, self.img_depth, axis=2)
        elif observation_type == 'gray':
            state = self.read_image(cam_id, 'lit')

            if self.plot == True:
                # path = 'log/ob%04d'%self.img_counter
                # path = path+'.png'

                path ='log/frame-{:06}.color.png'.format(self.img_counter)
                cv2.imwrite(path,state)
                self.img_counter += 1
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            # state = cv2.resize(state,(84,84))
            self.img_color = state = cv2.resize(state,(84,84))
            # print('state interaction ', state.shape)
            # self.img_color = state =np.expand_dims(state, axis=2)
            # print(state.shape)

        return state

    def define_observation(self,cam_id, observation_type):
        if observation_type == 'color':
            state = self.read_image(cam_id, 'lit')
            observation_space = spaces.Box(low=0, high=255, shape=state.shape)
        elif observation_type == 'depth':
            state = self.read_depth(cam_id)
            observation_space = spaces.Box(low=0, high=100, shape=state.shape)
        elif observation_type == 'rgbd':
            state = self.get_rgbd(cam_id)
            s_high = state
            s_high[:, :, -1] = 100.0  # max_depth
            s_high[:, :, :-1] = 255  # max_rgb
            s_low = np.zeros(state.shape)
            observation_space = spaces.Box(low=s_low, high=s_high)
        elif observation_type == 'gray':
            state = self.read_image(cam_id, 'lit')
            # print('state shape    ;;s;', state.shape)
            # cv2.imshow('state', state)
            # cv2.waitKey(0)
            state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            # state_reshaped = cv2.resize(state,(224,224))
            state = cv2.resize(state,(84,84))
            # cv2.imshow('state', state)
            # cv2.waitKey(0)
            state =np.expand_dims(state, axis=2)

            # print('state shape: ',state.shape)
            # cv2.imshow('state_reshaped', state)
            # cv2.waitKey(0)

            observation_space = spaces.Box(low=0, high=100, shape=state.shape)

        return observation_space


    def open_door(self):
        self.keyboard('RightMouseButton')
        time.sleep(2)
        self.keyboard('RightMouseButton')  # close the door
#nav = Navigation(env='test')
