ENV_NAME = 'curiosityChangePosDiscrete-v0'

CONTINUE = False #load a pre-trained model
RESTART_EP = 6000 # the episode number of the pre-trained model

TRAIN = True # train the network
USE_TARGET_NETWORK = False # use the target network
SHOW = True # show the current state, reward and action
MAP = False # show the trajectory in 2d map

MIN_elevation = 20
MAX_elevation = 89.8
# MIN_distance = 600
# MAX_distance = 2000
MIN_distance = 500
MAX_distance = 1500


TF_DEVICE = '/gpu:0'
MAX_EPOCHS = 10000 # max episode number
MEMORY_SIZE = 50000
# LEARN_START_STEP = 10000
# LEARN_START_STEP = 10000
LEARN_START_STEP = 400
# INPUT_SIZE = 84

# LEARN_START_STEP_ICM = 396
# LEARN_START_STEP_ICM = 200
LEARN_START_STEP_ICM = 200
# MAX_EXPLORE_STEPS = 50000
MAX_EXPLORE_STEPS = 21000

INPUT_WIDTH = 160 # pre 100
INPUT_HEIGHT = 120

BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # 1e6
GAMMA = 0.95
# INITIAL_EPSILON = 0.01  # starting value of epsilon
# FINAL_EPSILON = 0.1  # final value of epsilon
INITIAL_EPSILON = .5  # starting value of epsilon
FINAL_EPSILON = 0.01

MAX_EXPLORE_STEPS = 20000
# TEST_INTERVAL_EPOCHS = 100000
TEST_INTERVAL_EPOCHS = 1000
# SAVE_INTERVAL_EPOCHS = 500
SAVE_INTERVAL_EPOCHS = 20

LEARNINGRATE_ICM = 0.0001

LOG_NAME_SAVE = 'log'
MONITOR_DIR = LOG_NAME_SAVE + '/monitor/' #the path to save monitor file
MODEL_DIR = LOG_NAME_SAVE + '/model' # the path to save deep model
PARAM_DIR = LOG_NAME_SAVE + '/param' # the path to save the parameters
TRA_DIR = LOG_NAME_SAVE + '/trajectory.csv' # the path to save trajectory

LOG_NAME_READ = 'log'
#the path to reload weights, monitor and params
weights_path = LOG_NAME_READ + '/model/dqn_ep' + str(RESTART_EP)+ '.h5'
monitor_path = LOG_NAME_READ + '/monit` or/'+ str(RESTART_EP)
params_json = LOG_NAME_READ + '/param/dqn_ep' + str(RESTART_EP) + '.json'

PRETRAINED = True
ENC_SHAPE = (512,)
# ENC_PATH = '/hdd/AIRSCAN/icm_models/vae4_encoder_checkpointsmodel-7.hdf5'
# ENC_PATH = '../state_encoder/encoder-512.hdf5'
ENC_PATH = '/home/daryl/gym-unrealcv/example/ddpg_icm_sfm_combined_opt_rotreward_distTarget_enc/state_encoder/encoder-512.hdf5'
VAE =False
COLOR = True
