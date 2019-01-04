ENV_NAME = 'curiosityChangePos-v0'

CONTINUE = False
RESTART_EP = 300
TRAIN = True
SHOW = True
MAP = False
# MIN_elevation = 25
# MAX_elevation = 65
MIN_elevation = 20
MAX_elevation = 90
MIN_distance = 600
MAX_distance = 2000

TF_DEVICE = '/gpu:0'
MAX_EPOCHS = 10000
MAX_STEPS_PER_EPOCH = 10000
MEMORY_SIZE = 50000
# LEARN_START_STEP = 10000
# LEARN_START_STEP = 500
# LEARN_START_STEP = 1728
LEARN_START_STEP = 1800
# LEARN_START_STEP_ICM = 72
LEARN_START_STEP_ICM = 396
# MAX_EXPLORE_STEPS = 50000
MAX_EXPLORE_STEPS = 21000
# INPUT_SIZE = 84 # pre 100
INPUT_WIDTH = 320 # pre 100
INPUT_HEIGHT = 240 # pre 100

INPUT_WIDTH = 160 # pre 100
INPUT_HEIGHT = 120

BATCH_SIZE = 32
# LEARNINGRATE_CRITIC  = 0.001
LEARNINGRATE_CRITIC  = 0.0001
LEARNINGRATE_ACTOR = 0.0001
# LEARNINGRATE_ACTOR = 0.001
TARGET_UPDATE_RATE = 0.001
# LEARNINGRATE_ICM = 0.001
LEARNINGRATE_ICM = 0.0001
GAMMA = 0.95
# INITIAL_EPSILON = 1  # starting value of epsilon
# INITIAL_EPSILON = 0.9  # starting value of epsilon
INITIAL_EPSILON = 0.01  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon


TEST_INTERVAL_EPOCHS = 1000
# SAVE_INTERVAL_EPOCHS = 500
SAVE_INTERVAL_EPOCHS = 25

LOG_NAME_SAVE = 'log'
MONITOR_DIR = LOG_NAME_SAVE + '/monitor/' #the path to save monitor file
MODEL_DIR = LOG_NAME_SAVE + '/model' # the path to save deep model
PARAM_DIR = LOG_NAME_SAVE + '/param' # the path to save the parameters
TRA_DIR = LOG_NAME_SAVE + '/trajectory.csv' # the path to save trajectory

#the path to reload weights, monitor and params
LOG_NAME_READ = 'log'
critic_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'Critic_model.h5'
actor_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'Actor_model.h5'
state_encoder_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'StateEncoder.h5'
inverse_model_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'InverseModel.h5'
forward_model_weights_path = LOG_NAME_READ + '/model/ep' + str(RESTART_EP) + 'ForwardModel.h5'


monitor_path = LOG_NAME_READ + '/monitor/'+ str(RESTART_EP)
params_json = LOG_NAME_READ + '/param/' + str(RESTART_EP) + '.json'
PRETRAINED = True
ENC_SHAPE = (512,)
# ENC_PATH = '/hdd/AIRSCAN/icm_models/vae4_encoder_checkpointsmodel-7.hdf5'
ENC_PATH = '/home/daryl/gym-unrealcv/example/ddpg_icm_sfm_combined_opt_rotreward_distTarget_enc/state_encoder/encoder-512.hdf5'
VAE =False
COLOR = True
