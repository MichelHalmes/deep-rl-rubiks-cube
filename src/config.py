# RL
EPS_START = 0.95
EPS_END = 0.05
EPS_EXP = 1.3
GAMMA = 0.99

# TRAINING
GRADIENT_CLIP = .05
LEARNING_RATE = 1e-4
MAX_EPISODES = 60000
MAX_STEPS = 5
EVAL_STEPS = 100
DIFFICULTY_STEPS = 1500
DIFFICULTY_EXP = 1.5

# DQN
DQN_MEMORY_MAX_SIZE = 10000
DQN_TARGET_UPDATE = 10
DQN_BATCH_SIZE = 128

# A2C
A2C_NUM_EPISODES = 10

# PPO
PPO_NUM_EPISODES = 7
PPO_NUM_BATCHES = 3
PPO_BATCH_SIZE = 5
PPO_EPS_CLIP = .2

# NETWORK
CONV_NB_KERNELS = [256, 32]  # for 1x1 convolutions
LAYER_SIZES = [512, 512, 512]

# DATA
DATA_DIR = "./data"
MA_ALPHA = .9

